#include "train/trainer.h"

#include "data/transforms.h"
#include "model/checkpoint.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace mkd::train
{
	namespace
	{
		// 优化器记的是【构造那一刻】的 TensorImpl，之后没人维护这条边。
		// 破坏了不会报错，只会静默地训一份没人读的参数 —— 所以每次调用都对一次账。
		void verify_optimizer_binding(
			const model::ResNet& m,
			const torch::optim::Optimizer& optimizer,
			const char* tag)
		{
			std::unordered_set<const void*> owned;
			for (const auto& g : optimizer.param_groups())
				for (const auto& t : g.params())
					owned.insert(t.data_ptr());

			size_t n = 0;
			for (const auto& p : m->parameters())
			{
				TORCH_CHECK(owned.count(p.data_ptr()) != 0,
					tag, ": 优化器和模型的参数不配套 —— 模型那一侧变了，优化器还抓着旧的。"
					"要在模型构造完成（含 to(device)）之后再建优化器。");
				++n;
			}
			TORCH_CHECK(n == owned.size(),
				tag, ": 参数个数不一致（模型 ", n, " vs 优化器 ", owned.size(),
				"）—— 有新层没加进优化器？");
		}
	}  // namespace


	TrainResult train(
		model::ResNet student,
		std::optional<model::ResNet> teacher,
		const torch::Tensor& train_images,
		const torch::Tensor& train_labels,
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer,
		const TrainConfig& cfg,
		const kd::LossFn& loss_fn)
	{
		using namespace mkd::data;

		const auto device = student->parameters().front().device();
		const bool has_teacher = teacher.has_value();

		TORCH_CHECK(train_images.dim() == 4, "train: expect (N, C, H, W)");
		TORCH_CHECK(train_images.size(0) == train_labels.size(0), "train: images / labels count mismatch");
		TORCH_CHECK(train_images.scalar_type() == torch::kUInt8, "train: expect uint8 images, got ", train_images.scalar_type());
		TORCH_CHECK(train_labels.scalar_type() == torch::kInt64, "train: expect int64 labels");
		TORCH_CHECK(cfg.batch_size > 0, "train: batch_size must be > 0");
		TORCH_CHECK(train_images.device().is_cpu(), "train: 数据必须留在 CPU —— 变换在 CPU 上做，每批只搬结果到 device");

		// 空 std::function 调用会抛 bad_function_call，那种报错看不出是谁的问题
		TORCH_CHECK(loss_fn, "train: loss_fn 为空");

		// ---- teacher：有才查。有教师时这两条是硬约束，由调用者设 ----
		if (has_teacher)
		{
			// ⚠ teacher 是 optional<ResNet>。`teacher->X` 只能走到 ModuleHolder*，
			//   裸指针的 -> 不再递归 —— 所以先取出引用，让 holder 的 operator-> 接手。
			const model::ResNet& tea = *teacher;

			TORCH_CHECK(!tea->is_training(),"train: teacher 必须是 eval() 模式（BN 用 running stats）—— 由调用者设置");
			TORCH_CHECK(tea->parameters().front().device() == device,"train: teacher 在 ", tea->parameters().front().device()," 而 student 在 ", device);
		}

		verify_optimizer_binding(student, optimizer, "train");

		const auto norm = cifar100_norm();

		TrainResult res;
		const int64_t N = train_images.size(0);

		for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch)
		{
			// 到指定轮就把 lr 乘一次 decay_rate
			if (std::find(cfg.lr_decay_stages.begin(), cfg.lr_decay_stages.end(), epoch)
				!= cfg.lr_decay_stages.end())
			{
				for (auto& g : optimizer.param_groups())
					g.options().set_lr(g.options().get_lr() * cfg.lr_decay_rate);
			}

			// student 要用批统计 ⇒ train()；teacher 的模式由调用者保证，这里不碰
			student->train();

			const auto perm = torch::randperm(N, torch::kInt64);   // 洗牌就是这一行

			double  run_total = 0.0;
			double  last_scale = 1.0;
			int64_t nb = 0;

			// 分量累加器：按【方法自己报的顺序】保存，而不是 map 的字典序，
			// 这样日志里的分量顺序和方法里写的顺序一致。
			std::vector<std::pair<std::string, double>> acc;
			auto add_acc = [&acc](const std::string& name, double v)
				{
					for (auto& kv : acc)
						if (kv.first == name) { kv.second += v; return; }
					acc.emplace_back(name, v);
				};

			for (int64_t start = 0; start < N; start += cfg.batch_size)
			{
				const int64_t len = std::min(cfg.batch_size, N - start);
				if (cfg.drop_last && len < cfg.batch_size) break;

				const auto idx = perm.narrow(0, start, len);        // CPU
				const auto xb = train_images.index_select(0, idx);  // CPU uint8
				const auto yb = train_labels.index_select(0, idx).to(device);

				// 变换全在 CPU 上做完，只搬这一次
				const auto xf = transform_train(xb, norm).to(device);

				// ⭐ teacher 前向单独包一层：构造 → 关梯度，析构 → 恢复
				//   scope 必须收窄 —— student 的 forward / backward 在外面，否则图被丢掉
				//   没有教师时这一步整个跳过：不白烧 2/3 的算力
				torch::Tensor logits_tea;
				if (has_teacher)
				{
					torch::NoGradGuard ng;
					logits_tea = (*teacher)->forward(xf);
				}

				optimizer.zero_grad();

				// ce / 权重 / warmup 全在 loss_fn 内部，这里不要再算交叉熵
				const auto r = loss_fn(Batch{ student->forward(xf), logits_tea, yb, epoch });

				r.total.backward();
				optimizer.step();

				run_total += r.total.item<double>();
				for (const auto& kv : r.parts) add_acc(kv.first, kv.second.item<double>());
				last_scale = r.scale;
				++nb;

				if (cfg.log_every > 0 && (nb % cfg.log_every) == 0)
				{
					std::cout << "  epoch " << epoch << "  batch " << nb
						<< "  avg total " << (run_total / static_cast<double>(nb)) << "\n";
				}
			}

			const double inv_nb = (nb > 0) ? 1.0 / static_cast<double>(nb) : 0.0;
			res.last_train_loss = run_total * inv_nb;

			// 验证：BN 用 running stats ⇒ eval()；不建图 ⇒ NoGradGuard
			const EvalResult eval_result = evaluate(student, val_images, val_labels, cfg);
			const double acc_val = eval_result.acc();

			res.final_val_acc = acc_val;
			if (acc_val > res.best_val_acc)
			{
				res.best_val_acc = acc_val;
				res.best_epoch = epoch;

				if (cfg.save_path)
				{
					model::save_checkpoint(student, *cfg.save_path);
					std::cout << "  → best so far, saved -> " << *cfg.save_path << ".{txt,bin}\n";
				}
			}

			std::cout << "[epoch " << epoch << "/" << cfg.epochs << "]"
				<< "   lr " << optimizer.param_groups().front().options().get_lr();
			for (const auto& kv : acc)
				std::cout << "   " << kv.first << ' ' << kv.second * inv_nb;
			std::cout << "   w " << last_scale
				<< "   total " << res.last_train_loss
				<< "   acc " << acc_val
				<< "   (" << eval_result.correct << "/" << eval_result.total << ")\n";
		}

		return res;
	}


	EvalResult evaluate(
		model::ResNet model,
		const torch::Tensor& images,
		const torch::Tensor& labels,
		const TrainConfig& cfg)
	{
		TORCH_CHECK(cfg.batch_size > 0, "evaluate: batch_size must be > 0");
		TORCH_CHECK(images.size(0) == labels.size(0), "evaluate: images / labels count mismatch");
		TORCH_CHECK(images.device().is_cpu(), "evaluate: 数据必须留在 CPU");

		const auto device = model->parameters().front().device();
		const auto norm = data::cifar100_norm();

		torch::NoGradGuard no_grad;
		const bool was_training = model->is_training();
		model->eval();                                    // 会改变共享 impl 的模式

		int64_t correct = 0, total = 0;
		for (int64_t s = 0; s < images.size(0); s += cfg.batch_size)
		{
			const int64_t len = std::min(cfg.batch_size, images.size(0) - s);
			const auto xb = images.narrow(0, s, len);
			const auto yb = labels.narrow(0, s, len).to(device);
			const auto xf = data::transform_val(xb, norm).to(device);

			const auto pred = model->forward(xf).argmax(1);
			correct += pred.eq(yb).sum().item<int64_t>();
			total += len;
		}

		if (was_training) model->train();                 // 恢复调用者进来时的状态

		EvalResult r;
		r.correct = correct;
		r.total = total;
		return r;
	}
}
