// src/MultiKD_C++.cpp —— 冒烟 main
//
// 逐层往上验，每层都有个数字或一条不变量作为判据：
//   ① 模型指纹      62 / 1233540 / 206 / 7433860
//   ② 存档          loaded 206
//   ③ 损失层        make_kd 的 kd 项 == KL；DKD 恒等式；warmup 两端
//   ④ 数据          train 50000 / test 10000
//   ⑤ 训练循环      CE / KD / DKD 各 2 轮（用 2000 / 1000 的子集，几十秒）
//                   其中 CE 传空 teacher —— 验证"没有教师时跳过前向"
//   ⑥ 存档往返      训出来的学生存一次、读回另一个模型、逐张量比
//
// 注意：这里跑的是【子集 + 2 轮】，只验机制，不看准确率。

#include "MultiKD_C++.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace
{
	using torch::Tensor;

	// ---------------- ① 模型指纹 ----------------
	// 数键和元素的部分挪到 model/checkpoint.* 了 —— 它和「文件里到底有什么」
	// 是同一类问题。用法：mkd::model::fingerprint(model).print("名字");

	// ---------------- ③ 损失层 ----------------
	double comp(const mkd::kd::Res& r, const char* name)
	{
		for (const auto& kv : r.parts)
			if (kv.first == name) return kv.second.item<double>();
		TORCH_CHECK(false, "Res 里没有分量 ", name);
		return 0.0;
	}

	void selfcheck_losses()
	{
		using namespace mkd::kd;

		const int64_t C = 100;
		torch::manual_seed(7);

		const auto z = torch::randn({ 1, C });
		const auto v = torch::randn({ 1, C }) * 1.5;
		const auto y = torch::randint(0, C, { 1 },
			torch::TensorOptions().dtype(torch::kInt64));

		// 原始 KL，作为两个方法共同的对账基准。
		// 取 N = 1，这样 batchmean 就等于逐样本值，恒等式是【精确】的。
		const double T = 4.0, T2 = T * T;
		const double kl = (torch::nn::functional::kl_div(
			torch::log_softmax(z / T, 1),
			torch::softmax(v / T, 1),
			torch::nn::functional::KLDivFuncOptions()
				.reduction(torch::kBatchMean)) * T2).item<double>();

		// ① make_ce：就是交叉熵本身
		const double ce_only = make_ce(1.0)(Batch{ z, v, y, 20 }).total.item<double>();
		const double ce_ref = torch::nn::functional::cross_entropy(z, y).item<double>();
		std::cout << "[selfcheck] make_ce  vs  cross_entropy    diff "
			<< std::abs(ce_only - ce_ref) << "   (应 = 0)\n";

		// ② make_kd：kd 项必须正好等于那个 KL
		const double kd_only = make_kd(KdOpts{ T, /*ce_weight=*/0.0, /*kd_weight=*/1.0 })(
			Batch{ z, v, y, /*epoch=*/20 }).total.item<double>();
		std::cout << "[selfcheck] make_kd(ce=0,kd=1)  vs  KL    diff "
			<< std::abs(kd_only - kl) << "   (应 = 0)\n";

		// ③ make_dkd：KL = TCKD + (1 − p_t)·NCKD
		DkdOpts o;                                  // α=1 β=8 T=4 ce=1 warmup=20
		const auto res = make_dkd(o)(Batch{ z, v, y, /*epoch=*/20 });

		const double tckd = comp(res, "tckd");
		const double nckd = comp(res, "nckd");
		const auto yi = y[0].item<int64_t>();
		const double pt = torch::softmax(v / T, 1)[0][yi].item<double>();
		const double rhs = tckd + (1.0 - pt) * nckd;

		std::cout << "[selfcheck] KL " << kl
			<< "    TCKD + (1-pt)·NCKD " << rhs
			<< "    diff " << std::abs(kl - rhs) << "   (应 < 1e-5)\n";

		// ④ warmup 的两端
		std::cout << "[selfcheck] scale  epoch 20 -> " << res.scale
			<< "  (应为 1)    epoch 1 -> "
			<< make_dkd(o)(Batch{ z, v, y, /*epoch=*/1 }).scale
			<< "  (应为 0.05)\n";
	}

	// ---------------- ⑤ 训练循环冒烟 ----------------
	// 每种方法都用【全新的 student 和全新的优化器】——
	// 优化器在构造时抓参数，换一个模型就必须重建它。
	void smoke_train(
		const char* tag,
		const mkd::kd::LossFn& loss_fn,
		std::optional<mkd::model::ResNet> teacher,   // std::nullopt = 没有教师
		const Tensor& tr_x, const Tensor& tr_y,
		const Tensor& va_x, const Tensor& va_y,
		const torch::Device& device)
	{
		std::cout << "\n---- smoke " << tag
			<< (teacher.has_value() ? "  (有教师)" : "  (无教师)") << " ----\n";

		auto student = mkd::model::h_resnet8x4(100);
		student->to(device);

		torch::optim::SGD optimizer(
			student->parameters(),
			torch::optim::SGDOptions(/*lr=*/0.05).momentum(0.9).weight_decay(5e-4));

		mkd::train::TrainConfig cfg;
		cfg.epochs = 2;
		cfg.batch_size = 64;
		cfg.log_every = 0;                  // 不刷分批日志
		cfg.print();

		const auto res = mkd::train::train(
			student, teacher, tr_x, tr_y, va_x, va_y, optimizer, cfg, loss_fn);
		res.print();

		TORCH_CHECK(res.best_val_acc > 0.0, tag, ": 训练跑完了但 best_val_acc 是 0");
	}

	// ---------------- ⑥ 存档往返 ----------------
	// 比较两个模型的部分挪到 model/checkpoint.* 了 —— 它和「文件里的东西对不对」
	// 是同一类问题。用法：mkd::model::compare_models(a, b)
}

int main()
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif
	try {
		const auto cfg = mkd::cfg::defaults();
		std::cout << "[cfg] data_dir = " << cfg.data_dir << "\n";
		std::cout << "[cfg] out_dir  = " << cfg.out_dir << "\n";
		std::cout << "[cfg] device_index = " << cfg.device_index << "\n";

		const auto device = cfg.device_index < 0? torch::Device(torch::kCPU): torch::Device(torch::kCUDA, cfg.device_index);

		torch::manual_seed(cfg.seed);

		// ---- ① 模型指纹：必须和 DKD_C++ 里一模一样 ----
		mkd::model::fingerprint(mkd::model::h_resnet8x4(100)).print("resnet8x4 ");
		//   期望 keys 62   params 1233540   numel 1236302

		auto teacher = mkd::model::h_resnet32x4(100);
		mkd::model::fingerprint(teacher).print("resnet32x4");
		//   期望 keys 206  params 7433860   numel 7443814

		// ---- ② 存档：读那份已经验过三次的教师 ----
		const auto n = mkd::model::load_checkpoint(
			teacher, "D:/DATA/CODE/C++/DKD_C++/tool/resnet32x4_100");
		TORCH_CHECK(n == 206, "教师键数应为 206，实际 ", n);   // 路径由函数自己打
		teacher->to(device);
		teacher->eval();          // 必须由调用者设 —— train 里会校验

		// ---- ③ 损失层自检 ----
		selfcheck_losses();

		// ---- ④ 数据 ----
		const auto tr = mkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/true);
		const auto te = mkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/false);
		std::cout << "[data] train " << tr.images.size(0)
			<< "  test " << te.images.size(0) << "   (期望 50000 / 10000)\n";

		// ---- ⑤ 训练循环冒烟：用子集，几十秒 ----
		const auto tr_x = tr.images.narrow(0, 0, 2000);
		const auto tr_y = tr.labels.narrow(0, 0, 2000);
		const auto va_x = te.images.narrow(0, 0, 1000);
		const auto va_y = te.labels.narrow(0, 0, 1000);

		// CE：不传教师 —— std::nullopt 就是"没有"，应该完全跳过教师前向
		smoke_train("CE", mkd::kd::make_ce(),
			std::nullopt, tr_x, tr_y, va_x, va_y, device);

		smoke_train("KD", mkd::kd::make_kd(mkd::kd::KdOpts{}),
			teacher, tr_x, tr_y, va_x, va_y, device);

		smoke_train("DKD", mkd::kd::make_dkd(mkd::kd::DkdOpts{}),
			teacher, tr_x, tr_y, va_x, va_y, device);

		// ---- ⑥ 存档往返：训一个、存下来、读回另一个模型、逐张量比 ----
		std::cout << "\n---- smoke roundtrip ----\n";
		{
			const auto base = cfg.out_dir + "/smoke-student";

			auto a = mkd::model::h_resnet8x4(100);
			a->to(device);
			torch::optim::SGD opt(a->parameters(),
				torch::optim::SGDOptions(0.05).momentum(0.9).weight_decay(5e-4));

			mkd::train::TrainConfig c2;
			c2.epochs = 1;
			c2.batch_size = 64;
			c2.log_every = 0;
			c2.save_path = base;

			mkd::train::train(a, teacher, tr_x, tr_y, va_x, va_y,
				opt, c2, mkd::kd::make_dkd(mkd::kd::DkdOpts{}));

			auto b = mkd::model::h_resnet8x4(100);
			const auto got = mkd::model::load_checkpoint(b, base);

			const auto bad = mkd::model::compare_models(a, b);
			std::cout << "[roundtrip] loaded " << got << " tensors  不一致 " << bad
				<< " 个   (期望 62 / 0)\n";
			TORCH_CHECK(got == 62 && bad == 0, "存档往返失败");
		}

		std::cout << "\nOK\n";
	}
	catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
	return 0;
}
