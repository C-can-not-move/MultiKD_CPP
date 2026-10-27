// src/MultiKD_C++.cpp —— 环境体检 + CE 经典训练流程
//
// 前半段逐层确认底层是对的，每层一个数字或一条不变量：
//   ① 模型指纹   62 / 1233540 / 1236302   和  206 / 7433860 / 7443814
//   ② 存档       读那份已经验过三次的教师，键数必须是 206
//   ③ 损失层     test_tool::losses() —— 四条不变量
//   ④ 数据       train 50000 / test 10000
//
// 后半段是【经典训练流程】：CE + 切一份小数据，几十秒跑完。
// 换成 KD / DKD 只改最后一行那个 make_ce(...)，别的不用动。

#include "MultiKD_C++.h"

#ifdef _WIN32
#include <windows.h>
#endif

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

		const auto device = cfg.device_index < 0
			? torch::Device(torch::kCPU)
			: torch::Device(torch::kCUDA, cfg.device_index);

		torch::manual_seed(cfg.seed);   // 必须在所有随机消费之前（模型初始化会用它）

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

		// ---- ③ 损失层自检 ----
		mkd::test_tool::losses();

		// ---- ④ 数据 ----
		const auto tr = mkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/true);
		const auto te = mkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/false);
		std::cout << "[data] train " << tr.images.size(0)
			<< "  test " << te.images.size(0) << "   (期望 50000 / 10000)\n";

		// ---- ⑤ 经典训练流程：CE + 小数据 ----
		// 全量 240 轮太慢，这里切 2000 / 1000 只验流程。
		auto student = mkd::model::h_resnet8x4(100);// ← 必须在建优化器之前，否则优化器抓的是 CPU 上那批
		student->to(device);                 

		torch::optim::SGD optimizer(
			student->parameters(),
			torch::optim::SGDOptions(/*lr=*/0.05).momentum(0.9).weight_decay(5e-4));

		const auto tr_x = tr.images.narrow(0, 0, 2000);
		const auto tr_y = tr.labels.narrow(0, 0, 2000);
		const auto va_x = te.images.narrow(0, 0, 1000);
		const auto va_y = te.labels.narrow(0, 0, 1000);
		std::cout << "[subset] train " << tr_x.size(0) << "  val " << va_x.size(0) << "\n";

		mkd::train::TrainConfig tcfg;
		tcfg.epochs = 2;
		tcfg.batch_size = 64;
		tcfg.log_every = 0;
		tcfg.save_path = cfg.out_dir + "/ce-subset";
		tcfg.print();

		// CE 不需要教师 —— 传 std::nullopt，train 里会整个跳过教师前向
		const auto res = mkd::train::train(
			student, std::nullopt,
			tr_x, tr_y, va_x, va_y,
			optimizer, tcfg, mkd::kd::make_ce());
		res.print();

		std::cout << "\nOK\n";
	}
	catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
	return 0;
}
