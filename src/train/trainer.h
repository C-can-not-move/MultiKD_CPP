// src/train/trainer.h —— 通用训练循环
//
// 这个文件不认识任何具体方法。它只认 kd::LossFn。
// 换方法 = 换一个工厂函数，本文件不动。
#pragma once

#include "kd/loss.h"
#include "model/resnet.h"
#include "train/batch.h"

#include <torch/torch.h>
#include <torch/optim.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace mkd::train
{
	struct TrainConfig
	{
		int64_t epochs = 20;           // 官方是 240
		int64_t batch_size = 64;       // mdistiller 的 CIFAR-100 配置
		bool    drop_last = false;
		std::vector<int64_t> lr_decay_stages;   // 例如 {150, 180, 210}；空 = 不衰减
		double               lr_decay_rate = 0.1;
		int64_t log_every = 50;
		std::optional<std::string> save_path;   // 传 base，不带后缀

		void print() const
		{
			std::cout << "[train-cfg] epochs     = " << epochs << '\n'
				<< "[train-cfg] batch_size = " << batch_size << '\n'
				<< "[train-cfg] drop_last  = " << (drop_last ? "true" : "false") << '\n'
				<< "[train-cfg] log_every  = " << log_every << '\n'
				<< "[train-cfg] save_path  = " << (save_path ? save_path->c_str() : "(none)") << '\n';
			std::cout << "[train-cfg] lr_decay   = ";
			if (lr_decay_stages.empty()) { std::cout << "(off)\n"; }
			else
			{
				for (auto s : lr_decay_stages) std::cout << s << ' ';
				std::cout << " × " << lr_decay_rate << "\n";
			}
		}
	};

	struct EvalResult
	{
		int64_t correct = 0;
		int64_t total = 0;

		double acc() const
		{
			return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
		}
	};

	struct TrainResult
	{
		double  best_val_acc = 0.0;
		int64_t best_epoch = 0;
		double  final_val_acc = 0.0;
		double  last_train_loss = 0.0;

		void print() const
		{
			std::cout << "best_val_acc = " << best_val_acc
				<< "  (epoch " << best_epoch << ")"
				<< "   final_val_acc = " << final_val_acc
				<< "   last_train_loss = " << last_train_loss << '\n';
		}
	};

	// 评估准确率，没有训练循环。内部自己 eval() 并恢复原模式。
	EvalResult evaluate(
		model::ResNet model,
		const torch::Tensor& images,   // uint8 (N, 3, 32, 32)，CPU
		const torch::Tensor& labels,   // int64 (N)，CPU
		const TrainConfig& cfg);

	// 训练。名称里没有 "distill"，因为循环本身不知道有没有教师 —— 全靠 loss_fn。
	//
	//   kd::make_ce   → 纯交叉熵，teacher 传 std::nullopt，不会白跑教师前向
	//   kd::make_kd   → 经典 KD
	//   kd::make_dkd  → DKD
	//
	// teacher 是 std::optional：
	//   std::nullopt   → 跳过教师前向（省掉约 2/3 的每步算力）
	//   有值           → 要求 ① 已经 eval()  ② 和 student 同设备
	//
	// 为什么不用 ModuleHolder 的空状态（= nullptr）：
	//   ModuleHolder 没有默认构造函数，且不能和 nullptr 比较 ——
	//   std::optional 本身就表达"可以没有"，签名上也一眼看得出来。
	//
	// loss_fn 内部已经含 CE —— 循环里【不要】再单独算一次交叉熵。
	TrainResult train(
		model::ResNet student,
		std::optional<model::ResNet> teacher,   // std::nullopt = 没有教师
		const torch::Tensor& train_images,
		const torch::Tensor& train_labels,
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer,
		const TrainConfig& cfg,
		const kd::LossFn& loss_fn);
}
