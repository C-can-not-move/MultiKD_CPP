// src/kd/loss.h —— 蒸馏方法的【通用接口】，不含任何具体方法
//
//   Res     一次计算的产出
//   LossFn  方法的签名
//
// 具体方法在 kd/kd.h、kd/dkd.h ……。
// 加一个方法 = 加一对 .h/.cpp + 在 main 里加一个 case，【本文件不动】。
#pragma once

#include "train/batch.h"

#include <torch/torch.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mkd::kd
{
	// Batch 定义在 train/batch.h（训练器组装它，生产者定义契约）。
	using train::Batch;

	struct Res
	{
		torch::Tensor total;

		// 原始分量，供打印。不含 ce_weight / α / β / warmup ——
		// 这样日志里看到的是 KL 的真实量级。权重只出现在 total 里。
		// ⚠ 名字是运行期字符串：拼错编译期查不出来，取的时候请用辅助函数 + TORCH_CHECK。
		// ⚠ 顺序不保证：别按下标取。
		std::vector<std::pair<std::string, torch::Tensor>> parts;

		// 蒸馏项这一轮的实际缩放（warmup 之类）。没有就是 1.0。
		double scale = 1.0;
	};

	// 每个方法就是一个工厂：返回闭包住超参的 LossFn。
	// 参数是 const Batch& —— 以后往 Batch 加字段，所有方法的签名都不用改。
	using LossFn = std::function<Res(const Batch&)>;
}
