// src/train/batch.h —— 训练循环交给损失函数的一份「原料」
//
// 归属：放在 train 侧，因为【生产者定义契约】—— 是训练器组装它，
//       损失函数只是消费者。
//
// ⚠ 这是个叶子头：它不 include 任何项目内的东西，所以
//   kd/loss.h  -> train/batch.h 这一条单向依赖不会形成循环。
#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace mkd::train
{
	// ⚠ 按 const& 传：以后加特征蒸馏的 taps，只改这里，
	//   所有损失函数的签名都不用动。
	struct Batch
	{
		torch::Tensor logits_stu;   // (N, C) 未经 softmax
		torch::Tensor logits_tea;   // (N, C) 未经 softmax
		torch::Tensor target;       // (N,)   int64
		int64_t       epoch = 1;    // 1-based —— warmup 依赖它，别传 0-based
	};
}
