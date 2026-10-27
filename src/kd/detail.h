// src/kd/detail.h —— 各方法共用的实现细节
//
// ⚠ 内部头，【不进伞头】，只给 kd/*.cpp 用。外面不该 include 它。
//   全部 inline，不产生额外的翻译单元。
#pragma once

#include "kd/loss.h"

#include <torch/torch.h>

#include <algorithm>

namespace mkd::kd::detail
{
	inline void check_batch(const Batch& b, double T, const char* tag)
	{
		TORCH_CHECK(b.logits_stu.dim() == 2 && b.logits_tea.dim() == 2,
			tag, ": logits 必须是 (N, C)");
		TORCH_CHECK(b.logits_stu.sizes().vec() == b.logits_tea.sizes().vec(),
			tag, ": 师生 logits 形状不一致 ",
			b.logits_stu.sizes(), " vs ", b.logits_tea.sizes());
		TORCH_CHECK(b.target.scalar_type() == torch::kInt64,
			tag, ": target 必须是 int64");
		TORCH_CHECK(b.target.size(0) == b.logits_stu.size(0),
			tag, ": 样本数不一致");
		TORCH_CHECK(b.logits_stu.size(0) > 0, tag, ": batch 不能为 0");
		TORCH_CHECK(T > 0.0, tag, ": T 必须 > 0");
	}

	// 显式用 float 掩码而不是 bool：后面要拿它做算术（t * mask、logits - 1000*mask）。
	inline torch::Tensor gt_mask(const torch::Tensor& logits, const torch::Tensor& target)
	{
		return torch::zeros_like(logits).scatter(1, target.reshape({ -1, 1 }), 1);
	}

	inline torch::Tensor other_mask(const torch::Tensor& logits, const torch::Tensor& target)
	{
		return torch::ones_like(logits).scatter(1, target.reshape({ -1, 1 }), 0);
	}

	// 按两个掩码求和，把 (N, C) 压成 (N, 2)：
	//   第 0 列 = 目标类的概率，第 1 列 = 其余所有类的概率和
	inline torch::Tensor cat_mask(const torch::Tensor& t,
		const torch::Tensor& m1, const torch::Tensor& m2)
	{
		return torch::cat({ (t * m1).sum(1, /*keepdim=*/true),
							(t * m2).sum(1, /*keepdim=*/true) }, 1);
	}

	inline double warmup_scale(int64_t epoch, int64_t warmup)
	{
		if (warmup <= 0) return 1.0;
		// ⚠ warmup 是 int64_t，不做 double 转换会走整数除法，全程得 0
		return std::min(static_cast<double>(epoch) / static_cast<double>(warmup), 1.0);
	}
}
