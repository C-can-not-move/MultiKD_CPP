// src/kd/kd.h —— 经典 KD (Hinton et al., 2015)
#pragma once

#include "kd/loss.h"

namespace mkd::kd
{
	// kd_raw = KL(student ‖ teacher) 在温度 T 下，已乘 T²
	// total  = ce_weight · CE + kd_weight · kd_raw
	// 恒等式：total == ce_weight·parts["ce"] + kd_weight·parts["kd"]
	//
	// 默认值是 mdistiller 的 CIFAR-100 配置。
	// ⚠ 和 DKD 的关键差别：DKD 的 ce_weight 是 1.0，KD 是 0.1。别混。
	//
	// ⚠ 没有 warmup —— mdistiller 的 KD.py 里也没有，这是【有意】的空白，
	//   不是漏了。KD 一开始就整份蒸馏，DKD 才需要先让 CE 说话。
	struct KdOpts
	{
		double T = 4.0;
		double ce_weight = 0.1;
		double kd_weight = 0.9;
	};

	LossFn make_kd(const KdOpts& opts = {});
}
