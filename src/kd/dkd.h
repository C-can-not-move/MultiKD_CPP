// src/kd/dkd.h —— Decoupled Knowledge Distillation (CVPR 2022)
#pragma once

#include "kd/loss.h"

#include <cstdint>

namespace mkd::kd
{
	// 恒等式（论文 Sec 3）：KL_原始 = TCKD + (1 − p_t^T) · NCKD
	// 其中 p_t^T 是教师在温度 T 下给目标类的概率。
	//
	// total = ce_weight·CE + w·(α·TCKD + β·NCKD)，w = min(epoch/warmup, 1)
	// 恒等式：total == ce_weight·parts["ce"] + scale·(α·parts["tckd"] + β·parts["nckd"])
	//
	// TCKD / NCKD 都已乘 T²，但都【没有】乘 α / β。
	struct DkdOpts
	{
		double  alpha = 1.0;
		double  beta = 8.0;
		double  T = 4.0;
		double  ce_weight = 1.0;   // 恒为 1.0 —— 这是 DKD 相对经典 KD 的关键差别
		int64_t warmup = 20;       // 先让 CE 说话，再逐渐把教师的声音放大
	};

	LossFn make_dkd(const DkdOpts& opts = {});
}
