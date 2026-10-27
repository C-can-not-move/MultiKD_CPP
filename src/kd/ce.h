// src/kd/ce.h —— 纯交叉熵
#pragma once

#include "kd/loss.h"

namespace mkd::kd
{
	// 不用教师、不看 epoch —— 就是最普通的分类损失。
	// total = ce_weight · CE
	//
	// ⚠ 它不会调 detail::check_batch：那个函数会检查 logits_tea.dim() == 2，
	//   而 CE 压根没有教师。每个方法只校验自己用得到的原料。
	LossFn make_ce(double ce_weight = 1.0);
}
