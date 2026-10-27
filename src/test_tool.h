// src/test_tool.h —— 自检工具
//
// 放「用数字证明某一层是对的」那类检查。
// 它们不参与训练，但每次改动底层之后都该跑一遍。
#pragma once

namespace mkd
{
	// 用类而不是 namespace，是因为调用点和 namespace 一样短（mkd::test_tool::losses()），
	// 以后要加状态（比如收集"失败了几项"）也不用改调用点。
	class test_tool
	{
	public:
		// 损失层的四条不变量：
		//   ① make_ce == cross_entropy
		//   ② make_kd(ce=0, kd=1) 的 kd 项 == 原始 KL
		//   ③ KL == TCKD + (1 − p_t)·NCKD          （DKD 论文 Sec 3）
		//   ④ warmup 的两端：epoch 20 → 1，epoch 1 → 0.05
		//
		// 取 N = 1，这样 batchmean 就等于逐样本值，③ 是【精确】的（不是近似）。
		// 打印 diff，调用者自己看；任何一条炸了就是 TORCH_CHECK 的事。
		static void losses();
	};
}
