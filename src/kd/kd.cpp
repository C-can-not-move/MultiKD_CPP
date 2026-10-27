#include "kd/kd.h"

#include "kd/detail.h"

namespace mkd::kd
{
	LossFn make_kd(const KdOpts& opts)
	{
		return [opts](const Batch& b) -> Res
			{
				detail::check_batch(b, opts.T, "kd_loss");

				const double T = opts.T;
				const double T2 = T * T;

				// 软目标 KL。学生走 log 域、教师走概率域，方向不能反：
				//   KL(p_tea ‖ p_stu) = Σ p·(log p − log q)
				const auto kd_raw = torch::nn::functional::kl_div(
					torch::log_softmax(b.logits_stu / T, 1),
					torch::softmax(b.logits_tea / T, 1),
					torch::nn::functional::KLDivFuncOptions()
					.reduction(torch::kBatchMean)) * T2;

				// 硬标签 CE —— 用【原始】logits，T = 1。
				// cross_entropy 内部就是 log_softmax + nll，所以别传 z/T。
				const auto ce_raw =
					torch::nn::functional::cross_entropy(b.logits_stu, b.target);

				Res r;
				r.total = opts.ce_weight * ce_raw + opts.kd_weight * kd_raw;
				r.parts = { { "ce", ce_raw },{ "kd", kd_raw } };
				r.scale = 1.0;      // KD 没有 warmup，恒为 1
				return r;
			};
	}
}
