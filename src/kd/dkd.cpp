#include "kd/dkd.h"

#include "kd/detail.h"

namespace mkd::kd
{
	LossFn make_dkd(const DkdOpts& opts)
	{
		return [opts](const Batch& b) -> Res
			{
				detail::check_batch(b, opts.T, "dkd_loss");

				const double T = opts.T;
				const double T2 = T * T;

				const auto gt = detail::gt_mask(b.logits_stu, b.target);
				const auto other = detail::other_mask(b.logits_stu, b.target);

				// ---------- TCKD：目标类 vs 其余全体，退化成二分类 ----------
				const auto p_stu = torch::softmax(b.logits_stu / T, 1);
				const auto p_tea = torch::softmax(b.logits_tea / T, 1);

				const auto tckd_raw =
					torch::nn::functional::kl_div(
						torch::log(detail::cat_mask(p_stu, gt, other)),
						detail::cat_mask(p_tea, gt, other),
						torch::nn::functional::KLDivFuncOptions()
							.reduction(torch::kBatchMean)) * T2;

				// ---------- NCKD：非目标类内部的分布 ----------
				//
				// 目标类 logit 减去 1000：让 exp 在 float32 下精确下溢成 0，
				// 数学上等价于「把目标类从归一化里剔除」。
				// ⚠ 不能写 -inf：log_softmax 会给出 -inf，而 kl_div 里 0 * -inf = nan。
				const auto offset = 1000.0 * gt;

				const auto nckd_raw =
					torch::nn::functional::kl_div(
						torch::log_softmax(b.logits_stu / T - offset, 1),
						torch::softmax(b.logits_tea / T - offset, 1),
						torch::nn::functional::KLDivFuncOptions()
							.reduction(torch::kBatchMean)) * T2;

				const auto ce_raw =
					torch::nn::functional::cross_entropy(b.logits_stu, b.target);

				const double w = detail::warmup_scale(b.epoch, opts.warmup);

				Res r;
				r.total = opts.ce_weight * ce_raw
					+ w * (opts.alpha * tckd_raw + opts.beta * nckd_raw);
				r.parts = { {"ce", ce_raw}, {"tckd", tckd_raw}, {"nckd", nckd_raw} };
				r.scale = w;
				return r;
			};
	}
}
