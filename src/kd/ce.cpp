#include "kd/ce.h"

namespace mkd::kd
{
	LossFn make_ce(double ce_weight)
	{
		return [ce_weight](const Batch& b) -> Res
			{
				// 只校验自己用得到的原料
				TORCH_CHECK(b.logits_stu.dim() == 2, "ce_loss: logits_stu 必须是 (N, C)");
				TORCH_CHECK(b.target.scalar_type() == torch::kInt64, "ce_loss: target 必须是 int64");
				TORCH_CHECK(b.target.size(0) == b.logits_stu.size(0), "ce_loss: 样本数不一致");
				TORCH_CHECK(b.logits_stu.size(0) > 0, "ce_loss: batch 不能为 0");

				// cross_entropy 内部就是 log_softmax + nll，所以直接传原始 logits
				const auto ce_raw =
					torch::nn::functional::cross_entropy(b.logits_stu, b.target);

				Res r;
				r.total = ce_weight * ce_raw;
				r.parts = { {"ce", ce_raw} };
				r.scale = 1.0;
				return r;
			};
	}
}
