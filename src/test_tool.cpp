#include "test_tool.h"

#include "kd/ce.h"
#include "kd/dkd.h"
#include "kd/kd.h"
#include "kd/loss.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>

namespace mkd
{
	namespace
	{
		double comp(const kd::Res& r, const char* name)
		{
			for (const auto& kv : r.parts)
				if (kv.first == name) return kv.second.item<double>();
			TORCH_CHECK(false, "Res 里没有分量 ", name);
			return 0.0;
		}
	}

	void test_tool::losses()
	{
		using namespace kd;

		const int64_t C = 100;
		torch::manual_seed(7);

		const auto z = torch::randn({ 1, C });
		const auto v = torch::randn({ 1, C }) * 1.5;
		const auto y = torch::randint(0, C, { 1 },
			torch::TensorOptions().dtype(torch::kInt64));

		// 原始 KL，作为两个方法共同的对账基准
		const double T = 4.0, T2 = T * T;
		const double kl = (torch::nn::functional::kl_div(
			torch::log_softmax(z / T, 1),
			torch::softmax(v / T, 1),
			torch::nn::functional::KLDivFuncOptions()
				.reduction(torch::kBatchMean)) * T2).item<double>();

		// ① make_ce：就是交叉熵本身
		const double ce_only = make_ce(1.0)(Batch{ z, v, y, 20 }).total.item<double>();
		const double ce_ref = torch::nn::functional::cross_entropy(z, y).item<double>();
		std::cout << "[selfcheck] make_ce  vs  cross_entropy    diff "
			<< std::abs(ce_only - ce_ref) << "   (应 = 0)\n";

		// ② make_kd：kd 项必须正好等于那个 KL
		const double kd_only = make_kd(KdOpts{ T, /*ce_weight=*/0.0, /*kd_weight=*/1.0 })(
			Batch{ z, v, y, /*epoch=*/20 }).total.item<double>();
		std::cout << "[selfcheck] make_kd(ce=0,kd=1)  vs  KL    diff "
			<< std::abs(kd_only - kl) << "   (应 = 0)\n";

		// ③ make_dkd：KL = TCKD + (1 − p_t)·NCKD
		DkdOpts o;                                  // α=1 β=8 T=4 ce=1 warmup=20
		const auto res = make_dkd(o)(Batch{ z, v, y, /*epoch=*/20 });

		const double tckd = comp(res, "tckd");
		const double nckd = comp(res, "nckd");
		const auto yi = y[0].item<int64_t>();
		const double pt = torch::softmax(v / T, 1)[0][yi].item<double>();
		const double rhs = tckd + (1.0 - pt) * nckd;

		std::cout << "[selfcheck] KL " << kl
			<< "    TCKD + (1-pt)·NCKD " << rhs
			<< "    diff " << std::abs(kl - rhs) << "   (应 < 1e-5)\n";

		// ④ warmup 的两端
		std::cout << "[selfcheck] scale  epoch 20 -> " << res.scale
			<< "  (应为 1)    epoch 1 -> "
			<< make_dkd(o)(Batch{ z, v, y, /*epoch=*/1 }).scale
			<< "  (应为 0.05)\n";
	}
}
