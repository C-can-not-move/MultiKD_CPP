#include "data/transforms.h"

#include <cstdint>
#include <vector>

namespace mkd::data
{
    // 三个数与三个标准差是 CIFAR - 100 训练集（50000 张）的逐通道统计量
	// 这个项目会加载在CIFAR-100训练集上预训练的模型，所以要用CIFAR-100的均值和方差来归一化。
    Norm cifar100_norm()
    {
        return Norm{
            { 0.5071f, 0.4867f, 0.4408f },
            { 0.2675f, 0.2565f, 0.2761f }
        };
    }

    namespace
    {
        constexpr int64_t kSize = 32;
        constexpr int64_t kPad = 4;

        torch::Tensor normalize(torch::Tensor x, const Norm& norm)
        {
            const int64_t C = x.size(1);
            TORCH_CHECK(static_cast<int64_t>(norm.mean.size()) == C,"normalize: mean 长度 ", norm.mean.size(), " 与通道数 ", C, " 不符");

            const auto mean = torch::tensor(norm.mean).view({ 1, C, 1, 1 });
            const auto stddev = torch::tensor(norm.stddev).view({ 1, C, 1, 1 });
            return (x - mean) / stddev;
        }
    }

    torch::Tensor transform_train(torch::Tensor images, const Norm& norm)
    {
        TORCH_CHECK(images.dim() == 4&& images.size(2) == kSize && images.size(3) == kSize,"transform_train: expect (N, C, 32, 32)");
        TORCH_CHECK(images.device().is_cpu(), "transform_train: 只做 CPU");

        const int64_t N = images.size(0);
        if (N == 0) return images;

        // 先转 float，再做几何变换。
        // 这样 pad 填的是像素0= 黑；
        // 如果先归一化再 pad 0，填进去的是 -(mean/std) ≈ -1.9（浅灰），而且不报错。
        auto x = images.to(torch::kFloat32);

        // 四周各 pad 4：32x32 -> 40x40
        x = torch::nn::functional::pad(
            x,
            torch::nn::functional::PadFuncOptions({ kPad, kPad, kPad, kPad }).mode(torch::kConstant).value(0.0));

        // 每张图【独立】取样偏移，再裁回 32x32
        const int64_t span = 2 * kPad + 1;                  // 0..8，共 9 个位置
        const auto off_y = torch::randint(0, span, { N }, torch::kInt64);
        const auto off_x = torch::randint(0, span, { N }, torch::kInt64);
        const int64_t* py = off_y.data_ptr<int64_t>();
        const int64_t* px = off_x.data_ptr<int64_t>();

        std::vector<torch::Tensor> pieces;
        pieces.reserve(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i)
        {
            // x[i] 是 (C, 40, 40)；先切 H 再切 W
            pieces.push_back(x[i].slice(1, py[i], py[i] + kSize).slice(2, px[i], px[i] + kSize));
        }
        auto out = torch::stack(pieces, 0);

        // 每张图独立决定是否翻转，p = 0.5
        const auto flip_mask = torch::rand({ N, 1, 1, 1 }).lt(0.5);
        out = torch::where(flip_mask, out.flip(3), out);

        out = out / 255.0;
        return normalize(out, norm);
    }

    torch::Tensor transform_val(torch::Tensor images, const Norm& norm)
    {
        TORCH_CHECK(images.dim() == 4&& images.size(2) == kSize && images.size(3) == kSize,"transform_val: expect (N, C, 32, 32)");

        return normalize(images.to(torch::kFloat32) / 255.0, norm);
    }

}  // namespace mkd::data