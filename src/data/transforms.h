// src/data/transforms.h
#pragma once
#include <torch/torch.h>

#include <vector>

namespace mkd::data
{
    struct Norm
    {
        std::vector<float> mean;
        std::vector<float> stddev;
    };

    // CIFAR-100 自己的均值/方差 —— 不是 ImageNet 的
    Norm cifar100_norm();

    // 训练 pad 4 -> 随机裁32x32 -> 随机水平翻转（每张独立）-> /255 -> 归一化
    // 输入 uint8 (N, 3, 32, 32)，返回 float32 (N, 3, 32, 32)。全程 CPU。
	// DKD的这个输入是32*32,不需要扩成224，消耗小，先放在CPU上做几何变换，最后再送到GPU上。
    torch::Tensor transform_train(torch::Tensor images, const Norm& norm);

    // 验证：/255 -> 归一化。官方的测试集【没有】CenterCrop，一步几何变换都没有。
    torch::Tensor transform_val(torch::Tensor images, const Norm& norm);

}  // namespace mkd::data