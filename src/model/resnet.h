// src/model/resnet.h
#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace mkd::model
{
    // BasicBlock — mdistiller models/cifar/resnet.py BasicBlock，expansion == 1
    // CIFAR版的resnet：卷积全是 3x3，第一个 conv 走 stride=stride。
    // 官方代码relu是inplace=True。
    class BasicBlockImpl : public torch::nn::Module
    {
    public:
        static constexpr int64_t kExpansion = 1;

        BasicBlockImpl(
            int64_t in_channel,
            int64_t out_channel,
            int64_t stride = 1,
            torch::nn::Sequential downsample = nullptr);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d      conv1{ nullptr };   // 3x3, stride=stride, pad=1, no bias
        torch::nn::BatchNorm2d bn1{ nullptr };
        torch::nn::ReLU        relu{ nullptr };    // inplace = true
        torch::nn::Conv2d      conv2{ nullptr };   // 3x3, stride=1,      pad=1, no bias
        torch::nn::BatchNorm2d bn2{ nullptr };
        torch::nn::Sequential  downsample{ nullptr };
    };
    TORCH_MODULE(BasicBlock);


    // ------------------------------------------------------------------------
    // ResNet — mdistiller models/cifar/resnet.py
    // DKD 只要 logits ⇒ forward 直接返回 Tensor。
    // ------------------------------------------------------------------------
    class ResNetImpl : public torch::nn::Module
    {
    public:
        /* n = (depth - 2) / 6
         *   resnet8x4 : depth 8 -> n = 1
         * 三个 stage 用同一个 n；宽度固定 [32, 64, 128, 256]。
         */
        explicit ResNetImpl(int64_t n, int64_t num_classes = 100);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d      conv1{ nullptr };   // 3x3, stride 1, pad 1 —— 没有 maxpool
        torch::nn::BatchNorm2d bn1{ nullptr };
        torch::nn::ReLU        relu{ nullptr };
        torch::nn::Sequential  layer1{ nullptr };  // 32  ->  64, stride 1
        torch::nn::Sequential  layer2{ nullptr };  // 64  -> 128, stride 2
        torch::nn::Sequential  layer3{ nullptr };  // 128 -> 256, stride 2
        torch::nn::AvgPool2d   avgpool{ nullptr }; // kernel 8，写死 —— 只吃 32x32
        torch::nn::Linear      fc{ nullptr };
    };
    TORCH_MODULE(ResNet);

    // 学生
    ResNet h_resnet8x4(int64_t num_classes = 100);

	// 备用的老师，项目准备走python预训练->C++加载模型的路线，暂时不需要。
    ResNet h_resnet32x4(int64_t num_classes = 100);
}  // namespace mkd::model