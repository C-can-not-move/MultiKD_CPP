#include "model/resnet.h"

namespace mkd::model
{

    // ------------------------------------------------------------------------
    // make_layer — mdistiller models/cifar/resnet.py _make_layer
    // ------------------------------------------------------------------------
    torch::nn::Sequential make_layer(
        int64_t& in_channel,
        int64_t channel,
        int64_t block_num,
        int64_t stride)
    {
        const int64_t out_channel = channel * BasicBlockImpl::kExpansion;

        // 通道变了也要 downsample —— 不只是 stride != 1。
        // 所以 layer1（stride=1，32→64）同样带 downsample。
        torch::nn::Sequential downsample{ nullptr };
        if (stride != 1 || in_channel != out_channel)
        {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, /*kernel_size=*/1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channel));
        }

        torch::nn::Sequential layers;
        // push_back 自动命名 "0","1","2"。所以键会是 layer1.0.conv1.weight 这种形式。
        layers->push_back(BasicBlock(in_channel, channel, stride, downsample));
        in_channel = out_channel;

        for (int64_t i = 1; i < block_num; ++i)
        {
            layers->push_back(BasicBlock(in_channel, channel));
        }

        return layers;
    }


    // ------------------------------------------------------------------------
    // BasicBlock
    // ------------------------------------------------------------------------
    BasicBlockImpl::BasicBlockImpl(
        int64_t in_channel,
        int64_t out_channel,
        int64_t stride,
        torch::nn::Sequential downsample)
    {
        this->conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channel, out_channel, /*kernel_size=*/3)
            .stride(stride).padding(1).bias(false)));
        this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channel));
        this->relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        this->conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channel, out_channel, /*kernel_size=*/3)
            .padding(1).bias(false)));
        this->bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channel));

        if (downsample)
        {
            this->downsample = register_module("downsample", downsample);
        }
    }

    torch::Tensor BasicBlockImpl::forward(torch::Tensor x)
    {
        auto identity = x;
        if (downsample)
        {
            identity = downsample->forward(x);
        }

        // 残差相加故意用 'out + identity' 而不是 'out += identity'。避开原地操作
        auto out = relu->forward(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));
        out = out + identity;
        out = relu->forward(out);
        return out;
    }


    // ------------------------------------------------------------------------
    // ResNet
    // ------------------------------------------------------------------------
    ResNetImpl::ResNetImpl(int64_t n, int64_t num_classes)
    {
        TORCH_CHECK(n >= 1, "n must be >= 1  (n = (depth - 2) / 6)");

        this->conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(/*in=*/3, /*out=*/32, /*kernel_size=*/3)
            .padding(1).bias(false)));
        this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        this->relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

        // 通道数沿这条链往下传：32 → 64 → 128 → 256
        int64_t in_channel = 32;
        this->layer1 = register_module("layer1", make_layer(in_channel, 64, n, /*stride=*/1));
        this->layer2 = register_module("layer2", make_layer(in_channel, 128, n, /*stride=*/2));
        this->layer3 = register_module("layer3", make_layer(in_channel, 256, n, /*stride=*/2));

        // kernel 8 是写死的：32 -> 32 -> 16 -> 8，正好压成 1x1。
        // 喂 224 会得到 7x7，flatten 之后 256*49，fc形状不匹配。
        this->avgpool = register_module("avgpool", torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions(/*kernel_size=*/8)));

        this->fc = register_module("fc", torch::nn::Linear(256, num_classes));

        // named_parameters() 递归进已注册的子模块(register_module)
        // 凡是 4 维的参数，用 kaiming_normal 原地填一遍 —— 这里就是 conv。
        // BN 的 weight=1 / bias=0 是 PyTorch 默认，不用重复写。
        for (auto& item : named_parameters())
        {
            if (item.value().dim() == 4)
            {
                torch::nn::init::kaiming_normal_(item.value(), /*a=*/0, torch::kFanOut, torch::kReLU);
            }
        }
    }

    torch::Tensor ResNetImpl::forward(torch::Tensor x)
    {
        x = relu->forward(bn1->forward(conv1->forward(x)));   // (N,  32, 32, 32)
        x = layer1->forward(x);                               // (N,  64, 32, 32)
        x = layer2->forward(x);                               // (N, 128, 16, 16)
        x = layer3->forward(x);                               // (N, 256,  8,  8)
        x = avgpool->forward(x);                              // (N, 256,  1,  1)
        return fc->forward(torch::flatten(x, 1));             // (N, num_classes)
    }


    // ------------------------------------------------------------------------
    // 工厂
    // ------------------------------------------------------------------------
    ResNet h_resnet8x4(int64_t num_classes)
    {
        return ResNet(/*n=*/1, num_classes);
    }

    ResNet h_resnet32x4(int64_t num_classes)
    {
        return ResNet(/*n=*/5, num_classes);
    }
}  // namespace mkd::model