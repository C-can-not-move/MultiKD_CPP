// src/data/cifar100.h
#pragma once
#include <torch/torch.h>
#include <string>

namespace mkd::data
{
    struct Cifar100Raw
    {
        torch::Tensor images;   // (N, 3, 32, 32) uint8
        torch::Tensor labels;   // (N)            int64 —— fine label, 0..99
    };

    Cifar100Raw load_cifar100_bin(const std::string& dir, bool train);
}