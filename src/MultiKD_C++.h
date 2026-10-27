// src/MultiKD_C++.h  —— 伞头：只列已经搬进项目的东西
#pragma once

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <cmath>

#include "config.h"
#include "data/cifar100.h"
#include "data/transforms.h"
#include "kd/ce.h"      // 各方法自己一对头文件
#include "kd/dkd.h"
#include "kd/kd.h"
#include "kd/loss.h"    // 通用契约：Res / LossFn
#include "model/checkpoint.h"
#include "model/resnet.h"
#include "train/batch.h"    // 训练器交给损失函数的原料
#include "train/trainer.h"  // 通用训练循环
