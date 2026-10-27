# -*- coding: utf-8 -*-
"""
跨语言闭环验证：C++ 写出来的 checkpoint → Python 读 → 精度必须一模一样。

判据不是"能读"，而是"准确率分毫不差"。同时把三件事一起验了：
  ① 存档格式（键表 + 裸字节）能不能被第三方读出来
  ② C++ 的 register_module 名字和 mdistiller 官方 state_dict 是否逐个对上
  ③ 权重值、归一化、eval 模式是否一致

路径全部相对于本脚本，拷到别的机器也不用改。

用法：
    python verify_cpp_ckpt.py 7298      # 判定：必须正好 7298
    python verify_cpp_ckpt.py           # 只报告，不判定

依赖（放在同目录 / 同目录下的 ref\\）：
    read_flat.py      读 <base>.txt + <base>.bin
    ref\\resnet.py      mdistiller 的 cifar resnet（参照实现，来源见 README）
"""
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent      # …\DKD_C++\tool
PROJ = HERE.parent                                  # …\DKD_C++

sys.path.insert(0, str(HERE))                       # read_flat.py
sys.path.insert(0, str(HERE / 'ref'))               # mdistiller 的 resnet.py

from read_flat import read_flat                      # noqa: E402
import resnet as R                                   # noqa: E402

BASE = PROJ / 'run' / 'resnet8x4-baseline'
DATA = PROJ / 'dataset' / 'cifar-100-binary'
EXPECT = int(sys.argv[1]) if len(sys.argv) > 1 else None

for p in (pathlib.Path(f'{BASE}.txt'), pathlib.Path(f'{BASE}.bin'), DATA / 'test.bin'):
    if not p.exists():
        raise SystemExit(
            f'找不到 {p}\n'
            f'  存档要从 out\\build\\x64-release\\run\\ 挪到 {PROJ / "run"}\n'
            f'  （config.h 里的 out_dir 改成绝对路径，以后就不会再落进 build 目录）')

# ── ① 读 C++ 写的存档 ──────────────────────────────────────────
sd = read_flat(BASE)
n_elem = sum(t.numel() for t in sd.values())
n_bytes = sum(t.numel() * t.element_size() for t in sd.values())
print(f'张量   {len(sd)}')
print(f'元素   {n_elem:,}')
print(f'字节   {n_bytes:,}   (文件 {pathlib.Path(f"{BASE}.bin").stat().st_size:,})')
for name in ('conv1.weight', 'fc.bias'):
    if name in sd:
        t = sd[name]
        print(f'指纹   {name:<16} sum {t.sum().item():.10g}  abs-sum {t.abs().sum().item():.10g}')

# ── ② 装进 mdistiller 的 resnet8x4（strict：缺一个多一个都抛）────
net = R.resnet8x4(num_classes=100)
missing, unexpected = net.load_state_dict(sd, strict=True)
print(f'load   missing {len(missing)}  unexpected {len(unexpected)}')

# ── ③ 读 CIFAR-100 测试集（和 C++ 同一套：行宽 3074，fine 标签在第 1 列）
raw = np.fromfile(DATA / 'test.bin', dtype=np.uint8).reshape(-1, 3074)
labels = torch.from_numpy(raw[:, 1].astype(np.int64))
imgs = torch.from_numpy(raw[:, 2:].copy()).reshape(-1, 3, 32, 32).float().div_(255.0)

mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
stdv = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
imgs = (imgs - mean) / stdv

# ── ④ 评估 ────────────────────────────────────────────────────
net.eval()
correct = 0
with torch.no_grad():
    for i in range(0, imgs.size(0), 500):
        logit, _ = net(imgs[i:i + 500])
        correct += (logit.argmax(1) == labels[i:i + 500]).sum().item()

total = labels.numel()
print(f'测试集 {total}')
print(f'correct {correct} / {total}     acc {correct / total * 100:.2f} %')

if EXPECT is None:
    print('判定   （未给期望值，只报告）')
else:
    print(f'判定   期望 {EXPECT}')
    print('结果   ' + ('一致 ✓' if correct == EXPECT else f'不一致 ✗ 差 {correct - EXPECT} 张'))
