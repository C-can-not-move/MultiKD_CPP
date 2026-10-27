# -*- coding: utf-8 -*-
"""
把 mdistiller 发布的教师权重（PyTorch 旧存档格式）直接导成 libtorch 能读的
两个文件：

    <BASE>.txt   键表，一行一个张量：# name dtype shape off nbytes
    <BASE>.bin   裸字节，所有张量按 8 字节对齐首尾相接

一步到位，不需要中间 .pt。中间产物是多余的 —— 那一步唯一的用处是让
torch::serialize::InputArchive 去读，而它读不了（它是 TorchScript 归档的
读写器，内部成员是 jit::Module，必须要 constants.pkl；torch.save(dict) 从
来不写这个记录）。所以干脆绕开 pickle。

源文件自带两个历史问题，这里一并处理：

  ① 旧存档格式（PyTorch 1.6 之前的 _legacy_save）：头部 80 02 8A 0A 而不是
     50 4B 03 04。这一层 Python 的 torch.load 完全能处理，不用管。

  ② 权重里没有 bn.num_batches_tracked（这个 buffer 是 PyTorch 0.4.1 才有的）。
     它在 momentum=0.1 时不参与任何计算，但少了它就凑不齐 206 个键，
     C++ 侧会因缺键而抛 —— 所以这里补 0（初值）。

用法：
    python export_teacher_flat.py            # 输出到 .pth 所在目录
    python export_teacher_flat.py <out_dir>  # 输出到别处
"""
import pathlib
import sys

import torch

SRC = pathlib.Path(r'D:/DATA/CODE/C++/DKD_C++/tool/ckpt_epoch_240.pth')
OUT_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else SRC.parent
BASE = 'resnet32x4_100'

EXPECT_KEYS = 206          # mdistiller resnet32x4(num_classes=100).state_dict()
EXPECT_PARAMS = 7433860    # sum(p.numel() for p in net.parameters())  ← 只数参数
ALIGN = 8

DTYPE_NAME = {
    torch.float32: 'f32',
    torch.float64: 'f64',
    torch.float16: 'f16',
    torch.int64:   'i64',
    torch.int32:   'i32',
    torch.int16:   'i16',
    torch.int8:    'i8',
    torch.uint8:   'u8',
}

BUFFER_TAGS = ('running_mean', 'running_var', 'num_batches_tracked')


def head_bytes(p, n=4):
    with open(p, 'rb') as f:
        return ' '.join(f'{b:02X}' for b in f.read(n))


def load_state_dict():
    """读原始权重，剥掉外壳，补齐 num_batches_tracked。"""
    print(f'src    {SRC}')
    print(f'       头部 {head_bytes(SRC)}   （80 02 8A 0A = 旧格式；50 4B 03 04 = zip）')

    # weights_only=False：旧格式的 pickle 里除了 _rebuild_tensor_v2 还有别的全局引用
    ckpt = torch.load(SRC, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        print(f'外壳键 {list(ckpt.keys())}   ← 取出 model 那一层')
        sd = dict(ckpt['model'])
    else:
        sd = dict(ckpt)

    # 有的存档会带 DataParallel 的 module. 前缀，剥掉；没有就原样
    if sd and all(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        print('       剥掉 module. 前缀')

    added = 0
    for k in [k for k in sd if k.endswith('.running_var')]:
        nbt = k[: -len('running_var')] + 'num_batches_tracked'
        if nbt not in sd:
            sd[nbt] = torch.tensor(0, dtype=torch.int64)
            added += 1
    print(f'       补入 num_batches_tracked {added} 个')

    n_params = sum(v.numel() for k, v in sd.items()
                   if not any(t in k for t in BUFFER_TAGS))
    print(f'       keys {len(sd)}  params(仅参数) {n_params}')

    if len(sd) != EXPECT_KEYS or n_params != EXPECT_PARAMS:
        raise SystemExit('✗ 与 mdistiller resnet32x4(100) 的期望不符'
                         f'（{EXPECT_KEYS} 键 / {EXPECT_PARAMS} 参数），别往下走')
    return sd


def export(sd):
    keys = sorted(sd)                      # 排序 ⇒ 与 dict 内部顺序无关，可复现
    lines = ['# name dtype shape off nbytes']
    parts = []
    cur = 0

    for key in keys:
        t = sd[key].detach().to('cpu').contiguous()
        if t.dtype not in DTYPE_NAME:
            raise SystemExit(f'{key}: 不认识的 dtype {t.dtype}')

        pad = (-cur) % ALIGN               # Python 的 % 是 floor 取模，负数也对
        if pad:
            parts.append(b'\0' * pad)
            cur += pad

        raw = t.numpy().tobytes()
        shape = ','.join(str(int(d)) for d in t.shape) or '0'   # 0 维写成 0
        lines.append(f'{key} {DTYPE_NAME[t.dtype]} {shape} {cur} {len(raw)}')
        parts.append(raw)
        cur += len(raw)

    blob = b''.join(parts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = OUT_DIR / f'{BASE}.txt'
    bin_ = OUT_DIR / f'{BASE}.bin'
    txt.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    bin_.write_bytes(blob)

    print(f'键数   {len(keys)}')
    print(f'裸字节 {len(blob):,} B')
    print(f'键表   {txt}  ({txt.stat().st_size:,} B)')
    print(f'权重   {bin_}  ({bin_.stat().st_size:,} B)')

    # 下面两行是给 C++ 侧对账用的：数字必须一模一样，否则说明值没进来
    for name in ('conv1.weight', 'fc.bias'):
        w = sd[name].detach().to('cpu')
        print(f'指纹   {name:<16} sum {w.sum().item():.10g}  abs-sum {w.abs().sum().item():.10g}')


def main():
    export(load_state_dict())


if __name__ == '__main__':
    main()
