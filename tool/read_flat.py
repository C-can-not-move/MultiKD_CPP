# -*- coding: utf-8 -*-
"""
读 C++ 侧 save_checkpoint() 写出的 checkpoint（<base>.txt + <base>.bin）。

格式就五列，没有第六列：

    # name dtype shape off nbytes
    conv1.weight f32 32,3,3,3 0 3456
    bn1.num_batches_tracked i64 0 8 8

  · shape 写 '0' 表示零维张量
  · off 是相对 .bin 开头的字节偏移，8 字节对齐
  · 全部小端，没有 pickle、没有 zip、没有常量表

用法：
    python read_flat.py run/resnet8x4-baseline        # 只看清单
    python read_flat.py run/resnet8x4-baseline --diff <参考.pt>   # 和别的存档逐张量比

作为模块：
    from read_flat import read_flat
    sd = read_flat('run/resnet8x4-baseline')
    net.load_state_dict(sd)          # 键名和 Python 的 state_dict 一致，直接吃
"""
import pathlib
import sys

import numpy as np
import torch

DTYPE = {
    'f32': np.float32, 'f64': np.float64, 'f16': np.float16,
    'i64': np.int64, 'i32': np.int32,
    'i16': np.int16, 'i8': np.int8, 'u8': np.uint8,
}

ALIGN_HINT = '偏移不是 8 的倍数 —— 文件很可能不是这个格式'


def read_flat(base):
    """base 不带后缀。返回 {键: CPU 张量}，顺序即键表顺序。"""
    base = pathlib.Path(base)
    txt = pathlib.Path(f'{base}.txt')
    bin_ = pathlib.Path(f'{base}.bin')

    blob = bin_.read_bytes()
    out = {}

    for line in txt.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        name, dt, shape, off, nbytes = line.split()
        off, nbytes = int(off), int(nbytes)

        if off % 8:
            raise ValueError(f'{name}: {ALIGN_HINT}')
        if dt not in DTYPE:
            raise ValueError(f'{name}: 不认识的 dtype {dt!r}')

        dims = () if shape == '0' else tuple(int(d) for d in shape.split(','))
        count = nbytes // np.dtype(DTYPE[dt]).itemsize
        arr = np.frombuffer(blob, dtype=DTYPE[dt], count=count, offset=off).reshape(dims)

        # np.frombuffer 是只读视图，copy 一次才能交给 torch
        out[name] = torch.from_numpy(arr.copy())

    if not out:
        raise ValueError(f'{txt} 里没有数据行')
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    base = sys.argv[1]
    sd = read_flat(base)

    nbytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f'张量  {len(sd)}')
    print(f'元素  {sum(t.numel() for t in sd.values()):,}')
    print(f'字节  {nbytes:,}')

    for name in ('conv1.weight', 'fc.bias'):
        if name in sd:
            t = sd[name]
            print(f'指纹  {name:<16} sum {t.sum().item():.10g}  abs-sum {t.abs().sum().item():.10g}')

    if '--diff' in sys.argv:
        ref = torch.load(sys.argv[sys.argv.index('--diff') + 1], map_location='cpu')
        bad = [k for k, v in sd.items()
               if k not in ref or not torch.equal(v.reshape(ref[k].shape), ref[k])]
        print(f'对比  {sys.argv[sys.argv.index("--diff") + 1]}: '
              f'{"全等 ✓" if not bad else f"{len(bad)} 个不一致 ✗ {bad[:5]}"}')


if __name__ == '__main__':
    main()
