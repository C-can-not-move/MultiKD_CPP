# DKD_C++ / tool —— Python 侧工具集

C++ 是主场，这个目录只放**参照实现**和**验收脚本**。不参与编译，不影响 `src\`。

## 文件清单

| 文件 | 作用 | 来源 |
|---|---|---|
| `export_teacher_flat.py` | 教师权重（旧存档格式）→ 扁平契约（`.txt` + `.bin`） | 本项目自写 |
| `read_flat.py` | 读 `.txt` + `.bin` → `state_dict` | 本项目自写 |
| `verify_cpp_ckpt.py` | C++ 存档 → Python 读 → **精度对拍** | 本项目自写 |
| `ref\resnet.py` | CIFAR 版 ResNet 的**参照实现** | mdistiller（megvii-research/mdistiller），遵守其许可证 |
| `ckpt_epoch_240.pth` | mdistiller 发布的教师权重（原始，PyTorch 旧存档格式，59.5 MB） | mdistiller model zoo |

`ref\resnet.py` **只作对照，不进编译**。它存在的意义是：提供一个「官方认可的键名和结构」，用来判断我们自己写的 `h_resnet32x4` / `h_resnet8x4` 对不对。搬进 `src\` 会让人分不清哪个是实现、哪个是参照。

## 数据

```
dataset\cifar-100-binary\
    train.bin   153,700,000 B   = 50000 × 3074
    test.bin     30,740,000 B   = 10000 × 3074
```
行格式：`[1B 粗标签][1B 细标签][3072B 像素]`，行宽 **3074**（不是 3073）。

## 输出位置

存档写到 `run\`（**项目根下的绝对路径**）。如果看到文件落在
`out\build\x64-release\run\`，说明 `config.h` 里的 `out_dir` 还是相对路径 `"./run"`
——那个目录会被 CMake 清理时删掉，务必改成绝对路径。

## 三个常用动作

```powershell
# 1. 教师权重 → 扁平契约（一次就够，换教师才重跑）
python export_teacher_flat.py

# 2. 看一份存档里有什么（不跑模型）
python read_flat.py ..\run\resnet8x4-baseline

# 3. 验收：C++ 写完的存档，Python 读回来精度是否一致
python verify_cpp_ckpt.py 7298
```

## 已知的验收数字

| 方向 | 文件 | 判据 | 实测 |
|---|---|---|---|
| Python → C++ | `resnet32x4_100.{txt,bin}`（教师） | CIFAR-100 测试集准确率 | **7942 / 10000 = 79.42 %** |
| C++ → Python | `run\resnet8x4-baseline.{txt,bin}`（学生基线） | `verify_cpp_ckpt.py 7298` | **7298 / 10000 = 72.98 %** |

这两个数是**行为级**判据，不是“张量 allclose”——它同时证明了张量、结构、键名、归一化、BN 模式全都对。

## 格式（就五列）

```
# name dtype shape off nbytes
conv1.weight f32 3,32,3,3 0 3456
bn1.num_batches_tracked i64 1 3456 8
```

- `shape` 写 `0` 表示零维张量
- `off` 相对 `.bin` 开头，**8 字节对齐**
- 全部小端，没有 pickle、没有 zip、没有 `constants.pkl`

**为什么不用 `torch::serialize::InputArchive`**：它是 TorchScript 归档的读写器
（私有成员就是 `jit::Module module_`，`load_from` 直接转给 `torch::jit::load`），
而 TorchScript 归档必须带 `constants.pkl`。`torch.save(state_dict)` 从来不写那个记录，
所以它会报 `failed locating file constants.pkl ... your checkpoint file is corrupted`
——**文件没坏，是读档器不对口**。
