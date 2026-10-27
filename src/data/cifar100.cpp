#include "data/cifar100.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mkd::data
{
    // CIFAR-100 binary：每行 3074 字节 = coarse(1) + fine(1) + pixels(3072)
    // 与 CIFAR-10 的 3073 = label(1) + pixels(3072) 只差一个字节 —— 但全体错位。
    constexpr int64_t kRowBytes = 3074;
    constexpr int64_t kPixelBytes = 3072;
    constexpr int64_t kFineCol = 1;      // 用 fine（100 类），不是 coarse
    constexpr int64_t kPixelCol = 2;

    std::pair<torch::Tensor, torch::Tensor>load_one_file(const std::filesystem::path& file, int64_t n_rows)
    {
        const int64_t expected = n_rows * kRowBytes;

        std::ifstream in(file, std::ios::binary | std::ios::ate);
        if (!in)
        {
            throw std::runtime_error("load_cifar100_bin: cannot open " + file.string());
        }

        // 大小预检：行数 × 行宽必须精确相等，否则一行都不读
        const std::streamoff size = in.tellg();
        if (size != expected)
        {
            throw std::runtime_error(
                "load_cifar100_bin: " + file.string() + " 大小 " +
                std::to_string(size) + " 期望 " + std::to_string(expected));
        }
        in.seekg(0, std::ios::beg);

        std::vector<uint8_t> buf(static_cast<size_t>(expected));
        if (!in.read(reinterpret_cast<char*>(buf.data()),
            static_cast<std::streamsize>(expected)))
        {
            throw std::runtime_error("load_cifar100_bin: short read on " + file.string());
        }

        // from_blob 不接管内存，clone() 之后张量才持有数据
        torch::Tensor rows = torch::from_blob(
            buf.data(), { n_rows, kRowBytes }, torch::kUInt8).clone();

        // 第 1 列是 fine 标签
        torch::Tensor labels = rows.select(1, kFineCol).to(torch::kInt64);   // (N,)

        // 第 2..3073 列是像素。
        // 注意：narrow 出来的行步长仍是 3074 —— 【不连续】，
        //       所以必须 reshape（会拷贝），写成 view 会直接抛错。
        torch::Tensor images = rows.narrow(1, kPixelCol, kPixelBytes)
            .reshape({ n_rows, 3, 32, 32 });

        return { images, labels };
    }

    Cifar100Raw load_cifar100_bin(const std::string& dir, bool train)
    {
        const std::filesystem::path root(dir);

        auto [img, lab] = train
            ? load_one_file(root / "train.bin", 50000)
            : load_one_file(root / "test.bin", 10000);

        Cifar100Raw out;
        out.images = img;   // (50000 或 10000, 3, 32, 32) uint8
        out.labels = lab;   // (N,) int64
        return out;
    }
}