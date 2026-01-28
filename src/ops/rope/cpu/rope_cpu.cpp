#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, T *in, int64_t *pos_ids, size_t seq_len, size_t nkvhead, size_t d, float theta) {
    std::vector<double> base(d / 2);
    for (size_t k = 0; k < d / 2; ++k) {
        base[k] = std::pow(static_cast<double>(theta), -2.0 * static_cast<double>(k) / static_cast<double>(d));
    }

    for (size_t m = 0; m < seq_len; m++) {
        int64_t pos = pos_ids[m];
        for (size_t n = 0; n < nkvhead; n++) {
            for (size_t k = 0; k < d / 2; k++) {
                double freq = base[k];
                double angle = static_cast<double>(pos) * freq;
                // 减小角度到 [-pi, pi] 或用 remainder 减小到 2*pi 可提高精度
                angle = std::remainder(angle, 2.0 * M_PI);
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);

                double real_d = static_cast<double>(llaisys::utils::cast<float>(in[(m * nkvhead + n) * d + k]));
                double imag_d = static_cast<double>(llaisys::utils::cast<float>(in[(m * nkvhead + n) * d + d / 2 + k]));

                // 使用 fma 减少舍入误差（若不可用也可用简单乘加）
                float out0 = real_d * cos_val - imag_d * sin_val;
                float out1 = real_d * sin_val + imag_d * cos_val;

                out[(m * nkvhead + n) * d + k] = llaisys::utils::cast<T>(out0);
                out[(m * nkvhead + n) * d + d / 2 + k] = llaisys::utils::cast<T>(out1);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids, llaisysDataType_t type, size_t seq_len, size_t nkvhead, size_t d, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<int64_t *>(pos_ids), seq_len, nkvhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<int64_t *>(pos_ids), seq_len, nkvhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<int64_t *>(pos_ids), seq_len, nkvhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
