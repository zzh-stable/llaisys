#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, T *in, const T *weight, size_t M, size_t N, float eps) {
    for (size_t m = 0; m < M; m++) {
        // compute rms
        float rms = 0.0f;
        for (size_t n = 0; n < N; n++) {
            float val = llaisys::utils::cast<float>(in[m * N + n]);
            rms += val * val;
        }
        rms = std::sqrt(rms / N + eps);

        // normalize and scale
        for (size_t n = 0; n < N; n++) {
            float val = llaisys::utils::cast<float>(in[m * N + n]);
            float w = llaisys::utils::cast<float>(weight[n]);
            out[m * N + n] = llaisys::utils::cast<T>(val * w / rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, llaisysDataType_t type, size_t M, size_t N, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<const float *>(weight), M, N, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), M, N, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), M, N, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
