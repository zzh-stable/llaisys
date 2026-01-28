#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, T *gate, T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float gate_val = llaisys::utils::cast<float>(gate[i]);
        float up_val = llaisys::utils::cast<float>(up[i]);
        float res = up_val * gate_val / (1.0f + std::exp(-gate_val)); // sigmoid(up) * gate
        out[i] = llaisys::utils::cast<T>(res);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(gate), reinterpret_cast<float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(gate), reinterpret_cast<llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(gate), reinterpret_cast<llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
