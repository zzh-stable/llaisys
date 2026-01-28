#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(T *out, int64_t *index, const T *weight, size_t dim0, size_t dim1) {
    for (size_t i = 0; i < dim0; i++) {
        int64_t idx = index[i];
        for (size_t j = 0; j < dim1; j++) {
            out[i * dim1 + j] = weight[idx * dim1 + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t dim0, size_t dim1) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<int64_t *>(index), reinterpret_cast<const float *>(weight), dim0, dim1);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), dim0, dim1);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), dim0, dim1);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
