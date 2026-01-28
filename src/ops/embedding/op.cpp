#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // device check
    CHECK_SAME_DEVICE(out, index, weight);

    // Contiguous check
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: input must be contiguous.");

    // type check
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be of dtype int64.");

    // shape check
    ASSERT(out->ndim() == 2 && index->ndim() == 1 && weight->ndim() == 2, "Embedding: invalid tensor shape.");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: output first dimension must match index shape.");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: output second dimension must match weight second dimension.");

    
    if (out->shape()[0] == 0 || out->shape()[1] == 0) {
        return; // nothing to do
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], out->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], out->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
