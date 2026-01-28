#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // device check
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // Contiguous check
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Rope: input must be contiguous.");

    // type check
    CHECK_SAME_DTYPE(out->dtype(),in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids must be int64 type.");

    // shape check
    ASSERT(out->ndim() == 3 && in->ndim() == 3 && pos_ids->ndim() == 1, "Rope: invalid tensor shape.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(out->shape()[0] == pos_ids->shape()[0], "Rope: invalid tensor shape.");
    ASSERT(out->shape()[2] % 2 == 0, "Rope: the last dimension size must be even.");

    if (out->shape()[0] == 0 || out->shape()[1] == 0 || out->shape()[2] == 0) {
        return; // nothing to do
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), pos_ids->shape()[0], out->shape()[1], out->shape()[2], theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), pos_ids->shape()[0], out->shape()[1], out->shape()[2], theta);
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
