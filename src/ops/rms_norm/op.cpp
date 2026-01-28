#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // device check
    CHECK_SAME_DEVICE(out, in, weight);

    // Contiguous check
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMS Norm: input must be contiguous.");

    // type check
    CHECK_SAME_DTYPE(out->dtype(),in->dtype(), weight->dtype());

    // shape check
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 1, "RMS Norm: invalid tensor shape.");
    ASSERT(out->shape()[1] == weight->shape()[0], "RMS Norm: output shape must match weight shape.");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1], "RMS Norm: invalid tensor shape.");

    
    if (out->shape()[0] == 0 || out->shape()[1] == 0) {
        return; // nothing to do
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), out->shape()[0], out->shape()[1], eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), out->shape()[0], out->shape()[1], eps);
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
