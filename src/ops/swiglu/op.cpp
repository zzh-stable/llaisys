#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // device check
    CHECK_SAME_DEVICE(out, gate, up);

    // Contiguous check
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "Swiglu: input must be contiguous.");

    // type check
    CHECK_SAME_DTYPE(out->dtype(),gate->dtype(), up->dtype());

    // shape check
    ASSERT(out->numel() == gate->numel() && out->numel() == up->numel(), "Swiglu: invalid number size.");
    
    if (out->numel() == 0) {
        return; // nothing to do
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
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
