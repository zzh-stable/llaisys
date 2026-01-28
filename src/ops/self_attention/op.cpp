#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // device check
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    // Contiguous check
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: input must be contiguous.");

    // type check
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // shape check
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "SelfAttention: invalid tensor shape.");
    size_t seq_len = attn_val->shape()[0];
    size_t nhead = attn_val->shape()[1];
    size_t dv = attn_val->shape()[2];
    
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t d = k->shape()[2];
    ASSERT(q->shape()[0] == seq_len && q->shape()[1] == nhead && q->shape()[2] == d, "SelfAttention: invalid q tensor shape.");
    ASSERT(v->shape()[0] == total_len && v->shape()[1] == nkvhead && v->shape()[2] == dv, "SelfAttention: invalid v tensor shape.");


    if (seq_len == 0 || nhead == 0 || dv == 0) {
        return; // nothing to do
    }

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), seq_len, nhead, dv, total_len, nkvhead, d, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), seq_len, nhead, dv, total_len, nkvhead, d, scale);
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
