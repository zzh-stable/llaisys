#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

struct tensor3d_idx {
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t get_idx(size_t i, size_t j, size_t k) {
    return (i * dim1 + j) * dim2 + k;
}
}; 


template <typename T>
void self_attention_(T *attn_val, T *q, const T *k, const T *v, size_t seq_len, size_t nhead, size_t dv, size_t total_len, size_t nkvhead, size_t d, float scale) {
    tensor3d_idx q_idx{seq_len, nhead, d}, k_idx{total_len, nkvhead, d}, v_idx{total_len, nkvhead, dv}, attn_idx{seq_len, nhead, dv};
    size_t group_size = nhead / nkvhead;
    for( size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < nhead; h++) {
            // compute QK^T for one row
            std::vector<float> qk(total_len, 0.0f);
            float qk_sum = 0.0f;
            // mask end positions
            size_t valid_len =  total_len - seq_len + s + 1;
            for (size_t t = 0; t < valid_len; t++) {
                float sum = 0.0f;
                for (size_t i = 0; i < d; i++) {
                    float q_val = llaisys::utils::cast<float>(q[q_idx.get_idx(s, h, i)]);
                    float k_val = llaisys::utils::cast<float>(k[k_idx.get_idx(t, h / group_size, i)]);
                    sum += q_val * k_val;
                }
                qk[t] = std::exp(sum * scale);
                qk_sum += qk[t];
            }
            // softmax
            for (size_t t = 0; t < valid_len; t++) {
                qk[t] /= qk_sum;
            }
            // compute attention value
            std::vector<float> res(dv, 0.0f);
            for (size_t t = 0; t < valid_len; t++) {
                for (size_t i = 0; i < dv; i++) {
                    float v_val = llaisys::utils::cast<float>(v[v_idx.get_idx(t, h / group_size, i)]);
                    res[i] += qk[t] * v_val;
                }
            }
            for (size_t i = 0; i < dv; i++) {
                attn_val[attn_idx.get_idx(s, h, i)] = llaisys::utils::cast<T>(res[i]);
            }
        }
        
    }

}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v, llaisysDataType_t type, size_t seq_len, size_t nhead, size_t dv, size_t total_len, size_t nkvhead, size_t d, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<float *>(q), reinterpret_cast<float *>(k), reinterpret_cast<float *>(v), seq_len, nhead, dv, total_len, nkvhead, d, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<llaisys::bf16_t *>(q), reinterpret_cast<llaisys::bf16_t *>(k), reinterpret_cast<llaisys::bf16_t *>(v), seq_len, nhead, dv, total_len, nkvhead, d, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<llaisys::fp16_t *>(q), reinterpret_cast<llaisys::fp16_t *>(k), reinterpret_cast<llaisys::fp16_t *>(v), seq_len, nhead, dv, total_len, nkvhead, d, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
