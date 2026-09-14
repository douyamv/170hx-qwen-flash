#pragma once
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif
GGML_API struct ggml_tensor * ggml_qsa_pool_norm(struct ggml_context * ctx, struct ggml_tensor * raw, struct ggml_tensor * ids, struct ggml_tensor * gamma, float eps);
GGML_API struct ggml_tensor * ggml_qsa_expand(struct ggml_context * ctx, struct ggml_tensor * score, struct ggml_tensor * ids, struct ggml_tensor * mask);
GGML_API struct ggml_tensor * ggml_qsa_mask_select(struct ggml_context * ctx, struct ggml_tensor * mask, struct ggml_tensor * ids, int64_t padded);
// NEXT: fused Qwen hyper-connection elementwise chains (kinds 5 and 6)
//   mix tail: out[i,t] = (1/hc) * sum_h xn[h*n_embd+i, t] * sigmoid(glogits[h*n_embd+i, t])
GGML_API struct ggml_tensor * ggml_hc_mix_tail(struct ggml_context * ctx, struct ggml_tensor * xn, struct ggml_tensor * glogits, int hc);
//   combine: out[i,h,t] = residual[i,h,t] + block_out[i,t] * 2*sigmoid(inject[h,t] * (1/hc))
GGML_API struct ggml_tensor * ggml_hc_combine(struct ggml_context * ctx, struct ggml_tensor * residual, struct ggml_tensor * block_out, struct ggml_tensor * inject, int hc);
//   device-side causal KQ mask (kind 9): out[c,t] = (cell_pos[c] >= 0 && cell_pos[c] <= pos[t]) ? 0 : -inf
GGML_API struct ggml_tensor * ggml_kq_mask_dev(struct ggml_context * ctx, struct ggml_tensor * cell_pos, struct ggml_tensor * pos, struct ggml_tensor * kvs, int64_t n_kv, int64_t n_tps, int64_t n_ns, enum ggml_type type);
GGML_API int ggml_qsa_kind(const struct ggml_tensor * t);
GGML_API float ggml_qsa_epsilon(const struct ggml_tensor * t);
#ifdef __cplusplus
}
#endif
