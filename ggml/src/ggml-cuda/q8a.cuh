#pragma once
#include "common.cuh"

// NEXT: bandwidth-oriented Q8_0 GEMV for decode (1..8 columns). The weights are repacked once per tensor into a
// 16-byte-aligned split layout (int8 quants contiguous, fp16 block scales contiguous) so every lane issues 16-byte
// loads; the standard 34-byte Q8_0 blocks only allow 2-byte-aligned accesses, which leaves the 70-SM CMP 170HX
// instruction-bound at ~40% of its HBM bandwidth. Disable with NEXT_Q8A=0.
bool ggml_cuda_q8a_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                             const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
// returns false when the aligned copy is not available (allocation failed): the caller must use mmvq
bool ggml_cuda_mul_mat_q8a(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                           const ggml_cuda_mm_fusion_args_host * fusion);
