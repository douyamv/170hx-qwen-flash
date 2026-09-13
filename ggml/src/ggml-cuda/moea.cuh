#pragma once
#include "common.cuh"

// NEXT: expert-grouped decode GEMV for MoE experts (mul_mat_id with <= 8 tokens). One thread block streams an
// expert's rows once for every token that selected it, with 16-byte (Q4_K) / 8-byte (Q5_1) vector loads and the
// fused up*silu(gate) epilogue. Disable with NEXT_MOEA=0.
bool ggml_cuda_moea_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                              const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
bool ggml_cuda_mul_mat_moea(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                            ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
