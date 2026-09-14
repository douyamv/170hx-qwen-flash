# Upstream submission queue

Read by the scheduled task `llama-upstream-sync` (runs every 5 hours, at most ONE new pull request per run).
An entry is submitted only when its status is `ready`; after submission the task changes it to `opened <PR url>`.

| status | branch in douyamv/llama.cpp | title | body file |
|---|---|---|---|
| opened https://github.com/ggml-org/llama.cpp/pull/28871 | pr/cuda-deterministic-radix-topk | CUDA: deterministic radix select for top_k when cub::DeviceTopK is unavailable | next/upstream/pr-bodies/pr1.md |
| opened https://github.com/ggml-org/llama.cpp/pull/28872 | pr/llama-keep-sched-on-rereserve | llama : keep the backend scheduler on re-reserve instead of recreating it | next/upstream/pr-bodies/pr2.md |
| opened https://github.com/ggml-org/llama.cpp/pull/28873 | pr/kv-cache-honor-partial-only | kv-cache : honor LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY in state_write / state_read_sinfo | next/upstream/pr-bodies/pr3.md |
| opened https://github.com/ggml-org/llama.cpp/pull/28874 | pr/sched-user-inputs-first | ggml-backend : copy user inputs before cross-device inputs in compute_splits | next/upstream/pr-bodies/pr4.md |
| opened https://github.com/ggml-org/llama.cpp/pull/28875 | pr/mmvf-tiny-n-f32 | CUDA : use mmvf for tiny-N F32/F16 weights at decode batch sizes | next/upstream/pr-bodies/pr5.md |

Candidates not yet ready (need the branch, a body file and a compile/test check on upstream master first):
MMQ MoE expert-load grid; aligned Q8_0/Q4_0 GEMV layout (needs A100/H100 data); expert-grouped MoE GEMV (moea);
MTP acceptance-adaptive draft length; qwen4exp graph work (coordinate in issue #28734).
