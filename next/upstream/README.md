# Upstream candidates

Patches in this directory are `git diff` output against the base commit of this repository (Unsloth llama.cpp
`d1a9235`). Clean, self-contained branches rebased on `ggml-org/llama.cpp` master live in the fork
[douyamv/llama.cpp](https://github.com/douyamv/llama.cpp):

| Branch | Patch | Status |
|---|---|---|
| `pr/cuda-deterministic-radix-topk` | 0001 | applies to master (2026-09-14), ready for PR |
| `pr/llama-keep-sched-on-rereserve` | 0002 (fork-only tracer hunks removed) | ready for PR |
| `pr/kv-cache-honor-partial-only` | 0003 (only the two `PARTIAL_ONLY` early returns) | ready for PR |
| `pr/sched-user-inputs-first` | 0004 | ready for PR |
| `pr/mmvf-tiny-n-f32` | 0006 | ready for PR |
| — | 0005 MMQ expert-load grid | does not apply to current master (`mmq.cu` diverged); re-port pending |
| — | 0007 aligned Q8_0/Q4_0 GEMV (`q8a`) | design discussion first (extra VRAM copy; needs numbers on A100/H100) |

See [../docs/UPSTREAMING.md](../docs/UPSTREAMING.md) for the motivation, numbers and open questions of each item.
