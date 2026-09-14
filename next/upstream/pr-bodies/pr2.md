`llama_context::sched_reserve()` recreates the `ggml_backend_sched` whenever `sched_need_reserve` is set.
`llama_context::set_sampler()` sets that flag, and `llama-server` calls `llama_set_sampler()` for every request
(attach/detach of the backend sampler), so every request:

1. freed and reallocated all compute buffers and the pinned host input buffer — with `-ub 1024` at 262K context
   the host buffer is ~512 MB: `cudaFreeHost` 115 ms + `cudaMallocHost` 424 ms measured with CUPTI;
2. page-faulted on the fresh buffers during the first decode steps, which showed up as a per-request "warm-up".

`ggml_backend_sched_reserve()` on the existing scheduler re-splits the graph, re-plans the allocations and only
grows buffers that need to grow, so this PR keeps the scheduler unless it does not exist yet (the pipeline-parallel
fallback still recreates it as before).

Effect on a 4× GPU layer-split deployment of Qwen3.8-Flash-Next: time to first token of a 4–17 token continuation
0.8–1.1 s → 0.25–0.4 s, identical outputs; the same mechanism cost 0.2–0.4 s per request at `-ub 256`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
