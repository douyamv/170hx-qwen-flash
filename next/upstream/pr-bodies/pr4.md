In `ggml_backend_sched_compute_splits`, inputs of a split are copied in graph order. Without pipeline parallelism
(`n_copies == 1`, which is the case whenever `--override-tensor` is used since `has_tensor_overrides()` disables
it) the user-input path calls `ggml_backend_synchronize(split_backend)`. If a cross-device input (the previous
split's output) was already processed for the same split, that stream carries a wait event on the previous device,
so the synchronize blocks the host until the previous GPU has finished its entire graph — at every split boundary
of every decode step. The CUPTI trace of a 3-GPU layer split showed the host stuck in `cudaStreamSynchronize` for
the full duration of each device's graph before it could enqueue the next one.

Copying the user inputs first (their stream only holds finished work at that point) and the cross-device inputs
afterwards removes the stall with no extra memory and no change in results.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
