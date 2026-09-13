#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
int main(int argc, char ** argv) {
    for (int f = 1; f < argc; ++f) {
        ggml_context * meta = nullptr;
        gguf_init_params p = { /*no_alloc*/ true, &meta };
        gguf_context * ctx = gguf_init_from_file(argv[f], p);
        if (!ctx) { printf("FAILED to parse %s\n", argv[f]); return 1; }
        const int64_t n = gguf_get_n_tensors(ctx);
        size_t total = 0, end_max = 0;
        for (int64_t i = 0; i < n; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            size_t off = gguf_get_tensor_offset(ctx, i), sz = gguf_get_tensor_size(ctx, i);
            ggml_tensor * t = ggml_get_tensor(meta, name);
            if (ggml_nbytes(t) != sz) { printf("SIZE MISMATCH %s\n", name); return 1; }
            total += sz; end_max = std::max(end_max, off + sz);
            if (strstr(name, "shared_head_head")) printf("  %s type=%s ne=[%lld,%lld] size=%zu off=%zu\n", name, ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], sz, off);
        }
        FILE * fp = fopen(argv[f], "rb"); fseek(fp, 0, SEEK_END); long fsz = ftell(fp); fclose(fp);
        printf("%s: kv=%lld tensors=%lld data_offset=%zu tensor_bytes=%zu data_end=%zu file_size=%ld alignment=%zu -> %s\n",
               strrchr(argv[f], '/') + 1, (long long) gguf_get_n_kv(ctx), (long long) n, gguf_get_data_offset(ctx), total,
               gguf_get_data_offset(ctx) + end_max, fsz, gguf_get_alignment(ctx),
               (size_t) fsz >= gguf_get_data_offset(ctx) + end_max ? "OK" : "TRUNCATED");
        gguf_free(ctx); ggml_free(meta);
    }
    return 0;
}
