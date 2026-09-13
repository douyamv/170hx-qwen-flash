#pragma once
#include <cstdint>
extern "C" uint64_t next_trace_time() __attribute__((weak));
extern "C" void next_trace_cpu(const char *, uint64_t, uint64_t) __attribute__((weak));
struct llama_trace_local {
    const char * name;
    uint64_t start;
    explicit llama_trace_local(const char * label) : name(label), start(next_trace_time ? next_trace_time() : 0) {}
    ~llama_trace_local() { if (start && next_trace_time && next_trace_cpu) next_trace_cpu(name,start,next_trace_time()); }
};
