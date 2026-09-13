#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Minimum KV size for the NEXT_QSA_OPT paths, read from $NEXT_OPT_DIR/qsa_min_kv at most once per second.
inline int64_t next_qsa_min_kv() {
    static std::chrono::steady_clock::time_point next_check{};
    static int64_t value = 131072;
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_check) {
        next_check = now + std::chrono::seconds(1);
        int64_t v = 131072;
        if (const char * dir = getenv("NEXT_OPT_DIR")) {
            if (FILE * f = fopen((std::string(dir) + "/qsa_min_kv").c_str(), "r")) {
                long long x = 0;
                if (fscanf(f, "%lld", &x) == 1 && x > 0) {
                    v = x;
                }
                fclose(f);
            }
        }
        value = v;
    }
    return value;
}

#include <mutex>
#include <sys/stat.h>
#include <unordered_map>

// True when $NEXT_OPT_DIR/<name> exists; re-checked at most once per second per name.
inline bool next_opt_flag(const char * name) {
    struct entry { std::chrono::steady_clock::time_point next_check{}; bool value = false; };
    static std::mutex mtx;
    static std::unordered_map<std::string, entry> cache;
    std::lock_guard<std::mutex> lock(mtx);
    entry & e = cache[name];
    const auto now = std::chrono::steady_clock::now();
    if (now >= e.next_check) {
        e.next_check = now + std::chrono::seconds(1);
        const char * dir = getenv("NEXT_OPT_DIR");
        struct stat st;
        e.value = dir != nullptr && stat((std::string(dir) + "/" + name).c_str(), &st) == 0;
    }
    return e.value;
}
