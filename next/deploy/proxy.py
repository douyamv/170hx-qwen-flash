#!/usr/bin/env python3
"""
vLLM 负载均衡器 —— 在 3 个后端实例之间轮询分发请求。

特性：
  - 轮询调度（round-robin），跳过不健康的后端
  - 后台健康检查，自动摘除 / 恢复实例
  - 请求失败自动故障转移到下一个后端
  - SSE 流式响应透传（逐块 flush，不缓冲）
  - 模型路由别名重写（DSH 的 child32k 别名 -> vLLM 实际 served-model 名）
  - 无第三方依赖，仅用标准库

端口：8080 对外；后端 8081/8082/8083
附加端点：GET /lb-status 查看各后端状态与计数
"""
import http.client
import http.server
import itertools
import json
import threading
import time

BACKENDS = ["127.0.0.1:8093"]
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8080
HEALTH_INTERVAL = 5
HEALTH_TIMEOUT = 3
UPSTREAM_TIMEOUT = 7200
STREAM_CHUNK = 8192

MODEL_ALIASES = {
    "Qwen3.8-27B-FP8": "Qwen3.8-Flash-Next-UD-Q4_K_XL",
    "Qwen3.8-27B-FP8-child32k": "Qwen3.8-Flash-Next-UD-Q4_K_XL",
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-child32k": "Qwen3.8-Flash-Next-UD-Q4_K_XL",
}

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "access-control-allow-origin", "access-control-allow-headers",
    "access-control-allow-methods", "access-control-expose-headers",
    "access-control-max-age", "access-control-allow-credentials",
}

CORS_ORIGIN = "*"

_health = dict((b, True) for b in BACKENDS)
_stats = dict((b, {"ok": 0, "fail": 0}) for b in BACKENDS)
_lock = threading.Lock()
_rr = itertools.count()


def rewrite_model(body, path=""):
    """把请求体中的模型路由别名改写为 vLLM 实际注册的 served-model 名。

    vLLM 不认识 DSH 用来触发自动压缩的 child32k 别名，直接透传会 404。
    Content-Length 由 http.client 依据重写后的 body 自动重算。
    """
    if not body:
        return body
    try:
        obj = json.loads(body)
    except Exception:
        return body
    if not isinstance(obj, dict):
        return body
    changed = False
    target = MODEL_ALIASES.get(obj.get("model"))
    if target is not None:
        obj["model"] = target
        changed = True
    # llama-server 支持把采样放到 GPU 上（top_k/top_p/min_p/temp 全在显卡上算，输出与 CPU 采样逐 token 一致），
    # 每步省下 ~3-4ms 的 CPU 采样 + 4MB logits 回拷；服务端默认关闭，这里给所有补全请求默认打开（请求里显式指定时不覆盖）
    if path.startswith(("/v1/chat/completions", "/v1/completions", "/completion")) and "backend_sampling" not in obj:
        obj["backend_sampling"] = True
        changed = True
    if not changed:
        return body
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def health_loop():
    while True:
        for b in BACKENDS:
            host, port = b.split(":")
            ok = False
            try:
                c = http.client.HTTPConnection(host, int(port), timeout=HEALTH_TIMEOUT)
                c.request("GET", "/health")
                r = c.getresponse()
                r.read()
                ok = (200 <= r.status < 300)
                c.close()
            except Exception:
                ok = False
            with _lock:
                if _health[b] != ok:
                    print("[health] %s -> %s" % (b, "UP" if ok else "DOWN"), flush=True)
                _health[b] = ok
        time.sleep(HEALTH_INTERVAL)


def pick(exclude):
    with _lock:
        up = [b for b in BACKENDS if _health[b] and b not in exclude]
    if not up:
        return None
    return up[next(_rr) % len(up)]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vllm-lb/1.0"

    def log_message(self, fmt, *args):
        return

    def _local_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/lb-status":
            with _lock:
                st = {"backends": [
                    {"addr": b, "up": _health[b],
                     "ok": _stats[b]["ok"], "fail": _stats[b]["fail"]}
                    for b in BACKENDS
                ]}
            self._local_json(200, st)
            return
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def do_HEAD(self):
        self._proxy("HEAD")

    def _proxy(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n > 0 else None
        body = rewrite_model(body, self.path)

        fwd = {}
        for k, v in self.headers.items():
            if k.lower() not in HOP_BY_HOP:
                fwd[k] = v

        tried = []
        conn = None
        resp = None
        chosen = None

        for _ in range(len(BACKENDS)):
            b = pick(set(tried))
            if b is None:
                break
            tried.append(b)
            host, port = b.split(":")
            try:
                conn = http.client.HTTPConnection(host, int(port), timeout=UPSTREAM_TIMEOUT)
                conn.request(method, self.path, body=body, headers=fwd)
                resp = conn.getresponse()
                chosen = b
                break
            except Exception as e:
                with _lock:
                    _health[b] = False
                    _stats[b]["fail"] += 1
                print("[failover] %s unavailable: %s" % (b, e), flush=True)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                resp = None

        if resp is None:
            self._local_json(503, {"error": "no healthy backend", "tried": tried})
            return

        with _lock:
            _stats[chosen]["ok"] += 1

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("X-Served-By", chosen)
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Expose-Headers", "X-Served-By")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                chunk = resp.read1(STREAM_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def handle_one_request(self):
        try:
            http.server.BaseHTTPRequestHandler.handle_one_request(self)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def main():
    threading.Thread(target=health_loop, daemon=True).start()
    srv = http.server.ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    srv.daemon_threads = True
    print("vllm-lb listening on %s:%d -> %s"
          % (LISTEN_HOST, LISTEN_PORT, ", ".join(BACKENDS)), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
