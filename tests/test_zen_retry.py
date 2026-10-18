"""Local unit test (stdlib only, loopback server, no data) of the Zenodo retry/backoff policy in starcop_edge.zen."""
import http.server, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from starcop_edge import zen

PAYLOAD = bytes(range(256)) * 4
PLAN = []          # list of status codes to answer with, one per request (then 206)
HITS = []          # (t, method)
RETRY_AFTER = "1"

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _go(self):
        HITS.append((time.time(), self.command))
        code = PLAN.pop(0) if PLAN else 206
        if code in (429, 503):
            self.send_response(code)
            if code == 429 and RETRY_AFTER: self.send_header("Retry-After", RETRY_AFTER)
            self.send_header("Content-Length", "0"); self.end_headers(); return
        if code != 206:
            self.send_response(code); self.send_header("Content-Length", "0"); self.end_headers(); return
        rng = self.headers.get("Range")
        a, b = (int(v) for v in rng.split("=")[1].split("-")) if rng else (0, len(PAYLOAD) - 1)
        body = PAYLOAD[a:b + 1] if self.command == "GET" else b""
        self.send_response(206 if rng else 200); self.send_header("Content-Length", str(len(PAYLOAD) if self.command == "HEAD" else len(body))); self.end_headers()
        if self.command == "GET": self.wfile.write(body)
    do_GET = _go; do_HEAD = _go

def serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/f.zip"

def reset():
    PLAN.clear(); HITS.clear(); zen._COOL[0] = 0.0
    for k in zen.STATS: zen.STATS[k] = 0 if isinstance(zen.STATS[k], int) else 0.0

def test_all():
    global RETRY_AFTER
    srv, url = serve()
    # 1. 429 with Retry-After honoured: two 429s then success; wall time >= 2 x 1 s
    reset(); PLAN.extend([429, 429]); t = time.time()
    assert zen._range(url, 10, 19) == PAYLOAD[10:20]
    dt = time.time() - t
    # each cool-down sleeps the Retry-After (x U(1, 1.25)) plus U(0, 5) s of jitter (zen._wait_cooldown): two 429s take at most 2 x 6.25 s
    assert zen.STATS["http429"] == 2 and 2.0 <= dt < 14.0, (zen.STATS, dt)
    # 2. permanent error fails at once (one request)
    reset(); PLAN.extend([404])
    try: zen._range(url, 0, 5); assert False
    except IOError: pass
    assert len(HITS) == 1
    # 3. 503 is retried with backoff (patch sleep to keep the test fast)
    reset(); PLAN.extend([503, 503]); orig = time.sleep; slept = []
    zen.time.sleep = lambda s: (slept.append(s), orig(0.01))[0]
    try: assert zen._range(url, 0, 3) == PAYLOAD[:4]
    finally: zen.time.sleep = orig
    assert zen.STATS["http5xx"] == 2 and len(slept) == 2 and slept[0] >= 4.0 and slept[1] >= 8.0, slept
    # 4. HEAD returns the size; HTTPRangeFile reads through the same policy
    reset(); PLAN.extend([429]); f = zen.HTTPRangeFile(url, block=64)
    assert f.size == len(PAYLOAD); f.seek(100); assert f.read(10) == PAYLOAD[100:110]
    # 5. a 429 pauses the whole pool: 6 threads, first request answered 429 (Retry-After 2) -> the other threads wait for the cooldown
    #    instead of retrying immediately; total requests stay small and nobody fails
    reset(); RETRY_AFTER = "2"; PLAN.extend([429]); res = []; zen.set_concurrency(6)
    ths = [threading.Thread(target=lambda: res.append(zen._range(url, 0, 7))) for _ in range(6)]
    t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]; dt = time.time() - t
    assert len(res) == 6 and all(r == PAYLOAD[:8] for r in res) and zen.STATS["http429"] == 1 and dt >= 2.0, (zen.STATS, dt)
    assert len(HITS) <= 12, len(HITS)
    # 6. retries exhausted -> IOError mentioning the cause
    reset(); RETRY_AFTER = "1"; PLAN.extend([429] * 3)
    try: zen._range(url, 0, 5, tries=2); assert False
    except IOError as e: assert "429" in str(e)
    srv.shutdown()
    print("zen retry policy: all checks passed")

if __name__ == "__main__":
    test_all()
