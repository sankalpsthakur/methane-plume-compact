#!/usr/bin/env python3
"""Local (M4, 1 thread, batch 1, 512x512x4) latency of onnx files. Usage: latency_m4.py [--out results/latency_m4.json] a.onnx b.onnx ... Prints median/min ms and writes the json.
Owner exception to the no-local-compute rule: models under 50 MB only, through `bin/guard.sh -t 60 -m 512 --`, and only while `uptime` shows a 1-minute load average under 4
(otherwise write "latency deferred (machine busy)" in STATUS/SUBMISSION and skip). Repeats 3 rounds and reports the best round median; the load average is recorded."""
import json, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, onnxruntime as ort

def bench(path, n=25, warm=5, rounds=3):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]; shape = [1 if isinstance(d, str) else d for d in inp.shape]
    x = np.random.rand(*shape).astype(np.float32)
    t = time.perf_counter(); s.run(None, {inp.name: x}); one = time.perf_counter() - t   # first run also warms the session
    if one > 0.5: n, warm, rounds = 10, 1, 2   # keep a 6.6M-parameter model inside the 60 s guard
    best = None
    for _ in range(rounds):
        for _ in range(warm): s.run(None, {inp.name: x})
        ts = []
        for _ in range(n):
            t = time.perf_counter(); s.run(None, {inp.name: x}); ts.append((time.perf_counter() - t) * 1000)
        med = float(np.median(ts))
        if best is None or med < best[0]: best = (med, float(np.min(ts)))
    return best

args = sys.argv[1:]; out = "results/latency_m4.json"
if args[:1] == ["--out"]: out, args = args[1], args[2:]
res = {}
for p in args:
    assert os.path.getsize(p) < 50e6, ("model over 50 MB, not run locally", p)
    load1 = os.getloadavg()[0]
    assert load1 < 4.0, ("latency deferred (machine busy): 1-min load %.1f" % load1)
    med, mn = bench(p)
    res[os.path.basename(p)] = dict(size_mb=round(os.path.getsize(p) / 1e6, 3), median_ms=round(med, 2), min_ms=round(mn, 2), load1=round(load1, 2), threads=1, batch=1, input="1x4x512x512")
    print(f"{os.path.basename(p):50s} {os.path.getsize(p)/1e6:8.2f} MB  median {med:7.1f} ms  min {mn:7.1f} ms  (1 thread, batch 1, load {load1:.1f})")
if res:
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res); json.dump(old, open(out, "w"), indent=1)
