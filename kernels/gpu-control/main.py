# bench-gas-<name>: GPU training kernel driver. CFG = {"gpu": bool, "parallel": P, "threads": T, "runs": [ {...}, ... ], "edge": bool, "bench_sampler": bool}.
# Runs execute as separate processes, at most P at a time, T torch threads each. Test split (final/control kernels only) is read from the CRC-verified mirror.
# This file is the GPU twin of kernels/train/main.py (Amendment 4), kept separate so the CPU kernels stay byte-reproducible. Differences, and nothing else:
#  (1) GPU kernels print nvidia-smi / nproc and stop at once if no GPU is visible, or if more GPUs are visible than CFG max_gpus (the billing factor the caps were computed with), before the 9 GB page-cache warm-up burns quota;
#  (2) a run with device "cuda" gets CUDA_VISIBLE_DEVICES=<slot % n_gpu> in its environment when n_gpu > 1 (so every process sees its GPU as cuda:0, the single-GPU code path of train.py; n_gpu 0 or 1: nothing is set), slot = the free one of the P process slots;
#  (3) CFG bench_sampler (smoke only): timed CropSampler loop on the real train cache, writes sampler_bench.json;
#  (5) the page-cache warm-up is bounded (CFG warm_timeout_s, default 600 s) and the soft deadline escalates SIGTERM -> SIGKILL -> abandon, so a hung process or a slow mount cannot hold the session until the platform timeout;
#  (4) CFG edge_runs (optional) limits the ONNX/int8 edge stage to the named runs (the paper-recipe control shares G2's queue but is not exported); RUN_STATUS.json (device, rc, seconds of every run) is written after the runs; CFG deadline_s = soft deadline (running runs are terminated, pending runs not started, finished runs and the edge stage are kept).
# Training, evaluation, scoring and edge code (src/starcop_edge) are untouched by this file.
import json, os, subprocess, sys, time
from starcop_edge.kio import find_dir

cache = find_dir("train_X.npy")
assert cache, "train cache not mounted"
need_test = any(r.get("test_eval") for r in CFG["runs"]) or bool(CFG.get("need_test"))
mirror = find_dir("STARCOP_test") if need_test else None
assets = find_dir("zenodo_STARCOP_test.tsv") if need_test else None
assert need_test == (mirror is not None), "dev kernels must NOT mount the STARCOP mirror / test data"
print("train cache dir:", cache, "| mirror:", mirror, "| assets:", assets, "\nCFG:", json.dumps(CFG), flush=True)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print(sh("lscpu | grep -E 'Model name|^CPU\\(s\\)'; free -g | head -2"), flush=True)
NGPU = len([l for l in sh("nvidia-smi -L 2>/dev/null").splitlines() if l.startswith("GPU ")])
print("n_gpu:", NGPU, "\n" + sh("nvidia-smi 2>&1 | head -20; nproc"), flush=True)
if any(r.get("device", "cpu").startswith("cuda") for r in CFG["runs"]) and NGPU == 0:
    raise SystemExit("no GPU visible in this session: stopping before any quota is spent on the cache warm-up")
if CFG.get("max_gpus") is not None and NGPU > CFG["max_gpus"]:  # the quota is billed per GPU held (Amendment 5): the caps were computed for max_gpus GPUs
    raise SystemExit(f"{NGPU} GPUs visible but the budget assumes at most {CFG['max_gpus']} (billed per GPU): stopping before more quota is spent")
t0 = time.time(); WARM_S = CFG.get("warm_timeout_s", 600)
for f in ("train_X.npy", "train_Y.npy", "train_W.npy"):  # warm the page cache once, shared by all processes; measured 41-145 s for the 9 GB, a mount slower than WARM_S is a showstopper for the planned rates
    try: subprocess.run(["cat", f"{cache}/{f}"], stdout=subprocess.DEVNULL, timeout=max(1.0, WARM_S - (time.time() - t0)))
    except subprocess.TimeoutExpired: raise SystemExit(f"page-cache warm-up exceeded {WARM_S} s at {f}: the cache mount is too slow for the planned rates, stopping before more quota is spent")
print("page cache warmed in", round(time.time() - t0), "s", flush=True)

if CFG.get("bench_sampler"):  # smoke only: ceiling of the numpy crop sampler (one process, nothing else running) on the real train cache
    import numpy as np
    from starcop_edge import data as D
    Xs, Ys, Ws, ms = D.load_split(cache, "train", mmap=True)
    idx = np.arange(0, len(ms), 12)
    tb = time.time(); smp = D.CropSampler(Xs, Ys, Ws, idx, seed=0, jitter=32, batch=32); init_s = time.time() - tb
    smp.next(); tb = time.time(); n = 150
    for _ in range(n): smp.next()
    bench = {"batch32_batches_per_s": round(n / (time.time() - tb), 1), "sampler_init_s_per_tile": round(init_s / len(idx), 4), "tiles": int(len(idx))}
    smp.close(); json.dump(bench, open("/kaggle/working/sampler_bench.json", "w"), indent=1); print("sampler bench:", bench, flush=True)

def argv(r, dev):
    a = ["--data", cache, "--out", f"/kaggle/working/{r['name']}", "--tag", r["name"], "--widths", r["widths"], "--steps", str(r["steps"]),
         "--batch", str(r.get("batch", 32)), "--lr", str(r["lr"]), "--wd", str(r.get("wd", 0.0)), "--pos_weight", str(r.get("pos_weight", 1.0)),
         "--seed", str(r.get("seed", 0)), "--eval_every", str(r.get("eval_every", 4000)), "--threads", str(CFG.get("threads", 2)),
         "--jitter", str(r.get("jitter", 32)), "--stem_stride", str(r.get("stem_stride", 1)), "--hardneg", str(r.get("hardneg", 0.0)),
         "--arch", r.get("arch", "edge"), "--sched", r.get("sched", "cosine"), "--device", dev]
    if r.get("amp"): a += ["--amp"]
    if r.get("dec_widths"): a += ["--dec_widths", r["dec_widths"]]
    if r.get("val_dates"): a += ["--val_dates", r["val_dates"]]
    if r.get("val_max"): a += ["--val_max", str(r["val_max"])]
    if r.get("test_eval"): a += ["--test_eval", "--test_mirror", mirror, "--assets", assets]
    if r.get("allow_test_with_val"): a += ["--allow_test_with_val"]
    if r.get("plateau_every"): a += ["--plateau_every", str(r["plateau_every"])]
    if r.get("log_every"): a += ["--log_every", str(r["log_every"])]
    return a

def enforce_deadline(running, pending, status, now, t0, deadline, state, term_grace=20, give_up=90, log=print):
    """Soft deadline: after `deadline` seconds from t0 terminate the running runs and start no new one (pending runs are marked not started); SIGKILL the ones still alive after term_grace s;
    stop waiting for any that still have not exited after give_up s (they are marked abandoned; the session end reaps them). `running` is edited in place; returns the new `pending` list."""
    if not deadline or now - t0 <= deadline or not (pending or running): return pending
    if state.get("hit") is None:
        state["hit"] = now; log(f"[{now-t0:6.0f}s] SOFT DEADLINE {deadline}s: terminating {[it[0] for it in running]}, not starting {[r['name'] for r in pending]}")
        for it in running: it[1].terminate()
        for r in pending: status[r["name"]] = dict(device=None, rc=None, note="not started: soft deadline")
        return []
    if now - state["hit"] > term_grace:
        for it in running: it[1].kill()
    if now - state["hit"] > give_up and running:
        for it in list(running):
            status[it[0]] = dict(device=it[5], rc=None, note="did not exit after SIGKILL; abandoned at the soft deadline"); running.remove(it)
        log(f"[{now-t0:6.0f}s] gave up waiting for processes that did not exit after SIGKILL")
    return pending


P = CFG.get("parallel", 1)
DEADLINE = CFG.get("deadline_s"); DL_STATE = {}  # soft deadline: below the platform timeout (kaggle kernels push -t), so a misjudged budget keeps the finished runs instead of losing the whole session
pending, running, status = list(CFG["runs"]), [], {}
os.makedirs("/kaggle/working/logs", exist_ok=True)
while pending or running:
    pending = enforce_deadline(running, pending, status, time.time(), t0, DEADLINE, DL_STATE, log=lambda m: print(m, flush=True))
    while pending and len(running) < P:
        r = pending.pop(0)
        slot = min(set(range(P)) - {it[4] for it in running}); dev = r.get("device", "cpu")
        pin = dev.startswith("cuda") and NGPU > 1; env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot % NGPU)} if pin else None
        if pin: dev = f"{dev}[CUDA_VISIBLE_DEVICES={slot % NGPU}]"
        code = "import sys; sys.path.insert(0, '/tmp/pkg'); from starcop_edge import train; train.main(sys.argv[1:])"
        lf = open(f"/kaggle/working/logs/{r['name']}.log", "w")
        p = subprocess.Popen([sys.executable, "-c", code] + argv(r, r.get("device", "cpu")), stdout=lf, stderr=subprocess.STDOUT, env=env)
        running.append((r["name"], p, time.time(), lf, slot, dev)); print(f"[{time.time()-t0:6.0f}s] started {r['name']} (pid {p.pid}, slot {slot}, device {dev})", flush=True)
    time.sleep(10)
    for item in list(running):
        name, p, ts, lf, slot, dev = item
        if p.poll() is not None:
            running.remove(item); lf.close(); status[name] = dict(device=dev, rc=p.returncode, secs=round(time.time() - ts), **({'note': 'terminated by the soft deadline'} if DEADLINE and time.time() - t0 > DEADLINE and p.returncode != 0 else {}))
            tail = open(f"/kaggle/working/logs/{name}.log").read().strip().splitlines()[-3:]
            print(f"[{time.time()-t0:6.0f}s] finished {name} rc={p.returncode} in {time.time()-ts:.0f}s :: {' | '.join(tail)[:600]}", flush=True)
            json.dump(dict(n_gpu=NGPU, parallel=P, partial=True, secs=round(time.time() - t0), runs=status), open("/kaggle/working/RUN_STATUS.json", "w"), indent=1)
    if int(time.time() - t0) % 300 < 11:  # heartbeat every ~5 min with last train line of each running proc
        for name, p, ts, lf, slot, dev in running:
            last = [l for l in open(f"/kaggle/working/logs/{name}.log").read().splitlines() if '"event": "train"' in l or '"event": "val"' in l][-1:]
            print(f"[{time.time()-t0:6.0f}s] {name}: {last[0][:300] if last else 'no train line yet'}", flush=True)
print("ALL RUNS DONE", round(time.time() - t0), "s", flush=True)
json.dump(dict(n_gpu=NGPU, parallel=P, secs=round(time.time() - t0), runs=status), open("/kaggle/working/RUN_STATUS.json", "w"), indent=1)

if CFG.get("edge"):  # final kernels: ONNX fp32 + int8 export and scoring of every seed
    from starcop_edge import edge_eval
    try:
        edge_eval.edge_pipeline([f"/kaggle/working/{n}" for n in (CFG.get("edge_runs") or [r["name"] for r in CFG["runs"]])], cache, mirror, assets, "/kaggle/working/edge", lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True),
                                released_ckpt=os.path.join(assets, "hyperstarcop_mag1c_rgb_final_checkpoint_model.ckpt") if CFG.get("edge_released") else None)
    except Exception:
        import traceback; traceback.print_exc()
print("KERNEL DONE", round(time.time() - t0), "s", flush=True)
