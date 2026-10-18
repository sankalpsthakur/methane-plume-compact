# bench-gas-<name>: CPU training kernel. CFG = {"parallel": P, "threads": T, "runs": [ {...}, ... ]}; runs execute as separate
# processes, at most P at a time, T torch threads each. Test split (final kernels only) is read from the CRC-verified mirror.
import json, os, subprocess, sys, time
from starcop_edge.kio import find_dir

cache = find_dir("train_X.npy")
assert cache, "train cache not mounted"
need_test = any(r.get("test_eval") for r in CFG["runs"]) or bool(CFG.get("need_test"))  # need_test without runs: the reference kernel (released checkpoint rows only)
mirror = find_dir("STARCOP_test") if need_test else None
assets = find_dir("zenodo_STARCOP_test.tsv") if need_test else None
assert need_test == (mirror is not None), "dev kernels must NOT mount the STARCOP mirror / test data"
print("train cache dir:", cache, "| mirror:", mirror, "| assets:", assets, "\nCFG:", json.dumps(CFG), flush=True)
print(subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'; free -g | head -2", shell=True, capture_output=True, text=True).stdout, flush=True)
t0 = time.time()
for f in ("train_X.npy", "train_Y.npy", "train_W.npy"):  # warm the page cache once, shared by all processes
    subprocess.run(f"cat {cache}/{f} > /dev/null", shell=True)
print("page cache warmed in", round(time.time() - t0), "s", flush=True)

def argv(r):
    a = ["--data", cache, "--out", f"/kaggle/working/{r['name']}", "--tag", r["name"], "--widths", r["widths"], "--steps", str(r["steps"]),
         "--batch", str(r.get("batch", 32)), "--lr", str(r["lr"]), "--wd", str(r.get("wd", 0.0)), "--pos_weight", str(r.get("pos_weight", 1.0)),
         "--seed", str(r.get("seed", 0)), "--eval_every", str(r.get("eval_every", 4000)), "--threads", str(CFG.get("threads", 2)),
         "--jitter", str(r.get("jitter", 32)), "--stem_stride", str(r.get("stem_stride", 1)), "--hardneg", str(r.get("hardneg", 0.0)),
         "--arch", r.get("arch", "edge"), "--sched", r.get("sched", "cosine"), "--device", r.get("device", "cpu")]
    if r.get("amp"): a += ["--amp"]
    if r.get("dec_widths"): a += ["--dec_widths", r["dec_widths"]]
    if r.get("val_dates"): a += ["--val_dates", r["val_dates"]]
    if r.get("val_max"): a += ["--val_max", str(r["val_max"])]
    if r.get("test_eval"): a += ["--test_eval", "--test_mirror", mirror, "--assets", assets]
    if r.get("allow_test_with_val"): a += ["--allow_test_with_val"]
    if r.get("plateau_every"): a += ["--plateau_every", str(r["plateau_every"])]
    return a

pending, running = list(CFG["runs"]), []
os.makedirs("/kaggle/working/logs", exist_ok=True)
while pending or running:
    while pending and len(running) < CFG.get("parallel", 1):
        r = pending.pop(0)
        code = "import sys; sys.path.insert(0, '/tmp/pkg'); from starcop_edge import train; train.main(sys.argv[1:])"
        lf = open(f"/kaggle/working/logs/{r['name']}.log", "w")
        p = subprocess.Popen([sys.executable, "-c", code] + argv(r), stdout=lf, stderr=subprocess.STDOUT)
        running.append((r["name"], p, time.time(), lf)); print(f"[{time.time()-t0:6.0f}s] started {r['name']} (pid {p.pid})", flush=True)
    time.sleep(10)
    for item in list(running):
        name, p, ts, lf = item
        if p.poll() is not None:
            running.remove(item); lf.close()
            tail = open(f"/kaggle/working/logs/{name}.log").read().strip().splitlines()[-3:]
            print(f"[{time.time()-t0:6.0f}s] finished {name} rc={p.returncode} in {time.time()-ts:.0f}s :: {' | '.join(tail)[:600]}", flush=True)
    if int(time.time() - t0) % 300 < 11:  # heartbeat every ~5 min with last train line of each running proc
        for name, p, ts, lf in running:
            last = [l for l in open(f"/kaggle/working/logs/{name}.log").read().splitlines() if '"event": "train"' in l or '"event": "val"' in l][-1:]
            print(f"[{time.time()-t0:6.0f}s] {name}: {last[0][:300] if last else 'no train line yet'}", flush=True)
print("ALL RUNS DONE", round(time.time() - t0), "s", flush=True)

if CFG.get("edge"):  # final kernels: ONNX fp32 + int8 export and scoring of every seed
    from starcop_edge import edge_eval
    try:
        edge_eval.edge_pipeline([f"/kaggle/working/{r['name']}" for r in CFG["runs"]], cache, mirror, assets, "/kaggle/working/edge", lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True),
                                released_ckpt=os.path.join(assets, "hyperstarcop_mag1c_rgb_final_checkpoint_model.ckpt") if CFG.get("edge_released") else None)
    except Exception:
        import traceback; traceback.print_exc()

if CFG.get("official_check"):  # official starcop.validation.run_validation on the released checkpoint (cross-check of our scorer)
    from starcop_edge import official_check
    try:
        official_check.run(assets, mirror, "/kaggle/working", CFG["official_check"], lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True))
    except Exception:
        import traceback; traceback.print_exc()
print("KERNEL DONE", round(time.time() - t0), "s", flush=True)
