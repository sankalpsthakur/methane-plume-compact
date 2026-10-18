# bench-gas-mount-probe: does a fresh CPU kernel start and mount the 9 GB train cache (kernel source bench-gas-prep-cache) quickly? No GPU, no test data, no internet. Written 2026-10-10 because smoke v3 stayed RUNNING for 45+ min.
import os, subprocess, time, json
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
log(subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'; free -g | head -2; df -h /kaggle/working | tail -1", shell=True, capture_output=True, text=True).stdout)
found = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_X.npy" in files: found = root; break
    if root.count(os.sep) > 6: dirs[:] = []
log("cache dir:", found)
res = {"cache_dir": found}
if found:
    for f in ("train_X.npy", "train_Y.npy", "train_W.npy", "train_meta.csv"):
        p = os.path.join(found, f); res[f] = os.path.getsize(p)
    t = time.time(); n = 0
    with open(os.path.join(found, "train_X.npy"), "rb") as fh:
        while n < 512 * 2**20:
            b = fh.read(8 * 2**20)
            if not b: break
            n += len(b)
    res["read_512MB_s"] = round(time.time() - t, 2); log("read", n, "bytes in", res["read_512MB_s"], "s")
res["secs"] = round(time.time() - t0, 1)
json.dump(res, open("/kaggle/working/mount_probe.json", "w"), indent=1); log("DONE", json.dumps(res))
