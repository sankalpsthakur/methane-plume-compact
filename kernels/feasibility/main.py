# bench-gas-feasibility: CPU kernel. Answers: can STARCOP be used inside a private Kaggle kernel, within disk/time limits?
import collections, concurrent.futures as cf, io, json, os, platform, random, subprocess, sys, time, urllib.request, zlib
import numpy as np

T0 = time.time()
REP = {}
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def sh(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

# ---------------------------------------------------------------- 1. environment
log("== environment")
REP["python"] = sys.version.split()[0]
REP["nproc"] = os.cpu_count()
cpu = sh("lscpu | egrep 'Model name|Thread|Core|MHz'")
flags = sh("grep -m1 flags /proc/cpuinfo")
REP["cpu"] = cpu
REP["avx2"] = " avx2" in flags; REP["avx512f"] = " avx512f" in flags; REP["amx"] = "amx" in flags
log(cpu); log("avx2", REP["avx2"], "avx512f", REP["avx512f"], "amx", REP["amx"])
log(sh("free -g | head -2")); log(sh("df -h /kaggle/working /tmp /kaggle/input / 2>/dev/null"))
REP["df"] = sh("df -BG /kaggle/working /tmp / | tail -3")
for m in ["numpy", "torch", "torchvision", "tifffile", "imagecodecs", "rasterio", "segmentation_models_pytorch", "onnx", "onnxruntime", "jax", "sklearn", "scipy", "pandas"]:
    try:
        mod = __import__(m); REP["lib_" + m] = getattr(mod, "__version__", "?")
    except Exception as e:
        REP["lib_" + m] = "MISSING"
log({k: v for k, v in REP.items() if k.startswith("lib_")})

# ---------------------------------------------------------------- 2. mount + completeness vs Zenodo manifests
log("== mount")
IN = "/kaggle/input"
from starcop_edge.kio import find_dir
log("input tree (depth<=3):", sh("find /kaggle/input -maxdepth 3 | head -40"))
root = find_dir("STARCOP_test"); ASSETS = find_dir("zenodo_STARCOP_test.tsv")
assert root and ASSETS, ("mount not found", root, ASSETS)
log("starcop root:", root, "| assets:", ASSETS, sorted(os.listdir(ASSETS)))
ARCHS = ["STARCOP_test", "STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
t = time.time()
mirror = {}
for a in ARCHS:
    base = os.path.join(root, a, a)
    if not os.path.isdir(base):
        log("MISSING archive dir", a); continue
    n = 0
    for tile in os.scandir(base):
        if tile.is_dir():
            for f in os.scandir(tile.path):
                mirror[(a, tile.name, f.name)] = f.stat().st_size; n += 1
    log(a, "files", n)
log("walk secs", round(time.time() - t, 1), "total mirror files", len(mirror), "bytes", sum(mirror.values()))
zen = {}
for a in ARCHS:
    for l in open(os.path.join(ASSETS, f"zenodo_{a}.tsv")):
        n, s, crc, ct = l.rstrip("\n").split("\t"); p = n.split("/")
        if len(p) == 3:
            zen[(a, p[1], p[2])] = (int(s), crc)
comp = {}
for a in ARCHS:
    keys = [k for k in zen if k[0] == a]
    present = sum(1 for k in keys if k in mirror and mirror[k] == zen[k][0])
    comp[a] = {"official_files": len(keys), "present_same_size": present, "missing": len(keys) - present}
log("completeness vs Zenodo:", json.dumps(comp))
REP["completeness"] = comp

# ---------------------------------------------------------------- 3. CRC32 of every test-split file (byte-identical to Zenodo?) + throughput
log("== CRC32 of all STARCOP_test files")
def crc_file(k):
    c = 0
    with open(os.path.join(root, k[0], k[0], k[1], k[2]), "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b: break
            c = zlib.crc32(b, c)
    return k, f"{c:08x}"
t = time.time(); keys = [k for k in zen if k[0] == "STARCOP_test"]; bad = []; nbytes = 0
with cf.ThreadPoolExecutor(8) as ex:
    for k, c in ex.map(crc_file, keys):
        nbytes += zen[k][0]
        if c != zen[k][1]: bad.append(k)
dt = time.time() - t
REP["test_crc"] = {"files": len(keys), "mismatch": len(bad), "secs": round(dt, 1), "MBps": round(nbytes / dt / 1e6, 1)}
log("test CRC:", REP["test_crc"], bad[:3])

# ---------------------------------------------------------------- 4. tiff decode + data semantics
log("== tiff decode")
from starcop_edge.tif import read_tif
BANDS = ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm", "labelbinary", "weight_mag1c"]
import pandas as pd
tr = pd.read_csv(os.path.join(ASSETS, "train.csv")); te = pd.read_csv(os.path.join(ASSETS, "test.csv"))
def arch_of(tile):
    for a in ARCHS[1:]:
        if (a, tile, "mag1c.tif") in mirror: return a
    return None
random.seed(0)
sample = random.sample(list(tr[tr.has_plume].id), 15) + random.sample(list(tr[~tr.has_plume].id), 5)
t = time.time(); stats = []; wm = []
for tile in sample:
    a = arch_of(tile)
    if a is None: continue
    arr = {}
    for b in BANDS:
        arr[b] = read_tif(open(os.path.join(root, a, a, tile, b + ".tif"), "rb").read())
    stats.append({b: [str(arr[b].dtype), tuple(arr[b].shape), float(np.nanmin(arr[b])), float(np.nanmax(arr[b])), int(np.isnan(arr[b]).sum())] for b in BANDS})
    m, w = arr["mag1c"].astype(np.float64).ravel(), arr["weight_mag1c"].astype(np.float64).ravel()
    wm.append((m[::7], w[::7]))
dt = time.time() - t
REP["decode_secs_per_tile_6bands"] = round(dt / max(1, len(stats)), 3)
log("decode secs/tile", REP["decode_secs_per_tile_6bands"], "dtypes", {b: stats[0][b][0] for b in BANDS})
m = np.concatenate([x[0] for x in wm]); w = np.concatenate([x[1] for x in wm])
bins = [-1e9, 0, 1, 50, 100, 200, 400, 800, 1600, 3200, 1e9]
tab = []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = (m > lo) & (m <= hi) if lo > -1e8 else (m <= hi)
    if s.any(): tab.append([lo, hi, int(s.sum()), float(w[s].min()), float(np.median(w[s])), float(w[s].max())])
REP["weight_vs_mag1c"] = tab
log("weight vs mag1c (bin lo, hi, n, wmin, wmed, wmax):"); [log("   ", r) for r in tab]
REP["weight_unique_vals_head"] = np.unique(np.round(w, 3))[:12].tolist()
log("distinct weight values (rounded 3dp), first 12:", REP["weight_unique_vals_head"], "n distinct", len(np.unique(np.round(w, 3))))
REP["any_nan"] = any(s[b][4] > 0 for s in stats for b in BANDS)

# ---------------------------------------------------------------- 5. Zenodo reachability + throughput (for the missing part2)
log("== zenodo throughput")
URL = "https://zenodo.org/records/7863343/files/STARCOP_train_remaining_part2.zip?download=1"
def rng(off, n):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off+n-1}"})
    with urllib.request.urlopen(req, timeout=120) as r: return len(r.read())
try:
    t = time.time(); n = rng(1_000_000_000, 64 << 20); dt1 = time.time() - t
    t = time.time()
    with cf.ThreadPoolExecutor(8) as ex: n8 = sum(ex.map(lambda i: rng(2_000_000_000 + i * (16 << 20), 16 << 20), range(8)))
    dt8 = time.time() - t
    REP["zenodo_MBps_1thread"] = round(n / dt1 / 1e6, 1); REP["zenodo_MBps_8threads"] = round(n8 / dt8 / 1e6, 1)
except Exception as e:
    REP["zenodo_error"] = repr(e)
log("zenodo MB/s:", REP.get("zenodo_MBps_1thread"), REP.get("zenodo_MBps_8threads"), REP.get("zenodo_error"))

# ---------------------------------------------------------------- 6. released-architecture equivalence vs real smp + CPU speed
log("== smp equivalence + cpu speed")
import torch
REP["torch"] = torch.__version__
from starcop_edge.models import SMPLikeUnet, EdgeUNet, load_released_checkpoint, count_params
sd, ck = load_released_checkpoint(os.path.join(ASSETS, "hyperstarcop_mag1c_rgb_final_checkpoint_model.ckpt"))
mine = SMPLikeUnet(4, 1); mine.load_state_dict(sd, strict=True); mine.eval()
REP["released_params"] = count_params(mine)
print(sh("pip install -q segmentation-models-pytorch 2>&1 | tail -2"))
try:
    import segmentation_models_pytorch as smp
    ref = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=4, classes=1, activation=None)
    ref.load_state_dict(sd, strict=True); ref.eval()
    x = torch.rand(2, 4, 256, 256) * 2
    with torch.no_grad(): d = (ref(x) - mine(x)).abs().max().item()
    REP["smp_version"] = smp.__version__; REP["smp_vs_mine_maxabs"] = d
    log("smp", smp.__version__, "max abs diff", d)
except Exception as e:
    REP["smp_error"] = repr(e); log("smp error", e)
for nt in (4, 2):
    torch.set_num_threads(nt)
    for widths in [(4, 8, 16, 32, 64), (8, 16, 32, 64, 128), (12, 24, 48, 96, 192)]:
        net = EdgeUNet(4, widths); opt = torch.optim.Adam(net.parameters(), 1e-3); net.train()
        x = torch.rand(32, 4, 128, 128); y = (torch.rand(32, 1, 128, 128) > 0.9).float()
        for i in range(2 + 6):
            if i == 2: t = time.time()
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y); loss.backward(); opt.step()
        sps = 6 / (time.time() - t)
        REP[f"train_steps_per_s_threads{nt}_{widths[0]}"] = round(sps, 2)
        log(f"threads={nt} widths={widths} params={count_params(net)} train steps/s (batch32@128): {sps:.2f}")
torch.set_num_threads(4)
x = torch.rand(1, 4, 512, 512)
with torch.inference_mode():
    for name, net in [("released", mine), ("edge8", EdgeUNet(4, (8, 16, 32, 64, 128)).eval())]:
        net(x); t = time.time(); [net(x) for _ in range(3)]
        REP[f"infer_s_per_tile_{name}_4threads"] = round((time.time() - t) / 3, 3)
log({k: v for k, v in REP.items() if k.startswith("infer_")})
REP["total_secs"] = round(time.time() - T0, 1)
json.dump(REP, open("/kaggle/working/feasibility.json", "w"), indent=1, default=str)
log("DONE", REP["total_secs"], "s")
