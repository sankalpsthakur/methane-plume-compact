# bench-gas-prep: CPU kernel (private, internet on). Builds the compact STARCOP cache used by every later kernel.
#  - test + train_easy + remaining parts 1,3,4,5 are read from the Kaggle mirror chrisphamda/starcop
#  - remaining part2 (missing from the mirror) is fetched member-by-member from the official Zenodo archive
#  - every file used is CRC32-verified against the official zip central directories
import collections, concurrent.futures as cf, io, json, os, subprocess, sys, threading, time, zlib
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
from starcop_edge.tif import read_tif
from starcop_edge import zen
SPLIT = CFG["split"]  # "train" or "test": written by SEPARATE kernels so that dev kernels can mount the train cache only

IN = "/kaggle/input"
from starcop_edge.kio import find_dir
root = find_dir("STARCOP_test"); ASSETS = find_dir("zenodo_STARCOP_test.tsv")
assert root and ASSETS, (root, ASSETS)
OUT = "/kaggle/working/cache"; os.makedirs(OUT, exist_ok=True)
ARCHS = ["STARCOP_test", "STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
BANDS = ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
FACT = [1750.0, 60.0, 60.0, 60.0]
NEED = BANDS + ["labelbinary", "weight_mag1c"]

zen_m = {}; tile_arch = {}
for a in ARCHS:
    for l in open(os.path.join(ASSETS, f"zenodo_{a}.tsv")):
        n, s, crc, ct = l.rstrip("\n").split("\t"); p = n.split("/")
        if len(p) == 3:
            zen_m[(p[1], p[2])] = (int(s), crc); tile_arch[p[1]] = a
mirror_dir = {a: os.path.join(root, a, a) for a in ARCHS if os.path.isdir(os.path.join(root, a, a))}
log("archives on mirror:", sorted(mirror_dir))
need_zen = sorted({a for a in ARCHS if a not in mirror_dir}) if SPLIT == "train" else []
zinfo = {}; zsize = {}
for a in need_zen:
    zf = zen.open_archive(a); zsize[a] = zf.fp.size
    for i in zf.infolist():
        p = i.filename.split("/")
        if len(p) == 3: zinfo[(p[1], p[2])] = (a, i)
    log("zenodo central directory read for", a, len(zinfo))

report = collections.defaultdict(int); fails = []
rep_lock = threading.Lock()

def get_raw(tile, band):
    f = band + ".tif"; size, crc = zen_m[(tile, f)]; a = tile_arch[tile]
    if a in mirror_dir:
        with open(os.path.join(mirror_dir[a], tile, f), "rb") as fh: raw = fh.read(); src = "mirror"
    else:
        arch, zi = zinfo[(tile, f)]; raw = zen.read_member(zen.archive_url(arch), zi, zsize[arch]); src = "zenodo"
    ok = (len(raw) == size) and (f"{zlib.crc32(raw):08x}" == crc)
    with rep_lock:
        report[f"{src}_files"] += 1; report[f"{src}_bytes"] += len(raw)
        if not ok: report["crc_fail"] += 1; fails.append((tile, f, src))
    if not ok: raise ValueError(f"CRC/size mismatch {tile}/{f} from {src}")  # retried by process_retry
    return raw, src

def process(tile):
    arr = {b: read_tif(get_raw(tile, b)[0]) for b in NEED}
    x = np.empty((4, 512, 512), np.float16); nan = 0
    for i, (b, f) in enumerate(zip(BANDS, FACT)):
        v = arr[b].astype(np.float32); nan += int(np.isnan(v).sum()); v = np.nan_to_num(v, nan=0.0)
        x[i] = np.clip(v / f, 0.0, 2.0)
    y = arr["labelbinary"].astype(np.uint8); assert set(np.unique(y)).issubset({0, 1}), np.unique(y)
    w = arr["weight_mag1c"].astype(np.float64); wq = np.round(w * 255.0).astype(np.uint8)
    werr = float(np.abs(wq / 255.0 - w).max()); wmin, wmax = float(w.min()), float(w.max())
    valid = float((arr["TOA_AVIRIS_640nm"] > 0).mean())
    return x, y, wq, dict(n_label_px=int(y.sum()), nan_px=nan, w_quant_err=werr, w_min=wmin, w_max=wmax, valid_frac=valid,
                          src=("mirror" if tile_arch[tile] in mirror_dir else "zenodo"), mag1c_nonzero=float((arr["mag1c"] > 0).mean()))

for split, csv in ((SPLIT, f"{SPLIT}.csv"),):
    df = pd.read_csv(os.path.join(ASSETS, csv)); ids = list(df.id); N = len(ids)
    X = np.lib.format.open_memmap(f"{OUT}/{split}_X.npy", "w+", np.float16, (N, 4, 512, 512))
    Y = np.lib.format.open_memmap(f"{OUT}/{split}_Y.npy", "w+", np.uint8, (N, 512, 512))
    W = np.lib.format.open_memmap(f"{OUT}/{split}_W.npy", "w+", np.uint8, (N, 512, 512))
    extra = [None] * N; t = time.time(); done = 0; failed = []
    def process_retry(tid):
        last = None
        for k in range(4):
            try:
                return process(tid)
            except Exception as e:  # network hiccup etc.
                last = e; time.sleep(3 * (k + 1))
        raise last
    is_z = [tile_arch[i] not in mirror_dir for i in ids]
    def store(fu, i):
        global done
        try: x, y, w, info = fu.result()
        except Exception as e: failed.append((ids[i], repr(e))); extra[i] = {}; return
        X[i] = x; Y[i] = y; W[i] = w; extra[i] = info; done += 1
    with cf.ThreadPoolExecutor(12) as exm, cf.ThreadPoolExecutor(32) as exz:
        zfuts = {exz.submit(process_retry, ids[i]): i for i in range(N) if is_z[i]}  # slow Zenodo path (~1 MB/s per connection): many threads
        log("submitted", len(zfuts), "zenodo tiles")
        mir = [i for i in range(N) if not is_z[i]]
        for c0 in range(0, len(mir), 240):  # bounded number of mirror tiles in flight
            mf = {exm.submit(process_retry, ids[i]): i for i in mir[c0:c0 + 240]}
            for fu in cf.as_completed(mf): store(fu, mf[fu])
            log(split, "mirror tiles", min(c0 + 240, len(mir)), "/", len(mir), "| total stored", done, f"({done/(time.time()-t):.1f} tiles/s)")
        zd = 0
        for fu in cf.as_completed(zfuts):
            store(fu, zfuts[fu]); zd += 1
            if zd % 50 == 0: log(split, "zenodo tiles", zd, "/", len(zfuts), "| total stored", done)
    X.flush(); Y.flush(); W.flush(); del X, Y, W
    ok = [i for i in range(N) if extra[i]]
    meta = df[["id", "name", "date", "qplume", "has_plume"]].copy()
    for k in extra[ok[0]]: meta[k] = [extra[i].get(k) if extra[i] else None for i in range(N)]
    meta["ok"] = [bool(extra[i]) for i in range(N)]
    meta.to_csv(f"{OUT}/{split}_meta.csv", index=False)
    m_ok = meta[meta.ok]
    mism = int(((m_ok.n_label_px > 0) != m_ok.has_plume).sum())
    log(split, "done in", round(time.time() - t, 1), "s | failed tiles:", len(failed), "| has_plume vs label px mismatches:", mism, "| nan px:", int(m_ok.nan_px.sum()),
        "| max weight quant err:", float(m_ok.w_quant_err.max()), "| weight range:", float(m_ok.w_min.min()), float(m_ok.w_max.max()))
    report[f"{split}_tiles"] = N; report[f"{split}_failed"] = failed[:20]; report[f"{split}_has_plume_mismatch"] = mism
    report[f"{split}_src_counts"] = dict(m_ok.src.value_counts())

report = dict(report); report["fails"] = fails[:20]; report["secs"] = round(time.time() - T0, 1)
json.dump(report, open("/kaggle/working/prep_report.json", "w"), indent=1, default=str)
log("REPORT", json.dumps(report, default=str))
assert report.get("crc_fail", 0) == 0 and not report.get(f"{SPLIT}_failed"), "CRC/tile failures!"
log("DONE")
