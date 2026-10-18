# bench-gas-prep-cache: CPU kernel (private, internet on). Builds or COMPLETES the compact STARCOP TRAIN cache used by every dev/final kernel.
#  mode "patch"  : a mounted earlier cache (kernel source bench-gas-prep-train v2, which ended in ERROR with 12 of 3,425 tiles missing because
#                  Zenodo answered HTTP 429) is copied, and only the tiles marked ok=False are re-fetched from the official archives.
#  mode "rebuild": no usable base cache is mounted -> everything is built from the Kaggle mirror (+ Zenodo for remaining_part2).
#  - the test split is NOT cached here: final kernels read it from the CRC-verified mirror (starcop_edge.data.load_test_from_mirror)
#  - every source file is CRC32/size-verified against the official Zenodo zip central directories before it is stored
#  - afterwards (CFG["extras"]): CPU-throughput probe, edge-export self-test, and short train.py / edge-scoring smoke runs on TRAIN tiles only
import collections, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, threading, time, traceback, zlib
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
from starcop_edge.tif import read_tif
from starcop_edge import zen
from starcop_edge.kio import find_dir
assert CFG["split"] == "train", "the test split is never cached (final kernels read it from the CRC-verified mirror)"
zen.set_concurrency(CFG.get("zen_conc", 12))

root = find_dir("STARCOP_test"); ASSETS = find_dir("zenodo_STARCOP_test.tsv")
assert root and ASSETS, (root, ASSETS)
OUT = "/kaggle/working/cache"; os.makedirs(OUT, exist_ok=True)
ARCHS = ["STARCOP_test", "STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
BANDS = ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
FACT = [1750.0, 60.0, 60.0, 60.0]
NEED = BANDS + ["labelbinary", "weight_mag1c"]
INFO = ["n_label_px", "nan_px", "w_quant_err", "w_min", "w_max", "valid_frac", "src", "mag1c_nonzero"]

zen_m = {}; tile_arch = {}
for a in ARCHS:
    for l in open(os.path.join(ASSETS, f"zenodo_{a}.tsv")):
        n, s, crc, ct = l.rstrip("\n").split("\t"); p = n.split("/")
        if len(p) == 3:
            zen_m[(p[1], p[2])] = (int(s), crc); tile_arch[p[1]] = a
mirror_dir = {a: os.path.join(root, a, a) for a in ARCHS if os.path.isdir(os.path.join(root, a, a))}
log("archives on mirror:", sorted(mirror_dir))

df = pd.read_csv(os.path.join(ASSETS, "train.csv")); ids = list(df.id); N = len(ids)
SHAPES = {"train_X.npy": ((N, 4, 512, 512), np.float16), "train_Y.npy": ((N, 512, 512), np.uint8), "train_W.npy": ((N, 512, 512), np.uint8)}

# ------------------------------------------------------------------ base cache (patch mode) or fresh arrays (rebuild mode)
BASE = None; base_meta = None; base_report = None
bdir = find_dir("train_X.npy")
if bdir:
    try:
        bm = pd.read_csv(os.path.join(bdir, "train_meta.csv"))
        assert list(bm.id) == ids, "base cache tile ids differ from train.csv"
        for f, (shp, dt) in SHAPES.items():
            arr = np.load(os.path.join(bdir, f), mmap_mode="r"); assert arr.shape == shp and arr.dtype == dt, (f, arr.shape, arr.dtype)
        BASE, base_meta = bdir, bm
        rp = os.path.join(os.path.dirname(bdir), "prep_report.json")
        if os.path.exists(rp): base_report = json.load(open(rp))
    except Exception as e:
        log("base cache unusable -> REBUILD of the whole cache:", repr(e)); traceback.print_exc()
else:
    log("no base cache mounted -> REBUILD of the whole cache")
MODE = "patch" if BASE else "rebuild"
log("mode:", MODE, "| base:", BASE)

extra = [None] * N
if BASE:
    for f in SHAPES:
        t = time.time(); shutil.copyfile(os.path.join(BASE, f), os.path.join(OUT, f)); log("copied", f, round(os.path.getsize(os.path.join(OUT, f)) / 1e9, 2), "GB in", round(time.time() - t, 1), "s")
    X, Y, W = (np.load(os.path.join(OUT, f), mmap_mode="r+") for f in SHAPES)
    okb = base_meta.ok.astype(bool).values; rec = base_meta[INFO].to_dict("records")
    for i in range(N):
        if okb[i]: extra[i] = rec[i]
    todo = [i for i in range(N) if not okb[i]]
else:
    X = np.lib.format.open_memmap(f"{OUT}/train_X.npy", "w+", np.float16, (N, 4, 512, 512))
    Y = np.lib.format.open_memmap(f"{OUT}/train_Y.npy", "w+", np.uint8, (N, 512, 512))
    W = np.lib.format.open_memmap(f"{OUT}/train_W.npy", "w+", np.uint8, (N, 512, 512))
    todo = list(range(N))
log("tiles to fetch:", len(todo), "of", N)

# ------------------------------------------------------------------ Zenodo central directories (only for archives the todo tiles live in and the mirror lacks)
need_zen = sorted({tile_arch[ids[i]] for i in todo} - set(mirror_dir))
zinfo = {}; zsize = {}
for a in need_zen:
    zf = zen.open_archive(a); zsize[a] = zf.fp.size
    for i in zf.infolist():
        p = i.filename.split("/")
        if len(p) == 3: zinfo[(p[1], p[2])] = (a, i)
    log("zenodo central directory read for", a, len(zinfo))

report = collections.defaultdict(int); rep_lock = threading.Lock(); crc_events = []

def get_raw(tile, band, have):
    """verified bytes of one file; `have` keeps files already verified for this tile, so a tile-level retry does not re-fetch them"""
    f = band + ".tif"
    if f in have: return have[f]
    size, crc = zen_m[(tile, f)]; a = tile_arch[tile]
    if a in mirror_dir:
        with open(os.path.join(mirror_dir[a], tile, f), "rb") as fh: raw = fh.read(); src = "mirror"
    else:
        arch, zi = zinfo[(tile, f)]; raw = zen.read_member(zen.archive_url(arch), zi, zsize[arch]); src = "zenodo"
    ok = (len(raw) == size) and (f"{zlib.crc32(raw):08x}" == crc)
    with rep_lock:
        report[f"{src}_fetches"] += 1; report[f"{src}_bytes_fetched"] += len(raw)
        if not ok: report["crc_mismatch_events"] += 1; crc_events.append((tile, f, src))
        else: report[f"{src}_files_verified"] += 1
    if not ok: raise ValueError(f"CRC/size mismatch {tile}/{f} from {src}")
    have[f] = raw
    return raw

def process(tile, have):
    arr = {b: read_tif(get_raw(tile, b, have)) for b in NEED}
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

def process_retry(tid):
    have = {}; last = None
    for k in range(5):
        try: return process(tid, have)
        except Exception as e:
            last = e; time.sleep(min(60, 4 * 2 ** k))
    raise last

done = 0; failed = {}  # tile index -> repr(last error)
def store(fu, i):
    global done
    try: x, y, w, info = fu.result()
    except Exception as e: failed[i] = repr(e); return
    X[i] = x; Y[i] = y; W[i] = w; extra[i] = info; done += 1; failed.pop(i, None)

t = time.time()
is_z = {i: tile_arch[ids[i]] not in mirror_dir for i in todo}
with cf.ThreadPoolExecutor(12) as exm, cf.ThreadPoolExecutor(CFG.get("zen_workers", 16)) as exz:
    zfuts = {exz.submit(process_retry, ids[i]): i for i in todo if is_z[i]}
    log("submitted", len(zfuts), "zenodo tiles")
    mir = [i for i in todo if not is_z[i]]
    for c0 in range(0, len(mir), 240):  # bounded number of mirror tiles in flight
        mf = {exm.submit(process_retry, ids[i]): i for i in mir[c0:c0 + 240]}
        for fu in cf.as_completed(mf): store(fu, mf[fu])
        log("mirror tiles", min(c0 + 240, len(mir)), "/", len(mir), "| stored", done, f"({done/(time.time()-t):.1f} tiles/s)")
    zd = 0
    for fu in cf.as_completed(zfuts):
        store(fu, zfuts[fu]); zd += 1
        if zd % 50 == 0 or zd == len(zfuts): log("zenodo tiles", zd, "/", len(zfuts), "| stored", done, "| failed so far", len(failed), "| zen", json.dumps(zen.STATS))
first_pass_failed = len(failed)
for rnd in range(3):  # cooled-down sequential-ish second pass over whatever is still missing
    if not failed: break
    cool = 60 * (rnd + 1); log("second pass", rnd + 1, ":", len(failed), "tiles still missing, cooling down", cool, "s"); time.sleep(cool)
    with cf.ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(process_retry, ids[i]): i for i in list(failed)}
        for fu in cf.as_completed(futs): store(fu, futs[fu])
X.flush(); Y.flush(); W.flush()
log("fetch phase done in", round(time.time() - t, 1), "s | stored", done, "| still failed", len(failed), "| first-pass failures", first_pass_failed, "| zen", json.dumps(zen.STATS))

# ------------------------------------------------------------------ meta + audit
meta = df[["id", "name", "date", "qplume", "has_plume"]].copy()
for k in INFO: meta[k] = [extra[i].get(k) if extra[i] else None for i in range(N)]
meta["ok"] = [bool(extra[i]) for i in range(N)]
meta.to_csv(f"{OUT}/train_meta.csv", index=False)
m_ok = meta[meta.ok]
mism = m_ok[(m_ok.n_label_px > 0) != m_ok.has_plume]
mism[["id", "date", "qplume", "has_plume", "n_label_px", "src"]].to_csv("/kaggle/working/has_plume_label_mismatch.csv", index=False)
final_failed = [(ids[i], failed[i]) for i in sorted(failed)]
crc_fail = sum(1 for _, msg in final_failed if "CRC/size mismatch" in msg)

audit = {}
if BASE:  # every tile that was already ok in the base cache must be bit-identical in the copy
    Xb, Yb, Wb = (np.load(os.path.join(BASE, f), mmap_mode="r") for f in SHAPES); td = set(todo); bad = []
    for i in range(N):
        if i in td: continue
        if not (np.array_equal(X[i], Xb[i]) and np.array_equal(Y[i], Yb[i]) and np.array_equal(W[i], Wb[i])): bad.append(ids[i])
    audit["copied_tiles_identical_to_base"] = (N - len(td)) - len(bad); audit["copied_tiles_differing"] = bad[:20]
    del Xb, Yb, Wb
patched = [i for i in todo if i not in failed]
audit["patched_tiles"] = len(patched)
audit["patched_tile_checks_failed"] = [ids[i] for i in patched if not (np.abs(X[i].astype(np.float32)).max() > 0 and int(Y[i].sum()) == int(meta.n_label_px[i]) and int(W[i].min()) >= 25)]
audit["all_tiles_marked_ok"] = bool(meta.ok.all())
audit["tiles_with_valid_frac_below_1pct"] = int((meta.valid_frac < 0.01).sum())
del X, Y, W
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(64 << 20), b""): h.update(blk)
    return h.hexdigest()
with cf.ThreadPoolExecutor(4) as ex:
    hs = dict(zip(["train_X.npy", "train_Y.npy", "train_W.npy", "train_meta.csv"], ex.map(sha, [f"{OUT}/{f}" for f in ["train_X.npy", "train_Y.npy", "train_W.npy", "train_meta.csv"]])))

report = dict(report)
report.update(mode=MODE, base=BASE, base_report_train_failed=(len(base_report.get("train_failed", [])) if base_report else None), tiles_fetched_in_this_run=len(todo),
              train_tiles=N, train_ok=int(meta.ok.sum()), train_failed=final_failed, first_pass_failures=first_pass_failed,
              crc_fail=crc_fail, crc_mismatch_events=report.get("crc_mismatch_events", 0), crc_events=crc_events[:20],
              train_has_plume_mismatch=int(len(mism)), has_plume_mismatch_ids=list(mism.id), train_src_counts={k: int(v) for k, v in m_ok.src.value_counts().items()},
              nan_px=int(m_ok.nan_px.sum()), max_weight_quant_err=float(m_ok.w_quant_err.max()), weight_range=[float(m_ok.w_min.min()), float(m_ok.w_max.max())],
              audit=audit, sha256=hs, zen_stats=zen.STATS, base_report=base_report, secs=round(time.time() - T0, 1))
json.dump(report, open("/kaggle/working/prep_report.json", "w"), indent=1, default=str)
log("REPORT", json.dumps({k: v for k, v in report.items() if k not in ("base_report",)}, default=str))
CACHE_OK = not final_failed and crc_fail == 0 and audit["all_tiles_marked_ok"] and not audit["patched_tile_checks_failed"] and not audit.get("copied_tiles_differing")
log("CACHE_OK" if CACHE_OK else "CACHE_NOT_OK")

# ------------------------------------------------------------------ extras: probes and smoke runs (never part of model selection; TRAIN tiles only)
if CFG.get("extras") and CACHE_OK:
    from starcop_edge import probe
    for nm, fn in (("cpu probe", probe.run_cpu_probe), ("edge self-test", probe.edge_selftest)):
        try: fn(log)
        except Exception: log(nm, "FAILED"); traceback.print_exc()
    code = "import sys; sys.path.insert(0, '/tmp/pkg'); from starcop_edge import train; train.main(sys.argv[1:])"
    def smoke(name, more):
        out = f"/kaggle/working/smoke/{name}"; t1 = time.time()
        args = ["--data", OUT, "--out", out, "--tag", "smoke_" + name, "--steps", "100", "--batch", "16", "--lr", "0.002", "--widths", "8,16,32,64,128", "--stem_stride", "2",
                "--val_dates", "2019-10-22,2019-10-23", "--val_max", "12", "--eval_every", "50", "--log_every", "25", "--threads", "2"] + more
        r = subprocess.run([sys.executable, "-c", code] + args, capture_output=True, text=True)
        log("smoke", name, "rc", r.returncode, "wall", round(time.time() - t1, 1), "s |", " | ".join((r.stdout + r.stderr).strip().splitlines()[-4:])[:900])
        return out, r.returncode
    s1, rc1 = smoke("d2", []); s2, rc2 = smoke("d6hard", ["--hardneg", "0.5"])
    if rc1 == 0:
        try:  # scoring path of the final kernel on 4 TRAIN tiles standing in for test (the real test tiles are only read by the final kernels)
            from starcop_edge import edge_eval, data as D
            from starcop_edge.models import count_params
            net, ck = edge_eval.load_run(s1); calib = edge_eval.calib_tiles(OUT, n_each=4)
            Xs, Ys, _, ms = D.load_split(OUT, "train"); pl = np.where(ms.has_plume.values & (ms.n_label_px.values > 0))[0][:2]; ng = np.where(~ms.has_plume.values)[0][:2]; idx = np.concatenate([pl, ng])
            Xt = np.stack([np.asarray(Xs[i]) for i in idx]); Yt = np.stack([np.asarray(Ys[i]) for i in idx]); mt = ms.iloc[idx]
            os.makedirs("/kaggle/working/smoke/edge", exist_ok=True)
            rec = edge_eval._export_and_score("smoke", net, count_params(net), os.path.getsize(os.path.join(s1, "model.pt")), "/kaggle/working/smoke/edge", calib, Xt, Yt, mt.qplume.values.astype(np.float64), mt, log)
            json.dump(rec, open("/kaggle/working/smoke/edge/edge_smoke.json", "w"), indent=1, default=str); log("edge scoring smoke OK")
        except Exception: log("edge scoring smoke FAILED"); traceback.print_exc()

assert CACHE_OK, "CRC/tile failures or cache audit failed (see REPORT above)"
log("DONE")
