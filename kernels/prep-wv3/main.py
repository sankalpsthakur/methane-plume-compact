# bench-gas-prep-wv3: CPU kernel (private, internet on). Builds the WV3 FEATURE cache of Amendment 7 for the 3,425 train tiles: the three ratio channels of the paper's "Varon+Sanchez" input set, computed from the
# float32 TOA_WV3_SWIR1..8 bands of each tile (starcop_edge.wv3, a clean-room implementation), min-max normalised and stored as float16. Raw bands are NOT stored: float16 rounding of reflectance (relative step 5e-4 to 1e-3)
# is comparable to the weakest plume signals of the SWIR-7/8 ratios, while the features are stored after the division, scaled about 0 ("no anomaly" = exactly 0, where float16 is finest). (An 8-channel float16 band cache would be 14.4 GB, 15.3 GB with labels, which fits a kernel's 20 GB output; it is not built: choice of the builder, reversible before any push.)
#  - labels and tile metadata are COPIED from the mounted AVIRIS cache (kernel source bench-gas-prep-cache v1, labelbinary already CRC-verified; SHA-256 of train_Y.npy and train_meta.csv checked against CFG); no mag1c, no AVIRIS band
#  - every WV3 source file is CRC32/size-verified against the official Zenodo zip central directories before use; 2,852 tiles come from the Kaggle mirror, the 573 tiles of STARCOP_train_remaining_part2 (missing on the mirror) from Zenodo range reads
#  - the per-channel scale is fitted on the tiles OUTSIDE the validation dates (the fit set of the control and of G1), then applied to every tile; wv3_feat_norm.json carries it and the test loader of the stage kernels reads it
#  - the test split is never cached: the stage kernels read it from the CRC-verified mirror with the same code and the same limits (starcop_edge.wv3.load_wv3_test_from_mirror)
#  - at the end: a feature diagnostic (pixel AUROC plume vs non-plume per channel on TRAIN tiles; a bug check, never a selection input) and the crop-sampler throughput on this cache (sampler_bench.json, input of the stage planner)
import collections, concurrent.futures as cf, hashlib, json, os, shutil, sys, threading, time, traceback, zlib
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
from starcop_edge.tif import read_tif
from starcop_edge import zen, wv3 as W
from starcop_edge.kio import find_dir
assert CFG["split"] == "train", "the test split is never cached"
zen.set_concurrency(CFG.get("zen_conc", 12))

root = find_dir("STARCOP_test"); ASSETS = find_dir("zenodo_STARCOP_test.tsv"); AV = find_dir("train_Y.npy")
assert root and ASSETS and AV, ("mirror / assets / AVIRIS cache not mounted", root, ASSETS, AV)
OUT = "/kaggle/working/cache"; os.makedirs(OUT, exist_ok=True)
ARCHS = ["STARCOP_test", "STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
VAL_DATES = tuple(CFG.get("val_dates", ["2019-10-22", "2019-10-23"]))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(64 << 20), b""): h.update(blk)
    return h.hexdigest()

zen_m = {}; tile_arch = {}
for a in ARCHS:
    for l in open(os.path.join(ASSETS, f"zenodo_{a}.tsv")):
        n, s, crc, ct = l.rstrip("\n").split("\t"); p = n.split("/")
        if len(p) == 3:
            zen_m[(p[1], p[2])] = (int(s), crc); tile_arch[p[1]] = a
mirror_dir = {a: os.path.join(root, a, a) for a in ARCHS if os.path.isdir(os.path.join(root, a, a))}
log("archives on mirror:", sorted(mirror_dir))

df = pd.read_csv(os.path.join(ASSETS, "train.csv")); ids = list(df.id); N = len(ids)
# ------------------------------------------------------------------ labels + metadata from the mounted AVIRIS cache (verified by hash)
src_meta = pd.read_csv(os.path.join(AV, "train_meta.csv"))
assert list(src_meta.id) == ids, "AVIRIS cache tile ids differ from train.csv"
sha_y = sha(os.path.join(AV, "train_Y.npy")); sha_m = sha(os.path.join(AV, "train_meta.csv"))
assert sha_y == CFG["labels_sha256"] and sha_m == CFG["meta_sha256"], ("AVIRIS cache hashes differ from the recorded ones", sha_y, sha_m)
Ysrc = np.load(os.path.join(AV, "train_Y.npy"), mmap_mode="r"); assert Ysrc.shape == (N, 512, 512) and Ysrc.dtype == np.uint8
t = time.time(); shutil.copyfile(os.path.join(AV, "train_Y.npy"), os.path.join(OUT, "train_Y.npy")); log("copied labels", round(time.time() - t, 1), "s | sha256", sha_y[:16])
X = np.lib.format.open_memmap(f"{OUT}/train_X.npy", "w+", np.float16, (N, 3, 512, 512))
date_of = list(src_meta.date.astype(str)); fit_mask = [d not in VAL_DATES for d in date_of]

# ------------------------------------------------------------------ Zenodo central directories (only for archives the mirror lacks)
need_zen = sorted({tile_arch[t] for t in ids} - set(mirror_dir))
zinfo = {}; zsize = {}
for a in need_zen:
    zf = zen.open_archive(a); zsize[a] = zf.fp.size
    for i in zf.infolist():
        p = i.filename.split("/")
        if len(p) == 3: zinfo[(p[1], p[2])] = (a, i)
    log("zenodo central directory read for", a, len(zinfo))

report = collections.defaultdict(int); rep_lock = threading.Lock(); crc_events = []

def get_raw(tile, f, have):
    """verified bytes of one file; `have` keeps files already verified for this tile, so a tile-level retry does not re-fetch them"""
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
    raw = [read_tif(get_raw(tile, f, have)).astype(np.float32) for f in W.WV3_FILES]
    nan = int(sum(np.isnan(r).sum() for r in raw))
    bands = np.stack([np.nan_to_num(r, nan=0.0) for r in raw])
    feat, valid, info = W.ratio_features(bands)
    info = dict(info, nan_px=nan, src=("mirror" if tile_arch[tile] in mirror_dir else "zenodo"), band_mean=[float(bands[i][valid].mean()) if valid.any() else float("nan") for i in range(8)])   # band means: a scale sanity check (reflectance ~0.1-0.6)
    return feat.astype(np.float16), W.feature_sample(feat, valid), info

def process_retry(tid):
    have = {}; last = None
    for k in range(5):
        try: return process(tid, have)
        except Exception as e:
            last = e; time.sleep(min(60, 4 * 2 ** k))
    raise last

done = 0; failed = {}; infos = [None] * N; samples = [None] * N
def store(fu, i):
    global done
    try: x, smp, info = fu.result()
    except Exception as e: failed[i] = repr(e); return
    X[i] = x; samples[i] = smp; infos[i] = info; done += 1; failed.pop(i, None)

t = time.time(); todo = list(range(N))
is_z = {i: tile_arch[ids[i]] not in mirror_dir for i in todo}
with cf.ThreadPoolExecutor(CFG.get("mirror_workers", 8)) as exm, cf.ThreadPoolExecutor(CFG.get("zen_workers", 16)) as exz:
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
for rnd in range(3):  # cooled-down second pass over whatever is still missing
    if not failed: break
    cool = 60 * (rnd + 1); log("second pass", rnd + 1, ":", len(failed), "tiles still missing, cooling down", cool, "s"); time.sleep(cool)
    with cf.ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(process_retry, ids[i]): i for i in list(failed)}
        for fu in cf.as_completed(futs): store(fu, futs[fu])
X.flush()
log("fetch phase done in", round(time.time() - t, 1), "s | stored", done, "| still failed", len(failed), "| first-pass failures", first_pass_failed, "| zen", json.dumps(zen.STATS))
final_failed = [(ids[i], failed[i]) for i in sorted(failed)]
crc_fail = sum(1 for _, msg in final_failed if "CRC/size mismatch" in msg)

# ------------------------------------------------------------------ min-max limits from the fit tiles, in-place normalisation, metadata
okt = [i for i in range(N) if infos[i] is not None]
fit_idx = [i for i in okt if fit_mask[i] and infos[i]["ok"]]
assert fit_idx, "no tile to fit the normalisation limits on"
scale, p_lo, p_hi = W.fit_scale(np.concatenate([samples[i] for i in fit_idx], axis=1))
log("scale fitted on", len(fit_idx), "tiles (outside", VAL_DATES, ") | p_lo", np.round(p_lo, 5).tolist(), "p_hi", np.round(p_hi, 5).tolist(), "scale", np.round(scale, 5).tolist())
for c0 in range(0, N, 64):
    X[c0:c0 + 64] = W.normalise(np.asarray(X[c0:c0 + 64], np.float32), scale)
X.flush()
norm = dict(features=list(W.FEATURES), scale=[float(v) for v in scale], p_lo=[float(v) for v in p_lo], p_hi=[float(v) for v in p_hi], percentiles=list(W.NORM_PCT), sample_stride=W.SAMPLE_STRIDE, fit_tiles=len(fit_idx), fit_excludes_dates=list(VAL_DATES),
            fit_samples=int(sum(samples[i].shape[1] for i in fit_idx)), trim=list(W.TRIM), min_valid=W.MIN_VALID, raw_clip=W.RAW_CLIP, background_bands=list(W.BACKGROUND),
            wv3_py_sha256=hashlib.sha256(open(sys.modules["starcop_edge.wv3"].__file__, "rb").read()).hexdigest())
json.dump(norm, open(f"{OUT}/{W.NORM_FILE}", "w"), indent=1)
meta = src_meta.copy()
meta["wv3_ok"] = [bool(infos[i] and infos[i]["ok"]) for i in range(N)]
for k in ("valid_frac", "nan_px", "src", "c75", "c85", "c_sanchez"):
    meta["wv3_" + k] = [(infos[i].get(k) if infos[i] else None) for i in range(N)]
meta.to_csv(f"{OUT}/train_meta.csv", index=False)

# ------------------------------------------------------------------ diagnostics (TRAIN tiles only; a bug check, never used for selection)
diag = {}
try:
    Xd = np.load(f"{OUT}/train_X.npy", mmap_mode="r"); Yd = np.load(f"{OUT}/train_Y.npy", mmap_mode="r")
    plume = [i for i in okt if int(src_meta.n_label_px.iloc[i]) > 0 and infos[i]["ok"]]; pick = plume[::max(1, len(plume) // CFG.get("diag_tiles", 300))][:CFG.get("diag_tiles", 300)]
    pos = [[] for _ in range(3)]; neg = [[] for _ in range(3)]
    for i in pick:
        lab = np.asarray(Yd[i]).astype(bool); f = np.asarray(Xd[i], np.float32)
        for c in range(3): pos[c].append(f[c][lab]); neg[c].append(f[c][~lab][::37])
    for c, name in enumerate(W.FEATURES):
        p_, n_ = np.concatenate(pos[c]), np.concatenate(neg[c]); p_ = p_[::max(1, len(p_) // 200000)]; n_ = n_[::max(1, len(n_) // 200000)]
        a_ = W.auroc(p_, n_); diag[name] = dict(auroc_plume_gt_background=round(a_, 4), plume_low_separation=round(1 - a_, 4), plume_mean=float(p_.mean()), background_mean=float(n_.mean()), n_pos=int(len(p_)), n_neg=int(len(n_)))
    best = max(max(d["auroc_plume_gt_background"], d["plume_low_separation"]) for d in diag.values())
    diag["_summary"] = dict(tiles=len(pick), best_channel_separation=round(best, 4), min_required=CFG.get("min_auroc", 0.55), features_sane=bool(best >= CFG.get("min_auroc", 0.55)))
    log("feature diagnostic", json.dumps(diag))
except Exception:
    log("diagnostic FAILED"); traceback.print_exc(); diag["_summary"] = dict(features_sane=False, error="diagnostic crashed")
json.dump(diag, open("/kaggle/working/wv3_feature_diag.json", "w"), indent=1)

# ------------------------------------------------------------------ crop-sampler throughput on this cache (CPU kernel with the same host class as the GPU kernels; input of the stage planner)
bench = {}
if CFG.get("bench", True):
    try:
        from starcop_edge import data as D
        Xs, Ys, Ws, ms = D.load_split(OUT, "train", mmap=True); idx = np.arange(0, len(ms), 12)
        tb = time.time(); smp = D.CropSampler(Xs, Ys, Ws, idx, seed=0, jitter=32, batch=32); init_s = time.time() - tb
        smp.next(); tb = time.time(); nb = 150
        for _ in range(nb): smp.next()
        bench = {"batch32_batches_per_s": round(nb / (time.time() - tb), 1), "sampler_init_s_per_tile": round(init_s / len(idx), 4), "tiles": int(len(idx)), "channels": 3}
        smp.close(); json.dump(bench, open("/kaggle/working/sampler_bench.json", "w"), indent=1); log("sampler bench:", bench)
    except Exception:
        log("sampler bench FAILED"); traceback.print_exc()

# ------------------------------------------------------------------ report
del X
hs = {f: sha(f"{OUT}/{f}") for f in ["train_X.npy", "train_Y.npy", "train_meta.csv", W.NORM_FILE]}
bm = np.array([infos[i]["band_mean"] for i in range(N) if infos[i] and infos[i]["ok"]], np.float64)
report = dict(report)
report.update(train_tiles=N, tiles_stored=done, train_failed=final_failed, first_pass_failures=first_pass_failed, crc_fail=crc_fail, crc_mismatch_events=report.get("crc_mismatch_events", 0), crc_events=crc_events[:20],
              tiles_features_ok=int(meta.wv3_ok.sum()), tiles_features_not_ok=[ids[i] for i in range(N) if not meta.wv3_ok.iloc[i]][:50], wv3_nan_px=int(meta.wv3_nan_px.fillna(0).sum()),
              valid_frac_min=float(meta.wv3_valid_frac.min()), valid_frac_mean=float(meta.wv3_valid_frac.mean()), band_mean_over_tiles=[round(float(v), 5) for v in np.nanmean(bm, axis=0)] if len(bm) else None, src_counts={k: int(v) for k, v in meta.wv3_src.value_counts().items()},
              labels_source=dict(path=AV, sha256=sha_y, meta_sha256=sha_m), norm=norm, diag_summary=diag.get("_summary"), sampler_bench=bench, sha256=hs, zen_stats=zen.STATS, secs=round(time.time() - T0, 1))
json.dump(report, open("/kaggle/working/prep_report.json", "w"), indent=1, default=str)
log("REPORT", json.dumps({k: v for k, v in report.items() if k not in ("norm",)}, default=str))
CACHE_OK = not final_failed and crc_fail == 0 and done == N and int(meta.wv3_ok.sum()) >= 0.99 * N
log("CACHE_OK" if CACHE_OK else "CACHE_NOT_OK", "| features_sane:", (diag.get("_summary") or {}).get("features_sane"))
assert CACHE_OK, "tile failures, CRC mismatches or too many tiles without usable features (see REPORT above)"
log("DONE")
