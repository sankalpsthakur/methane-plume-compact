"""Simulated-WorldView-3 (WV3) inputs of the STARCOP release (Amendment 7).

Clean-room re-implementation of the three ratio channels of the paper's "Varon+Sanchez" input set (Sci. Rep. 13:19999, section "Feature extraction from multispectral data"). It was written from the
paper text and the Zenodo description ONLY: the STARCOP source (GPL-3.0) was not read and nothing was copied from it. Where the paper is silent the choice is ours and is fixed here (and in Amendment 7
item 3): the outlier rule, the intercept of the regression, the valid-pixel mask and the normalisation. numpy only (rasterio is imported lazily by the test loader).

Bands: B1..B8 are the files TOA_WV3_SWIR1.tif .. TOA_WV3_SWIR8.tif of each tile (simulated WV3 top-of-atmosphere reflectance, float32, 512 x 512).
Channels (the paper's "Varon+Sanchez" variant, in this order):
  0  VaronRatio(B7, B5)       = (c * B7 - B5) / B5
  1  VaronRatio(B8, B5)       = (c * B8 - B5) / B5
  2  SanchezRatio(B7)         = VaronRatio(B7, B7_MLR) with B7_MLR the per-tile multiple linear regression of B7 on B1, B2, B4, B5, B6 (with intercept)
with c = sum(B') / sum(S') over the kept pixels (the paper's simplified formula; S' and B' are the signal and background values "with removal of outliers").
Our outlier rule: a pixel is kept when every array entering the sum (signal, background) lies inside its own 1st-99th percentile range over the valid pixels of the tile.
A pixel is valid when all of B1, B2, B4-B8 are finite and > 0 (no-data outside the swath is 0 or NaN); features at invalid pixels are 0 ("no anomaly"). A tile with fewer than MIN_VALID valid pixels gets all-zero features.
Raw features are clipped to +-RAW_CLIP (physical anomalies are below 0.2; larger values come from near-zero background pixels).
Normalisation (the paper: "min-max normalisation ... using the statistics from the training dataset"; the exact form is not given). Ours is a symmetric max-abs scaling about the value "no anomaly": scale_c = max(|p_lo|, |p_hi|) with
p_lo, p_hi the 0.1st and 99.9th percentile of the raw feature over a strided sample of the valid pixels of the training tiles (fit set = train minus the validation dates), x_norm = clip(x / scale_c, -1, 1), stored as float16.
Reason for departing from a [0, 1] min-max: raw 0 ("no anomaly", also the value of invalid pixels) stays exactly 0, where float16 is finest, so storage rounding stays far below the sensor noise whatever the width of the
tails; with a [0, 1] map the neutral value would sit near 0.5-1, where one float16 step (5e-4) is comparable to the weakest plume signals of the ratios. A per-channel affine map is absorbed by the first convolution and its BatchNorm.
Train cache and test loader follow the same numeric path: raw float32 features -> float16 -> float32 -> x / scale -> clip -> float16 (the cache kernel stores the raw features as float16 first and scales them in place afterwards).
"""
import hashlib, json, os, zlib
import numpy as np

WV3_FILES = tuple(f"TOA_WV3_SWIR{i}.tif" for i in range(1, 9))
FEATURES = ("varon_b7_b5", "varon_b8_b5", "sanchez_b7")
BACKGROUND = (1, 2, 4, 5, 6)           # MLR regressors (paper: "B1-B2 and B4-B6 as background bands")
USED = (1, 2, 4, 5, 6, 7, 8)           # bands any channel uses (B3 is not used)
TRIM = (1.0, 99.0)                     # percentile range of the outlier rule
MIN_VALID = 1000
RAW_CLIP = 10.0
NORM_PCT = (0.1, 99.9)
SAMPLE_STRIDE = 16
NORM_FILE = "wv3_feat_norm.json"


def valid_mask(bands):
    """bands (8,H,W): True where every band used by a channel is finite and > 0."""
    sel = np.asarray(bands)[[i - 1 for i in USED]]
    return np.isfinite(sel).all(axis=0) & (sel > 0).all(axis=0)


def _trim(valid, *arrays, trim=TRIM):
    """pixels of `valid` whose value is inside the [trim] percentile range of every given array (computed over the valid pixels)."""
    keep = valid.copy()
    for a in arrays:
        lo, hi = np.percentile(a[valid], trim)
        keep &= (a >= lo) & (a <= hi)
    return keep


def varon_ratio(S, B, keep, valid):
    """(c * S - B) / B with c = sum(B[keep]) / sum(S[keep]); 0 outside `valid` (B must be > 0 there). Returns (float32 array, c)."""
    ss = S[keep].sum(dtype=np.float64)
    if not ss > 0: raise ValueError("signal band sums to zero over the kept pixels")
    c = B[keep].sum(dtype=np.float64) / ss
    out = np.zeros(S.shape, np.float32)
    s = S[valid].astype(np.float64); b = B[valid].astype(np.float64)
    out[valid] = ((c * s - b) / b).astype(np.float32)
    return out, float(c)


def mlr_background(S, regressors, keep):
    """Per-tile multiple linear regression (least squares, with intercept) of S on `regressors`, fitted on the pixels of `keep`; the fit is evaluated at every pixel. Returns (float64 array, beta)."""
    A = np.column_stack([np.ones(int(keep.sum()))] + [r[keep].astype(np.float64) for r in regressors])
    beta = np.linalg.lstsq(A, S[keep].astype(np.float64), rcond=None)[0]
    pred = beta[0] + sum(bk * r.astype(np.float64) for bk, r in zip(beta[1:], regressors))
    return pred, beta


def ratio_features(bands):
    """bands (8,H,W) TOA reflectance of SWIR1..8 -> (raw features (3,H,W) float32, valid mask (H,W), info dict). All-zero features (info ok False) for a tile with too few valid pixels or a degenerate fit."""
    b = np.asarray(bands, np.float32); H, W = b.shape[1:]
    valid = valid_mask(b); n = int(valid.sum())
    out = np.zeros((3, H, W), np.float32); info = dict(n_valid=n, valid_frac=n / float(H * W), ok=False)
    if n < MIN_VALID: return out, valid, info
    B = {i: b[i - 1] for i in range(1, 9)}
    try:
        out[0], c75 = varon_ratio(B[7], B[5], _trim(valid, B[7], B[5]), valid)
        out[1], c85 = varon_ratio(B[8], B[5], _trim(valid, B[8], B[5]), valid)
        keep_m = _trim(valid, B[7])
        pred, beta = mlr_background(B[7], [B[r] for r in BACKGROUND], keep_m)
        v2 = valid & (pred > 1e-6); keep2 = keep_m & v2
        out[2], c_s = varon_ratio(B[7], pred, keep2, v2)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.zeros((3, H, W), np.float32), valid, info
    np.clip(out, -RAW_CLIP, RAW_CLIP, out=out)
    info.update(ok=bool(np.isfinite(out).all()), c75=c75, c85=c85, c_sanchez=c_s, beta=[float(x) for x in beta], n_pred_nonpositive=int((valid & ~v2).sum()))
    if not info["ok"]: out = np.zeros((3, H, W), np.float32)
    return out, valid, info


def feature_sample(feat, valid, stride=SAMPLE_STRIDE):
    """deterministic strided sample (every `stride`-th valid pixel in raster order) of the raw features: (3, n) float32."""
    idx = np.flatnonzero(valid.ravel())[::stride]
    return feat.reshape(3, -1)[:, idx]


def fit_scale(samples, pct=NORM_PCT):
    """per-channel scale = max(|p_lo|, |p_hi|) of the pooled training sample (3, n); returns (scale, p_lo, p_hi) as float64 arrays"""
    p_lo = np.percentile(samples, pct[0], axis=1); p_hi = np.percentile(samples, pct[1], axis=1)
    scale = np.maximum(np.abs(p_lo), np.abs(p_hi))
    if not (scale > 0).all(): raise ValueError(("degenerate normalisation scale", p_lo, p_hi))
    return scale.astype(np.float64), p_lo.astype(np.float64), p_hi.astype(np.float64)


def normalise(feat, scale):
    """clip(x / scale, -1, 1) per channel, float16. feat (3,H,W) or (N,3,H,W)."""
    inv = (1.0 / np.asarray(scale, np.float32)).astype(np.float32)
    shp = (3, 1, 1) if feat.ndim == 3 else (1, 3, 1, 1)
    return np.clip(np.asarray(feat, np.float32) * inv.reshape(shp), -1.0, 1.0).astype(np.float16)


def auroc(pos, neg):
    """rank-based AUROC of scores `pos` vs `neg` (1-D); ties get mid-ranks."""
    s = np.concatenate([pos, neg]).astype(np.float64); o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if len(u) < len(s):
        sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def load_norm(path):
    d = json.load(open(path))
    return np.array(d["scale"], np.float64), d


def load_wv3_test_from_mirror(mirror_root, assets_dir, norm_path, verbose=True):
    """The 342 official test tiles read straight from the Kaggle mirror, every file CRC32-verified against the official Zenodo zip central directory (as data.load_test_from_mirror), turned into the normalised
    three-channel WV3 features with the SAME code and the SAME limits as the train cache. Returns X (N,3,512,512) float16, Y (N,512,512) uint8, None, meta."""
    import pandas as pd
    from .tif import read_tif
    scale, _ = load_norm(norm_path)
    meta = pd.read_csv(os.path.join(assets_dir, "test.csv"))
    crc = {}
    for l in open(os.path.join(assets_dir, "zenodo_STARCOP_test.tsv")):
        n, sz, c, ct = l.rstrip("\n").split("\t"); p = n.split("/")
        if len(p) == 3: crc[(p[1], p[2])] = c
    base = os.path.join(mirror_root, "STARCOP_test", "STARCOP_test")
    def rd(tid, f):
        raw = open(os.path.join(base, tid, f), "rb").read()
        assert f"{zlib.crc32(raw):08x}" == crc[(tid, f)], ("CRC mismatch", tid, f)
        return read_tif(raw)
    N = len(meta); X = np.empty((N, 3, 512, 512), np.float16); Y = np.empty((N, 512, 512), np.uint8); bad = 0
    for i, tid in enumerate(meta.id):
        bands = np.stack([np.nan_to_num(rd(tid, f).astype(np.float32), nan=0.0) for f in WV3_FILES])
        feat, _, info = ratio_features(bands); bad += int(not info["ok"])
        X[i] = normalise(feat.astype(np.float16), scale); Y[i] = rd(tid, "labelbinary.tif").astype(np.uint8)
    if verbose: print("WV3 test tiles loaded from mirror (CRC verified):", X.shape, "| tiles with all-zero features:", bad, flush=True)
    return X, Y, None, meta


def load_wv3_test_cached(mirror_root, assets_dir, norm_path, cache_dir, verbose=True):
    """load_wv3_test_from_mirror, but the computed features are kept in `cache_dir` (a temporary directory of the kernel, never /kaggle/working: the test split is not written to a kernel's output) so that the next run of the same kernel
    reads them in a second instead of recomputing them (about 150 s per run). The cache is valid only for the same normalisation file and the same wv3.py (marker file, written last; files are replaced atomically, so two runs
    that finish together may both compute, and neither can read a half-written file)."""
    import pandas as pd
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    key = dict(norm_sha256=sha(norm_path), wv3_py_sha256=sha(__file__)); meta = pd.read_csv(os.path.join(assets_dir, "test.csv")); key["n"] = len(meta)
    xp, yp, mp = (os.path.join(cache_dir, f) for f in ("wv3_test_X.npy", "wv3_test_Y.npy", "wv3_test_cache.json"))
    try:
        if os.path.exists(mp) and json.load(open(mp)) == key:
            X, Y = np.load(xp), np.load(yp)
            if verbose: print("WV3 test features read from the kernel-local cache", cache_dir, X.shape, flush=True)
            return X, Y, None, meta
    except (OSError, ValueError): pass
    X, Y, _, meta = load_wv3_test_from_mirror(mirror_root, assets_dir, norm_path, verbose)
    os.makedirs(cache_dir, exist_ok=True); pid = os.getpid()
    for path, arr in ((xp, X), (yp, Y)):
        tmp = f"{path}.tmp{pid}.npy"; np.save(tmp, arr); os.replace(tmp, path)
    tmp = f"{mp}.tmp{pid}"; json.dump(key, open(tmp, "w")); os.replace(tmp, mp)
    return X, Y, None, meta
