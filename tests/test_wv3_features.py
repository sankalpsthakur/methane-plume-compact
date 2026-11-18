"""Amendment 7: the clean-room WV3 ratio channels (src/starcop_edge/wv3.py) on synthetic tiles (numpy only, rasterio faked for the test loader).
Checks what the paper's products are supposed to do: a methane-like absorption in B7/B8 (a few percent) shows up as a negative anomaly in the three channels; the channels are blind to the surface albedo
(a common multiplicative field), invariant to a global gain, robust to a few outlier pixels, zero at invalid pixels; the normalisation maps into [0, 1]; the test loader reads the mirror layout with the CRC guard
and gives the same values as the direct computation."""
import io, json, os, shutil, sys, tempfile, types, zlib
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

class _DS:
    def __init__(s, a): s.a = a
    def __enter__(s): return s
    def __exit__(s, *x): pass
    def read(s): return s.a[None] if s.a.ndim == 2 else s.a
class _MF:
    def __init__(s, raw): s.a = np.load(io.BytesIO(raw))
    def __enter__(s): return s
    def __exit__(s, *x): pass
    def open(s): return _DS(s.a)
fake = types.ModuleType("rasterio"); fake.MemoryFile = _MF; sys.modules["rasterio"] = fake
from starcop_edge import wv3 as W

H = 128
SPEC = np.array([0.90, 1.00, 1.02, 1.05, 1.10, 1.08, 1.00, 0.92])   # smooth spectral shape of the surface across SWIR1..8


def smooth_field(rng, h=H, lo=0.1, hi=0.6, k=16):
    g = rng.uniform(0, 1, (h // k + 2, h // k + 2))
    f = np.kron(g, np.ones((k, k)))[:h, :h]
    return lo + (hi - lo) * f


def make_bands(rng, plume=None, noise=1e-4, gain=1.0, h=H):
    albedo = smooth_field(rng, h)
    b = np.stack([albedo * s for s in SPEC]).astype(np.float64) * (1.0 + rng.normal(0, noise, (8, h, h)))
    if plume is not None:                                                    # absorption: B7 -1.5 %, B8 -3 % under the plume (a "weak methane plume" of the SWIR-2 window)
        b[6] *= 1.0 - 0.015 * plume; b[7] *= 1.0 - 0.030 * plume
    return (gain * b).astype(np.float32)


def blob(h=H):
    y, x = np.mgrid[:h, :h]
    return np.exp(-(((x - 70) / 9.0) ** 2 + ((y - 50) / 6.0) ** 2))


def test_channels_respond_to_absorption_and_ignore_albedo():
    rng = np.random.default_rng(0); p = blob()
    feat, valid, info = W.ratio_features(make_bands(rng, p))
    assert feat.shape == (3, H, H) and feat.dtype == np.float32 and valid.all() and info["ok"] and info["n_valid"] == H * H
    core = p > 0.8; bg = p < 0.01
    for c, name in enumerate(W.FEATURES):
        assert feat[c][core].mean() < feat[c][bg].mean() - 0.004, (name, feat[c][core].mean(), feat[c][bg].mean())        # plume = negative anomaly
        assert abs(feat[c][bg].mean()) < 1.5e-3 and feat[c][bg].std() < 2.5e-3, (name, feat[c][bg].mean(), feat[c][bg].std())   # albedo and its spatial structure cancel
    assert feat[1][core].mean() < feat[0][core].mean(), "B8 (3 %) must respond more strongly than B7 (1.5 %)"
    # the same tile without the plume: no structure at the plume location
    f0, _, _ = W.ratio_features(make_bands(np.random.default_rng(0), None))
    assert abs(f0[1][core].mean()) < 1.5e-3
    print("channels: negative anomaly under the plume, background mean/std", [round(float(feat[c][bg].mean()), 5) for c in range(3)], [round(float(feat[c][bg].std()), 5) for c in range(3)])


def test_gain_invariance_and_determinism():
    rng = np.random.default_rng(1); p = blob(); b = make_bands(rng, p)
    f1, _, _ = W.ratio_features(b); f2, _, _ = W.ratio_features(b); f3, _, _ = W.ratio_features(b * np.float32(2.5))
    assert np.array_equal(f1, f2)                                              # deterministic
    assert np.abs(f1 - f3).max() < 2e-4, np.abs(f1 - f3).max()                 # a global gain cancels (float32 rounding only)


def test_outliers_do_not_move_the_scale_factor():
    rng = np.random.default_rng(2); b = make_bands(rng, blob()); f_clean, _, i_clean = W.ratio_features(b)
    b2 = b.copy(); idx = rng.integers(0, H, (40, 2))
    for (r, c) in idx: b2[6, r, c] *= 8.0; b2[4, r, c] *= 0.2                 # 40 spike pixels in B7 and B5
    f_out, _, i_out = W.ratio_features(b2)
    keep = np.ones((H, H), bool); keep[idx[:, 0], idx[:, 1]] = False
    assert abs(i_out["c75"] - i_clean["c75"]) < 2e-3 and np.abs((f_out - f_clean)[:, keep]).max() < 6e-3, (i_out["c75"], i_clean["c75"], np.abs((f_out - f_clean)[:, keep]).max())


def test_invalid_pixels_and_degenerate_tiles():
    rng = np.random.default_rng(3); b = make_bands(rng, blob())
    b[:, :, :20] = 0.0                                                         # outside the swath: 0
    b[3, 5, 40] = np.nan; b[0, 60, 60] = 0.0                                    # single bad pixels in bands that matter / do not matter (B4 and B1 are regressors: both used)
    feat, valid, info = W.ratio_features(b)
    assert not valid[:, :20].any() and not valid[5, 40] and not valid[60, 60] and info["n_valid"] == int(valid.sum())
    assert (feat[:, ~valid] == 0).all() and np.isfinite(feat).all() and info["ok"]
    b3 = b.copy(); b3[2, :, :] = 0.0                                           # B3 is never used: all-zero B3 must not invalidate anything
    assert W.valid_mask(b3).sum() == valid.sum()
    tiny = np.zeros((8, H, H), np.float32); tiny[:, :10, :10] = 0.3
    f, v, i = W.ratio_features(tiny); assert not i["ok"] and (f == 0).all() and i["n_valid"] == 100
    flat = np.full((8, H, H), 0.3, np.float32)                                  # perfectly flat tile: the regression is rank deficient but must not crash or return NaN
    f, v, i = W.ratio_features(flat); assert np.isfinite(f).all()


def test_normalisation():
    rng = np.random.default_rng(4); feats, samples = [], []
    for _ in range(6):
        f, v, _ = W.ratio_features(make_bands(rng, blob() * rng.uniform(0, 1))); feats.append(f); samples.append(W.feature_sample(f, v))
    s = np.concatenate(samples, axis=1)
    assert s.shape[0] == 3 and s.shape[1] == sum(len(np.flatnonzero(np.ones(H * H))[::W.SAMPLE_STRIDE]) for _ in range(6))
    scale, p_lo, p_hi = W.fit_scale(s); assert (scale > 0).all() and (p_lo < 0).all() and (p_hi > 0).all() and np.allclose(scale, np.maximum(-p_lo, p_hi))
    x = W.normalise(feats[0], scale)
    assert x.dtype == np.float16 and x.shape == (3, H, H) and float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert (W.normalise(np.zeros((3, 4, 4), np.float32), scale) == 0).all()                                   # "no anomaly" (and every invalid pixel) is exactly 0, where float16 is finest
    assert float(x[:, 50:56, 62:78].min()) < -0.02, "the plume (negative anomaly) survives the scaling"
    xb = W.normalise(np.stack(feats[:2]), scale); assert xb.shape == (2, 3, H, H) and np.array_equal(xb[0], W.normalise(feats[0], scale))      # batched form
    big = np.zeros((3, 2, 2), np.float32); big[:, 0, 0] = 1e3; big[:, 1, 1] = -1e3; z = W.normalise(big, scale); assert float(z[0, 0, 0]) == 1.0 and float(z[0, 1, 1]) == -1.0   # clipped to [-1, 1]
    try: W.fit_scale(np.zeros((3, 100))); assert False
    except ValueError: pass


def test_auroc():
    a = np.arange(100.0); assert W.auroc(a + 1000, a) == 1.0 and W.auroc(a, a + 1000) == 0.0
    r = np.random.default_rng(0); x = r.normal(size=20000); y = r.normal(size=20000); assert abs(W.auroc(x, y) - 0.5) < 0.02
    assert abs(W.auroc(np.zeros(10), np.zeros(10)) - 0.5) < 1e-12              # all ties


def npy(a):
    b = io.BytesIO(); np.save(b, a); return b.getvalue()


def test_test_loader_matches_direct_computation():
    from starcop_edge import data as D
    tmp = tempfile.mkdtemp(); rng = np.random.default_rng(5)
    try:
        assets = os.path.join(tmp, "assets"); os.makedirs(assets); base = os.path.join(tmp, "mirror", "STARCOP_test", "STARCOP_test"); lines, rows, bands_by_tile = [], [], {}
        for t in range(2):
            tid = f"tile{t}"; d = os.path.join(base, tid); os.makedirs(d)
            bands = np.pad(make_bands(rng, blob() * (t + 1) / 2.0), ((0, 0), (0, 512 - H), (0, 512 - H)))       # real tiles are 512 x 512; the pad is no-data (0)
            bands_by_tile[tid] = bands
            files = {f: bands[i] for i, f in enumerate(W.WV3_FILES)}; files["labelbinary.tif"] = (np.arange(512 * 512).reshape(512, 512) % 97 == 0).astype(np.uint8)
            for f, a in files.items():
                raw = npy(a); open(os.path.join(d, f), "wb").write(raw); lines.append(f"STARCOP_test/{tid}/{f}\t{len(raw)}\t{zlib.crc32(raw):08x}\t8")
            rows.append(dict(id=tid, name=tid, date="2019-10-18", qplume=10.0 * t, has_plume=bool(t)))
        open(os.path.join(assets, "zenodo_STARCOP_test.tsv"), "w").write("\n".join(lines) + "\n"); pd.DataFrame(rows).to_csv(os.path.join(assets, "test.csv"), index=False)
        scale = np.array([0.01, 0.02, 0.01]); norm = os.path.join(tmp, W.NORM_FILE); json.dump(dict(scale=list(scale)), open(norm, "w"))
        X, Y, _, meta = W.load_wv3_test_from_mirror(os.path.join(tmp, "mirror"), assets, norm, verbose=False)
        assert X.shape == (2, 3, 512, 512) and X.dtype == np.float16 and Y.shape == (2, 512, 512) and Y.dtype == np.uint8 and list(meta.id) == ["tile0", "tile1"]
        for i, tid in enumerate(meta.id):
            f, _, _ = W.ratio_features(bands_by_tile[tid]); assert np.array_equal(X[i], W.normalise(f.astype(np.float16), scale)), tid       # the train-cache path: float32 -> float16 -> scale
        p = os.path.join(base, "tile1", "TOA_WV3_SWIR7.tif"); raw = bytearray(open(p, "rb").read()); raw[-5] ^= 0xFF; open(p, "wb").write(bytes(raw))
        try: W.load_wv3_test_from_mirror(os.path.join(tmp, "mirror"), assets, norm, verbose=False); assert False
        except AssertionError as e: assert "CRC mismatch" in str(e)
    finally: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f()
    print("wv3 features: all checks passed")
