"""Local dry run (no network, no GPU, synthetic 512 x 512 tiles) of kernels/prep-wv3/main.py (Amendment 7): mirror + Zenodo routes with CRC verification, labels and metadata copied from a mounted AVIRIS cache after a hash check,
features and min-max limits (fit tiles = outside the validation dates), in-place normalisation, diagnostic, sampler bench, failure handling. rasterio is faked (tiles are .npy bytes), Zenodo is a local zip, /kaggle paths are redirected."""
import contextlib, hashlib, importlib, io, json, os, shutil, sys, tempfile, time, types, zipfile, zlib
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from test_wv3_features import make_bands, blob

ARCHS = ["STARCOP_test", "STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
SPEC = [(f"t_easy_{i}", "STARCOP_train_easy", i % 2 == 0, "2019-10-0%d" % (1 + i)) for i in range(4)] + [(f"t_z_{i}", "STARCOP_train_remaining_part2", i % 2 == 1, "2019-10-2%d" % (1 + i)) for i in range(4)]
# dates: t_easy_0..3 = 10-01..10-04, t_z_0..3 = 10-21, 10-22, 10-23, 10-24  -> t_z_1 and t_z_2 are validation tiles


def npy(a):
    b = io.BytesIO(); np.save(b, a); return b.getvalue()


def plume_blob(h=512):
    y, x = np.mgrid[:h, :h]
    return np.exp(-(((x - 300) / 40.0) ** 2 + ((y - 200) / 25.0) ** 2))


def build(tmp):
    rng = np.random.default_rng(0)
    inp = os.path.join(tmp, "input"); assets = os.path.join(inp, "datasets/sankalpsthakur/bench-gas-assets"); mirror = os.path.join(inp, "datasets/chrisphamda/starcop"); av = os.path.join(inp, "notebooks/sankalpsthakur/bench-gas-prep-cache/cache")
    os.makedirs(assets); os.makedirs(av); os.makedirs(os.path.join(mirror, "STARCOP_test", "STARCOP_test")); tiles, lines, rows, labels = {}, {a: [] for a in ARCHS}, [], []
    for tid, arch, plume, date in SPEC:
        p = plume_blob() if plume else None
        bands = make_bands(rng, p, h=512); lab = ((p > 0.3).astype(np.uint8) if plume else np.zeros((512, 512), np.uint8))
        tiles[tid] = {f: npy(bands[i]) for i, f in enumerate(W.WV3_FILES)}; tiles[tid]["bands"] = bands; labels.append(lab)
        for f in W.WV3_FILES:
            raw = tiles[tid][f]; lines[arch].append(f"{arch}/{tid}/{f}\t{len(raw)}\t{zlib.crc32(raw):08x}\t8")
            if arch != "STARCOP_train_remaining_part2":
                d = os.path.join(mirror, arch, arch, tid); os.makedirs(d, exist_ok=True); open(os.path.join(d, f), "wb").write(raw)
        rows.append(dict(id=tid, name=tid, date=date, qplume=100.0 if plume else 0.0, has_plume=plume))
    for a in ARCHS: open(os.path.join(assets, f"zenodo_{a}.tsv"), "w").write("\n".join(lines[a]) + ("\n" if lines[a] else ""))
    pd.DataFrame(rows).to_csv(os.path.join(assets, "train.csv"), index=False)
    Y = np.stack(labels); np.save(os.path.join(av, "train_Y.npy"), Y)
    meta = pd.DataFrame(rows).assign(n_label_px=[int(y.sum()) for y in Y], nan_px=0, valid_frac=1.0); meta.to_csv(os.path.join(av, "train_meta.csv"), index=False)
    zp = os.path.join(tmp, "part2.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for tid, arch, _, _ in SPEC:
            if arch == "STARCOP_train_remaining_part2":
                for f in W.WV3_FILES: z.writestr(f"{arch}/{tid}/{f}", tiles[tid][f])
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    return inp, zp, tiles, Y, dict(split="train", labels_sha256=sha(os.path.join(av, "train_Y.npy")), meta_sha256=sha(os.path.join(av, "train_meta.csv")), diag_tiles=10, mirror_workers=2, zen_workers=2, zen_conc=2)


class FakeArch:
    def __init__(s, zp): s.z = zipfile.ZipFile(zp); s.fp = types.SimpleNamespace(size=os.path.getsize(zp))
    def infolist(s): return s.z.infolist()


def run_main(tmp, inp, zp, cfg, failing=(), tag="run"):
    work = os.path.join(tmp, "work_" + tag); os.makedirs(work)
    from starcop_edge import kio, zen
    importlib.reload(kio); importlib.reload(zen)
    kio.find_dir.__defaults__ = (os.path.join(inp),) + kio.find_dir.__defaults__[1:]
    attempts = {}; zf = zipfile.ZipFile(zp)
    zen.open_archive = lambda name: FakeArch(zp)
    def read_member(url, zinfo, size_total):
        tile = zinfo.filename.split("/")[1]; attempts[tile] = attempts.get(tile, 0) + 1
        if tile in failing: raise IOError("fake 429 for " + tile)
        return zf.read(zinfo.filename)
    zen.read_member = read_member
    src = open(os.path.join(ROOT, "kernels/prep-wv3/main.py")).read().replace("/kaggle/working", work)
    time_sleep = time.sleep; time.sleep = lambda s: None
    g = {"CFG": cfg, "__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "prep-wv3/main.py", "exec"), g)
    except AssertionError as e: g["__assert__"] = repr(e)
    finally: time.sleep = time_sleep
    return work, g, attempts


def test_all():
    tmp = tempfile.mkdtemp(prefix="prepwv3_")
    try:
        inp, zp, tiles, Y, cfg = build(tmp)
        # 1. a clean run: both routes, hash-checked labels, features equal to a direct computation
        work, g, att = run_main(tmp, inp, zp, cfg, tag="a")
        assert "__assert__" not in g, g.get("__assert__")
        rep = json.load(open(os.path.join(work, "prep_report.json")))
        assert rep["tiles_stored"] == 8 and rep["train_failed"] == [] and rep["crc_fail"] == 0 and rep["src_counts"] == {"mirror": 4, "zenodo": 4} and rep["tiles_features_ok"] == 8, rep
        assert rep["mirror_files_verified"] == 32 and rep["zenodo_files_verified"] == 32 and set(att) == {f"t_z_{i}" for i in range(4)}
        X = np.load(os.path.join(work, "cache/train_X.npy")); Yc = np.load(os.path.join(work, "cache/train_Y.npy")); meta = pd.read_csv(os.path.join(work, "cache/train_meta.csv"))
        assert X.shape == (8, 3, 512, 512) and X.dtype == np.float16 and float(X.min()) >= -1.0 and float(X.max()) <= 1.0 and np.array_equal(Yc, Y) and list(meta.id) == [s[0] for s in SPEC] and meta.wv3_ok.all()
        assert not os.path.exists(os.path.join(work, "cache/train_W.npy")), "no mag1c loss weights in the WV3 cache"
        raws, samples = [], []
        for tid, arch, plume, date in SPEC:
            f, v, _ = W.ratio_features(tiles[tid]["bands"]); raws.append(f.astype(np.float16)); samples.append(W.feature_sample(f, v) if date not in ("2019-10-22", "2019-10-23") else None)
        scale, p_lo, p_hi = W.fit_scale(np.concatenate([s for s in samples if s is not None], axis=1))
        norm = json.load(open(os.path.join(work, "cache", W.NORM_FILE)))
        assert norm["fit_tiles"] == 6 and norm["fit_excludes_dates"] == ["2019-10-22", "2019-10-23"] and np.allclose(norm["scale"], scale) and np.allclose(norm["p_lo"], p_lo) and np.allclose(norm["p_hi"], p_hi)
        for i in range(8):
            exp = W.normalise(raws[i], scale); assert np.array_equal(X[i], exp), SPEC[i][0]     # raw float16 -> scale, exactly the path of the test loader (load_wv3_test_from_mirror)
        for k in ("train_X.npy", "train_Y.npy", "train_meta.csv", W.NORM_FILE): assert rep["sha256"][k] == hashlib.sha256(open(os.path.join(work, "cache", k), "rb").read()).hexdigest()
        d = rep["diag_summary"]; assert d["features_sane"] and d["best_channel_separation"] > 0.8, d                                      # the synthetic plumes are strong: every channel separates them
        assert os.path.exists(os.path.join(work, "sampler_bench.json")) and rep["sampler_bench"]["channels"] == 3 and rep["sampler_bench"]["batch32_batches_per_s"] > 0
        assert abs(sum(rep["band_mean_over_tiles"]) / 8 - float(np.mean([t["bands"].mean() for t in tiles.values()]))) < 0.05
        # 2. a Zenodo tile that never arrives -> the kernel fails by assertion and reports it (no silent hole)
        work2, g2, att2 = run_main(tmp, inp, zp, cfg, failing=("t_z_1",), tag="b")
        assert "__assert__" in g2
        rep2 = json.load(open(os.path.join(work2, "prep_report.json")))
        assert [t for t, _ in rep2["train_failed"]] == ["t_z_1"] and rep2["tiles_stored"] == 7 and att2["t_z_1"] >= 5
        # 3. a corrupted mirror file -> CRC mismatch counted, tile failed, kernel assertion
        bad = os.path.join(inp, "datasets/chrisphamda/starcop/STARCOP_train_easy/STARCOP_train_easy/t_easy_2/TOA_WV3_SWIR7.tif"); raw = bytearray(open(bad, "rb").read()); raw[-9] ^= 0xFF; open(bad, "wb").write(bytes(raw))
        work3, g3, att3 = run_main(tmp, inp, zp, cfg, tag="c")
        rep3 = json.load(open(os.path.join(work3, "prep_report.json")))
        assert "__assert__" in g3 and rep3["crc_fail"] == 1 and rep3["train_failed"][0][0] == "t_easy_2" and rep3["crc_mismatch_events"] >= 1, rep3
        # 4. the labels or the metadata of the mounted AVIRIS cache differ from the recorded hashes -> stop before any tile is read
        work4, g4, att4 = run_main(tmp, inp, zp, dict(cfg, labels_sha256="0" * 64), tag="d")
        assert "__assert__" in g4 and "hashes differ" in g4["__assert__"] and not att4
        print("prep-wv3 logic: both routes, hash checks, limits on the fit tiles, normalisation, diagnostic, failure handling passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_all()
