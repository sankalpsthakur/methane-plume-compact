"""Local dry run (no network, no GPU, tiny synthetic tiles) of kernels/prep-cache/main.py: rebuild mode, failure -> patch mode, audit.
rasterio is faked (tiles are .npy bytes), Zenodo is faked with a local zip, /kaggle paths are redirected into a temp dir."""
import importlib, io, os, shutil, sys, tempfile, time, types, zipfile, zlib, json
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

# ---- fake rasterio
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

NEEDF = ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm", "labelbinary", "weight_mag1c"]
ARCHS = ["STARCOP_test", "STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
rng = np.random.default_rng(0)

def npy(a):
    b = io.BytesIO(); np.save(b, a); return b.getvalue()

def make_tile(has_label):
    mag = rng.uniform(0, 3000, (512, 512)); lab = np.zeros((512, 512), np.uint8)
    if has_label: lab[100:130, 200:220] = 1
    return {"mag1c": npy(mag), "TOA_AVIRIS_640nm": npy(rng.uniform(1, 90, (512, 512)).astype(np.float32)),
            "TOA_AVIRIS_550nm": npy(rng.uniform(1, 90, (512, 512)).astype(np.float32)), "TOA_AVIRIS_460nm": npy(rng.uniform(1, 90, (512, 512)).astype(np.float32)),
            "labelbinary": npy(lab), "weight_mag1c": npy(np.clip(mag / 400, 0.1, 1.0))}

def build(tmp):
    inp = os.path.join(tmp, "input"); assets = os.path.join(inp, "datasets/sankalpsthakur/bench-gas-assets"); mirror = os.path.join(inp, "datasets/chrisphamda/starcop")
    os.makedirs(assets); tiles = {}
    spec = [("t_easy_%d" % i, "STARCOP_train_easy", i % 2 == 0, "2019-10-0%d" % (1 + i)) for i in range(4)] + [("t_z_%d" % i, "STARCOP_train_remaining_part2", i % 2 == 1, "2019-10-1%d" % i) for i in range(4)]
    lines = {a: [] for a in ARCHS}; rows = []
    for tid, arch, plume, date in spec:
        tiles[tid] = make_tile(plume)
        for f in NEEDF:
            raw = tiles[tid][f]; lines[arch].append(f"{arch}/{tid}/{f}.tif\t{len(raw)}\t{zlib.crc32(raw):08x}\t8")
            if arch != "STARCOP_train_remaining_part2":
                d = os.path.join(mirror, arch, arch, tid); os.makedirs(d, exist_ok=True); open(os.path.join(d, f + ".tif"), "wb").write(raw)
        rows.append(dict(id=tid, name=tid, date=date, qplume=100.0 if plume else 0.0, has_plume=plume))
    os.makedirs(os.path.join(mirror, "STARCOP_test", "STARCOP_test"), exist_ok=True)
    for a in ARCHS: open(os.path.join(assets, f"zenodo_{a}.tsv"), "w").write("\n".join(lines[a]) + ("\n" if lines[a] else ""))
    pd.DataFrame(rows).to_csv(os.path.join(assets, "train.csv"), index=False)
    zp = os.path.join(tmp, "part2.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for tid, arch, _, _ in spec:
            if arch == "STARCOP_train_remaining_part2":
                for f in NEEDF: z.writestr(f"{arch}/{tid}/{f}.tif", tiles[tid][f])
    return inp, zp, spec, tiles

class FakeArch:
    def __init__(s, zp): s.z = zipfile.ZipFile(zp); s.fp = types.SimpleNamespace(size=os.path.getsize(zp))
    def infolist(s): return s.z.infolist()

def run_main(tmp, inp, zp, failing=(), base=None, tag="run"):
    work = os.path.join(tmp, "work_" + tag); os.makedirs(work)
    from starcop_edge import kio, zen
    importlib.reload(kio); importlib.reload(zen)
    kio.find_dir.__defaults__ = (os.path.join(inp),) + kio.find_dir.__defaults__[1:]   # base="/kaggle/input" -> temp input
    attempts = {}
    zf = zipfile.ZipFile(zp)
    zen.open_archive = lambda name: FakeArch(zp)
    def read_member(url, zinfo, size_total):
        tile = zinfo.filename.split("/")[1]; attempts[tile] = attempts.get(tile, 0) + 1
        if tile in failing: raise IOError("fake 429 for " + tile)
        return zf.read(zinfo.filename)
    zen.read_member = read_member
    src = open(os.path.join(ROOT, "kernels/prep-cache/main.py")).read().replace("/kaggle/working", work)
    time_sleep = time.sleep; time.sleep = lambda s: None
    g = {"CFG": {"split": "train"}, "__name__": "__main__"}
    try: exec(compile(src, "prep-cache/main.py", "exec"), g)
    except AssertionError as e: g["__assert__"] = repr(e)
    finally: time.sleep = time_sleep
    return work, g, attempts

def test_all():
    tmp = tempfile.mkdtemp(prefix="prepcache_")
    try:
        inp, zp, spec, tiles = build(tmp)
        # reference arrays computed directly
        # 1. rebuild with two Zenodo tiles that never succeed -> cache not ok, kernel assertion fires, base-like output exists
        work1, g1, att1 = run_main(tmp, inp, zp, failing=("t_z_1", "t_z_3"), tag="a")
        assert "__assert__" in g1, "kernel should fail when tiles are missing"
        rep1 = json.load(open(os.path.join(work1, "prep_report.json")))
        assert sorted(t for t, _ in rep1["train_failed"]) == ["t_z_1", "t_z_3"] and rep1["mode"] == "rebuild" and rep1["train_ok"] == 6, rep1
        assert rep1["crc_fail"] == 0 and rep1["first_pass_failures"] == 2
        assert att1["t_z_1"] >= 5, att1      # retried at tile level + second pass
        # 2. publish that output as a mounted kernel source and run patch mode
        base = os.path.join(inp, "notebooks/sankalpsthakur/bench-gas-prep-train"); os.makedirs(base)
        shutil.copytree(os.path.join(work1, "cache"), os.path.join(base, "cache")); shutil.copy(os.path.join(work1, "prep_report.json"), base)
        work2, g2, att2 = run_main(tmp, inp, zp, failing=(), tag="b")
        assert "__assert__" not in g2, g2.get("__assert__")
        rep2 = json.load(open(os.path.join(work2, "prep_report.json")))
        assert rep2["mode"] == "patch" and rep2["tiles_fetched_in_this_run"] == 2 and rep2["train_failed"] == [] and rep2["crc_fail"] == 0 and rep2["train_ok"] == 8, rep2
        assert rep2["audit"]["copied_tiles_identical_to_base"] == 6 and rep2["audit"]["patched_tiles"] == 2 and not rep2["audit"]["patched_tile_checks_failed"]
        assert set(att2) == {"t_z_1", "t_z_3"}, att2   # only the missing tiles were fetched again
        # 3. final arrays equal a direct computation for every tile
        X = np.load(os.path.join(work2, "cache/train_X.npy")); Y = np.load(os.path.join(work2, "cache/train_Y.npy")); W = np.load(os.path.join(work2, "cache/train_W.npy"))
        meta = pd.read_csv(os.path.join(work2, "cache/train_meta.csv")); assert meta.ok.all() and list(meta.id) == [s[0] for s in spec]
        for i, (tid, arch, plume, date) in enumerate(spec):
            r = lambda f: np.load(io.BytesIO(tiles[tid][f]))
            exp = np.stack([np.clip(np.nan_to_num(r(b).astype(np.float32)) / f, 0, 2) for b, f in zip(["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"], [1750., 60., 60., 60.])]).astype(np.float16)
            assert np.array_equal(X[i], exp), tid
            assert np.array_equal(Y[i], r("labelbinary")) and np.array_equal(W[i], np.round(r("weight_mag1c") * 255).astype(np.uint8)), tid
        assert rep2["sha256"]["train_X.npy"] and rep2["train_has_plume_mismatch"] == 0
        # 4. base unusable (ids differ) -> falls back to a full rebuild
        shutil.rmtree(os.path.join(base, "cache")); os.makedirs(os.path.join(base, "cache"))
        pd.DataFrame({"id": ["x"], "ok": [True]}).to_csv(os.path.join(base, "cache/train_meta.csv"), index=False); open(os.path.join(base, "cache/train_X.npy"), "wb").write(b"junk")
        work3, g3, att3 = run_main(tmp, inp, zp, failing=(), tag="c")
        rep3 = json.load(open(os.path.join(work3, "prep_report.json")))
        assert "__assert__" not in g3 and rep3["mode"] == "rebuild" and rep3["train_ok"] == 8 and rep3["tiles_fetched_in_this_run"] == 8, rep3
        print("prep-cache logic: rebuild/fail/patch/audit/fallback checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    test_all()
