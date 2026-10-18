"""Local synthetic test (no data, rasterio faked) of starcop_edge.data.load_test_from_mirror: layout, CRC check, preprocessing identical to the train cache."""
import io, os, shutil, sys, tempfile, types, zlib
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
from starcop_edge import data as D
def npy(a):
    b = io.BytesIO(); np.save(b, a); return b.getvalue()
def test_all():
    tmp = tempfile.mkdtemp(); rng = np.random.default_rng(1)
    try:
        assets = os.path.join(tmp, "assets"); os.makedirs(assets); base = os.path.join(tmp, "mirror", "STARCOP_test", "STARCOP_test"); lines = []; rows = []; exp = {}
        for t in range(3):
            tid = f"tile{t}"; d = os.path.join(base, tid); os.makedirs(d)
            files = {"mag1c": rng.uniform(0, 4000, (512, 512)), "TOA_AVIRIS_640nm": rng.uniform(0, 100, (512, 512)).astype(np.float32), "TOA_AVIRIS_550nm": rng.uniform(0, 100, (512, 512)).astype(np.float32),
                     "TOA_AVIRIS_460nm": rng.uniform(0, 100, (512, 512)).astype(np.float32), "labelbinary": (rng.random((512, 512)) > 0.99).astype(np.uint8)}
            for f, a in files.items():
                raw = npy(a); open(os.path.join(d, f + ".tif"), "wb").write(raw); lines.append(f"STARCOP_test/{tid}/{f}.tif\t{len(raw)}\t{zlib.crc32(raw):08x}\t8")
            exp[tid] = files; rows.append(dict(id=tid, name=tid, date="2019-10-18", qplume=10.0 * t, has_plume=bool(t)))
        open(os.path.join(assets, "zenodo_STARCOP_test.tsv"), "w").write("\n".join(lines) + "\n"); pd.DataFrame(rows).to_csv(os.path.join(assets, "test.csv"), index=False)
        X, Y, _, meta = D.load_test_from_mirror(os.path.join(tmp, "mirror"), assets, verbose=False)
        assert X.shape == (3, 4, 512, 512) and X.dtype == np.float16 and Y.shape == (3, 512, 512) and Y.dtype == np.uint8 and list(meta.id) == ["tile0", "tile1", "tile2"]
        for i, tid in enumerate(meta.id):
            e = exp[tid]
            for c, (b, f) in enumerate(zip(["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"], [1750.0, 60.0, 60.0, 60.0])):
                assert np.array_equal(X[i, c], np.clip(e[b].astype(np.float32) / np.float32(f), 0.0, 2.0).astype(np.float16)), (tid, b)
            assert np.array_equal(Y[i], e["labelbinary"])
        # corrupted file -> CRC assertion
        p = os.path.join(base, "tile1", "mag1c.tif"); raw = bytearray(open(p, "rb").read()); raw[-5] ^= 0xFF; open(p, "wb").write(bytes(raw))
        try: D.load_test_from_mirror(os.path.join(tmp, "mirror"), assets, verbose=False); assert False
        except AssertionError as e: assert "CRC mismatch" in str(e)
        print("load_test_from_mirror: layout, preprocessing and CRC guard OK")
    finally: shutil.rmtree(tmp, ignore_errors=True)
if __name__ == "__main__": test_all()
