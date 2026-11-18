"""Local dry run of kernels/train_gpu/main.py (the driver every stage kernel ships) with the WV3 additions, on CPU, with tiny synthetic data and /kaggle redirected into a temp directory:
ImageNet download through a file:// URL with the SHA-256 prefix check, the warm-up skipping the absent train_W.npy, the run keys (pretrained, test_kind, in_ch) and the kernel-local test cache reaching train.py, RUN_STATUS.json.
HEAVY (the control is the 6.6 M parameter network): run it with `guard.sh -t 180 -m 1500`. Not part of the quick suite."""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import test_nchannel as T                      # fakes rasterio, builds the synthetic 3-channel cache and the fake mirror (torch is NOT imported here: the control child needs the memory)
import mk_gpu_wave as G
from starcop_edge import kio


def main():
    tmp = tempfile.mkdtemp(prefix="drv_")
    try:
        inp = os.path.join(tmp, "input"); work = os.path.join(tmp, "work"); os.makedirs(work); cache = os.path.join(inp, "notebooks/sankalpsthakur/bench-gas-prep-wv3/cache"); os.makedirs(cache)
        T.write_cache(cache)
        mirror, assets = T.fake_mirror(tmp, n=1, plume_first=True)
        os.makedirs(os.path.join(inp, "datasets/chrisphamda")); shutil.move(os.path.join(mirror, "STARCOP_test"), os.path.join(inp, "datasets/chrisphamda/STARCOP_test"))
        os.makedirs(os.path.join(inp, "datasets/sankalpsthakur")); shutil.move(assets, os.path.join(inp, "datasets/sankalpsthakur/bench-gas-assets"))
        fake = os.path.join(tmp, "imagenet.pth")      # a torchvision-style state dict, written by a child interpreter
        subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path[:0] = [%r, %r]; import torch, test_nchannel as T; from starcop_edge.models import SMPLikeUnet; torch.save(T.torchvision_style(SMPLikeUnet(3, 1)), %r)" % (HERE, os.path.join(ROOT, "src"), fake)], check=True)
        prefix = hashlib.sha256(open(fake, "rb").read()).hexdigest()[:8]
        common = {**G.COMMON, "device": "cpu", "amp": False}
        ctrl = dict(name="ctrl_wv3", arch="mnv2", sched="plateau", plateau_every=2, seed=0, steps=4, batch=2, lr=1e-4, wd=0.0, jitter=0, widths="8,16,32,64,128", pos_weight=15.0, device="cpu", amp=False, val_dates=G.VAL, val_max=1, eval_every=4,
                    test_eval=True, allow_test_with_val=True, test_kind="wv3", pretrained="imagenet", in_ch=3)
        g1 = dict(name="g_d1", seed=0, val_dates=G.VAL, val_max=1, eval_every=4, steps=4, in_ch=3, **{**common, "batch": 4}, **G.CANDS["d1"])
        fakedir = os.path.join(tmp, "fakepy"); os.makedirs(os.path.join(fakedir, "rasterio"))      # the child interpreters of the driver need the fake rasterio too (tiles are .npy bytes)
        open(os.path.join(fakedir, "rasterio", "__init__.py"), "w").write("import io\nimport numpy as np\nclass _DS:\n    def __init__(s, a): s.a = a\n    def __enter__(s): return s\n    def __exit__(s, *x): pass\n    def read(s): return s.a[None] if s.a.ndim == 2 else s.a\nclass MemoryFile:\n    def __init__(s, raw): s.a = np.load(io.BytesIO(raw))\n    def __enter__(s): return s\n    def __exit__(s, *x): pass\n    def open(s): return _DS(s.a)\n")
        os.environ["PYTHONPATH"] = fakedir
        tcache = os.path.join(tmp, "tcache")
        cfg = dict(gpu=False, parallel=1, threads=1, warm_timeout_s=60, internet=True, imagenet=dict(url="file://" + fake, sha256_prefix=prefix), test_cache=tcache, runs=[ctrl, g1])
        src = open(os.path.join(ROOT, "kernels", "train_gpu", "main.py")).read().replace("/kaggle/working", work).replace("/tmp/pkg", os.path.join(ROOT, "src")).replace("time.sleep(10)", "time.sleep(1)")
        kio.find_dir.__defaults__ = (inp,) + kio.find_dir.__defaults__[1:]
        g = {"CFG": cfg, "__name__": "__main__"}
        import contextlib, io
        buf = io.StringIO(); t = time.time()
        with contextlib.redirect_stdout(buf): exec(compile(src, "train_gpu/main.py", "exec"), g)
        out = buf.getvalue(); print("driver dry run:", round(time.time() - t), "s")
        rs = json.load(open(os.path.join(work, "RUN_STATUS.json")))
        assert {k: v["rc"] for k, v in rs["runs"].items()} == {"ctrl_wv3": 0, "g_d1": 0}, (rs, out[-1500:])
        assert hashlib.sha256(open(os.path.join(work, "imagenet_mobilenet_v2.pth"), "rb").read()).hexdigest()[:8] == prefix and "ImageNet MobileNetV2 weights" in out
        assert os.path.exists(os.path.join(work, "ctrl_wv3", "test_metrics.json")) and os.path.exists(os.path.join(work, "g_d1", "val_metrics.json")) and not os.path.exists(os.path.join(work, "g_d1", "test_metrics.json"))
        assert os.path.exists(os.path.join(tcache, "wv3_test_cache.json")), "the control's test features go to the kernel-local cache"
        start = json.loads(open(os.path.join(work, "ctrl_wv3", "log.jsonl")).readline()); assert start["pretrained"]["tensors_loaded"] == start["pretrained"]["encoder_tensors"] and start["in_ch"] == 3
        assert "train_W.npy" not in out or "No such file" not in out
        # a wrong hash stops the kernel before any run starts
        cfg2 = dict(cfg, imagenet=dict(url="file://" + fake, sha256_prefix="deadbeef")); g2 = {"CFG": cfg2, "__name__": "__main__"}
        shutil.rmtree(work); os.makedirs(work)
        try:
            with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "train_gpu/main.py", "exec"), g2)
            assert False
        except SystemExit as e: assert "does not start with deadbeef" in str(e) and not os.path.exists(os.path.join(work, "RUN_STATUS.json"))
        print("driver dry run with the WV3 additions: ImageNet hash check, absent train_W.npy, run keys, test cache, RUN_STATUS: passed")
    finally: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
