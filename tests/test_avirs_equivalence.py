"""AVIRIS equivalence golden (Amendment 7). The WV3 work extends data.py, train.py, models.py and edge_eval.py for N-channel inputs; this test pins that the AVIRIS code paths still
produce the same numbers. The golden was recorded on src 54170b5e7b1e2859 (the src every AVIRIS wave of Amendments 3-6 ran) BEFORE any WV3 change, by `python3 tests/test_avirs_equivalence.py --record`.
  * sampler: the SHA-256 of 5 batches of CropSampler (hardneg 0 and 0.5) on a seeded synthetic 4-channel cache. numpy only, platform independent: always checked.
  * models: SHA-256 of the seeded initial state_dict and of one forward pass of EdgeUNet (d1) and SMPLikeUnet (released architecture), 4 input channels.
  * training: SHA-256 of the final state_dict bytes after a few CPU steps of train.main (d1 with the cosine schedule) on the same synthetic cache, without validation (the default run stays inside the guard's 512 MiB).
    `--heavy` adds the released architecture trained with pos_weight 15, d1 with validation dates and the released architecture with the plateau schedule and its validation crops; it needs `guard.sh -m 1024`
    and pins val_metrics.json as well. Float kernels differ between platforms and torch builds, so the model and training parts are compared only where torch version, OS and CPU architecture
    equal the recorded ones; elsewhere they print SKIP.
usage: python3 tests/test_avirs_equivalence.py [--record] [--heavy]"""
import hashlib, json, os, platform, shutil, sys, tempfile
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
GOLDEN = os.path.join(ROOT, "tests", "golden", "avirs_equivalence.json")
from starcop_edge import data as D

N = 6
DATES = ["2019-10-18"] * 4 + ["2019-10-22"] * 2
QPLUME = [0.0, 1500.0, 300.0, 0.0, 800.0, 0.0]


def synth(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2, (N, 4, 512, 512)).astype(np.float16)
    X[:, 0] = rng.uniform(0, 0.05, (N, 512, 512)).astype(np.float16)   # background mag1c below the hard-negative threshold, as in tests/test_sampler.py
    Y = np.zeros((N, 512, 512), np.uint8); W = rng.integers(26, 256, (N, 512, 512)).astype(np.uint8)
    for t in (1, 2, 4):
        r, c = rng.integers(40, 380, 2); Y[t, r:r + 50, c:c + 70] = 1; X[t, 0, r:r + 50, c:c + 70] = 1.5
    X[3, 0, 300:380, 100:200] = 0.5
    meta = pd.DataFrame(dict(id=[f"tile{t}" for t in range(N)], name=[f"tile{t}" for t in range(N)], date=DATES, qplume=QPLUME, has_plume=[q > 0 for q in QPLUME],
                             n_label_px=[int(y.sum()) for y in Y]))
    return X, Y, W, meta


def h(*arrays):
    m = hashlib.sha256()
    for a in arrays: m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


def sampler_hashes():
    X, Y, W, _ = synth()
    out = {}
    for name, hn in (("hardneg0", 0.0), ("hardneg05", 0.5)):
        s = D.CropSampler(X, Y, W, np.arange(N), seed=3, jitter=8, batch=4, hardneg=hn)
        out[name] = h(*[a for _ in range(5) for a in s.next()]); s.close()
    return out


def sd_hash(sd):
    m = hashlib.sha256()
    for k in sorted(sd): m.update(k.encode()); m.update(sd[k].detach().cpu().numpy().tobytes())
    return m.hexdigest()


def state_hash(path):
    import torch
    return sd_hash(torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)["state_dict"])


def model_hashes():
    import torch
    from starcop_edge.models import EdgeUNet, SMPLikeUnet
    out = {}
    for name, mk in (("edge_d1", lambda: EdgeUNet(4, (4, 8, 16, 32, 64), stem_stride=2)), ("mnv2", lambda: SMPLikeUnet(4, 1))):
        torch.manual_seed(5); net = mk().eval(); x = torch.from_numpy(np.random.default_rng(1).uniform(0, 2, (1, 4, 128, 128)).astype(np.float32))
        with torch.no_grad(): y = net(x).numpy()
        out[name] = dict(init_state_sha256=sd_hash(net.state_dict()), forward_sha256=h(y))
    return out


def train_hashes(heavy=False):
    from starcop_edge import train
    import contextlib, io
    X, Y, W, meta = synth(); tmp = tempfile.mkdtemp(); res = {}
    try:
        np.save(os.path.join(tmp, "train_X.npy"), X); np.save(os.path.join(tmp, "train_Y.npy"), Y); np.save(os.path.join(tmp, "train_W.npy"), W); meta.to_csv(os.path.join(tmp, "train_meta.csv"), index=False)
        base = ["--data", tmp, "--threads", "1", "--device", "cpu", "--log_every", "100", "--eval_every", "1000"]
        val = ["--val_dates", "2019-10-22", "--val_max", "1"]
        edge = ["--widths", "4,8,16,32,64", "--stem_stride", "2", "--steps", "6", "--batch", "4", "--lr", "0.002", "--jitter", "8", "--seed", "3"]
        mnv2 = ["--arch", "mnv2", "--steps", "3", "--batch", "2", "--lr", "0.0001", "--pos_weight", "15", "--jitter", "0", "--seed", "1", "--amp"]
        runs = {"train_edge": edge}
        if heavy:
            runs.update({"train_mnv2": mnv2, "train_edge_val": edge + val, "train_mnv2_plateau": mnv2 + val + ["--sched", "plateau", "--plateau_every", "2"]})
        for i, (name, extra) in enumerate(runs.items()):
            out = os.path.join(tmp, f"o{i}")
            with contextlib.redirect_stdout(io.StringIO()): train.main(base + ["--out", out, "--tag", name] + extra)
            res[name] = dict(state_sha256=state_hash(out))
            if os.path.exists(os.path.join(out, "val_metrics.json")):
                res[name]["val_metrics_sha256"] = hashlib.sha256(json.dumps(json.load(open(os.path.join(out, "val_metrics.json"))), sort_keys=True).encode()).hexdigest()
    finally: shutil.rmtree(tmp, ignore_errors=True)
    return res


def env():
    import torch
    return dict(torch=torch.__version__, os=sys.platform, machine=platform.machine(), numpy=np.__version__)


def check(heavy=False):
    g = json.load(open(GOLDEN))
    sh = sampler_hashes()
    assert sh == g["sampler"], ("sampler output changed", sh, g["sampler"])
    print("AVIRIS sampler golden: identical (hardneg 0 and 0.5)")
    try: e = env()
    except Exception as ex:
        print("SKIP model and training goldens: torch unavailable", repr(ex)[:80]); return
    rec = g["recorded"]
    if any(e[k] != rec[k] for k in ("torch", "os", "machine")):
        print("SKIP model and training goldens: recorded on", {k: rec[k] for k in ("torch", "os", "machine")}, "this run:", {k: e[k] for k in ("torch", "os", "machine")}); return
    mh = model_hashes()
    assert mh == g["models"], ("model construction or forward changed", mh, g["models"])
    print("AVIRIS model goldens: identical (EdgeUNet d1 and SMPLikeUnet, 4 channels: seeded init and forward)")
    th = train_hashes(heavy)
    want = {k: v for k, v in g["train"].items() if heavy or k == "train_edge"}
    assert th == want, ("training output changed", th, want)
    print("AVIRIS training goldens: identical (" + ", ".join(th) + ")")


def record():
    e = env(); sh = sampler_hashes(); mh = model_hashes(); th = train_hashes(heavy=True)
    src = hashlib.sha256(b"".join(open(os.path.join(ROOT, "src", "starcop_edge", f), "rb").read() for f in sorted(os.listdir(os.path.join(ROOT, "src", "starcop_edge"))) if f.endswith(".py"))).hexdigest()[:16]
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    json.dump(dict(recorded=e, src_sha256_16=src, sampler=sh, models=mh, train=th), open(GOLDEN, "w"), indent=1); print("recorded", GOLDEN, src, json.dumps(th)[:300])


if __name__ == "__main__":
    record() if "--record" in sys.argv else check(heavy="--heavy" in sys.argv)
