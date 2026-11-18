"""Amendment 7: N-channel input support (the WV3 feature cache has 3 channels and no mag1c loss weights). The AVIRIS code paths are pinned separately by tests/test_avirs_equivalence.py.
Light mode (default; run it with `guard.sh -t 60 -m 1024`: torch's transient allocations reach about 700 MiB): sampler without loss weights, loader without train_W.npy, model constructors with 3 channels, the ImageNet encoder loader (synthetic torchvision-style state dict),
checkpoint in_ch, edge_eval.load_run, train.main end to end on a 3-channel cache (d1; with the WV3 test loader on a fake mirror).
--heavy (needs `guard.sh -t 120 -m 1500`): the control's flag combination end to end (released architecture, ImageNet-style encoder file, plateau schedule on validation dates, WV3 test read, --allow_test_with_val)."""
import contextlib, io, json, os, shutil, sys, tempfile, types, zlib
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
from starcop_edge import data as D, wv3 as W

N = 5
DATES = ["2019-10-18", "2019-10-18", "2019-10-22", "2019-10-18", "2019-10-22"]      # the validation dates (10-22) hold tiles 2 and 4, both with a plume
QPLUME = [0.0, 1500.0, 300.0, 0.0, 800.0]


def synth3(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.3, 0.7, (N, 3, 512, 512)).astype(np.float16)
    Y = np.zeros((N, 512, 512), np.uint8)
    for t in (1, 2, 4):
        r, c = rng.integers(40, 380, 2); Y[t, r:r + 50, c:c + 70] = 1; X[t, 0, r:r + 50, c:c + 70] = 0.0       # plume = low normalised value in channel 0 (aligned with the label)
    meta = pd.DataFrame(dict(id=[f"tile{t}" for t in range(N)], name=[f"tile{t}" for t in range(N)], date=DATES, qplume=QPLUME, has_plume=[q > 0 for q in QPLUME], n_label_px=[int(y.sum()) for y in Y]))
    return X, Y, meta


def write_cache(tmp, with_w=False):
    X, Y, meta = synth3(); np.save(os.path.join(tmp, "train_X.npy"), X); np.save(os.path.join(tmp, "train_Y.npy"), Y); meta.to_csv(os.path.join(tmp, "train_meta.csv"), index=False)
    if with_w: np.save(os.path.join(tmp, "train_W.npy"), np.full(Y.shape, 255, np.uint8))
    json.dump(dict(scale=[0.01, 0.02, 0.01]), open(os.path.join(tmp, W.NORM_FILE), "w"))
    return X, Y, meta


def test_sampler_without_weights_and_loader():
    tmp = tempfile.mkdtemp()
    try:
        X, Y, meta = write_cache(tmp)
        Xl, Yl, Wl, ml = D.load_split(tmp, "train")
        assert Wl is None and Xl.shape == (N, 3, 512, 512) and len(ml) == N
        s = D.CropSampler(Xl, Yl, None, np.arange(N), seed=3, jitter=16, batch=8)
        npos = 0
        for _ in range(8):
            x, y, w = s.next()
            assert x.shape == (8, 3, 128, 128) and y.shape == (8, 1, 128, 128) and w.shape == (8, 1, 128, 128) and x.dtype == np.float32
            assert (w == 1.0).all(), "no loss weights: uniform weight 1"
            for i in range(8):
                assert np.array_equal(x[i, 0] < 0.001, y[i, 0] > 0.5), "label no longer aligned with channel 0 after augmentation"
                npos += int(y[i].sum() > D.POS_PIX)
        s.close(); assert 0.3 < npos / 64 < 0.75
        np.save(os.path.join(tmp, "train_W.npy"), np.full(Y.shape, 128, np.uint8))                    # with a W file the loader returns it, as before
        assert D.load_split(tmp, "train")[2].shape == Y.shape
    finally: shutil.rmtree(tmp, ignore_errors=True)


def test_sampler_fails_loudly():
    """an empty window pool is refused at construction; a producer failure reaches the consumer instead of hanging it (it used to block until the platform timeout)"""
    X, Y, _ = synth3(); Y0 = np.zeros_like(Y)
    try: D.CropSampler(X, Y0, None, np.arange(N), seed=0, jitter=0, batch=4); assert False
    except ValueError as e: assert "empty window pool" in str(e)
    s = D.CropSampler(X, Y, None, np.arange(N), seed=0, jitter=0, batch=4); s.next()
    s.X = None                                             # break the producer
    try:
        for _ in range(20): s.next()
        assert False
    except RuntimeError as e: assert "producer thread failed" in str(e)


def test_models_with_three_channels():
    import torch
    from starcop_edge.models import EdgeUNet, SMPLikeUnet, count_params
    torch.manual_seed(0)
    n4 = count_params(EdgeUNet(4, (4, 8, 16, 32, 64), stem_stride=2)); n3 = count_params(EdgeUNet(3, (4, 8, 16, 32, 64), stem_stride=2))
    assert n4 == 123649 and n3 == 123649 - 2 * 4 * 9, (n4, n3)       # one input channel less: 4 x 9 weights in the first encoder conv and 4 x 9 in the skip conv
    x = torch.rand(2, 3, 128, 128)
    assert EdgeUNet(3, (4, 8, 16, 32, 64), stem_stride=2)(x).shape == (2, 1, 128, 128) and EdgeUNet(3, (8, 16, 32, 64, 128), stem_stride=1)(x).shape == (2, 1, 128, 128)
    m = SMPLikeUnet(3, 1); assert m(x[:1]).shape == (1, 1, 128, 128) and count_params(m) == 6629233 - 32 * 9, count_params(m)


def torchvision_style(model, seed=0):
    """what mobilenet_v2-b0353104.pth looks like: 'features.N...' keys (BatchNorm num_batches_tracked included or not) plus the classifier head, built from a model's own encoder tensors"""
    import torch
    g = torch.Generator().manual_seed(seed)
    sd = {k[len("encoder."):]: torch.randn(v.shape, generator=g).to(v.dtype) if v.dtype.is_floating_point else v.clone() for k, v in model.state_dict().items() if k.startswith("encoder.features.")}
    sd["classifier.1.weight"] = torch.randn(1000, 1280, generator=g); sd["classifier.1.bias"] = torch.randn(1000, generator=g)
    return sd


def test_imagenet_encoder_loader():
    import torch
    from starcop_edge.models import SMPLikeUnet, load_imagenet_encoder
    torch.manual_seed(1); m = SMPLikeUnet(3, 1); sd = torchvision_style(SMPLikeUnet(3, 1))
    dec_before = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith("encoder.")}
    rep = load_imagenet_encoder(m, sd)
    assert rep["classifier_keys_ignored"] == 2 and rep["tensors_loaded"] == rep["encoder_tensors"] and rep["first_conv_shape"] == [32, 3, 3, 3], rep
    for k, v in m.state_dict().items():
        if k.startswith("encoder."): assert torch.equal(v, sd[k[len("encoder."):]]), k
        else: assert torch.equal(v, dec_before[k]), "decoder and head must be untouched: " + k
    # old files have no num_batches_tracked: still fine; a missing weight, an unexpected key or a wrong first-conv shape must raise
    sd2 = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}; load_imagenet_encoder(SMPLikeUnet(3, 1), sd2)
    for bad in ({k: v for k, v in sd.items() if k != "features.5.conv.0.0.weight"}, {**sd, "features.99.0.weight": torch.zeros(1)}):
        try: load_imagenet_encoder(SMPLikeUnet(3, 1), bad); assert False
        except ValueError: pass
    try: load_imagenet_encoder(SMPLikeUnet(4, 1), sd); assert False                                 # a 4-channel first conv cannot take the 3-channel ImageNet weights
    except ValueError as e: assert "shape mismatch" in str(e)


def fake_mirror(tmp, n=2, plume_first=False):
    """a fake STARCOP_test mirror (npy bytes behind the faked rasterio) with WV3 bands, labels, test.csv and the CRC manifest; returns (mirror_root, assets_dir)"""
    from test_wv3_features import make_bands, blob, npy
    assets = os.path.join(tmp, "assets"); os.makedirs(assets); base = os.path.join(tmp, "mirror", "STARCOP_test", "STARCOP_test"); lines, rows = [], []; rng = np.random.default_rng(9)
    for t in range(n):
        tid = f"test{t}"; d = os.path.join(base, tid); os.makedirs(d)
        has = bool(t) or plume_first
        bands = np.pad(make_bands(rng, blob() * (1.0 if has else 0.0)), ((0, 0), (0, 512 - 128), (0, 512 - 128)))
        lab = np.zeros((512, 512), np.uint8)
        if has: lab[40:60, 60:80] = 1
        files = {f: bands[i] for i, f in enumerate(W.WV3_FILES)}; files["labelbinary.tif"] = lab
        for f, a in files.items():
            raw = npy(a); open(os.path.join(d, f), "wb").write(raw); lines.append(f"STARCOP_test/{tid}/{f}\t{len(raw)}\t{zlib.crc32(raw):08x}\t8")
        rows.append(dict(id=tid, name=tid, date="2019-10-18", qplume=1200.0 if has else 0.0, has_plume=has))
    open(os.path.join(assets, "zenodo_STARCOP_test.tsv"), "w").write("\n".join(lines) + "\n"); pd.DataFrame(rows).to_csv(os.path.join(assets, "test.csv"), index=False)
    return os.path.join(tmp, "mirror"), assets


def test_wv3_test_feature_cache():
    """the kernel-local cache of the computed WV3 test features: filled by the first run, read by the next, invalidated by a different normalisation file or a different marker"""
    tmp = tempfile.mkdtemp()
    try:
        mirror, assets = fake_mirror(tmp); norm = os.path.join(tmp, W.NORM_FILE); json.dump(dict(scale=[0.01, 0.02, 0.01]), open(norm, "w")); cdir = os.path.join(tmp, "cache")
        X1, Y1, _, m1 = W.load_wv3_test_cached(mirror, assets, norm, cdir, verbose=False)
        assert sorted(os.listdir(cdir)) == ["wv3_test_X.npy", "wv3_test_Y.npy", "wv3_test_cache.json"] and X1.shape == (2, 3, 512, 512)
        real = W.load_wv3_test_from_mirror
        def boom(*a, **k): raise AssertionError("must read the cache, not the mirror")
        W.load_wv3_test_from_mirror = boom
        try:
            X2, Y2, _, m2 = W.load_wv3_test_cached(mirror, assets, norm, cdir, verbose=False)
            assert np.array_equal(X1, X2) and np.array_equal(Y1, Y2) and list(m2.id) == list(m1.id)
            json.dump(dict(scale=[0.02, 0.02, 0.01]), open(norm, "w"))                    # other limits: the cache is stale
            try: W.load_wv3_test_cached(mirror, assets, norm, cdir, verbose=False); assert False
            except AssertionError as e: assert "must read the cache" in str(e)
            open(os.path.join(cdir, "wv3_test_cache.json"), "w").write("{broken")                                # a broken marker is not trusted either
            try: W.load_wv3_test_cached(mirror, assets, norm, cdir, verbose=False); assert False
            except AssertionError as e: assert "must read the cache" in str(e)
        finally: W.load_wv3_test_from_mirror = real
        X3, _, _, _ = W.load_wv3_test_cached(mirror, assets, norm, cdir, verbose=False)                          # recomputed with the new limits and stored again
        assert not np.array_equal(X1, X3) and not [f for f in os.listdir(cdir) if ".tmp" in f]
    finally: shutil.rmtree(tmp, ignore_errors=True)


def run_train(argv):
    from starcop_edge import train
    with contextlib.redirect_stdout(io.StringIO()): train.main(argv)


def test_train_end_to_end_three_channels():
    import torch
    tmp = tempfile.mkdtemp()
    try:
        write_cache(tmp); common = ["--data", tmp, "--threads", "1", "--device", "cpu", "--log_every", "100", "--eval_every", "1000", "--widths", "4,8,16,32,64", "--stem_stride", "2", "--jitter", "8", "--batch", "4"]
        # (1) validation run: in_ch inferred from the cache, no loss weights, checkpoint and log carry in_ch
        out = os.path.join(tmp, "o1"); run_train(common + ["--out", out, "--steps", "4", "--val_dates", "2019-10-22", "--val_max", "1", "--tag", "wv3_val"])
        ck = torch.load(os.path.join(out, "model.pt"), map_location="cpu", weights_only=False)
        assert ck["in_ch"] == 3 and ck["state_dict"]["enc.0.0.0.weight"].shape[1] == 3 and os.path.exists(os.path.join(out, "val_metrics.json"))
        start = json.loads(open(os.path.join(out, "log.jsonl")).readline()); assert start["in_ch"] == 3 and start["event"] == "start"
        from starcop_edge import edge_eval
        net, ck2 = edge_eval.load_run(out); assert net(torch.rand(1, 3, 64, 64)).shape == (1, 1, 64, 64)       # the edge stage rebuilds a 3-channel network from the checkpoint
        # (2) --hardneg needs the AVIRIS mag1c channel and a W file
        try: run_train(common + ["--out", os.path.join(tmp, "o2"), "--steps", "2", "--hardneg", "0.5"]); assert False
        except SystemExit as e: assert "hardneg" in str(e)
        # (3) test read through the WV3 loader (no validation dates: dev runs never touch the test split)
        mirror, assets = fake_mirror(tmp)
        out3 = os.path.join(tmp, "o3"); run_train(common + ["--out", out3, "--steps", "3", "--test_eval", "--test_kind", "wv3", "--test_mirror", mirror, "--assets", assets, "--tag", "wv3_test"])
        out3b = os.path.join(tmp, "o3b"); cdir = os.path.join(tmp, "tcache")                                       # the same through the kernel-local cache (as the stage kernels do): second run reads it
        for k in range(2): run_train(common + ["--out", out3b + str(k), "--steps", "2", "--test_eval", "--test_kind", "wv3", "--test_mirror", mirror, "--assets", assets, "--test_cache", cdir, "--tag", f"wv3_cache{k}"])
        assert os.path.exists(os.path.join(cdir, "wv3_test_cache.json")) and json.load(open(os.path.join(out3b + "0", "test_metrics.json"))).keys() == json.load(open(os.path.join(out3b + "1", "test_metrics.json"))).keys()
        tm = json.load(open(os.path.join(out3, "test_metrics.json"))); assert "AUPRC_ap" in tm and "F1_strong_q" in tm and os.path.exists(os.path.join(out3, "test_summaries.npz"))
        z = np.load(os.path.join(out3, "test_summaries.npz"), allow_pickle=True); assert len(z["ids"]) == 2 and list(z["ids"]) == ["test0", "test1"]
        # (4) the integrity guard of Amendment 1 still stands: test_eval with validation dates is refused
        try: run_train(common + ["--out", os.path.join(tmp, "o4"), "--steps", "2", "--val_dates", "2019-10-22", "--test_eval", "--test_kind", "wv3", "--test_mirror", mirror, "--assets", assets]); assert False
        except SystemExit as e: assert "integrity guard" in str(e)
        # (5) --pretrained refuses anything but the 3-channel released architecture
        try: run_train(common + ["--out", os.path.join(tmp, "o5"), "--steps", "2", "--pretrained", "x.pth"]); assert False
        except SystemExit as e: assert "mnv2" in str(e)
    finally: shutil.rmtree(tmp, ignore_errors=True)


def test_control_flags_end_to_end():
    """the stage-A control run: released architecture, ImageNet-style encoder file, plateau schedule on validation dates, pos_weight 15, WV3 test read (heavy)"""
    import torch
    from starcop_edge.models import SMPLikeUnet
    tmp = tempfile.mkdtemp()
    try:
        write_cache(tmp); pre = os.path.join(tmp, "imagenet.pth"); torch.save(torchvision_style(SMPLikeUnet(3, 1)), pre); mirror, assets = fake_mirror(tmp, n=1, plume_first=True)
        out = os.path.join(tmp, "ctrl")
        run_train(["--data", tmp, "--out", out, "--arch", "mnv2", "--pretrained", pre, "--sched", "plateau", "--plateau_every", "2", "--steps", "4", "--batch", "2", "--lr", "0.0001", "--pos_weight", "15", "--jitter", "0",
                   "--val_dates", "2019-10-22", "--val_max", "1", "--eval_every", "4", "--threads", "1", "--device", "cpu", "--amp", "--test_eval", "--test_kind", "wv3", "--allow_test_with_val", "--test_mirror", mirror, "--assets", assets,
                   "--tag", "ctrl_wv3"])
        lines = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
        start = lines[0]; assert start["pretrained"]["tensors_loaded"] == start["pretrained"]["encoder_tensors"] and start["in_ch"] == 3 and start["params"] == 6629233 - 32 * 9
        assert any(l["event"] == "plateau" for l in lines) and any(l["event"] == "test" for l in lines) and os.path.exists(os.path.join(out, "test_metrics.json"))
    finally: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_sampler_without_weights_and_loader(); test_sampler_fails_loudly(); test_wv3_test_feature_cache(); test_models_with_three_channels(); test_imagenet_encoder_loader(); test_train_end_to_end_three_channels()
    if "--heavy" in sys.argv: test_control_flags_end_to_end()
    print("n-channel support: all checks passed" + (" (heavy mode)" if "--heavy" in sys.argv else ""))
