"""Local check (numpy only) of the window sampler: x/y/w stay aligned under D4 augmentation and jitter, plume windows are drawn about half of the time, hard negatives work."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from starcop_edge import data as D

def test_sampler():
    rng = np.random.default_rng(0); N = 6
    X = np.zeros((N, 4, 512, 512), np.float16); Y = np.zeros((N, 512, 512), np.uint8); W = np.zeros((N, 512, 512), np.uint8)
    for t in range(N):
        X[t, 0] = rng.uniform(0, 0.05, (512, 512)).astype(np.float16)           # background mag1c
        W[t] = rng.integers(26, 255, (512, 512)).astype(np.uint8)
        if t % 2 == 0:                                                           # plume tiles: a blob with high mag1c, label = blob
            r, c = rng.integers(50, 400, 2); X[t, 0, r:r + 40, c:c + 60] = 1.0; Y[t, r:r + 40, c:c + 60] = 1
        if t == 1:                                                               # a hard negative: high mag1c over a large area WITHOUT label
            X[t, 0, 300:380, 100:200] = 0.5
        X[t, 1] = X[t, 0] * 0.5 + 0.25                                           # a channel that is a function of mag1c (alignment probe)
    for hn in (0.0, 0.5):
        s = D.CropSampler(X, Y, W, np.arange(N), seed=3, jitter=32, batch=32, hardneg=hn)
        npos = 0; nhard = 0; tot = 0
        for _ in range(20):
            x, y, w = s.next(); assert x.shape == (32, 4, 128, 128) and y.shape == (32, 1, 128, 128) and w.shape == (32, 1, 128, 128)
            for i in range(32):
                bl = x[i, 0] >= 0.99
                assert np.array_equal(bl, y[i, 0] > 0.5), "label no longer aligned with the input after augmentation"   # blob == label for every crop
                assert np.allclose(x[i, 1], x[i, 0] * 0.5 + 0.25, atol=1e-2)
                assert w[i].min() >= 26 / 255 - 1e-6 and w[i].max() <= 1.0
                npos += int(y[i].sum() > D.POS_PIX); nhard += int(((x[i, 0] > D.ACT_THR) & (y[i, 0] < 0.5)).sum() >= D.ACT_PIX); tot += 1
        s.close()
        frac = npos / tot; print(f"hardneg={hn}: plume crops {frac:.2f}, crops with >= {D.ACT_PIX} active unlabelled px {nhard/tot:.2f}")
        assert 0.3 < frac < 0.75
    print("sampler: alignment, balance, hard negatives OK")
if __name__ == "__main__": test_sampler()
