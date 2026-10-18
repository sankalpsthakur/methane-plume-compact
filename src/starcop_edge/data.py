"""Compact STARCOP cache + training sampler.

Cache layout (written by the prep kernel), one set per split in {train, test}:
  {split}_X.npy  float16 (N,4,512,512)  channels = mag1c, TOA_AVIRIS_640nm, 550nm, 460nm, already normalised with the official
                                        constants: clip(raw/factor, 0, 2), factor = (1750, 60, 60, 60)
  {split}_Y.npy  uint8   (N,512,512)    labelbinary
  {split}_W.npy  uint8   (N,512,512)    weight_mag1c * 255 (official per-pixel loss weight, in [0.1, 1])
  {split}_meta.csv                      id,name,date,qplume,has_plume,n_label_px,...
"""
import os
import threading, queue
import numpy as np
import pandas as pd

WIN, STRIDE, POS_PIX = 128, 64, 40  # 10/64^2 * 128^2 = 40 px, as in the official tiled_dataframe


def load_split(d, split, mmap=True):
    mm = "r" if mmap else None
    X = np.load(os.path.join(d, f"{split}_X.npy"), mmap_mode=mm)
    Y = np.load(os.path.join(d, f"{split}_Y.npy"), mmap_mode=mm)
    Wt = np.load(os.path.join(d, f"{split}_W.npy"), mmap_mode=mm)
    meta = pd.read_csv(os.path.join(d, f"{split}_meta.csv"))
    return X, Y, Wt, meta


ACT_THR = 200.0 / 1750.0   # normalised mag1c of 200 ppm.m (the enhancement level used by the official mask code)
ACT_PIX = 50               # a negative window is "hard" if >= 50 of its 16384 pixels exceed ACT_THR


def _window_sums(a, offs):
    ii = np.zeros((513, 513), np.int32)
    ii[1:, 1:] = a.astype(np.int32).cumsum(0).cumsum(1)
    return np.array([[ii[r + WIN, c + WIN] - ii[r, c + WIN] - ii[r + WIN, c] + ii[r, c] for c in offs] for r in offs]).ravel()


def window_table(Y, idx, X=None):
    """For tiles `idx`, count label pixels (and, if X given, mag1c-active pixels) inside each of the 7x7 official windows
    (128 px, stride 64). Returns rows [tile, r, c], positive-pixel counts, active-pixel counts."""
    offs = np.arange(0, 512 - WIN + 1, STRIDE)  # 0..384
    rr, cc = np.meshgrid(offs, offs, indexing="ij")
    rows, cnts, acts = [], [], []
    for t in idx:
        cnts.append(_window_sums(np.asarray(Y[t]), offs))
        acts.append(_window_sums(np.asarray(X[t, 0]) > ACT_THR, offs) if X is not None else np.zeros(len(offs) ** 2, np.int32))
        rows.append(np.stack([np.full(rr.size, t), rr.ravel(), cc.ravel()], 1))
    return np.concatenate(rows).astype(np.int32), np.concatenate(cnts).astype(np.int32), np.concatenate(acts).astype(np.int32)


class CropSampler:
    """Class-balanced window sampler mirroring the official WeightedRandomSampler (plume windows and non-plume windows
    each get half of the probability mass), optional jitter, D4 augmentation."""

    def __init__(self, X, Y, Wt, idx, seed=0, jitter=32, pos_frac=0.5, batch=32, prefetch=6, hardneg=0.0):
        self.X, self.Y, self.Wt = X, Y, Wt
        self.rng = np.random.default_rng(seed)
        rows, cnts, acts = window_table(Y, idx, X if hardneg > 0 else None)
        self.pos_w = rows[cnts > POS_PIX]
        neg = cnts <= POS_PIX
        self.hard_w = rows[neg & (acts >= ACT_PIX)]; self.neg_w = rows[neg & ~(acts >= ACT_PIX)] if hardneg > 0 else rows[neg]
        self.jitter, self.pos_frac, self.batch, self.hardneg = jitter, pos_frac, batch, hardneg
        self.n_pos_windows, self.n_neg_windows, self.n_hard_windows = len(self.pos_w), len(self.neg_w), len(self.hard_w)
        self._q = queue.Queue(maxsize=prefetch)
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _one(self):
        if self.rng.random() < self.pos_frac: pool = self.pos_w
        elif self.hardneg > 0 and len(self.hard_w) and self.rng.random() < self.hardneg: pool = self.hard_w
        else: pool = self.neg_w
        t, r, c = pool[self.rng.integers(len(pool))]
        if self.jitter:
            r = int(np.clip(r + self.rng.integers(-self.jitter, self.jitter + 1), 0, 512 - WIN))
            c = int(np.clip(c + self.rng.integers(-self.jitter, self.jitter + 1), 0, 512 - WIN))
        x = np.asarray(self.X[t, :, r:r + WIN, c:c + WIN]); y = np.asarray(self.Y[t, r:r + WIN, c:c + WIN]); w = np.asarray(self.Wt[t, r:r + WIN, c:c + WIN])
        k = int(self.rng.integers(4))
        if k:
            x, y, w = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(w, k)
        if self.rng.random() < 0.5:
            x, y, w = x[:, :, ::-1], y[:, ::-1], w[:, ::-1]
        return x, y, w

    def _run(self):
        while not self._stop:
            xs, ys, ws = zip(*[self._one() for _ in range(self.batch)])
            self._q.put((np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)[:, None],
                         (np.stack(ws).astype(np.float32) / 255.0)[:, None]))

    def next(self):
        return self._q.get()

    def close(self):
        self._stop = True


def load_test_from_mirror(mirror_root, assets_dir, verbose=True):
    """The 342 official test tiles read straight from the Kaggle mirror, every file CRC32-verified against the official Zenodo zip
    central directory, preprocessed exactly like the train cache: X = clip(raw/factor, 0, 2) stored as float16, Y = labelbinary."""
    import zlib
    from .tif import read_tif
    from .models import INPUT_ORDER, INPUT_FACTORS
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
    N = len(meta)
    X = np.empty((N, 4, 512, 512), np.float16); Y = np.empty((N, 512, 512), np.uint8)
    for i, tid in enumerate(meta.id):
        for c, (b, f) in enumerate(zip(INPUT_ORDER, INPUT_FACTORS)):
            X[i, c] = np.clip(rd(tid, b + ".tif").astype(np.float32) / np.float32(f), 0.0, 2.0)
        Y[i] = rd(tid, "labelbinary.tif").astype(np.uint8)
    if verbose: print("test tiles loaded from mirror (CRC verified):", X.shape, flush=True)
    return X, Y, None, meta
