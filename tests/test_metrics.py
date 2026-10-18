"""Local check (numpy only, small random tiles) of the scorer against direct brute-force computations."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from starcop_edge import metrics as M

def exact_ap(scores, labels):
    o = np.argsort(-scores, kind="stable"); l = labels[o]; tp = np.cumsum(l); fp = np.cumsum(~l.astype(bool))
    # AP = sum over distinct score thresholds of (R_k - R_{k-1}) * P_k (step-wise, like sklearn average_precision_score)
    sc = scores[o]; last = np.r_[np.where(np.diff(sc) != 0)[0], len(sc) - 1]
    tp, fp = tp[last], fp[last]; prec = tp / (tp + fp); rec = tp / labels.sum()
    return 100.0 * np.sum(np.diff(np.r_[0.0, rec]) * prec)

def test_scorer():
    rng = np.random.default_rng(0); N, H = 40, 64
    labs, logs, sums = [], [], []
    q = np.zeros(N)
    for i in range(N):
        lab = np.zeros((H, H), np.uint8)
        if i % 3:  # plume tiles of different size
            s = 3 + (i % 5) * 4; lab[10:10 + s, 12:12 + s] = 1; q[i] = 300.0 if i % 2 else 1500.0
        lg = rng.normal(-2.0, 2.0, (H, H)).astype(np.float32) + 3.5 * lab
        labs.append(lab); logs.append(lg); sums.append(M.tile_summary(lg, lab))
    S = M.stack_summaries(sums); m = M.scalar_metrics(S, q)
    # pooled pixel F1 for the emission-rate groups, computed directly at p > 0.5 (logit > 0)
    for g, mask in (("strong_q", q >= 1000), ("weak_q", (q < 1000) & (np.array([l.sum() for l in labs]) > 0))):
        tp = sum(((logs[i] > 0) & (labs[i] > 0)).sum() for i in np.where(mask)[0]); fp = sum(((logs[i] > 0) & (labs[i] == 0)).sum() for i in np.where(mask)[0])
        fn = sum(((logs[i] <= 0) & (labs[i] > 0)).sum() for i in np.where(mask)[0]); f1 = 100 * 2 * tp / (2 * tp + fp + fn)
        assert abs(m[f"F1_{g}"] - f1) < 1e-6, (g, m[f"F1_{g}"], f1)
    # FPR by tile with the 10-pixel rule among tiles without label pixels
    neg = [i for i in range(N) if labs[i].sum() == 0]; pos10 = [(logs[i] > 0).sum() > 10 for i in neg]
    assert abs(m["FPR_tile_text10"] - 100 * np.mean(pos10)) < 1e-9
    # AP (threshold free) against the exact step-wise AP over all pixels pooled
    sc = np.concatenate([l.ravel() for l in logs]); lb = np.concatenate([l.ravel() for l in labs]).astype(bool)
    assert abs(m["AUPRC_ap"] - exact_ap(sc, lb)) < 0.5, (m["AUPRC_ap"], exact_ap(sc, lb))   # 0.01-logit histogram bins
    # bootstrap with all weights 1 reproduces the point estimate
    full = M.compute_metrics(S, q, weights=np.ones((1, N)))
    assert abs(full["F1_weak_q"][0] - m["F1_weak_q"]) < 1e-9
    print("metrics: F1 (strong/weak by emission rate), FPR (10 px), AP vs brute force OK")
if __name__ == "__main__": test_scorer()
