"""STARCOP scorer (numpy only). Re-implements the official starcop.validation.run_validation aggregation
(spaceml-org/STARCOP @ c478926) from per-tile summaries, so that one pass over the model outputs yields:

  * official-code grouping: has_plume = label_pixels > 0; "strong"(easy) = label_pixels > 1000 px, "weak"(hard) = 1..1000 px
  * paper-text grouping    : strong = qplume >= 1000 kg/h, weak = qplume < 1000 kg/h (paper Sec. Metrics / Zenodo description)
  * pixel metrics at p > 0.5, pooled over the tiles of the group (TP/FP/FN sums), F1 = 2PR/(P+R)
  * tile classification: tile is positive if #predicted px > 10*H*W/64^2 (= 640 px at 512x512; official code);
    the paper text says "more than 10 active pixels" -> both variants reported. FPR by tile = fraction of no-plume tiles positive.
  * AUPRC: (a) average precision from pooled logit histograms (threshold free), (b) trapezoid over the 16 thresholds
    of the official validation code ([0,1e-3,1e-2]+[0.5..0.95 step .05]+[.99,.995,.999]), pooled over ALL tiles.
Per-tile summaries make a tile-level bootstrap exact and cheap."""
import numpy as np

THRESHOLDS = np.array([0.0, 1e-3, 1e-2] + [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)] + [0.99, 0.995, 0.999], dtype=np.float64)
THR32 = THRESHOLDS.astype(np.float32)
# histogram of logits for the threshold-free AP
LMIN, LMAX, LSTEP = -30.0, 30.0, 0.01
NB = int(round((LMAX - LMIN) / LSTEP))


def sigmoid32(z):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float32)))


def tile_summary(logit, label):
    """logit: (H,W) float32 model output (pre-sigmoid); label: (H,W) {0,1}. Returns dict of small arrays."""
    H, W = logit.shape
    lab = label.astype(bool)
    prob = sigmoid32(logit)
    pb = prob > np.float32(0.5)
    tp = int(np.count_nonzero(pb & lab)); fp = int(np.count_nonzero(pb & ~lab))
    npos = int(lab.sum()); pred_px = int(pb.sum())
    pos_above = np.array([np.count_nonzero(prob[lab] > t) for t in THR32], dtype=np.int64)
    neg_above = np.array([np.count_nonzero(prob[~lab] > t) for t in THR32], dtype=np.int64)
    idx = np.clip(((np.clip(logit, LMIN, LMAX - 1e-4) - LMIN) / LSTEP).astype(np.int64), 0, NB - 1)
    hp = np.bincount(idx[lab], minlength=NB).astype(np.int32)
    hn = np.bincount(idx[~lab], minlength=NB).astype(np.int32)
    return dict(n_pos=npos, tp=tp, fp=fp, fn=npos - tp, tn=H * W - npos - fp, pred_px=pred_px, hw=H * W,
                pos_above=pos_above, neg_above=neg_above, hist_pos=hp, hist_neg=hn)


def stack_summaries(summaries):
    keys = summaries[0].keys()
    return {k: np.stack([np.asarray(s[k]) for s in summaries]) for k in keys}


def _f1(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = tp / (tp + fp); r = tp / (tp + fn)
        return 100.0 * 2 * p * r / (p + r), 100.0 * p, 100.0 * r


def _ap_from_hist(hp, hn):
    """hp, hn: (..., NB) pooled counts per logit bin. Average precision with thresholds descending over bins."""
    cp = np.cumsum(hp[..., ::-1], axis=-1).astype(np.float64); cn = np.cumsum(hn[..., ::-1], axis=-1).astype(np.float64)
    P = cp[..., -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(cp + cn > 0, cp / (cp + cn), 1.0)
        rec = cp / P
    drec = np.diff(np.concatenate([np.zeros_like(rec[..., :1]), rec], axis=-1), axis=-1)
    return 100.0 * np.nansum(prec * drec, axis=-1)


def _trapz16(pos_above, neg_above):
    """PR points at the 16 official thresholds (pooled), trapezoid over recall (ascending). pos/neg_above: (..., T)."""
    P = pos_above[..., :1].astype(np.float64)  # thr=0 -> everything positive (prob > 0)
    P = np.maximum(P, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = pos_above / P
        prec = np.where(pos_above + neg_above > 0, pos_above / (pos_above + neg_above), 1.0)
    # thresholds ascending -> recall descending; flip to ascending recall
    rec, prec = rec[..., ::-1], prec[..., ::-1]
    return 100.0 * np.sum((rec[..., 1:] - rec[..., :-1]) * (prec[..., 1:] + prec[..., :-1]) / 2.0, axis=-1)


def compute_metrics(S, qplume, weights=None):
    """S: dict from stack_summaries (arrays with leading N). qplume: (N,) kg/h from the CSV.
    weights: optional (B,N) tile multiplicities (bootstrap) -> every metric becomes an array of shape (B,)."""
    N = len(S["n_pos"])
    Wt = np.ones((1, N)) if weights is None else np.asarray(weights, dtype=np.float64)

    def ssum(a, mask=None):
        a = np.asarray(a, dtype=np.float64)
        m = Wt if mask is None else Wt * mask[None, :]
        return m @ a

    n_pos = S["n_pos"]; has = n_pos > 0
    out = {}
    groups = {
        "strong_px": has & (n_pos > 1000), "weak_px": has & (n_pos <= 1000),
        "strong_q": has & (qplume >= 1000), "weak_q": has & (qplume < 1000),
        "all_plume": has,
    }
    for g, mask in groups.items():
        f1, p, r = _f1(ssum(S["tp"], mask), ssum(S["fp"], mask), ssum(S["fn"], mask))
        out[f"F1_{g}"], out[f"P_{g}"], out[f"R_{g}"] = f1, p, r
    f1, p, r = _f1(ssum(S["tp"]), ssum(S["fp"]), ssum(S["fn"]))
    out["F1_all_tiles"], out["P_all_tiles"], out["R_all_tiles"] = f1, p, r
    neg = ~has
    out["n_tiles"] = np.array([N]); 
    for thr_name, thr in (("official640", 10.0 * S["hw"] / 64.0 ** 2), ("text10", np.full(N, 10.0))):
        pos_tile = (S["pred_px"] > thr).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"FPR_tile_{thr_name}"] = 100.0 * ssum(pos_tile, neg) / ssum(np.ones(N), neg)
            out[f"TPR_tile_{thr_name}_all_plume"] = 100.0 * ssum(pos_tile, has) / ssum(np.ones(N), has)
            out[f"TPR_tile_{thr_name}_strong_q"] = 100.0 * ssum(pos_tile, groups["strong_q"]) / ssum(np.ones(N), groups["strong_q"])
            out[f"TPR_tile_{thr_name}_weak_q"] = 100.0 * ssum(pos_tile, groups["weak_q"]) / ssum(np.ones(N), groups["weak_q"])
    out["FPR_px_no_plume"] = 100.0 * ssum(S["fp"], neg) / ssum(S["fp"] + S["tn"], neg)
    pa, na = ssum(S["pos_above"]), ssum(S["neg_above"])
    out["AUPRC_trapz16"] = _trapz16(pa, na)
    out["AUPRC_ap"] = _ap_from_hist(ssum(S["hist_pos"]), ssum(S["hist_neg"]))
    return {k: (np.atleast_1d(v) if not isinstance(v, np.ndarray) else v) for k, v in out.items()}


def scalar_metrics(S, qplume):
    return {k: float(np.asarray(v).ravel()[0]) for k, v in compute_metrics(S, qplume).items()}


def bootstrap(S, qplume, B=1000, seed=0):
    """Tile-level bootstrap (resample tiles with replacement). Returns {metric: (B,) array}."""
    rng = np.random.default_rng(seed)
    N = len(S["n_pos"])
    counts = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)
    return compute_metrics(S, qplume, weights=counts)


def summarize_bootstrap(boot, keys, lo=2.5, hi=97.5):
    return {k: (float(np.nanpercentile(boot[k], lo)), float(np.nanpercentile(boot[k], hi))) for k in keys}


HEADLINE = ["F1_strong_q", "F1_weak_q", "FPR_tile_official640", "AUPRC_ap"]
