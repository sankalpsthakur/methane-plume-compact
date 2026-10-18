#!/usr/bin/env python3
"""Aggregate per-seed test summaries into the paper-style table.
usage: aggregate.py out.json --seeds s0.npz s1.npz ... [--ref released.npz --ref-name NAME] [--B 2000]
Reports mean +- sample std (ddof=1, also ddof=0) over seeds, a 95% tile-bootstrap interval of the seed-mean, and paired
differences to a reference model (same tiles, same bootstrap draws)."""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from starcop_edge import metrics as M

HEAD = ["F1_strong_q", "F1_weak_q", "FPR_tile_text10", "AUPRC_ap"]
EXTRA = ["F1_strong_px", "F1_weak_px", "FPR_tile_official640", "AUPRC_trapz16", "F1_all_plume", "TPR_tile_text10_all_plume", "TPR_tile_text10_weak_q", "TPR_tile_text10_strong_q"]
PUB = {"F1_strong_q": (81.96, 3.71), "F1_weak_q": (43.42, 5.72), "FPR_tile_text10": (43.66, 7.36), "AUPRC_ap": (51.99, 2.76)}
PUB_M1 = {"F1_strong_q": (74.15, 6.10), "F1_weak_q": (47.57, 4.17), "FPR_tile_text10": (52.11, 10.98), "AUPRC_ap": (49.41, 5.49)}

def load(p):
    z = np.load(p, allow_pickle=False)
    S = {k: z[k] for k in z.files if k not in ("qplume", "ids")}
    return S, z["qplume"].astype(np.float64)

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", nargs="+", required=True)
ap.add_argument("--ref"); ap.add_argument("--ref-name", default="released R3"); ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--chunk", type=int, default=100, help="bootstrap draws per block (keeps peak RSS of the 6000-bin histograms well under 512 MiB)")
a = ap.parse_args()
runs = [load(p) for p in a.seeds]; q = runs[0][1]
res = {"seeds": len(runs), "per_seed": [], "mean": {}, "std_ddof1": {}, "std_ddof0": {}, "boot95_of_mean": {}, "published": {"mag1c_rgb": PUB, "mag1c_only": PUB_M1}}
per = [M.scalar_metrics(S, q) for S, _ in runs]
res["per_seed"] = [{k: round(p[k], 3) for k in HEAD + EXTRA} for p in per]
for k in HEAD + EXTRA:
    v = np.array([p[k] for p in per]); res["mean"][k] = float(v.mean()); res["std_ddof1"][k] = float(v.std(ddof=1)) if len(v) > 1 else 0.0; res["std_ddof0"][k] = float(v.std())
N = len(q); rng = np.random.default_rng(0); W = rng.multinomial(N, np.full(N, 1.0 / N), size=a.B).astype(np.float64)
KEYS = HEAD + EXTRA

def boot_metrics(S, qq):
    """(K, B) metric values for every bootstrap draw, computed block by block (same draws W for every model: paired comparisons are exact)."""
    out = []
    for b0 in range(0, a.B, a.chunk):
        m = M.compute_metrics(S, qq, weights=W[b0:b0 + a.chunk]); out.append(np.stack([np.asarray(m[k]).ravel() for k in KEYS], 0))
    return np.concatenate(out, axis=1)

boot = np.mean([boot_metrics(S, q) for S, _ in runs], axis=0)  # (K,B) seed-mean per draw
for i, k in enumerate(HEAD + EXTRA):
    res["boot95_of_mean"][k] = [float(np.nanpercentile(boot[i], 2.5)), float(np.nanpercentile(boot[i], 97.5))]
if a.ref:
    Sr, qr = load(a.ref); mr = M.scalar_metrics(Sr, qr); br = boot_metrics(Sr, qr)
    res["ref_name"] = a.ref_name; res["ref"] = {k: float(mr[k]) for k in HEAD + EXTRA}; res["paired_diff_vs_ref"] = {}
    for i, k in enumerate(HEAD + EXTRA):
        d = boot[i] - br[i]
        res["paired_diff_vs_ref"][k] = {"diff_of_means": float(res["mean"][k] - mr[k]), "boot95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
json.dump(res, open(a.out, "w"), indent=1)
print(f"{'metric':26s} {'ours mean +- std':>20s} {'boot95(mean)':>18s}   published mag1c+RGB   mag1c-only")
for k in HEAD:
    lo, hi = res["boot95_of_mean"][k]; p = PUB[k]; p1 = PUB_M1[k]
    print(f"{k:26s} {res['mean'][k]:9.2f} +- {res['std_ddof1'][k]:5.2f}   [{lo:6.2f},{hi:6.2f}]   {p[0]:.2f} +- {p[1]:.2f}   {p1[0]:.2f} +- {p1[1]:.2f}")
if a.ref:
    print("paired diff vs", a.ref_name); [print(f"  {k:24s} {v['diff_of_means']:7.2f}  boot95 [{v['boot95'][0]:6.2f},{v['boot95'][1]:6.2f}]") for k, v in res["paired_diff_vs_ref"].items() if k in HEAD]
