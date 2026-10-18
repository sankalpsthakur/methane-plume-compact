#!/usr/bin/env python3
"""Corrected-protocol STARCOP scorer.

Usage:  python scripts/score.py --pred-dir DIR --test-dir /path/to/STARCOP_test --test-csv data_meta/test.csv [--kind logit|prob] [--out res.json]

DIR holds one <tile_id>.npy per official test tile (512x512, float, logits by default). Labels are read from
<test-dir>/<tile_id>/labelbinary.tif (official STARCOP_test.zip, CC BY-NC 4.0). Needs numpy + pandas + (rasterio or tifffile).
Headline = paper-text protocol (strong/weak by emission rate, tile positive at > 10 predicted px, AUPRC = average precision), with the
variants of the public code (label-pixel grouping, 640 px rule, 16-threshold trapezoid) reported next to it. 95% tile bootstrap included."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from starcop_edge import metrics as M


def read_label(path):
    try:
        import rasterio
        with rasterio.open(path) as ds: return ds.read(1).astype(np.uint8)
    except ImportError:
        import tifffile
        return tifffile.imread(path).astype(np.uint8)


ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", required=True); ap.add_argument("--test-dir", required=True); ap.add_argument("--test-csv", required=True)
ap.add_argument("--kind", default="logit", choices=["logit", "prob"]); ap.add_argument("--out"); ap.add_argument("--B", type=int, default=1000)
a = ap.parse_args()
te = pd.read_csv(a.test_csv); sums = []
for tid in te.id:
    p = np.load(os.path.join(a.pred_dir, tid + ".npy")).astype(np.float32)
    if a.kind == "prob":
        p = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    sums.append(M.tile_summary(p, read_label(os.path.join(a.test_dir, tid, "labelbinary.tif"))))
S = M.stack_summaries(sums); q = te.qplume.values.astype(np.float64)
m = M.scalar_metrics(S, q); boot = M.bootstrap(S, q, B=a.B)
rows = [("F1 strong (emission >= 1000 kg/h)", "F1_strong_q"), ("F1 weak (emission < 1000 kg/h)", "F1_weak_q"), ("FPR by tile (> 10 px)", "FPR_tile_text10"),
        ("AUPRC (average precision)", "AUPRC_ap"), ("[code variant] F1 strong (> 1000 label px)", "F1_strong_px"), ("[code variant] F1 weak (<= 1000 label px)", "F1_weak_px"),
        ("[code variant] FPR by tile (> 640 px)", "FPR_tile_official640"), ("[code variant] AUPRC trapz over 16 thresholds", "AUPRC_trapz16")]
res = {k: {"value": m[k], "ci95": M.summarize_bootstrap(boot, [k])[k]} for _, k in rows}
for name, k in rows: print(f"{name:48s} {m[k]:7.2f}   95% CI [{res[k]['ci95'][0]:6.2f}, {res[k]['ci95'][1]:6.2f}]")
print("published (Sci. Rep. 13:19999 Table 2, mag1c+RGB, 5-run mean +- std): F1 strong 81.96 +- 3.71 | F1 weak 43.42 +- 5.72 | FPR 43.66 +- 7.36 | AUPRC 51.99 +- 2.76")
print("published mag1c-only: F1 strong 74.15 +- 6.10 | F1 weak 47.57 +- 4.17 | FPR 52.11 +- 10.98 | AUPRC 49.41 +- 5.49")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
