#!/usr/bin/env python3
"""Apply the pre-registered selection rule (Amendments 1, 2, 2b, 2c) to the dev runs. Validation data only.
usage: select_config.py <dev_output_dir> [--out protocol/selection.json] [--tol 2.0]
<dev_output_dir>/<run>/val_metrics.json (written by train.py at the end of a completed run) and <run>/log.jsonl (start event: params).
C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap); band = runs with C >= best - tol; choose the fewest parameters in the band, ties -> highest C.
Amendment 4 (GPU wave) adds one clause, written before any GPU number: band members whose parameter count is within 1 % of the smallest are size-equal, and among size-equal members those with stem stride 2
are preferred over stem stride 1 (about 4x the multiply-accumulates at equal parameters: d1 vs d5, d2 vs d2w). It changes nothing for the CPU wave (Amendment 3: d1 alone is within 1 % of itself)."""
import argparse, glob, json, os
SIZE_EQ = 0.01   # Amendment 4 size-equality tolerance (parameters)

def composite(m, fpr="FPR_tile_text10"):
    return (m["F1_strong_q"] + m["F1_weak_q"] + (100.0 - m[fpr]) + m["AUPRC_ap"]) / 4.0

def load_runs(d):
    runs, incomplete = {}, []
    for rd in sorted(glob.glob(os.path.join(d, "*"))):
        if not os.path.isdir(rd): continue
        name = os.path.basename(rd)
        if name == "logs": continue  # per-run stdout logs of the dev kernel, not a run directory
        mp = os.path.join(rd, "val_metrics.json"); lp = os.path.join(rd, "log.jsonl")
        if not (os.path.exists(mp) and os.path.exists(lp)): incomplete.append(name); continue
        start = next(json.loads(l) for l in open(lp) if '"event": "start"' in l)
        m = json.load(open(mp)); runs[name] = dict(params=int(start["params"]), stem_stride=int(start["args"].get("stem_stride", 2)), C=composite(m), C_official640_logged_only=composite(m, "FPR_tile_official640"), metrics=m)
    return runs, incomplete

def select(runs, tol=2.0):
    best = max(r["C"] for r in runs.values())
    band = {n: r for n, r in runs.items() if r["C"] >= best - tol}
    fewest = min(r["params"] for r in band.values())
    equal = {n: r for n, r in band.items() if r["params"] <= fewest * (1 + SIZE_EQ)}   # size-equal: within 1 % of the smallest band member
    cands = {n: r for n, r in equal.items() if r.get("stem_stride", 2) == 2} or equal
    chosen = max(cands, key=lambda n: cands[n]["C"])
    return chosen, best, sorted(band), sorted(cands)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out", default="protocol/selection.json"); ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()
    runs, incomplete = load_runs(a.dir)
    assert runs, "no completed dev run"
    chosen, best, band, cands = select(runs, a.tol)
    print(f"{'run':6s} {'params':>9s} {'C':>7s}  F1s    F1w    FPR10  AUPRC  (C with 640-rule FPR, logged only)")
    for n, r in sorted(runs.items()):
        m = r["metrics"]; print(f"{n:6s} {r['params']:9d} {r['C']:7.2f}  {m['F1_strong_q']:5.1f}  {m['F1_weak_q']:5.1f}  {m['FPR_tile_text10']:5.1f}  {m['AUPRC_ap']:5.1f}  ({r['C_official640_logged_only']:.2f})")
    print("not completed (excluded):", incomplete or "none"); print(f"best C {best:.2f} | band (>= best - {a.tol}): {band} | smallest-size candidates: {cands} -> CHOSEN {chosen}")
    json.dump(dict(chosen=chosen, best_C=best, tol=a.tol, band=band, smallest_size_candidates=cands, not_completed=incomplete,
                   runs={n: {k: v for k, v in r.items() if k != "metrics"} | {"val_metrics": r["metrics"]} for n, r in runs.items()}), open(a.out, "w"), indent=1)
