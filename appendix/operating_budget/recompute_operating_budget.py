#!/usr/bin/env python3
# Copyright 2026 Sankalp Thakur. Licensed under the Apache License, Version 2.0.
"""Recompute the operating-budget numbers of the plume audit from its saved per-tile records (numpy + stdlib only).

Original code of this repository; it contains no code from the audit kit (which was labelled GPL-3.0).
It reads two kinds of saved files and scores nothing new: no model is loaded and no test tile is read.

  --zip   PATH   the audit's original output archive (plume-budget-output.zip: <ref>-tile-scores.json and
                 <ref>-<rule>-rows.json for 3 references x 4 rules, 342 tiles each)
  --tables DIR   the CSV copy of the same records (data-cc-by-nc/tile_scores.csv, pixel_counts.csv)
  --emit-tables DIR  (with --zip) write that CSV copy
  --baseline DIR optional: gas kernel_outputs/baseline (the track's own per-tile summaries), to compare the
                 released references' standard-rule counts and to print the program's 110-tile weak F1

Rule recomputed: for each held-out date, threshold = ascending order statistic of the tile statistic (11th largest
pixel score) of the mask-negative tiles of the OTHER dates at rank ceil((n_neg + 1) * (1 - alpha)), +inf if the
rank exceeds n_neg; a tile is an indication iff its statistic is strictly above its date's threshold. Weak = mask
has positives and 0 < qplume < 1000 kg/h (audit definition, 108 tiles); program definition = mask has positives and
qplume < 1000 (110 tiles). Localized detection = indication and mask IoU >= 0.1 (IoU = tp / (tp + fp + fn), from the
saved pixel counts at the fold threshold).
"""
import argparse, csv, io, json, math, sys, zipfile
from fractions import Fraction
import numpy as np

REFS = ["released-mag1c-rgb", "released-mag1c-only", "raw-mag1c"]
RULES = ["standard", "0.01", "0.05", "0.1"]
BAR = {"worst_date_fpr": 0.05, "weak_localized_recall": 0.80, "weak_mean_iou": 0.25}
IOU_MIN = 0.1


def load_zip(path):
    z = zipfile.ZipFile(path)
    tiles, counts = {}, {}
    for r in REFS:
        tiles[r] = json.loads(z.read(f"{r}-tile-scores.json"))
        for rule in RULES:
            counts[(r, rule)] = json.loads(z.read(f"{r}-{rule}-rows.json"))
    return tiles, counts


TILE_COLS = ["id", "date", "flight", "qplume", "metadata_has_plume", "label_pixels", "tile_score"]
COUNT_COLS = ["id", "tp", "fp", "fn", "tn", "indication"]


def emit_tables(tiles, counts, out):
    import os
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tile_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference"] + TILE_COLS)
        for r in REFS:
            for t in tiles[r]:
                w.writerow([r] + [repr(t[c]) if c in ("qplume", "tile_score") else t[c] for c in TILE_COLS])
    with open(os.path.join(out, "pixel_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference", "rule"] + COUNT_COLS)
        for r in REFS:
            for rule in RULES:
                for t in counts[(r, rule)]:
                    w.writerow([r, rule] + [int(t[c]) if c == "indication" else t[c] for c in COUNT_COLS])


def load_tables(d):
    tiles = {r: [] for r in REFS}
    counts = {(r, k): [] for r in REFS for k in RULES}
    with open(f"{d}/tile_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            tiles[row["reference"]].append({
                "id": row["id"], "date": row["date"], "flight": row["flight"], "qplume": float(row["qplume"]),
                "metadata_has_plume": row["metadata_has_plume"], "label_pixels": int(row["label_pixels"]),
                "tile_score": float(row["tile_score"])})
    with open(f"{d}/pixel_counts.csv", newline="") as f:
        for row in csv.DictReader(f):
            counts[(row["reference"], row["rule"])].append({
                "id": row["id"], **{k: int(row[k]) for k in ("tp", "fp", "fn", "tn")}, "indication": bool(int(row["indication"]))})
    return tiles, counts


def thresholds(tiles, alpha):
    """Per-date threshold from the mask-negative tiles of the other dates (exact rational rank)."""
    a = Fraction(alpha)
    dates = sorted({t["date"] for t in tiles})
    out = {}
    for d in dates:
        cal = sorted(t["tile_score"] for t in tiles if t["date"] != d and t["label_pixels"] == 0)
        rank = math.ceil((len(cal) + 1) * (1 - a))
        out[d] = (cal[rank - 1] if rank <= len(cal) else math.inf, rank, len(cal))
    return out


def iou(c):
    den = c["tp"] + c["fp"] + c["fn"]
    return c["tp"] / den if den else 0.0


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def aggregate(tiles, cnt):
    """Aggregates for one (reference, rule): tiles and cnt are in the same order (checked by id)."""
    assert [t["id"] for t in tiles] == [c["id"] for c in cnt]
    neg = [i for i, t in enumerate(tiles) if t["label_pixels"] == 0]
    pos = [i for i, t in enumerate(tiles) if t["label_pixels"] > 0]
    weak = [i for i in pos if 0 < tiles[i]["qplume"] < 1000]
    weak110 = [i for i in pos if tiles[i]["qplume"] < 1000]
    strong = [i for i in pos if tiles[i]["qplume"] >= 1000]
    unknown = [i for i in pos if tiles[i]["qplume"] <= 0]
    ind = lambda i: cnt[i]["indication"]
    fpr_date = {}
    for d in sorted({t["date"] for t in tiles}):
        n = [i for i in neg if tiles[i]["date"] == d]
        fpr_date[d] = (sum(ind(i) for i in n), len(n))
    g = lambda idx, k: sum(cnt[i][k] for i in idx)
    return {
        "n_negative": len(neg), "false_indications": sum(ind(i) for i in neg),
        "worst_date_fpr": max(a / b for a, b in fpr_date.values()), "fpr_by_date": fpr_date,
        "n_weak": len(weak), "n_weak_program": len(weak110), "n_strong": len(strong), "n_positive_unknown_rate": len(unknown),
        "weak_localized": sum(1 for i in weak if ind(i) and iou(cnt[i]) >= IOU_MIN),
        "weak_localized_recall": sum(1 for i in weak if ind(i) and iou(cnt[i]) >= IOU_MIN) / len(weak),
        "weak_mean_iou": float(np.mean([iou(cnt[i]) for i in weak])),
        "weak_pixel_f1_audit108": f1(g(weak, "tp"), g(weak, "fp"), g(weak, "fn")),
        "weak_pixel_f1_program110": f1(g(weak110, "tp"), g(weak110, "fp"), g(weak110, "fn")),
        "strong_pixel_f1": f1(g(strong, "tp"), g(strong, "fp"), g(strong, "fn")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip"); ap.add_argument("--tables"); ap.add_argument("--emit-tables"); ap.add_argument("--baseline")
    ap.add_argument("--published", help="RESULTS.json of the audit kit: compare every printed number to it")
    ap.add_argument("--json", help="write the recomputed numbers here")
    a = ap.parse_args()
    if bool(a.zip) == bool(a.tables):
        sys.exit("give exactly one of --zip or --tables")
    tiles, counts = load_zip(a.zip) if a.zip else load_tables(a.tables)
    if a.emit_tables:
        assert a.zip, "--emit-tables needs --zip"
        emit_tables(tiles, counts, a.emit_tables)
    res, bad = {}, []
    for r in REFS:
        T = tiles[r]
        for rule in RULES:
            C = counts[(r, rule)]
            m = aggregate(T, C)
            if rule != "standard":
                th = thresholds(T, rule)
                m["thresholds"] = {d: v[0] for d, v in th.items()}
                m["ranks"] = {d: (v[1], v[2]) for d, v in th.items()}
                # the saved indication flags must follow from threshold(date) and the tile statistic
                flags = [t["tile_score"] > th[t["date"]][0] for t in T]
                if flags != [c["indication"] for c in C]:
                    bad.append(f"{r} {rule}: saved indication flags differ from the recomputed rule on "
                               f"{sum(x != c['indication'] for x, c in zip(flags, C))} tiles")
            else:
                thr = 200.0 if r == "raw-mag1c" else 0.0   # fixed diagnostic: logit 0 (p = 0.5) or 200 ppm.m
                if [t["tile_score"] > thr for t in T] != [c["indication"] for c in C]:
                    bad.append(f"{r} standard: saved indication flags differ from tile_score > {thr}")
            res[(r, rule)] = m
    pub = json.load(open(a.published))["results"] if a.published else None
    print("reference | rule | neg indications | worst-date FPR | weak localized (n/108, recall) | weak mean IoU | weak F1 pixel (108 / 110) | strong F1")
    for r in REFS:
        for rule in RULES:
            m = res[(r, rule)]
            print(f"{r} | {rule} | {m['false_indications']}/{m['n_negative']} | {100*m['worst_date_fpr']:.2f}% | "
                  f"{m['weak_localized']}/{m['n_weak']} = {100*m['weak_localized_recall']:.2f}% | {m['weak_mean_iou']:.6f} | "
                  f"{100*m['weak_pixel_f1_audit108']:.2f} / {100*m['weak_pixel_f1_program110']:.2f} | {100*m['strong_pixel_f1']:.2f}")
            if pub:
                p = pub[r][rule]["metrics"]
                for ours, theirs in (("false_indications", "false_indications"), ("worst_date_fpr", "worst_date_fpr"),
                                     ("weak_localized", "weak_localized_indications"),
                                     ("weak_localized_recall", "weak_localized_recall_iou_0_1"),
                                     ("weak_mean_iou", "weak_mean_iou"), ("weak_pixel_f1_audit108", "weak_pixel_f1")):
                    if abs(m[ours] - p[theirs]) > 1e-9:
                        bad.append(f"{r} {rule}: {ours} {m[ours]!r} vs published {theirs} {p[theirs]!r}")
                if rule != "standard":
                    for d, v in pub[r][rule]["folds"].items():
                        if v["threshold"] != m["thresholds"][d] and not (math.isinf(m["thresholds"][d]) and v["threshold_is_positive_infinity"]):
                            bad.append(f"{r} {rule} {d}: threshold {m['thresholds'][d]!r} vs published {v['threshold']!r}")
    print("\nprimary 5% rule, per held-out date (negative indications, weak localized, calibration tiles, rank, threshold)")
    for r in REFS:
        C = counts[(r, "0.05")]; T = tiles[r]; th = thresholds(T, "0.05")
        for d in sorted(th):
            n = [i for i, t in enumerate(T) if t["date"] == d and t["label_pixels"] == 0]
            w = [i for i, t in enumerate(T) if t["date"] == d and t["label_pixels"] > 0 and 0 < t["qplume"] < 1000]
            wl = sum(1 for i in w if C[i]["indication"] and iou(C[i]) >= IOU_MIN)
            print(f"{r} {d}: {sum(C[i]['indication'] for i in n)}/{len(n)} | {wl}/{len(w)} | {th[d][2]} | {th[d][1]} | {th[d][0]!r}")
    print("\nresearch bar (<= 5% on every date, >= 80% weak localized recall, >= 0.25 mean weak IoU), 5% rule")
    for r in REFS:
        m = res[(r, "0.05")]
        ok = (m["worst_date_fpr"] <= BAR["worst_date_fpr"], m["weak_localized_recall"] >= BAR["weak_localized_recall"], m["weak_mean_iou"] >= BAR["weak_mean_iou"])
        print(f"{r}: budget {ok[0]}, recall {ok[1]}, IoU {ok[2]} -> {'cleared' if all(ok) else 'failed'}")
    if a.baseline:
        print("\ncomparison with the track's own per-tile summaries (standard rule = logit > 0, i.e. p > 0.5)")
        for ref, fn in (("released-mag1c-rgb", "baseline_test_summaries.npz"), ("released-mag1c-only", "baseline_mag1c_only_test_summaries.npz")):
            S = np.load(f"{a.baseline}/{fn}")
            order = {i: k for k, i in enumerate(S["ids"].tolist())}
            T, C = tiles[ref], counts[(ref, "standard")]
            idx = [order[t["id"]] for t in T]
            same = {k: int(sum(int(S[k][j]) != c[k] for j, c in zip(idx, C))) for k in ("tp", "fp", "fn", "tn")}
            same["n_pos_vs_label_pixels"] = int(sum(int(S["n_pos"][j]) != t["label_pixels"] for j, t in zip(idx, T)))
            same["qplume"] = int(sum(float(S["qplume"][j]) != t["qplume"] for j, t in zip(idx, T)))
            print(f"{ref}: tiles {len(idx)}, tiles with a differing count: {same}")
            tp, fp, fn = (np.array([S[k][j] for j in idx]) for k in ("tp", "fp", "fn"))
            has = np.array([t["label_pixels"] > 0 for t in T]); q = np.array([t["qplume"] for t in T])
            w110 = has & (q < 1000); w108 = has & (q > 0) & (q < 1000)
            F = lambda mk: 100 * f1(tp[mk].sum(), fp[mk].sum(), fn[mk].sum())
            print(f"  from the track's counts: weak F1 {F(w110):.2f} on {int(w110.sum())} tiles, {F(w108):.2f} on {int(w108.sum())} tiles")
            if not all(v == 0 for v in same.values()):
                bad.append(f"{ref}: counts differ from the baseline summaries: {same}")
    print("\nMISMATCHES: " + ("none" if not bad else ""))
    for b in bad:
        print("  " + b)
    if a.json:
        json.dump({f"{r}|{k}": {kk: (vv if not isinstance(vv, float) or math.isfinite(vv) else str(vv)) for kk, vv in v.items()}
                   for (r, k), v in res.items()}, open(a.json, "w"), indent=1, default=str)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
