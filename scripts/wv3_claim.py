#!/usr/bin/env python3
"""Amendment 7 claim rule, applied as written (committed before any WV3 test number of ours exists).
usage: wv3_claim.py out.json --seeds s0.npz s1.npz s2.npz s3.npz s4.npz --gate gate.json [--published protocol/wv3_published.json] [--B 2000]
Primary metric: AUPRC (average precision, threshold free) on the 342 test tiles, PyTorch fp32, last-step weights, seeds 0-4. Comparator: the paper's "Our (Varon+Sanchez)" row of Table 1 (mean p, standard deviation sigma over 5 runs).
Interval level 97.5 % TWO-SIDED (owner, 2026-10-19: the labels are shared with the AVIRIS evaluations): tile-bootstrap percentiles 1.25 / 98.75 of the seed-mean (2,000 draws, default_rng(0), the draws of scripts/aggregate.py),
and the t interval over seeds, mean +- t(0.9875; n-1) * s / sqrt(n). A claim needs ALL of:
  (a) the reproduction gate PASSED (scripts/wv3_gate_decision.py output);  (b) all 5 seeds completed;  (c) mean - p > sigma (a margin smaller than one published sd is not a win: the house rule of the first pre-registration);
  (d) the lower tile-bootstrap bound of the seed-mean > p;  (e) the lower t bound over seeds > p.
Strong F1, weak F1 and tile FPR are reported with the same intervals and are NEVER claimed. No released WV3 checkpoint exists, so there is no paired test."""
import argparse, datetime, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from starcop_edge import metrics as M

HEAD = ["AUPRC_ap", "F1_strong_q", "F1_weak_q", "FPR_tile_text10"]
EXTRA = ["AUPRC_trapz16", "F1_strong_px", "F1_weak_px", "FPR_tile_official640", "F1_all_plume", "TPR_tile_text10_all_plume", "TPR_tile_text10_weak_q", "TPR_tile_text10_strong_q"]
LEVEL = 0.975


def _betacf(a, b, x, itmax=500, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap; d = 1e-300 if abs(d) < 1e-300 else d; d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < eps: break
    return h


def betai(a, b, x):
    """regularized incomplete beta function I_x(a, b)"""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1.0) / (a + b + 2.0) else 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    p = 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_ppf(p, df):
    """quantile of Student's t for p in (0.5, 1), by bisection"""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files if k not in ("qplume", "ids")}, z["qplume"].astype(np.float64)


def analyse(seed_files, pub, gate, B=2000, chunk=100, level=LEVEL, date=None):
    runs = [load(p) for p in seed_files]; q = runs[0][1]; n = len(runs); N = len(q)
    prim = pub["primary_metric"]; row = pub["rows"][pub["comparator_row"]]; p, sigma = row[prim]
    per = [M.scalar_metrics(S, q) for S, _ in runs]; keys = HEAD + EXTRA
    rng = np.random.default_rng(0); Wt = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)    # the draws of scripts/aggregate.py
    def boot(S):
        out = []
        for b0 in range(0, B, chunk):
            m = M.compute_metrics(S, q, weights=Wt[b0:b0 + chunk]); out.append(np.stack([np.asarray(m[k]).ravel() for k in keys], 0))
        return np.concatenate(out, axis=1)
    bm = np.mean([boot(S) for S, _ in runs], axis=0)      # (K, B): seed-mean per bootstrap draw
    tq = t_ppf(1.0 - (1.0 - level) / 2.0, n - 1) if n > 1 else float("nan")
    lo_p, hi_p = round(100.0 * (1.0 - level) / 2.0, 6), round(100.0 * (1.0 - (1.0 - level) / 2.0), 6)
    res = dict(n_seeds=n, level_two_sided=level, boot_percentiles=[lo_p, hi_p], t_quantile=tq, per_seed={k: [round(m[k], 3) for m in per] for k in keys}, metrics={})
    for i, k in enumerate(keys):
        v = np.array([m[k] for m in per]); mean = float(v.mean()); s = float(v.std(ddof=1)) if n > 1 else float("nan")
        res["metrics"][k] = dict(mean=mean, std_ddof1=s, boot_lo=float(np.nanpercentile(bm[i], lo_p)), boot_hi=float(np.nanpercentile(bm[i], hi_p)), boot95=[float(np.nanpercentile(bm[i], 2.5)), float(np.nanpercentile(bm[i], 97.5))],
                                 t_lo=mean - tq * s / math.sqrt(n), t_hi=mean + tq * s / math.sqrt(n))
    r = res["metrics"][prim]
    conds = {"a_gate_passed": bool(gate and gate.get("verdict") == "PASS"), "b_all_5_seeds": n == 5, "c_margin_over_published_sd": bool(r["mean"] - p > sigma),
             "d_lower_tile_bootstrap_above_published_mean": bool(r["boot_lo"] > p), "e_lower_t_bound_above_published_mean": bool(r["t_lo"] > p)}
    res["claim"] = dict(primary=prim, published_row=pub["comparator_row"], published_mean=p, published_sd=sigma, margin=r["mean"] - p, margin_in_published_sd=(r["mean"] - p) / sigma, conditions=conds, claim=all(conds.values()),
                        gate=(gate or {}).get("verdict", "not supplied"),
                        wording=("best published result on STARCOP test, simulated-WV3 split, paper-text protocol, self-evaluated, as of " + (date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")) if all(conds.values()) else
                                 ("not distinguishable from the published row" if r["mean"] > p else "below the published row (negative result)")))
    res["vs_published_rows"] = {name: {c: dict(published=list(rw[c]), ours_mean=res["metrics"][c]["mean"]) for c in HEAD} for name, rw in pub["rows"].items()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", nargs="+", required=True); ap.add_argument("--gate", required=True)
    ap.add_argument("--published", default=os.path.join(HERE, "..", "protocol", "wv3_published.json")); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--date", default="", help="the 'as of' date of the claim wording (default: today, UTC)")
    a = ap.parse_args(); pub = json.load(open(a.published)); gate = json.load(open(a.gate)); res = analyse(a.seeds, pub, gate, a.B, date=a.date or None)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'metric':22s} {'ours mean +- std':>18s} {'tile-bootstrap 97.5 %':>26s} {'t 97.5 % (seeds)':>20s}   published (Varon+Sanchez)")
    for k in HEAD:
        m = res["metrics"][k]; pr = pub["rows"][pub["comparator_row"]][k]
        print(f"{k:22s} {m['mean']:8.2f} +- {m['std_ddof1']:5.2f}   [{m['boot_lo']:7.2f}, {m['boot_hi']:7.2f}]   [{m['t_lo']:7.2f}, {m['t_hi']:7.2f}]   {pr[0]:.2f} +- {pr[1]:.2f}")
    c = res["claim"]; print("claim conditions:", json.dumps(c["conditions"])); print("CLAIM" if c["claim"] else "NO CLAIM", "|", c["wording"])
