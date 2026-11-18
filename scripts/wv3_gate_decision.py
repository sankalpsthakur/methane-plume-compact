#!/usr/bin/env python3
"""Amendment 7 reproduction gate (committed before the control has run): does the paper-recipe control (released architecture, 3 Varon+Sanchez channels, ImageNet-pretrained encoder, 15 epochs, 1 run)
land within +-1 published standard deviation of the paper's "Our (Varon+Sanchez)" row of Table 1 on the PRIMARY metric, AUPRC (average precision), scored on the 342 test tiles by our scorer?
usage: wv3_gate_decision.py <stage-A output dir> [--published protocol/wv3_published.json]
Reads <dir>/RUN_STATUS.json (rc of ctrl_wv3) and <dir>/ctrl_wv3/test_metrics.json. Prints a JSON verdict and exits 0 (PASS), 1 (FAIL) or 2 (UNFINISHED: rc not 0 or no test metrics).
The other three columns and the label-pixel / 640-px / 16-threshold variants are reported, never gated. FAIL or UNFINISHED ends the wave: G2 is not pushed and the test rows #44-#48 are closed as not run."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "ctrl_wv3"
COLS = ("AUPRC_ap", "F1_strong_q", "F1_weak_q", "FPR_tile_text10")
VARIANTS = ("AUPRC_trapz16", "F1_strong_px", "F1_weak_px", "FPR_tile_official640")


def decide(out_dir, published):
    pub = json.load(open(published)); row = pub["rows"][pub["comparator_row"]]; prim = pub["primary_metric"]
    work = os.path.join(out_dir, "kaggle", "working"); work = work if os.path.isdir(work) else out_dir
    rs_p = os.path.join(work, "RUN_STATUS.json"); tm_p = os.path.join(work, RUN, "test_metrics.json")
    rs = json.load(open(rs_p)) if os.path.exists(rs_p) else {}
    rc = (rs.get("runs") or {}).get(RUN, {}).get("rc")
    if rc != 0 or not os.path.exists(tm_p):
        return dict(verdict="UNFINISHED", rc=rc, note="the control did not finish with a test evaluation: the wave ends (Amendment 7 item 7); nothing is re-run without the owner's yes")
    m = json.load(open(tm_p)); mean, sd = row[prim]
    lo, hi = mean - sd, mean + sd
    rep = {c: dict(control=round(m[c], 3), published_mean=row[c][0], published_sd=row[c][1], z=(round((m[c] - row[c][0]) / row[c][1], 2) if row[c][1] else None)) for c in COLS}
    return dict(verdict="PASS" if lo <= m[prim] <= hi else "FAIL", primary=prim, control=round(m[prim], 3), band=[round(lo, 3), round(hi, 3)], published_row=pub["comparator_row"], z=round((m[prim] - mean) / sd, 2),
                columns=rep, variants_labelled_not_gated={k: round(m[k], 3) for k in VARIANTS if k in m}, rc=rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--published", default=os.path.join(HERE, "..", "protocol", "wv3_published.json"))
    a = ap.parse_args(); r = decide(a.dir, a.published); print(json.dumps(r, indent=1))
    sys.exit({"PASS": 0, "FAIL": 1, "UNFINISHED": 2}[r["verdict"]])
