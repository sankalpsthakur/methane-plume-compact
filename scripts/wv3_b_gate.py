#!/usr/bin/env python3
"""Amendment 7 stage-B timing gate and feasibility restriction (timing only, never metrics; read-only).
usage: wv3_b_gate.py <stage-A output dir> [--cfg kernels/gpu-wv3b/cfg.json] [--out protocol/wv3_feasible.json]
For every validation run g_<c> of stage A that finished (rc 0) it takes the measured wall seconds (same step count, same two-worker layout with the control beside it, includes init and one validation pass), turns them into the
projected wall time of the five seeds of c in stage B: K_OVER + 3 rounds x seed job, where the validation pass of the measured run is swapped for the test pass (the first round computes the WV3 test features, later rounds read the
kernel-local cache), and compares it with the soft deadline (cfg deadline_s) of the stage-B kernel. A candidate whose seeds would not fit is NOT feasible and is excluded from the selection (select_config.py --exclude), so selection can
never pick a configuration whose five seeds do not fit. Prints the exact --exclude list. Exit 0 = d1 is feasible (stage B may be generated), 1 = stop and ask the owner."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mk_wv3_wave as V


def project(secs):
    return V.K_OVER + 3 * secs + (V.SEED_TEST_FIRST_S - V.EVAL_S) + 2 * (V.SEED_TEST_CACHED_S - V.EVAL_S)


def gate(a_dir, deadline):
    work = os.path.join(a_dir, "kaggle", "working"); work = work if os.path.isdir(work) else a_dir
    rs = json.load(open(os.path.join(work, "RUN_STATUS.json"))).get("runs", {}); out = {}
    for c in V.G1_ORDER:
        r = rs.get(f"g_{c}")
        if not r or r.get("rc") != 0: out[c] = dict(finished=False, feasible=False, note="validation run missing or unfinished: excluded"); continue
        w = project(r["secs"]); out[c] = dict(finished=True, secs=r["secs"], projected_wall_s=round(w), deadline_s=deadline, feasible=bool(w <= deadline))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--cfg", default=os.path.join(HERE, "..", "kernels", "gpu-wv3b", "cfg.json")); ap.add_argument("--out", default="")
    a = ap.parse_args(); deadline = json.load(open(a.cfg))["deadline_s"]; res = gate(a.dir, deadline)
    excl = ["ctrl_wv3"] + [f"g_{c}" for c in V.G1_ORDER if not res[c]["feasible"]]
    res = dict(candidates=res, exclude_from_selection=",".join(excl)); print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if res["candidates"]["d1"]["feasible"] else 1)
