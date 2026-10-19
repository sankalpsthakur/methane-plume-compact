#!/usr/bin/env python3
"""Amendment 6 gate (committed before the gate result existed): does the E = 15 validation run beat G1's E = 10 d1 run by at least
2.0 points of the validation composite C (scripts/select_config.composite, FPR rule text10)?
usage: colab_gate_decision.py <out_dir of the gate job>      (reads [kaggle/working/]RUN_STATUS.json and [kaggle/working/]c_d1_e15_val/val_metrics.json)
Prints a JSON verdict: PASS, FAIL, or UNFINISHED (rc not 0 or no validation numbers: Amendment 6 note 1 a)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from select_config import composite

TOL = 2.0
RUN = "c_d1_e15_val"
COMPONENTS = ("F1_weak_q", "F1_strong_q", "FPR_tile_text10", "AUPRC_ap")


def decide(out_dir):
    ref = json.load(open(os.path.join(HERE, "..", "protocol", "selection_gpu.json")))["runs"]["g_d1"]
    assert abs(composite(ref["val_metrics"]) - ref["C"]) < 1e-9, "composite() does not reproduce the recorded G1 value"
    work = os.path.join(out_dir, "kaggle", "working")   # Colab tarball layout; a Kaggle watcher fetch has the files directly in out_dir
    work = work if os.path.isdir(work) else out_dir
    rs_p = os.path.join(work, "RUN_STATUS.json")
    vm_p = os.path.join(work, RUN, "val_metrics.json")
    rs = json.load(open(rs_p)) if os.path.exists(rs_p) else {}
    rc = (rs.get("runs") or {}).get(RUN, {}).get("rc")
    if rc != 0 or not os.path.exists(vm_p):
        return dict(verdict="UNFINISHED", rc=rc, run_status=rs, note="Amendment 6 note 1 a: re-run once; a second unfinished run ends wave 3")
    m = json.load(open(vm_p))
    c = composite(m)
    return dict(verdict="PASS" if c - ref["C"] >= TOL else "FAIL", C_gate=round(c, 3), C_g1=round(ref["C"], 3), dC=round(c - ref["C"], 3), tol=TOL,
                deltas_val={k: round(m[k] - ref["val_metrics"][k], 3) for k in COMPONENTS},
                gate_val={k: round(m[k], 3) for k in COMPONENTS}, g1_val={k: round(ref["val_metrics"][k], 3) for k in COMPONENTS}, rc=rc)


if __name__ == "__main__":
    print(json.dumps(decide(sys.argv[1]), indent=1))
