#!/usr/bin/env python3
"""TEST-LOG.md rows #43-#48 of the WV3 wave (Amendment 7) from the fetched small outputs; failures are listed too so nothing is silently omitted.
#43 the paper-recipe control (stage A, bench-gas-gpu-wv3a), #44-#48 PyTorch fp32 seeds 0-4 of the frozen candidate (stage B, bench-gas-gpu-wv3b; closed as not run when the gate failed).
usage: testlog_rows_wv3.py --stage-a kernel_outputs/gpu-wv3a [--stage-b kernel_outputs/gpu-wv3b --config d1] [--time "HH:MM-HH:MM UTC (kernel window)"] [--gate-failed]"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--stage-a", required=True); ap.add_argument("--stage-b", default=""); ap.add_argument("--config", default="d1"); ap.add_argument("--time", default="see kernel log")
ap.add_argument("--gate-failed", action="store_true"); ap.add_argument("--ka", default="bench-gas-gpu-wv3a v1"); ap.add_argument("--kb", default="bench-gas-gpu-wv3b v1")
a = ap.parse_args()
H = lambda m: "F1 strong/weak (rate) %.2f/%.2f, FPR(>10 px) %.2f, AUPRC(AP) %.2f" % (m["F1_strong_q"], m["F1_weak_q"], m["FPR_tile_text10"], m["AUPRC_ap"])
rows = []
cm = os.path.join(a.stage_a, "ctrl_wv3", "test_metrics.json")
rows.append(f"| 43 | {a.time} | {a.ka} | ctrl_wv3 PyTorch fp32, last-step weights (paper-recipe control: released architecture, ImageNet encoder, 3 Varon+Sanchez channels, 15 epochs, plateau on the validation dates) | reproduction gate | " +
            (("OK. " + H(json.load(open(cm)))) if os.path.exists(cm) else "FAILED or missing: no test_metrics.json (see logs/ctrl_wv3.log and RUN_STATUS.json)") + " |")
for k in range(5):
    n = 44 + k; nm = f"wf_{a.config}_s{k}"
    if a.gate_failed or not a.stage_b: rows.append(f"| {n} | not run | (bench-gas-gpu-wv3b, never pushed) | {nm} | final test evaluation (one per seed) | NOT RUN: the reproduction gate failed or the control did not finish (Amendment 7 item 6); no test file was read for it |"); continue
    tm = os.path.join(a.stage_b, nm, "test_metrics.json")
    rows.append(f"| {n} | {a.time} | {a.kb} | {nm} PyTorch fp32, last-step weights | final test evaluation (one per seed) | " + (("OK. " + H(json.load(open(tm)))) if os.path.exists(tm) else f"FAILED or missing: no test_metrics.json (see logs/{nm}.log and RUN_STATUS.json)") + " |")
print("\n".join(rows))
