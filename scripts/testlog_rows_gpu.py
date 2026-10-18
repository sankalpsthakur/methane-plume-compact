#!/usr/bin/env python3
"""TEST-LOG.md rows #22-#37 of the GPU wave (Amendment 4) from the fetched small outputs, failures listed too so nothing is silently omitted.
#22-26 PyTorch fp32 seeds 0-4 (bench-gas-gpu-compact), #27-31 ONNX fp32, #32-36 ONNX int8, #37 the control (bench-gas-gpu-control).
usage: testlog_rows_gpu.py --compact kernel_outputs/gpu-compact --control kernel_outputs/gpu-control [--prefix gf_] [--time "HH:MM-HH:MM UTC (kernel window)"] [--kernel "bench-gas-gpu-compact v1"] [--ckernel "bench-gas-gpu-control v1"]"""
import argparse, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--compact", default=""); ap.add_argument("--control", default=""); ap.add_argument("--prefix", default="gf_"); ap.add_argument("--time", default="see kernel log")
ap.add_argument("--kernel", default="bench-gas-gpu-compact v1"); ap.add_argument("--ckernel", default="bench-gas-gpu-control v1")
a = ap.parse_args(); rows = {}
H = lambda m: "F1 strong/weak (rate) %.2f/%.2f, FPR(>10 px) %.2f, AUPRC %.2f" % (m["F1_strong_q"], m["F1_weak_q"], m["FPR_tile_text10"], m["AUPRC_ap"])
def row(n, kernel, model, purpose, outcome): rows[n] = f"| {n} | {a.time} | {kernel} | {model} | {purpose} | {outcome} |"
dirs = sorted(d for d in glob.glob(os.path.join(a.compact, a.prefix + "*_s[0-9]*")) if os.path.isdir(d)) if a.compact else []
E = json.load(open(os.path.join(a.compact, "edge", "edge_report.json"))) if a.compact and os.path.exists(os.path.join(a.compact, "edge", "edge_report.json")) else {}
for k in range(5):
    nm = next((os.path.basename(d) for d in dirs if os.path.basename(d).endswith(f"_s{k}")), f"{a.prefix}<cfg>_s{k}")
    tm = os.path.join(a.compact, nm, "test_metrics.json")
    row(22 + k, a.kernel, f"{nm} PyTorch fp32, last-step weights", "final test evaluation (one per seed)", ("OK. " + H(json.load(open(tm)))) if os.path.exists(tm) else f"FAILED or missing: no test_metrics.json (see logs/{nm}.log and RUN_STATUS.json)")
    rec = E.get(nm, {})
    for base, tag in ((27, "onnx_fp32"), (32, "onnx_int8")):
        row(base + k, a.kernel, f"{nm} {tag.replace('_', ' ')}", "edge variant, same test tiles", ("OK. " + H(rec[tag])) if tag in rec else "FAILED or missing: " + str(rec.get("error", "no result in edge_report.json"))[:140])
cm = os.path.join(a.control, "ctrl_mnv2", "test_metrics.json") if a.control else ""
row(37, a.ckernel, "ctrl_mnv2 PyTorch fp32, last-step weights (paper-recipe-like control, plateau on the validation dates)", "control row for reading the compact results", ("OK. " + H(json.load(open(cm)))) if cm and os.path.exists(cm) else "FAILED or missing: no test_metrics.json (see logs/ctrl_mnv2.log and RUN_STATUS.json)")
print("\n".join(rows[n] for n in sorted(rows)))
