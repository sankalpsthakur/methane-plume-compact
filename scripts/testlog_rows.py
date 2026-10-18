#!/usr/bin/env python3
"""Print TEST-LOG.md rows for every test evaluation of the final wave, from the fetched outputs, in the numbering announced in TEST-LOG.md / Amendment 3:
#4 official run_validation (ref kernel), #5-9 PyTorch fp32 seeds, #10-14 ONNX fp32 seeds, #15-19 ONNX int8 seeds (final kernel), #20-21 released checkpoint ONNX fp32 / int8 (ref kernel).
Failures are listed too, so nothing is silently omitted.
usage: testlog_rows.py <final_output_dir> [--ref-dir <ref_output_dir>] [--time "HH:MM-HH:MM UTC"] [--kernel "bench-gas-final v1"] [--ref-kernel "bench-gas-final-ref v1"]"""
import argparse, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--ref-dir", default=""); ap.add_argument("--time", default="see kernel log")
ap.add_argument("--kernel", default="bench-gas-final v1"); ap.add_argument("--ref-kernel", default="bench-gas-final-ref v1")
a = ap.parse_args(); rows = {}
H = lambda m: "F1 strong/weak (rate) %.2f/%.2f, FPR(>10 px) %.2f, AUPRC %.2f" % (m["F1_strong_q"], m["F1_weak_q"], m["FPR_tile_text10"], m["AUPRC_ap"])
def row(n, kernel, model, purpose, outcome): rows[n] = f"| {n} | {a.time} | {kernel} | {model} | {purpose} | {outcome} |"
# official cross-check (ref kernel)
oc = os.path.join(a.ref_dir, "official_check.json") if a.ref_dir else ""
if oc and os.path.exists(oc):
    r = json.load(open(oc)); o = r["official"]; d = r["abs_diff"]
    row(4, a.ref_kernel, "released mag1c+RGB, official `run_validation` (commit %s)" % r["commit"][:8], "cross-check of our scorer (redo of the official part of #3)",
        "OK. official F1 strong(px)/weak(px) %.2f/%.2f, FPR(640 px) %.2f, AUPRC(trapz16) %.2f; abs diff to our scorer: %s" % (o["F1_strong_px"], o["F1_weak_px"], o["FPR_tile_official640"], o["AUPRC_trapz16"], {k: float('%.2g' % v) for k, v in d.items()}))
else:
    row(4, a.ref_kernel, "released mag1c+RGB, official `run_validation`", "cross-check of our scorer", "NOT PRODUCED (no official_check.json; ref kernel not run or it failed): see its log")
# PyTorch fp32 per seed (final kernel)
seeds = sorted(d for d in glob.glob(os.path.join(a.dir, "f_*_s[0-9]*")) if os.path.isdir(d))
for k in range(5):
    nm = next((os.path.basename(d) for d in seeds if os.path.basename(d).endswith(f"_s{k}")), f"f_d1_s{k}")
    tm = os.path.join(a.dir, nm, "test_metrics.json")
    row(5 + k, a.kernel, f"{nm} PyTorch fp32, last-step weights", "final test evaluation (one per seed)", ("OK. " + H(json.load(open(tm)))) if os.path.exists(tm) else "FAILED or missing: no test_metrics.json (see logs/%s.log)" % nm)
# edge variants per seed (final kernel)
er = os.path.join(a.dir, "edge", "edge_report.json"); E = json.load(open(er)) if os.path.exists(er) else {}
for g, tag in ((10, "onnx_fp32"), (15, "onnx_int8")):
    for k in range(5):
        nm = next((n for n in E if n.endswith(f"_s{k}") and n.startswith("f_")), f"f_d1_s{k}"); rec = E.get(nm, {})
        row(g + k, a.kernel, f"{nm} {tag.replace('_', ' ')}", "edge variant, same test tiles", ("OK. " + H(rec[tag])) if tag in rec else "FAILED or missing: " + str(rec.get("error", "no result in edge_report.json"))[:140])
# released checkpoint through the edge export (ref kernel)
er2 = os.path.join(a.ref_dir, "edge", "edge_report.json") if a.ref_dir else ""; E2 = json.load(open(er2)) if er2 and os.path.exists(er2) else {}
rec = next((v for n, v in E2.items() if n.startswith("released")), {})
for n, tag in ((20, "onnx_fp32"), (21, "onnx_int8")):
    row(n, a.ref_kernel, f"released mag1c+RGB {tag.replace('_', ' ')} (same export as ours)", "size, latency and score of the leader under the identical edge pipeline", ("OK. " + H(rec[tag])) if tag in rec else "NOT PRODUCED: " + str(rec.get("error", "ref kernel not run or failed"))[:140])
print("\n".join(rows[n] for n in sorted(rows)))
