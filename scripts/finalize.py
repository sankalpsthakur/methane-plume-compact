#!/usr/bin/env python3
"""Turn the fetched outputs of a final wave into the results tables (local, numpy only, a few seconds).
usage: finalize.py <final_output_dir> [--prefix f_d2_s] [--out results/final] [--baseline kernel_outputs/baseline] [--B 2000]
Reads  <dir>/<prefix>{0..4}/test_summaries.npz           (PyTorch fp32 per-seed test summaries)
       <dir>/edge/<prefix>{k}_onnx_{fp32,int8}_test_summaries.npz, <dir>/edge/edge_report.json   (edge pipeline)
       <baseline>/baseline_test_summaries.npz (released mag1c+RGB, run R3) and baseline_mag1c_only_test_summaries.npz
Writes <out>_<variant>_vs_<ref>.json (scripts/aggregate.py output) and <out>_TABLE.md (paper-style table, paired differences, sizes)."""
import argparse, glob, json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--prefix", default="f_d2_s"); ap.add_argument("--out", default="results/final")
ap.add_argument("--baseline", default="kernel_outputs/baseline"); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--latency", default="results/latency_m4.json"); ap.add_argument("--ref-dir", default="kernel_outputs/final-ref")
a = ap.parse_args()
pt = sorted(glob.glob(f"{a.dir}/{a.prefix}[0-9]*/test_summaries.npz"))
variants = {"pytorch_fp32": pt, "onnx_fp32": sorted(glob.glob(f"{a.dir}/edge/{a.prefix}[0-9]*_onnx_fp32_test_summaries.npz")), "onnx_int8": sorted(glob.glob(f"{a.dir}/edge/{a.prefix}[0-9]*_onnx_int8_test_summaries.npz"))}
refs = {"released_mag1c_rgb": (f"{a.baseline}/baseline_test_summaries.npz", "released mag1c+RGB (run R3)"), "released_mag1c_only": (f"{a.baseline}/baseline_mag1c_only_test_summaries.npz", "released mag1c-only")}
agg = {}
for v, files in variants.items():
    if not files: continue
    for r, (rp, rn) in refs.items():
        out = f"{a.out}_{v}_vs_{r}.json"
        subprocess.run([sys.executable, os.path.join(here, "aggregate.py"), out, "--seeds", *files, "--ref", rp, "--ref-name", rn, "--B", str(a.B)], check=True, stdout=subprocess.DEVNULL)
        agg[(v, r)] = json.load(open(out))
edge = json.load(open(f"{a.dir}/edge/edge_report.json")) if os.path.exists(f"{a.dir}/edge/edge_report.json") else {}
if os.path.exists(f"{a.ref_dir}/edge/edge_report.json"): edge.update(json.load(open(f"{a.ref_dir}/edge/edge_report.json")))  # released checkpoint rows from the reference kernel
lat = json.load(open(a.latency)) if os.path.exists(a.latency) else {}
HEAD = ["F1_strong_q", "F1_weak_q", "FPR_tile_text10", "AUPRC_ap"]; LAB = ["F1 strong", "F1 weak", "FPR tile (>10 px)", "AUPRC"]
def cell(d, k): return f"{d['mean'][k]:.2f} +- {d['std_ddof1'][k]:.2f} [{d['boot95_of_mean'][k][0]:.1f}, {d['boot95_of_mean'][k][1]:.1f}]"
L = ["| model (test, 342 tiles) | " + " | ".join(LAB) + " |", "|---|" + "---|" * len(HEAD)]
for v, nm in (("pytorch_fp32", "ours, PyTorch fp32"), ("onnx_fp32", "ours, ONNX fp32"), ("onnx_int8", "ours, ONNX int8 (static QDQ)")):
    d = agg.get((v, "released_mag1c_only"))
    if d: L.append(f"| {nm}, {d['seeds']}-seed mean +- std [tile-bootstrap 95 % of the mean] | " + " | ".join(cell(d, k) for k in HEAD) + " |")
d0 = next(iter(agg.values()), None)
for r, (rp, rn) in refs.items():
    any_d = next((agg[x] for x in agg if x[1] == r), None)
    if any_d: L.append(f"| {rn}, one run (our scorer) | " + " | ".join(f"{any_d['ref'][k]:.2f}" for k in HEAD) + " |")
if d0:
    for nm, key in (("published mag1c+RGB (Sci. Rep. 13:19999 Table 2, 5-run mean +- std)", "mag1c_rgb"), ("published mag1c-only (same table)", "mag1c_only")):
        L.append(f"| {nm} | " + " | ".join(f"{d0['published'][key][k][0]:.2f} +- {d0['published'][key][k][1]:.2f}" for k in HEAD) + " |")
P = ["", "Paired difference (ours - released, same 342 tiles, same bootstrap draws; interval = tile-bootstrap 95 % of the difference of the seed-mean and the single released run):", "",
     "| ours | reference | " + " | ".join(LAB) + " |", "|---|---|" + "---|" * len(HEAD)]
for (v, r), d in agg.items():
    P.append(f"| {v} | {d['ref_name']} | " + " | ".join(f"{d['paired_diff_vs_ref'][k]['diff_of_means']:+.2f} [{d['paired_diff_vs_ref'][k]['boot95'][0]:+.1f}, {d['paired_diff_vs_ref'][k]['boot95'][1]:+.1f}]" for k in HEAD) + " |")
S = ["", "Per-seed values (PyTorch fp32):", "", "| seed | " + " | ".join(LAB) + " |", "|---|" + "---|" * len(HEAD)]
if ("pytorch_fp32", "released_mag1c_only") in agg:
    for i, ps in enumerate(agg[("pytorch_fp32", "released_mag1c_only")]["per_seed"]): S.append(f"| {i} | " + " | ".join(f"{ps[k]:.2f}" for k in HEAD) + " |")
E = ["", "Size and edge cost:", "", "| model | params | PyTorch state (KB) | ONNX fp32 (KB) | ONNX int8 (KB) | Kaggle EPYC 1-thread median ms fp32 / int8 | M4 1-thread median ms fp32 / int8 |", "|---|---|---|---|---|---|---|"]
for name, rec in edge.items():
    if "error" in rec: E.append(f"| {name} | error: {rec['error'][:80]} | | | | | |"); continue
    kl = lambda t: rec.get(f"{t}_latency_ms_1thread_kaggle(median,min)", [float('nan')])[0]
    fname = "released_mag1c_rgb" if name.startswith("released") else name   # ONNX file stem differs from the edge_report key for the released checkpoint
    m4 = lambda t: lat.get(f"{fname}_{t.split('_')[1]}.onnx", {}).get("median_ms", "not measured")
    E.append(f"| {name} | {rec['params']:,} | {rec['pt_state_bytes']/1e3:.0f} | {rec['onnx_fp32_bytes']/1e3:.0f} | {rec['onnx_int8_bytes']/1e3:.0f} | {kl('onnx_fp32'):.0f} / {kl('onnx_int8'):.0f} | {m4('onnx_fp32')} / {m4('onnx_int8')} |")
txt = "\n".join(L + P + S + E) + "\n"; open(f"{a.out}_TABLE.md", "w").write(txt); print(txt)
