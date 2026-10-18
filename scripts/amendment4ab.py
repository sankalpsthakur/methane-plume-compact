#!/usr/bin/env python3
"""Render Amendment 4a (written after G0 and BEFORE G1 is pushed) or Amendment 4b (after G1 and BEFORE G2 is pushed) from the fetched outputs; prints the text, --write appends it to protocol/PREREGISTRATION.md.
usage: amendment4ab.py 4a [--probe kernel_outputs/gpu-probe/gpuprobe.json] [--budget protocol/gpu_budget.json] [--ledger results/gpu_ledger.csv] [--write]
       amendment4ab.py 4b [--sel protocol/selection_gpu.json] [--budget protocol/gpu_budget.json] [--status kernel_outputs/gpu-wide/RUN_STATUS.json] [--write]
Local, stdlib only (run through bin/guard.sh)."""
import argparse, datetime, glob, hashlib, json, os, subprocess
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["4a", "4b"]); ap.add_argument("--probe", default="kernel_outputs/gpu-probe/gpuprobe.json"); ap.add_argument("--budget", default="protocol/gpu_budget.json")
ap.add_argument("--ledger", default="results/gpu_ledger.csv"); ap.add_argument("--sel", default="protocol/selection_gpu.json"); ap.add_argument("--status", default="kernel_outputs/gpu-wide/RUN_STATUS.json"); ap.add_argument("--write", action="store_true")
a = ap.parse_args()
P = lambda p: os.path.join(root, p)
sha = lambda p: hashlib.sha256(open(P(p), "rb").read()).hexdigest()[:16]
git = lambda *c: subprocess.run(["git", "-C", root, *c], capture_output=True, text=True).stdout.strip()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
bud = json.load(open(P(a.budget))); w = bud["wave"]; ctl = bud["control"]
mg = bud.get("merged"); capw = dict(bud["caps_wall_s"]); capw["compact"] = mg["wall_cap_s"] if mg else capw["compact"]
ledger = [l.rstrip("\n").split(",") for l in open(P(a.ledger))][1:] if os.path.exists(P(a.ledger)) else []
led = "; ".join(f"{r[1]} {r[2].split('bench-gas-')[-1]}: billed {r[6]} h ({r[7]})" for r in ledger) or "none yet"
if a.which == "4a":
    pr = json.load(open(P(a.probe))); env = pr.get("env", {})
    wc = json.load(open(P("kernels/gpu-wide/cfg.json")))
    assert sorted(r["name"] for r in wc["runs"]) == sorted("g_" + c for c in w["cands"]) and all(r["steps"] == w["steps"] for r in wc["runs"]), "kernels/gpu-wide is not generated for this plan: regenerate it first"
    assert wc["billing_b"] == bud["billing_b"], "kernels/gpu-wide was generated with another billing factor"
    rows = "\n".join(f"   | {k.split('|')[0]} | amp {k.split('amp')[-1]} | {v.get('steps_per_s', v.get('error'))} | {v.get('params', '')} | {v.get('peak_mem_gb', '')} |" for k, v in pr.items() if isinstance(v, dict) and "|bs" in k)
    txt = f"""
---
## Amendment 4a ({now} UTC by the session clock). Written after G0 and BEFORE G1 is pushed. No validation or test number of any GPU model exists at this time.

1. G0 `bench-gas-gpu-probe` (script sha256 {sha('kernels/gpu-probe/script.py')}): hardware obtained: {pr.get('device')}, n_gpu {env.get('n_gpu')}, torch {env.get('torch')}, cuda {env.get('cuda')}, arch list {env.get('arch_list')}, capability {env.get('capability')}, host CPUs {env.get('cpus')};
   imports: rasterio {env.get('rasterio')}, pandas {env.get('pandas')}, numpy {env.get('numpy')}, onnx {env.get('onnx')}, onnxruntime {env.get('onnxruntime')}; probe seconds {pr.get('secs')}. Gas ledger so far: {led}.
   Synthetic throughput (batch 32, 128 x 128 crops, steps/s per process):

   | case | AMP | steps/s | parameters | peak GB |
   |---|---|---|---|---|
{rows}
2. Plan (`scripts/mk_gpu_wave.py plan`, `{a.budget}` sha256 {sha(a.budget)}; sampler ceiling {bud['sampler_batches_per_s']} batches/s from `results/gpu_smoke/sampler_bench.json`; billing factor b = {bud['billing_b']}, n_gpu {bud['n_gpu']}, per-process slowdown {bud['model']['RHO']}):
   planning rates (steps/s) {bud['rates_steps_per_s']}; **E = {w['epochs']} paper epochs = {w['steps']:,} steps** for G1 and G2; candidate set {w['cands']}; **control epochs Ec = {ctl['epochs']}** ({ctl['steps']:,} steps, plateau check every {ctl['plateau_every']} steps) {'inside the G2 kernel' if ctl.get('merged_into') else 'as its own kernel'};
   projected billed seconds: G1 {w['projected_wide_billed_s']:,}; worst-case G2 + control {w['projected_compact_worst_billed_s']:,}; total {bud.get('projected_total_billed_h')} billed h of the 7.0 h of caps; verdict {bud['verdict']}.
   Wall caps for `-t` (billed caps / b; compact = G2 + control merged: {mg['billed_cap_s'] if mg else 'n/a'} billed s): {capw}; soft deadlines (wall s): {w['soft_deadline_wall_s']}.
3. Kernels generated for the plan (not pushed before this amendment is committed): `kernels/gpu-wide` sha256 {sha('kernels/gpu-wide/script.py')} (candidates {w['cands']}, {w['steps']:,} steps, seed 0, validation dates only, no test mirror, CFG deadline_s {json.load(open(P('kernels/gpu-wide/cfg.json')))['deadline_s']}).
4. Nothing else changes: Amendments 4 and 5 and their notes apply as written.
"""
else:
    sel = json.load(open(P(a.sel))); ch = sel["chosen"].replace("g_", "", 1)
    st = json.load(open(P(a.status))) if os.path.exists(P(a.status)) else {}
    rows = "\n".join(f"   | {n.replace('g_', '', 1)} | {r['params']:,} | {r['val_metrics']['F1_strong_q']:.2f} | {r['val_metrics']['F1_weak_q']:.2f} | {r['val_metrics']['FPR_tile_text10']:.2f} | {r['val_metrics']['AUPRC_ap']:.2f} | {r['C']:.2f} | {st.get('runs', {}).get(n, {}).get('secs', '')} |" for n, r in sorted(sel["runs"].items()))
    gc = json.load(open(P("kernels/gpu-compact/cfg.json"))); seeds = [r["name"] for r in gc["runs"] if r["name"].startswith("gf_")]
    assert len(seeds) == 5 and all(n.startswith(f"gf_{ch}_s") for n in seeds) and all(r["steps"] == w["steps"] for r in gc["runs"] if r["name"].startswith("gf_")), "kernels/gpu-compact is not generated for the chosen config and E: regenerate it first"
    assert any(r["name"] == "ctrl_mnv2" and r["steps"] == ctl["steps"] for r in gc["runs"]) and gc["billing_b"] == bud["billing_b"], "kernels/gpu-compact lacks the control job of the plan: regenerate with --with-control"
    txt = f"""
---
## Amendment 4b ({now} UTC by the session clock). Written after G1 and BEFORE G2 (with the control) is pushed, and before any test evaluation of a GPU model.

1. G1 `bench-gas-gpu-wide` (validation dates 2019-10-22/23 only, no test access, E = {w['epochs']} = {w['steps']:,} steps, seed 0, last-step weights). Completed runs (C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap)); wall seconds of each run in the last column:

   | candidate | parameters | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC | C | secs |
   |---|---|---|---|---|---|---|---|
{rows}
   Not completed (excluded by the rule): {sel['not_completed'] or 'none'}. Gas ledger so far: {led}.
2. Rule (Amendment 4 item 5 with its size-equality clause, applied by `scripts/select_config.py`, output `protocol/selection_gpu.json` sha256 {sha(a.sel)}): best C = {sel['best_C']:.2f}; band (C >= best - {sel['tol']}) = {sel['band']}; smallest-size set = {sel['smallest_size_candidates']}; highest C among those -> **{sel['chosen']}**.
3. Frozen configuration of G2: {ch} (see `kernels/gpu-compact/cfg.json`), E = {w['epochs']} ({w['steps']:,} steps, batch 32), seeds {', '.join(s.rsplit('_s', 1)[1] for s in seeds)}, trained on ALL 3,425 train tiles, last-step weights, AMP; the paper-recipe-like control ({ctl['epochs']} epochs = {ctl['steps']:,} steps) is the last job of the same queue (Amendment 5 note 1).
   `kernels/gpu-compact/script.py` sha256 {sha('kernels/gpu-compact/script.py')}, CFG billed cap {gc['billed_cap_s']} s, wall cap {gc['wall_cap_s']} s, soft deadline {gc['deadline_s']} s.
4. Test evaluations G2 will make (announced in TEST-LOG.md before this push): rows #22-#26 PyTorch fp32 (seeds 0-4), #27-#31 ONNX fp32, #32-#36 ONNX int8, #37 the control. No test number of any GPU model exists at the time of writing.
5. G2 timing gate (`scripts/g2_timing_gate.sh {ch}`) output is recorded in STATUS.md before the push; a FAIL means G2 is not pushed.
"""
print(txt)
if a.write:
    open(P("protocol/PREREGISTRATION.md"), "a").write(txt); print("appended to protocol/PREREGISTRATION.md")
