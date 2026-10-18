#!/usr/bin/env python3
"""Render Amendment 3 (freeze after the dev wave) from the dev outputs, the selection rule and the git/cache hashes; appends it to protocol/PREREGISTRATION.md with --write.
usage: amendment3.py <dev_output_dir> [--write]   (run select_config.py first: it writes protocol/selection.json)"""
import argparse, datetime, glob, hashlib, json, os, subprocess, sys
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--write", action="store_true"); ap.add_argument("--dev-start", default="?"); ap.add_argument("--dev-end", default="?")
a = ap.parse_args()
sel = json.load(open(os.path.join(root, "protocol/selection.json"))); ch = sel["chosen"]
CANDS = {"d1": "widths 4,8,16,32,64; stem stride 2; pos_weight 1; hardneg 0", "d2": "widths 8,16,32,64,128; stem stride 2; pos_weight 1; hardneg 0", "d4": "widths 8,16,32,64,128; stem stride 2; pos_weight 0.5; hardneg 0",
         "d6": "widths 8,16,32,64,128; stem stride 2; pos_weight 1; hardneg 0.5"}
git = lambda *c: subprocess.run(["git", "-C", root, *c], capture_output=True, text=True).stdout.strip()
src_hash = hashlib.sha256(b"".join(open(p, "rb").read() for p in sorted(glob.glob(os.path.join(root, "src/starcop_edge/*.py"))))).hexdigest()[:16]
cache = json.load(open(os.path.join(root, "results/prep_cache/prep_report.json")))["sha256"]
rows = "\n".join(f"   | {n} | {r['params']:,} | {r['val_metrics']['F1_strong_q']:.2f} | {r['val_metrics']['F1_weak_q']:.2f} | {r['val_metrics']['FPR_tile_text10']:.2f} | {r['val_metrics']['AUPRC_ap']:.2f} | {r['C']:.2f} | {r['C_official640_logged_only']:.2f} |"
                 for n, r in sorted(sel["runs"].items()))
txt = f"""
---
## Amendment 3 (2026-10-07 {datetime.datetime.now(datetime.timezone.utc):%H:%M} UTC). Written after the dev wave and BEFORE any test evaluation of a model of ours.

1. Dev wave: kernel `bench-gas-dev` v1 (started {a.dev_start}, ended {a.dev_end} UTC), validation dates 2019-10-22/23 only, no test access, 4 processes x 1 thread, mounted cache `bench-gas-prep-cache` v1
   (SHA-256 X {cache['train_X.npy'][:16]}..., Y {cache['train_Y.npy'][:16]}..., W {cache['train_W.npy'][:16]}...), repository commit {git('rev-parse', '--short', 'HEAD')}, hash of src/starcop_edge/*.py {src_hash}.
   Completed runs (validation, single seed 0; C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap); last column logged only):

   | run | parameters | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC | C | C with 640-px FPR |
   |---|---|---|---|---|---|---|---|
{rows}
   Not completed (excluded by the rule): {sel['not_completed'] or 'none'}.
2. Rule (Amendments 1, 2, 2b, 2c, applied by `scripts/select_config.py`, output `protocol/selection.json`): best C = {sel['best_C']:.2f}; band (C >= best - {sel['tol']}) = {sel['band']}; fewest parameters in the band -> {sel['smallest_size_candidates']};
   highest C among those -> **{ch}**.
3. Frozen configuration of the final wave: {ch} = {CANDS[ch]}; EdgeUNet(4 inputs: mag1c, TOA 640/550/460 nm, normalised as in the official code and clipped to [0, 2]); 8,000 steps, batch 16, AdamW (peak lr 2e-3, weight decay 0),
   200 warmup steps then cosine to 2 %, jitter 32 px, D4 augmentation, class-balanced window sampler; seeds 0, 1, 2, 3, 4; trained on ALL 3,425 train tiles; last-step weights (no checkpoint selection, no early stopping).
   Edge variants as in Amendment 2 item 4 (BN folded, ONNX fp32 opset 17, static int8 QDQ, 48 TRAIN calibration tiles, nothing tuned on test).
4. Test evaluations the final kernel `bench-gas-final` will make (announced here, logged in TEST-LOG.md rows #4 onward whatever their outcome): 5 PyTorch fp32 (one per seed), 5 ONNX fp32 and 5 ONNX int8 (same seeds),
   the released mag1c+RGB checkpoint through the same ONNX fp32 and int8 export (2), and the official `run_validation` cross-check of the released mag1c+RGB checkpoint (1, TEST-LOG #4).
5. At the time of writing no test number of any model of ours exists; the only test numbers seen are the released checkpoints (TEST-LOG #2-#3). Claims follow PLAN.md wording; a margin inside the published std or the bootstrap interval is not claimed as a win.
"""
print(txt)
if a.write:
    open(os.path.join(root, "protocol/PREREGISTRATION.md"), "a").write(txt); print("appended to protocol/PREREGISTRATION.md")
