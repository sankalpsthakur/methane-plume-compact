#!/usr/bin/env python3
"""Generate (and optionally push) the dev wave, the final wave or the reference kernel.
dev:   mk_wave.py dev --parallel 4 --threads 1 [--push]
final: mk_wave.py final --config d1 --parallel 5 --threads 1 [--push]   (config must be one of the dev candidates; frozen after selection; add --with-ref to also run the released-model rows in the same kernel)
ref:   mk_wave.py ref [--push]   (no training: released mag1c+RGB through the same ONNX fp32/int8 export + the official run_validation cross-check; reads the test split)"""
import argparse, json, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("wave", choices=["dev", "final", "ref"]); ap.add_argument("--parallel", type=int, default=2)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--config", default=""); ap.add_argument("--push", action="store_true"); ap.add_argument("--only", default="")
ap.add_argument("--with-ref", action="store_true")
a = ap.parse_args()
COMMON = dict(steps=8000, batch=16, lr=2e-3, wd=0.0, jitter=32)
OFFICIAL = {"F1_strong_px": 84.228, "F1_weak_px": 39.0232, "FPR_tile_official640": 12.5714, "AUPRC_trapz16": 37.1474}  # our scorer on the released mag1c+RGB (TEST-LOG #2-#3)
CANDS = {
    "d1": dict(widths="4,8,16,32,64", stem_stride=2, pos_weight=1.0, hardneg=0.0),
    "d2": dict(widths="8,16,32,64,128", stem_stride=2, pos_weight=1.0, hardneg=0.0),
    "d4": dict(widths="8,16,32,64,128", stem_stride=2, pos_weight=0.5, hardneg=0.0),
    "d6": dict(widths="8,16,32,64,128", stem_stride=2, pos_weight=1.0, hardneg=0.5),
}
if a.wave == "dev":
    names = [n for n in ["d2", "d6", "d4", "d1"] if not a.only or n in a.only.split(",")]
    runs = [dict(name=n, seed=0, val_dates="2019-10-22,2019-10-23", eval_every=4000, **COMMON, **CANDS[n]) for n in names]
    cfg = dict(parallel=a.parallel, threads=a.threads, runs=runs); kname = "dev"
elif a.wave == "final":
    c = CANDS[a.config]
    runs = [dict(name=f"f_{a.config}_s{s}", seed=s, test_eval=True, **COMMON, **c) for s in range(5)]
    cfg = dict(parallel=a.parallel, threads=a.threads, edge=True, runs=runs); kname = "final"
    if a.with_ref: cfg.update(edge_released=True, official_check=OFFICIAL)
else:
    runs = []; cfg = dict(parallel=1, threads=2, edge=True, edge_released=True, need_test=True, official_check=OFFICIAL, runs=runs); kname = "final-ref"
cmd = [sys.executable, str(root / "scripts/push_train.py"), kname, json.dumps(cfg)] + ([] if a.push else ["--no-push"])
subprocess.run(cmd, check=True)
print("generated kernels/%s (push=%s):" % (kname, a.push), json.dumps({k: v for k, v in cfg.items() if k != "runs"}), [r["name"] for r in runs])
