#!/usr/bin/env python3
"""Wave 3 (Amendment 6): generate the Colab kernels from the GPU wave's definitions (scripts/mk_gpu_wave.py).
usage: mk_colab_wave3.py gate|test|kaggle-gate|kaggle-test

gate: kernels/colab-gate, one run c_d1_e15_val: d1, seed 0, E = 15 (78,675 steps), trained on train minus the validation dates,
      final validation only. No run reads the test split.
test: kernels/colab-test-s0 .. s4, one run each: d1, seed s, E = 15, all train tiles, test split scored once (generated only after
      Amendment 6a records a passed gate).
Every kernel runs on one T4 with one worker, through the same driver (kernels/train_gpu/main.py) and src as the GPU wave; only the
CFG line differs. No control run and no ONNX edge stage in this wave."""
import json, pathlib, subprocess, sys
from typing import Any
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mk_gpu_wave as W

ROOT = pathlib.Path(__file__).resolve().parent.parent
E = 15
STEPS = E * W.EPOCH_STEPS                       # 78,675
BASE = dict(gpu=True, machine_shape="ColabT4", max_gpus=1, billing_b=1, parallel=1, threads=2, warm_timeout_s=900)
DEADLINE_S = 5400                               # soft deadline per kernel job (one run each): 1.5 h of T4 wall time


# Kaggle (Amendment 6 note 6): the 2 x T4 shape the GPU wave used, billed per GPU held (b = 2); caps are billed seconds and the
# wall caps are billed / b. The gate's single run uses one worker; the test stage runs the five seeds on two workers, as G2 did.
KAGGLE = dict(gpu=True, machine_shape="NvidiaTeslaT4", max_gpus=2, billing_b=2, threads=2)


def gate_run():
    return dict(name="c_d1_e15_val", seed=0, val_dates=W.VAL, eval_every=STEPS, steps=STEPS, **W.COMMON, **W.CANDS["d1"])


def test_run(s):
    return dict(name=f"c_d1_e15_s{s}", seed=s, test_eval=True, steps=STEPS, **W.COMMON, **W.CANDS["d1"])


def configs(stage: str) -> list[tuple[str, dict[str, Any]]]:
    """[(kernel name, CFG)] for the stage; pure (writes nothing)."""
    if stage == "gate":
        return [("colab-gate", dict(BASE, deadline_s=DEADLINE_S, runs=[gate_run()]))]
    if stage == "test":
        return [(f"colab-test-s{s}", dict(BASE, deadline_s=DEADLINE_S, runs=[test_run(s)])) for s in range(5)]
    if stage == "kaggle-gate":
        return [("gpu-gate3", dict(KAGGLE, billed_cap_s=6000, wall_cap_s=3000, parallel=1, deadline_s=2700, runs=[gate_run()]))]
    if stage == "kaggle-test":
        return [("gpu-test3", dict(KAGGLE, billed_cap_s=14400, wall_cap_s=7200, parallel=2, deadline_s=6600, runs=[test_run(s) for s in range(5)]))]
    raise SystemExit(f"unknown stage {stage!r} (gate|test|kaggle-gate|kaggle-test)")


if __name__ == "__main__":
    for name, cfg in configs(sys.argv[1]):
        subprocess.run([sys.executable, str(ROOT / "scripts/push_train.py"), name, json.dumps(cfg), "--no-push"], check=True)
        print("generated kernels/%s (not pushed):" % name, [(r["name"], r["seed"], r["steps"], bool(r.get("test_eval"))) for r in cfg["runs"]])
