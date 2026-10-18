#!/usr/bin/env python3
"""Create + push one CPU training kernel: push_train.py <kernel-name> '<json {"runs":[{...},...]}>' [--no-push]. Slug: bench-gas-<name>.
CFG keys read here: gpu (enable_gpu), machine_shape (e.g. NvidiaTeslaT4; Amendment 4), driver ("train" = kernels/train/main.py, the CPU driver; "train_gpu" = kernels/train_gpu/main.py;
default: train_gpu when gpu is true, else train), edge (internet), test_eval/need_test (STARCOP mirror + assets datasets).
GPU kernels (gpu true) are NEVER pushed from here: the GPU day runbook in STATUS.md pushes them by hand with `kaggle kernels push -t <seconds>` after the quota/lock gates."""
import json, os, pathlib, shutil, subprocess, sys
root = pathlib.Path(__file__).resolve().parent.parent
name, cfg = sys.argv[1], json.loads(sys.argv[2])
if cfg.get("gpu") and "--no-push" not in sys.argv:
    raise SystemExit("refusing to push a GPU kernel from push_train.py: follow the GPU day runbook (STATUS.md): quota gate, locks/kaggle-gpu, kaggle kernels push -t <cap seconds>")
kd = root / "kernels" / name; kd.mkdir(parents=True, exist_ok=True)
driver = cfg.get("driver") or ("train_gpu" if cfg.get("gpu") else "train")
shutil.copy(root / "kernels" / driver / "main.py", kd / "main.py")
test = any(r.get("test_eval") for r in cfg["runs"]) or bool(cfg.get("need_test"))
CACHE_KERNEL = "sankalpsthakur/bench-gas-prep-cache"  # train cache provider (v2 of bench-gas-prep-train ended in ERROR; prep-cache patches its output). Dev and final MUST name the same one.
meta = {"id": f"sankalpsthakur/bench-gas-{name}", "title": f"bench-gas-{name}", "code_file": "script.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": ("true" if cfg.get("gpu") else "false"), "enable_tpu": "false", "enable_internet": ("true" if cfg.get("edge") else "false"),
        "machine_shape": cfg.get("machine_shape", ""), "dataset_sources": (["chrisphamda/starcop", "sankalpsthakur/bench-gas-assets"] if test else []), "competition_sources": [],
        "kernel_sources": [CACHE_KERNEL], "model_sources": []}
(kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
subprocess.run([sys.executable, str(root / "scripts/make_kernel.py"), str(kd), "--cfg", json.dumps(cfg)], check=True)
(kd / "cfg.json").write_text(json.dumps(cfg, indent=1))
if "--no-push" not in sys.argv:
    subprocess.run([os.environ.get("KAGGLE_BIN", str(pathlib.Path.home() / ".local/bin/kaggle")), "kernels", "push", "-p", str(kd)], check=True)
