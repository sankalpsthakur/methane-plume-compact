# results/ index (gas / STARCOP)

| file | what | source |
|---|---|---|
| `baseline_reproduction.md`, `baseline_report.json` | reproduction of the published evaluation with the released checkpoints (TEST-LOG #2-#3), inconsistencies found | kernel `bench-gas-baseline` v2-v3 |
| `CORRECTED-PROTOCOL.md` | corrected protocol, scorer, reproduced baselines (organizer baseline, self-evaluated) | derived from the above and `protocol/PREREGISTRATION.md` |
| `prep_cache/` | train-cache report (3,425 tiles, CRC32 verified, SHA-256), CPU throughput probe, edge self-test, smoke run on TRAIN tiles, the 10 upstream `has_plume`/label mismatches | kernel `bench-gas-prep-cache` v1 |
| `final_*.json`, `final_TABLE.md` | 5-seed aggregation against the released references (written by `scripts/finalize.py`) | kernel `bench-gas-final` v1 |
| `official_check.json`, `official_eval/` | official `run_validation` on the released mag1c+RGB checkpoint vs our scorer (TEST-LOG #4) | kernel `bench-gas-final-ref` v2 |
| `testlog_rows.md` | TEST-LOG rows generated from the final outputs (paste into TEST-LOG.md) | `scripts/testlog_rows.py` |
| `latency_m4.json` | M4 1-thread batch-1 latency (only if the machine was quiet) | `scripts/latency_m4.py` |
| `gpu_final_TABLE.md`, `gpu_final_*_vs_released_*.json` | GPU-wave 5-seed aggregation (d1, E = 10, Amendments 4a/4b): PyTorch fp32, ONNX fp32, ONNX int8 against both released references; `final_TABLE.md` has the CPU wave, the GPU wave and the side-by-side summary | kernel `bench-gas-gpu-compact` v1, `scripts/finalize.py` |
| `gpu_control_vs_released_*.json` | paper-recipe-like control (released architecture, 5 epochs, 1 run) against the released runs | kernel `bench-gas-gpu-compact` v1 (last queued job), `scripts/aggregate.py` |
| `gpu_runs/` | small per-run summaries of the GPU wave: `gpuprobe.json` (G0), `RUN_STATUS.json`, `val_metrics.json` (G1), `test_metrics.json`, `edge_report.json`, `log.jsonl`, kernel logs | fetched with the watcher and the `.npz`/ONNX patterns of the runbook |
| `gpu_ledger.csv` | billed GPU hours per kernel (GPUs x wall, next to the raw quota delta) | `scripts/gpu_gate.sh record` |

Protocol files: `protocol/PREREGISTRATION.md` (amendments 1, 2, 2b, 2c, 3 are dated before the runs they govern), `protocol/selection.json` (dev-wave selection), `protocol/GPU-PLAN.md` and Amendments 4, 4a, 4b, 5 (GPU wave, run 2026-10-17), `protocol/gpu_budget.json` (plan), `protocol/selection_gpu.json` (G1 selection).
Raw Kaggle outputs live in `kernel_outputs/` (gitignored); nothing from STARCOP itself is on this Mac.
