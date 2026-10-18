# STARCOP evaluation: corrected protocol, scorer and reproduced baselines (organizer baseline, self-evaluated)

Scope: the AVIRIS-NG STARCOP test split (342 tiles, Zenodo 10.5281/zenodo.7863343, CC BY-NC 4.0), the paper's Table 2 metrics (Sci. Rep. 13:19999) and the public code (spaceml-org/STARCOP, commit c4789268).
Everything below was measured by `bench-gas-baseline` (private Kaggle kernel) on the released checkpoints (`isp-uv-es/starcop`, CC BY-NC 4.0); numbers and bootstrap intervals are in `baseline_reproduction.md`.

## What does not match between the paper text, the public code and the released checkpoints
Code locations are `file:line` in spaceml-org/STARCOP at commit c4789268a3fa0395f92357429052f6f5fc748acb (checked 2026-10-18).
1. Tile rule for the false-positive rate: the paper says a tile is positive above 10 active pixels; the public code uses `10*H*W/64^2` = 640 px at 512x512 (`starcop/models/model_module.py:211`). The published FPR (43.7) matches the 10-px rule (released run: 40.6), not the code (12.6).
2. Strong/weak split: the paper and the Zenodo description split by emission rate (>= 1000 kg/h); `run_validation` splits by label pixel count (> 1000 px; `starcop/validation.py:159`). The two groupings move 40 of the 166 plume tiles between groups and 3 tiles disagree on plume vs no plume.
3. One sentence of the paper's dataset section swaps strong and weak; the Metrics section is correct.
4. Learning rate: paper text Adam 1e-3, released config 1e-4 (`starcop/config.yaml:43`; the checkpoint's optimizer state agrees with 1e-4).
5. Leakage: in the public training code the validation set is the test set (`starcop/data/datamodule.py:245`, `self.val_dataset = self.test_dataset`) and the ReduceLROnPlateau schedule monitors that loss (`starcop/models/model_module.py:178-185`; the checkpoint callback in `scripts/train.py:89-96` monitors it too). The released model is the last-epoch weights, so the leak is limited to the learning-rate schedule, but a corrected protocol must not do it.
6. `run_validation` crashes on current torchmetrics (`ConfusionMatrix(num_classes=2)` without `task`, `starcop/validation.py:45` and `:67`) and does not create its output directory.

## Corrected protocol used here (fixed before any model of ours was trained: `protocol/PREREGISTRATION.md`)
- Model selection on validation only: whole acquisition dates (2019-10-22/23) held out from train (no flight line shared with the rest of train or with test; `protocol/val_split.json`). Final models are retrained on all train, last-step weights, and the test split is scored once per model.
- Headline metrics (the variants that reproduce the published numbers): F1 strong / weak by emission rate (>= / < 1000 kg/h); FPR by tile with the 10-pixel rule; AUPRC as threshold-free average precision. The public-code variants (label-pixel grouping, 640-px rule, 16-threshold trapezoid) are reported next to them, labelled.
- 5 seeds, mean +- sample std, plus a tile-level bootstrap 95 % interval of the seed-mean and paired differences to the released runs on the same tiles. The 342-tile test split resolves only large differences (weak-plume F1 interval of a single model spans about 25-64).
- Scorer: `src/starcop_edge/metrics.py` (numpy only, per-tile summaries, exact bootstrap) and `scripts/score.py` (`--pred-dir` of 512x512 logits, `--test-dir`, `--test-csv data_meta/test.csv`).
- Cross-check against the official code (TEST-LOG #4, kernel `bench-gas-final-ref` v2, 2026-10-07): the official `starcop.validation.run_validation` (commit c4789268, released mag1c+RGB checkpoint loaded into the official `ModelModule`, official `STARCOPDataset`, all 342 test tiles) gives
F1 strong / weak by label pixels 84.228 / 39.023, FPR by tile (640-px rule) 12.571, AUPRC (16-threshold trapezoid) 37.147, identical to our scorer's numbers for the same checkpoint: identical to all printed digits on F1 strong, F1 weak and FPR, and within 4e-7 on AUPRC (trapz16) (full-precision values: results/baseline_report.json vs results/official_check.json).
The headline variants (grouping by emission rate, 10-pixel tile rule, average precision) have no official implementation; they are our scorer's reading of the paper text, reproduce the published numbers (table below) and agree with brute-force computations on random tiles (`tests/test_metrics.py`).
The first attempt (v1) crashed before reading any test tile because `segmentation_models_pytorch` was not installed (TEST-LOG #4a).

## Reproduced baselines (our scorer, one run each, 95 % tile-bootstrap interval)
| model | F1 strong | F1 weak | FPR (>10 px) | AUPRC (AP) |
|---|---|---|---|---|
| released mag1c+RGB (run R3 of 5) | 83.08 [73.1, 88.2] | 42.34 [24.6, 64.3] | 40.57 [33.0, 47.5] | 50.60 [25.5, 78.7] |
| released mag1c-only | 66.74 [48.0, 78.6] | 48.76 [30.9, 66.4] | 36.00 [28.8, 42.8] | 40.91 [23.5, 65.4] |
| published mag1c+RGB (5-run mean +- std) | 81.96 +- 3.71 | 43.42 +- 5.72 | 43.66 +- 7.36 | 51.99 +- 2.76 |
| published mag1c-only (5-run mean +- std) | 74.15 +- 6.10 | 47.57 +- 4.17 | 52.11 +- 10.98 | 49.41 +- 5.49 |
Each released checkpoint is a single run, so it is compared with the paper's 5-run mean +- std, not with the mean itself.
