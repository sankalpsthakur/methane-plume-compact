# methane-plume-compact

This repository holds the code, protocol, test log and results behind a compact methane-plume segmentation model for STARCOP. The model is a 123,649-parameter U-Net for AVIRIS-NG airborne hyperspectral tiles, 53.6 times smaller than the released STARCOP model. The weights and the model card are at https://huggingface.co/sankalpsthakur/methane-plume-compact (CC BY-NC 4.0).

**Result:**
- **Setting:** STARCOP AVIRIS test split (342 tiles), paper-text protocol, 5 seeds, self-evaluated on 2026-10-17.
- **Weak-plume F1:** 54.47 ± 2.08, against 47.57 ± 4.17 for the paper's mag1c-only row.
- **Comparison:** the paired tile-bootstrap interval against the released mag1c-only model includes zero (+5.71 [−2.5, +19.2]). Under the pre-registered rule, the result is therefore **not distinguishable from the published row**.
- **Weakness:** its tile-level false-alarm rate is worse than both released models.

This is a self-evaluation, not an official ranking.

## Results

The protocol is the paper-text one: strong and weak plumes are split by emission rate, a tile counts as positive above 10 predicted pixels, and AUPRC is average precision. Values are mean ± standard deviation over 5 seeds, with the 95 % tile-bootstrap interval of the mean in brackets. The tables are written by `scripts/finalize.py` into `results/gpu_final_TABLE.md` (GPU run) and `results/final_TABLE.md` (CPU run).

| model | training | F1 strong | F1 weak | tile FPR (> 10 px) | AUPRC |
|---|---|---|---|---|---|
| **this model, fp32** (GPU run, 5 seeds) | 52,450 × 32 = 1.68 M windows | 82.78 ± 2.21 [70.8, 89.3] | **54.47 ± 2.08** [30.1, 78.2] | 50.51 ± 1.65 [43.7, 57.1] | 55.94 ± 8.08 [39.1, 74.9] |
| this model, ONNX int8 (GPU run, 5 seeds) | same weights | 82.85 ± 1.97 | 55.46 ± 2.57 | 50.06 ± 2.51 | 55.50 ± 7.50 |
| same architecture, CPU run (5 seeds, 2026-10-07) | 8,000 × 16 = 0.128 M windows | 74.73 ± 1.34 | 42.70 ± 2.30 | 58.40 ± 9.73 | 51.29 ± 8.31 |
| control: released architecture, paper-like recipe, 5 of 15 epochs, 1 run | 22,215 × 32 | 71.83 | 54.42 | 46.29 | 56.18 |
| released mag1c-only, 1 run, same scorer | paper recipe | 66.74 | 48.76 | 36.00 | 40.91 |
| published mag1c-only (paper, Table 2, 5 runs) | paper recipe | 74.15 ± 6.10 | 47.57 ± 4.17 | 52.11 ± 10.98 | 49.41 ± 5.49 |
| released mag1c+RGB (run R3), 1 run, same scorer | paper recipe | 83.08 | 42.34 | 40.57 | 50.60 |
| published mag1c+RGB (paper, Table 2, 5 runs) | paper recipe | 81.96 ± 3.71 | 43.42 ± 5.72 | 43.66 ± 7.36 | 51.99 ± 2.76 |

**Paired differences** (this model's seed mean minus the released run, same 342 tiles, 95 % tile bootstrap):

| reference | F1 strong | F1 weak | tile FPR | AUPRC |
|---|---|---|---|---|
| released mag1c-only (the comparator of the claim rule) | +16.04 [+6.6, +25.6] | +5.71 [−2.5, +19.2] | +14.51 [+8.3, +20.6] | +15.03 [+2.3, +24.1] |
| released mag1c+RGB (R3) | −0.30 [−5.7, +4.0] | +12.13 [+3.8, +25.3] | +9.94 [+4.5, +15.2] | +5.34 [−7.0, +16.8] |

**Official-code variants** of the same predictions use a different split, tile rule and AUPRC: plumes split by label size, tiles positive above 640 pixels, and AUPRC over the official code's 16 thresholds. They give:
- this model: F1 strong 83.91 ± 2.15, F1 weak 49.88 ± 2.24, tile FPR 11.31 ± 0.63, AUPRC 56.16 ± 8.24;
- released mag1c-only: 66.74, 47.51, 9.14, 39.00;
- released R3: 84.23, 39.02, 12.57, 37.15.

**What the result does and does not show:**
- **False alarms:** the weak-F1 gain comes with more false-alarm tiles than either released model. The FPR is level with the paper's mag1c-only mean.
- **Comparator noise:** the released mag1c-only comparator is a single run. The tile bootstrap does not capture its seed noise.
- **Control:** the control row carries no claim. Trained with this pipeline at this budget, the released architecture reaches the same weak F1 as the small model (54.42 against 54.47). That suggests the gain over the released runs comes from the training pipeline rather than from the small architecture, but it rests on one control run.
- **Two runs:** the same architecture was evaluated on the same test split twice. The GPU run was planned, with the same target and rule, before the CPU run's test evaluation, and it changed only the training budget.

## Protocol

- **Benchmark and target:** the STARCOP AVIRIS-NG test split (342 tiles). The target is the best published weak-plume F1, the paper's mag1c-only row: 47.57 ± 4.17.
- **Claim rule, fixed before any GPU run:** "best published result on STARCOP test" is claimed only if both conditions hold:
  - the 5-seed mean weak F1 exceeds 47.57;
  - the paired tile-bootstrap 95 % interval against the released mag1c-only run excludes zero.

  Otherwise the result is reported as "not distinguishable".
- **Selection on validation only:** whole acquisition dates 2019-10-22/23 are held out from training (`protocol/val_split.json`). The configuration is chosen by a rule fixed in advance (`scripts/select_config.py`), then frozen and retrained on all 3,425 training tiles.
- **Test reads:** each model is scored on the test split once, and every read is logged in `TEST-LOG.md`.
- **Pre-registration:** `protocol/PREREGISTRATION.md` and its amendments.
  - Amendments 1–3 govern the CPU run.
  - Amendments 4, 4a, 4b and 5, together with `protocol/GPU-PLAN.md`, govern the GPU run.
- **Scorer:** `src/starcop_edge/metrics.py`, numpy only.
  - It works from per-tile summaries, so the tile bootstrap is exact.
  - It reports the paper-text variants as the headline and the official-code variants next to them.
  - On the released mag1c+RGB checkpoint it matches the official `run_validation` exactly on three metrics and within 4e-7 on AUPRC (`results/official_check.json`, TEST-LOG #4).

The paper text, the public code and the released checkpoints disagree in six places: the tile rule, the strong/weak split, one swapped sentence, the learning rate, the use of the test set as validation in the public training code, and an incompatibility of `run_validation` with current torchmetrics. `results/CORRECTED-PROTOCOL.md` lists each one with the file and line at the pinned commit. The scorer reports both readings. These points, plus how `run_validation` computes AUPRC, were reported to the maintainers on 2026-10-18 as [spaceml-org/STARCOP#9](https://github.com/spaceml-org/STARCOP/issues/9).

## Verify the tooling (no data needed)

```bash
pip install numpy pandas pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests
python tests/test_driver_deadline.py
bash tests/test_gpu_gate.sh && bash tests/test_gpu_launch.sh && bash tests/test_g2_gate.sh
sha256sum --check --quiet MANIFEST.txt     # on macOS: shasum -a 256 --check --quiet MANIFEST.txt
```

The tests cover:
- the scorer against brute-force F1, FPR and AP;
- the sampler's augmentation alignment;
- the test-split loader and the train-cache logic on synthetic tiles;
- the Zenodo retry policy;
- the selection rule and the GPU planner;
- the GPU push gate and launcher, against a fake `kaggle`;
- the soft deadline of the GPU driver, with toy processes.

CI runs the same commands on Python 3.12 and 3.13.

## Reproduce

**Score your own predictions** with the same scorer. You need the official test split from Zenodo (`STARCOP_test.zip`) and one `<tile_id>.npy` of 512 × 512 logits per test tile:

```bash
python scripts/score.py --pred-dir PRED_DIR --test-dir STARCOP_test --test-csv data_meta/test.csv [--kind logit|prob] [--out res.json]
```

**Re-run the training and evaluation.** The runs used private Kaggle kernels, and their scripts are in `kernels/`; `kernels/VERSIONS.md` lists every version and outcome. The kernels read three sources, which you need to recreate under your own account:

1. The public Kaggle mirror `chrisphamda/starcop`. Every test file is CRC32-checked against the Zenodo archives (`manifests/`). The mirror lacks `STARCOP_train_remaining_part2.zip`, which the prep kernels fetch from Zenodo. See `FEASIBILITY.md`.
2. A Kaggle dataset holding:
   - `test.csv`, `train.csv` and `train_easy.csv` (Zenodo; copies in `data_meta/`);
   - `zenodo_STARCOP_*.tsv` (copies in `manifests/`);
   - the released mag1c+RGB checkpoint and config from https://huggingface.co/isp-uv-es/starcop (`models/hyperstarcop_mag1c_rgb/final_checkpoint_model.ckpt` and `config.yaml`), renamed `hyperstarcop_mag1c_rgb_final_checkpoint_model.ckpt` and `hyperstarcop_mag1c_rgb_config.yaml`.

   The mag1c-only checkpoint is downloaded from the same Hugging Face repository by `kernels/baseline`.
3. The train cache, built by `kernels/prep-cache` in rebuild mode, which reads the mirror and fetches the missing archive from Zenodo. In patch mode it can instead repair the output of `kernels/prep-train`. Run it first; it needs internet.

In every `kernel-metadata.json`, point `id`, `dataset_sources` and `kernel_sources` at your copies. The kernels find their inputs under `/kaggle/input` by marker files (`src/starcop_edge/kio.py`), so no other path needs changing. The generators `scripts/mk_gpu_wave.py` and `scripts/push_train.py` and the fetch helper `scripts/after_final.sh` also name the author's account (`sankalpsthakur`); change them as well if you regenerate kernels or fetch outputs.

Push the kernels in this order:
- **CPU run:** `prep-cache` → `dev` → `scripts/select_config.py` → `final` and `final-ref`.
- **GPU run:** `gpu-probe` → `scripts/mk_gpu_wave.py plan` → `gpu-wide` → `scripts/select_config.py` → `gpu-compact`.

The GPU kernels pin `NvidiaTeslaT4`; Kaggle serves 2 × T4. `gpu-compact` needs internet for pip.

Build the tables from the fetched outputs with `scripts/finalize.py` and `scripts/aggregate.py`. `scripts/after_final.sh` shows the commands. Measure latency with `scripts/latency_m4.py` (1 thread, batch 1).

Training is not bit-reproducible across hardware.

## Repository layout

| path | what |
|---|---|
| `src/starcop_edge/` | data loading and the train cache, the compact U-Net and a replica of the released architecture, training, ONNX export and int8 quantisation, the scorer, the official-code cross-check |
| `kernels/` | the Kaggle kernel scripts as run (`script.py` is generated from `main.py` and `src/`), with their metadata and `VERSIONS.md` |
| `protocol/` | pre-registration and amendments, the GPU plan, the validation split, selection records, the compute budget |
| `results/` | tables, paired comparisons, the official-code cross-check, the corrected protocol, small run logs (index: `results/README.md`) |
| `TEST-LOG.md` | every read of the test split, in order |
| `scripts/` | scoring, aggregation, kernel generators, selection, latency, and the helpers that pushed and watched the kernels |
| `tests/` | tests of the tooling |
| `data_meta/`, `manifests/` | STARCOP metadata and Zenodo file lists (CC BY-NC 4.0, see `NOTICE`) |
| `FEASIBILITY.md` | the data route: Zenodo, the Kaggle mirror and its missing archive |

## What differs from the private working repository

- **Fresh history.** The pre-registration and its amendments were committed in the author's private working repository before the runs they govern. The commit hashes cited in `protocol/` and `kernels/VERSIONS.md` (for example `93727d1` and `24267cc`) refer to that repository. This repository is a fresh export, so its git history does not prove that order.
- **Not included:**
  - the two working notes `STATUS.md` and `SUBMISSION.md`, which are logs of a Kaggle account shared with other projects; their results are in this README and in `results/`;
  - the Hugging Face model card, which is published separately;
  - the program-level files the logs refer to: `PLAN.md` and `RESOURCES.md` (plan and ledgers of the whole program), `bin/guard.sh` (the local resource guard) and `../locks/` (the lock directories).
- **Redactions:** names of unrelated kernels on the same Kaggle account are replaced by `[name redacted]` or `[names redacted]`, three times in `protocol/PREREGISTRATION.md` and twice in `kernels/VERSIONS.md`. Nothing else in those files differs.
- **Edited before export** (the same change was made in the private repository on 2026-10-18):
  - six helper scripts (`scripts/after_final.sh`, `finish_wave.sh`, `run_wave.sh`, `slot.sh`, `watch_kernel.sh`, `push_train.py`) had the author's absolute paths replaced by relative defaults;
  - `results/CORRECTED-PROTOCOL.md` gained `file:line` citations;
  - `tests/test_zen_retry.py`: the upper time bound of one check rose from 9 s to 14 s, the most the code's cool-down jitter allows for two 429 responses. CI on Python 3.12 took 9.03 s. The retry code itself is unchanged.
- **Terms used in the logs:**
  - *owner* is the author;
  - *lead* or *coordinator* is the coordinating session of the AI-assisted workflow below, and a *builder* is a session working on this track;
  - `locks/` holds the lock directories that stopped the program's tracks from using the shared Kaggle account's GPU sessions at the same time.

## How this was made

The work was run with AI coding agents (Claude Code) under the author's direction. Agents wrote the code, built and monitored the Kaggle kernels, kept the pre-registration and the test log, analysed the results and drafted the documents.

## Licences

The code is licensed Apache-2.0 (`LICENSE`). The STARCOP-derived files listed in `NOTICE` are CC BY-NC 4.0 and for non-commercial use only. `NOTICE` also covers:
- the STARCOP code (GPL-3.0), which is fetched at run time and not included;
- the architecture replica, with the notices of segmentation_models_pytorch (MIT) and torchvision (BSD-3-Clause) in `LICENSES/`.

The weights are on Hugging Face under CC BY-NC 4.0. This is independent work, not affiliated with or endorsed by the STARCOP authors.

## Citation

Please cite the STARCOP paper and dataset:

```bibtex
@article{ruzicka2023starcop,
  title   = {Semantic segmentation of methane plumes with hyperspectral machine learning models},
  author  = {R{\r{u}}{\v{z}}i{\v{c}}ka, V{\'\i}t and Mateo-Garcia, Gonzalo and G{\'o}mez-Chova, Luis and Vaughan, Anna and Guanter, Luis and Markham, Andrew},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {19999},
  year    = {2023},
  doi     = {10.1038/s41598-023-44918-6}
}

@misc{starcop2023dataset,
  title     = {{STARCOP} dataset: Semantic Segmentation of Methane Plumes with Hyperspectral Machine Learning Models},
  author    = {R{\r{u}}{\v{z}}i{\v{c}}ka, V{\'\i}t and Mateo-Garcia, Gonzalo and G{\'o}mez-Chova, Luis and Vaughan, Anna and Guanter, Luis and Markham, Andrew},
  publisher = {Zenodo},
  version   = {v1.0},
  year      = {2023},
  doi       = {10.5281/zenodo.7863343}
}
```
