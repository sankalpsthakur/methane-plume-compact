# Pre-registration (written before any model was trained; see git history for the timestamp)

Track: gas / STARCOP. Author: build-gas. Date: 2026-10-07.

## What is compared
Published reference (Sci. Rep. 13:19999, Table 2, AVIRIS test split, 5-run mean +- std, mag1c + RGB):
F1 strong 81.96 +- 3.71, F1 weak 43.42 +- 5.72, FPR by tile 43.66 +- 7.36, AUPRC 51.99 +- 2.76.
Same table, mag1c-only variant: F1 weak 47.57 +- 4.17 (the best published weak-plume F1 in that table).
Our results are always reported against both rows.

## Split and selection (frozen now)
- Fit/validation split: hold out whole dates 2019-10-22 and 2019-10-23 of train.csv (`protocol/val_split.json`, `val_ids.csv`).
  No flight line is shared between the held-out dates and the rest of train, and none with the test split.
- The official test split (test.csv, 342 tiles) is never read during development. Dev kernels receive only the train cache split.
- Model-selection score on validation: C = mean(F1_strong, F1_weak, 100 - FPR_tile, AUPRC), each computed with the scorer in
  `src/starcop_edge/metrics.py` (strong/weak by emission rate qplume >= 1000 kg/h, FPR_tile with the official 640 px rule,
  AUPRC = threshold-free average precision).
- Rule: the chosen configuration is the SMALLEST (fewest parameters) whose validation C is within 1.0 point of the best C among the
  dev runs. Ties go to fewer parameters. Dev runs use 1 seed.
- After selection the configuration is frozen (architecture, widths, steps, lr, pos_weight, augmentation) and retrained on ALL of
  train.csv with seeds 0..4, last-step weights (no checkpoint selection, no early stopping), then scored on test.csv once per seed.
- Every test evaluation is logged in TEST-LOG.md, including failed or discarded ones.

## Reported numbers
5-seed mean +- sample std (ddof=1) for F1 strong, F1 weak, FPR by tile, AUPRC, plus a 95% tile-level bootstrap interval of the
5-seed mean. A margin smaller than the published std or the bootstrap interval is NOT claimed as a win.
Parameter count, fp32 and int8 size on disk, and M4 1-thread batch-1 latency per 512x512 tile are reported for every model that is scored.

---
## Amendment 1 (2026-10-07 ~04:20 UTC). Written after the baseline reproduction, BEFORE any model of ours was trained or scored on validation or test.

1. Wording fix. Dev kernels mount only the output of `bench-gas-prep-train`. The test cache is written by a separate kernel,
   `bench-gas-prep-test`, and is mounted only by the final-seed kernels. `train.py` raises if `--test_eval` is combined with `--val_dates`.
   (The first sentence above, "Dev kernels receive only the train cache split", is now literally true.)
2. Scorer variants chosen from the reproduction of the released checkpoint (kernel `bench-gas-baseline`, test split, released
   `hyperstarcop_mag1c_rgb`, run "R3" of the paper's five). The released model must land inside the published mean +- 1 std:

   | metric | variant | released R3 | paper Table 2 mean +- std | inside? |
   |---|---|---|---|---|
   | F1 strong | by emission rate (>=1000 kg/h) | 83.08 | 81.96 +- 3.71 | yes |
   | F1 strong | by label pixels (>1000 px, official code) | 84.23 | same | yes |
   | F1 weak | by emission rate (<1000 kg/h) | 42.34 | 43.42 +- 5.72 | yes |
   | F1 weak | by label pixels (<=1000 px, official code) | 39.02 | same | yes |
   | FPR by tile | tile positive if >10 predicted px (paper text) | 40.57 | 43.66 +- 7.36 | yes |
   | FPR by tile | tile positive if >640 px (public code, 10*H*W/64^2) | 12.57 | same | NO |
   | AUPRC | threshold-free average precision (pooled pixels) | 50.60 | 51.99 +- 2.76 | yes |
   | AUPRC | trapezoid over the 16 thresholds of the official validation code | 37.15 | same | NO |

   Therefore the HEADLINE protocol is: strong/weak by emission rate, tile positive at >10 predicted pixels, AUPRC = average precision.
   The variants that did not reproduce are still reported (labelled "official-code variant").
3. Dev candidates are fixed here (no additions after validation numbers are seen). All: 8,000 steps, batch 32, AdamW (peak lr 2e-3,
   weight decay 0), 200 warmup steps then cosine to 2%, jitter 32 px, D4 augmentation, one seed (0), trained on train.csv minus the
   validation dates:
   | id | stem_stride | widths | pos_weight |
   |---|---|---|---|
   | d1 | 2 | 4,8,16,32,64 | 1 |
   | d2 | 2 | 8,16,32,64,128 | 1 |
   | d3 | 2 | 16,32,64,128,256 | 1 |
   | d4 | 2 | 8,16,32,64,128 | 0.5 |
   | d5 | 1 | 4,8,16,32,64 | 1 |
4. Tolerance in the selection rule changed from 1.0 to 2.0 composite points because dev runs have a single seed (composite noise is of
   that order). Rule otherwise unchanged: smallest parameter count within 2.0 points of the best composite.
5. Steps are part of the frozen configuration: the final 5-seed runs use the same 8,000 steps as the dev runs.

---
## Amendment 2 (2026-10-07 ~04:15 UTC). Still before any dev run; written after measuring CPU training speed in the feasibility kernel.

Why: the account's only usable compute is CPU kernels (2 physical cores each, one kernel at a time per program rule). The schedule of Amendment 1
(8,000 steps at batch 32) is replaced by a cheaper, frozen one, and the candidate list is cut to what the CPU budget can both develop and retrain 5 times.

1. Frozen schedule for dev AND final runs: 8,000 steps, batch 16 (not 32), AdamW peak lr 2e-3, wd 0, 200 warmup steps, cosine to 2%, jitter 32 px, D4 augmentation.
   Samples seen: 128k crops, about 0.8 of one official epoch; the paper trained 15 epochs. All our numbers are therefore from a ~5% training budget.
2. Dev candidates (single seed 0, trained on train.csv minus validation dates). d3 (16..256 widths) and d5 (stem stride 1) of Amendment 1 are removed
   (too slow on CPU to also retrain 5 times; staged for the GPU after 2026-10-10). Added: d6, hard-negative sampling.
   | id | stem_stride | widths | pos_weight | hardneg |
   |---|---|---|---|---|
   | d1 | 2 | 4,8,16,32,64 | 1 | 0 |
   | d2 | 2 | 8,16,32,64,128 | 1 | 0 |
   | d4 | 2 | 8,16,32,64,128 | 0.5 | 0 |
   | d6 | 2 | 8,16,32,64,128 | 1 | 0.5 |
   hardneg 0.5: half of the non-plume window draws (the 50% not drawn from plume windows) come from "hard" negative windows, i.e. windows without plume
   label that contain >= 50 pixels with mag1c >= 200 ppm.m (normalised 0.114); the rest from the remaining negative windows. This targets false-positive tiles.
3. Selection rule unchanged from Amendment 1 (smallest parameter count within 2.0 points of the best validation composite C).
   Any dev run that has not finished when the dev kernel ends is reported as not completed and excluded.
4. Edge-ready variant, fixed now: BN folded into convs, ONNX fp32 (opset 17), and static int8 (QDQ, int8 activations and weights, per-channel weights, MinMax
   calibration on 48 TRAIN tiles: 24 plume + 24 no-plume at 512x512, seed 0). No calibration or quantisation choices are tuned on test; whatever the int8
   model scores is reported.
5. Final wave (frozen after selection): seeds 0..4, trained on ALL of train.csv, last-step weights; test tiles are read in the same kernel from the
   CRC-verified mirror; per-seed fp32 (PyTorch) test summaries plus ONNX fp32 and int8 test summaries are written; TEST-LOG lists every one of them.

---
## Amendment 2b (2026-10-07 ~10:10 UTC). Still before any dev run: no validation number of any model of ours exists yet.

1. Test cache retired. `bench-gas-prep-test` (v1 crashed on its first config line, before touching data) is never re-pushed; no test cache is built. The final kernels read the
   342 test tiles straight from the CRC-verified mirror (`starcop_edge.data.load_test_from_mirror`, used by `train.py --test_eval` and by `edge_eval`). This supersedes the wording of
   Amendment 1 item 1 ("the test cache is written by `bench-gas-prep-test`"); the intent (dev kernels cannot see test data) is unchanged and still enforced in `kernels/train/main.py`.
2. Train cache provenance. `bench-gas-prep-train` v2 ended in ERROR after caching 3,413 of 3,425 train tiles (12 tiles lost to Zenodo HTTP 429 after the retry budget). Every stored tile in
   that output had passed the CRC32/size check against the official archive directories. Kernel `bench-gas-prep-cache` mounts that output, copies it, re-fetches only the 12 missing tiles
   (backoff that honours Retry-After), proves the other 3,413 tiles are bit-identical to the source, and writes `prep_report.json` with SHA-256 of the cache. Dev and final kernels mount
   `bench-gas-prep-cache` only. If the v2 output cannot be mounted, the same kernel rebuilds the entire cache from the mirror + Zenodo.
3. Tie-break of the selection rule (Amendment 1 item 4 / Amendment 2 item 3), stated now because d2, d4 and d6 have identical parameter counts: let B be the best validation composite C among
   completed dev runs; the band is every completed run with C >= B - 2.0; choose the run with the fewest parameters in the band; if several runs share that parameter count, choose the one with the
   highest C. Nothing else (no test number, no per-metric preference, no second look) enters the decision.
4. Smoke runs. `bench-gas-prep-cache` also runs 100-step `train.py` smoke runs (configs d2 and d6, 12 validation tiles, TRAIN data only) and one edge-scoring smoke on 4 TRAIN tiles, only to prove the
   code path runs on Kaggle. They are not dev runs, their numbers are never used for selection, and they are not reported as results.
5. `train.py` gained `--log_every` (logging interval only; default 200 = previous behaviour). No optimisation, sampling or evaluation code changed.

---
## Amendment 2c (2026-10-07 ~10:16 UTC). Still before any dev run.

The composite of the original text names "FPR_tile with the official 640 px rule", written before the baseline reproduction showed that this rule does not reproduce the published FPR
(Amendment 1 item 2 made the 10-pixel rule the headline). To remove the ambiguity before any validation number exists: the selection composite is
**C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap)**, i.e. the headline variants of Amendment 1. C computed with FPR_tile_official640 is logged next to it but never used.
The rule of Amendments 1, 2 and 2b is applied by `scripts/select_config.py` (deterministic; its output `protocol/selection.json` is committed together with Amendment 3).

**Erratum to Amendment 2b item 2 (10:16 UTC).** Kaggle refused the errored `bench-gas-prep-train` as a kernel source ("not valid kernel sources"), so the patch route is unavailable and
`bench-gas-prep-cache` v1 runs in its fallback, REBUILD mode (whole cache from the mirror + Zenodo, CRC32-verified per file, with the 429-aware fetch). The cache content is the same by construction
(the same files, the same preprocessing, the same code path for `process`); its SHA-256 is recorded in `prep_report.json`.

---
## Amendment 3 (2026-10-07 12:29 UTC). Written after the dev wave and BEFORE any test evaluation of a model of ours.

1. Dev wave: kernel `bench-gas-dev` v1 (pushed 10:55 UTC, COMPLETE before 12:26 UTC; 5,079 s inside the kernel; d1 3,360 s, d4 4,980 s, d2 4,990 s, d6 5,040 s; all four runs rc 0), validation dates 2019-10-22/23 only, no test access,
   4 processes x 1 thread, mounted cache `bench-gas-prep-cache` v1 (SHA-256 X 5e6f1a9aab35992e..., Y d55da93f701c2372..., W 0c553912d906f27e...), kernel script generated from repository commit 24267cc, SHA-256 of src/starcop_edge/*.py 5ac8a9bbb1969aa1...
   (unchanged since the dev kernel was generated: `git diff 24267cc HEAD -- src` is empty). Validation results at step 8,000 (last-step weights, one seed; C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap); last column logged only):

   | run | parameters | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC | C | C with 640-px FPR |
   |---|---|---|---|---|---|---|---|
   | d1 | 123,649 | 70.24 | 72.87 | 55.16 | 56.08 | 61.01 | 70.53 |
   | d2 | 492,197 | 70.59 | 73.38 | 52.31 | 57.44 | 62.27 | 70.99 |
   | d4 | 492,197 | 63.25 | 68.82 | 37.37 | 56.50 | 62.80 | 69.11 |
   | d6 | 492,197 | 71.36 | 74.09 | 69.75 | 56.66 | 58.09 | 70.19 |

   All four runs completed; none excluded. (Raw numbers: `protocol/selection.json`, `kernel_outputs/dev/*/val_metrics.json`.)
2. Rule (Amendments 1, 2, 2b, 2c): best C = 62.80 (d4); band C >= 60.80 = {d1, d2, d4}; fewest parameters in the band = d1 (123,649; its C is 1.79 below the best, inside the 2.0 tolerance by 0.21);
   no tie in size. **Chosen: d1.** The same choice results if the composite used the 640-px FPR (best d2 70.99, d1 70.53, inside the band). The arithmetic was done with `jq` because the Mac's free disk was under the 30 GiB floor of
   `bin/guard.sh` at the time (no local python allowed); `scripts/select_config.py` is to be re-run when the disk allows and must reproduce `protocol/selection.json`.
3. Frozen configuration of the final wave: d1 = EdgeUNet, 4 input channels (mag1c, TOA 640/550/460 nm, normalised as in the official code and clipped to [0, 2]), widths 4,8,16,32,64, stem stride 2, 123,649 parameters, pos_weight 1, no hard-negative sampling;
   8,000 steps, batch 16, AdamW (peak lr 2e-3, weight decay 0), 200 warmup steps then cosine to 2 %, jitter 32 px, D4 augmentation, class-balanced window sampler; seeds 0, 1, 2, 3, 4; trained on ALL 3,425 train tiles; last-step weights (no checkpoint selection, no early stopping);
   decision threshold p > 0.5. Edge variants as in Amendment 2 item 4 (BN folded, ONNX fp32 opset 17, static int8 QDQ S8/S8 per-channel, MinMax calibration on 48 TRAIN tiles, nothing tuned on test).
4. Final wave = two kernels, so that the primary result is not held back by reference rows (a re-ordering of Amendment 2 item 5, not a change of its content):
   (a) `bench-gas-final`: 5 seeds of d1, test scored once per seed in PyTorch fp32, then ONNX fp32 and int8 of every seed scored on the test split (15 test evaluations: TEST-LOG rows #5-#19);
   (b) `bench-gas-final-ref`: no training; the released mag1c+RGB checkpoint through the same ONNX fp32 and int8 export (2 evaluations, rows #20-#21) and the official `run_validation` cross-check of our scorer (1 evaluation, row #4).
   Every one of these is logged in TEST-LOG.md whatever its outcome (announced there before the push).
5. At the time of writing no test number of any model of ours exists; the only test numbers seen are the released checkpoints (TEST-LOG #2-#3). Claims follow PLAN.md wording; a margin inside the published std or the bootstrap interval is not claimed as a win.
   Note for readers: validation F1 values (about 70-74 for weak and strong) are not comparable to test values (the released models score about 42-49 weak F1 on test): the validation dates are a different, apparently easier, slice. Selection used only differences between candidates on the same slice.

**Verification of Amendment 3 item 2 (12:44 UTC).** After the guard's disk floor was lowered by the coordinator (30 -> 20 GiB; the floor was not touched by the builder), `scripts/select_config.py kernel_outputs/dev` was run under `bin/guard.sh`:
it reproduces the jq computation (best C 62.80, band [d1, d2, d4], smallest-size candidates [d1], chosen d1). Its only change since the amendment is that it ignores the kernel's `logs/` directory.

---
## Amendment 4 (2026-10-09 ~01:45 UTC by the session clock, which has jumped by +5 h before: the git history, not this stamp, orders it). The GPU wave. Written BEFORE any GPU kernel of this track has ever been pushed or run, and before any GPU number exists.

Context at the time of writing: `kaggle quota` shows GPU 44.45 h used of 30 h (0.00 h remaining) until the refresh at 2026-10-10T00:00:00 UTC; no GPU kernel is pushed before then. The only models of ours scored on test are the CPU wave (Amendment 3: d1, 123,649 parameters, 8,000 steps x batch 16 = 128k crops, about 5 % of the paper's budget; TEST-LOG #5-#19). Everything below is fixed now; the only inputs still to come are the G0 throughput numbers and the G1 validation numbers, and the rules that turn them into decisions are written here.

### 1. Target and what is reported
- Target (unchanged, never lowered): **weak-plume F1 above 47.57 +- 4.17**, the best published weak-plume row (Sci. Rep. 13:19999 Table 2, AVIRIS test split, 5-run mean +- std, mag1c-only), under the paper's protocol as reproduced in Amendment 1 item 2 (strong/weak by emission rate, tile positive above 10 predicted pixels, AUPRC = average precision), on the 342 official test tiles, 5 seeds.
- Reported beside it for every GPU model, with the same scorer: **F1 strong, F1 weak, FPR by tile (>10 px), AUPRC** as 5-seed mean +- sample std (ddof=1) with the tile-level bootstrap 95 % interval of the mean (2,000 draws), the paired difference to the released mag1c-only and mag1c+RGB runs (same tiles), parameters, fp32 and int8 size, Kaggle and M4 latency; the official-code variants (label-pixel grouping, 640-px rule, 16-threshold AUPRC) labelled as such. Published rows for reference: mag1c+RGB 81.96 +- 3.71 / 43.42 +- 5.72 / 43.66 +- 7.36 / 51.99 +- 2.76; mag1c-only 74.15 +- 6.10 / 47.57 +- 4.17 / 52.11 +- 10.98 / 49.41 +- 5.49 (strong F1 / weak F1 / FPR / AUPRC). CPU rows (d1: 42.70 +- 2.30 weak F1) stay in the table as the 5 %-budget rows; GPU rows are labelled with their step count. Neither set is dropped.
- Claim rule (GPU-PLAN, verbatim): "best published result on STARCOP test, protocol Z, self-evaluated, as of DATE" only if the 5-seed mean weak F1 exceeds 47.57 AND the paired bootstrap interval against the released mag1c-only run excludes zero; otherwise the result is reported as "not distinguishable from the published row" or as a negative. The other three columns are reported either way; a weak-F1 gain bought with a worse tile-level FPR is described as that. If fewer than 5 seeds of G2 complete, no claim is made. Never "#1" or "SOTA".

### 2. The wave (stages, kernels, data access, caps)
[Amendment 5, 2026-10-10: the caps in this table are read as BILLED seconds with the numbers and `-t` values of Amendment 5 item 2; read as wall seconds they would bill about twice as much on the T4 shape.]
Hardware: every GPU kernel pins `machine_shape` NvidiaTeslaT4 (the shape the owner's own kernels use on this account; metadata `enable_gpu` true, `is_private` true). One GPU session at a time under `locks/kaggle-gpu`. All kernels mount the same train cache `sankalpsthakur/bench-gas-prep-cache` v1 (SHA-256 in Amendment 3) and read the test split only where stated.

| stage | kernel (private) | what | seeds | test split | cap (`kaggle kernels push -t`) | soft deadline (CFG) |
|---|---|---|---|---|---|---|
| G0 | `bench-gas-gpu-probe` (kernels/gpu-probe) | synthetic-batch training throughput of d1, d2, d3, d5, d2w and the released architecture (mnv2), AMP on/off; GPU model, torch/CUDA, `nvidia-smi`. No inputs, no data | - | not mounted | 900 s (0.25 h) | - |
| G1 | `bench-gas-gpu-wide` (kernels/gpu-wide) | validation-only dev runs of the candidates below on train minus the validation dates | 0 (one per candidate) | NOT mounted (no dataset source) | 8,100 s (2.25 h) | 7,500 s |
| G2 | `bench-gas-gpu-compact` (kernels/gpu-compact) | the ONE config frozen after G1, trained on ALL 3,425 train tiles, last-step weights, ONNX fp32 + static int8 edge export | 0, 1, 2, 3, 4 | mounted, scored once per seed | 10,800 s (3.00 h) | 9,600 s |
| G3 | `bench-gas-gpu-control` (kernels/gpu-control) | paper-recipe control, released architecture (section 4) | 0 | mounted, scored once | 5,400 s (1.50 h) | 4,800 s |

Push order: G0, plan (section 3), record 4a, G1, selection (section 5), record 4b, G2, G3. **Sum of caps = 25,200 s = 7.00 GPU-h; 1.00 GPU-h of the 8 GPU-h gas cap stays in reserve for section 8.** Every push carries `-t <cap>`, so the platform stops a runaway kernel: "the wave fits 8 GPU-h" holds whatever the throughput turns out to be. A kernel that dies at its cap is treated as lost in full (it is not known that Kaggle keeps `/kaggle/working` after a platform timeout); the soft deadline (CFG `deadline_s`, below the cap) makes the driver terminate running processes and start no new one, which keeps every finished run and, for G2, the edge stage.
G1 candidates (all single seed 0, same hyper-parameters; ids as in Amendments 1-2): **d1** (widths 4,8,16,32,64, stem stride 2; the CPU-selected config), **d2** (8,16,32,64,128, stem 2; the CPU runner-up and the natural middle size, added here because without it the family jumps from 0.12 M to 1.96 M parameters), **d3** (16,32,64,128,256, stem 2; the wider model deferred from Amendment 1), **d5** (4,8,16,32,64, stem stride 1: full-resolution stem, deferred from Amendment 1), **d2w** (8,16,32,64,128, stem stride 1, new). d4 and d6 (pos_weight 0.5 / hard negatives) were CPU variants and are not repeated.

### 3. Schedule and the budget rule (fixed now, executed by `scripts/mk_gpu_wave.py plan` after G0)
- Schedule for G1 and G2 (identical in steps): AdamW, peak lr 2e-3, weight decay 0, **batch 32**, 200 warmup steps then cosine to 2 %, jitter 32 px, D4 augmentation, class-balanced window sampler, pos_weight 1, no hard negatives, fp16 autocast with GradScaler (AMP; the CPU wave was fp32), p > 0.5, last-step weights, no early stopping. Number of steps = **E x 5,245**, E in {15, 10, 5} paper-epoch equivalents (one paper epoch = 167,840 windows = 5,245 steps at batch 32): 78,675 / 52,450 / 26,225 steps (2.52 M / 1.68 M / 0.84 M crops; the CPU wave saw 0.128 M). The step count is the same in G1 (train minus validation dates, 142,149 windows) and G2 (all train, 167,825 windows), so G1 passes over its smaller set about 18 % more often per step; the cosine schedule is defined in steps, which is why steps, not epochs, are matched. G1 evaluates the validation dates once, at the last step (no mid-run evaluations: they cost GPU minutes and selection never uses them).
- Planning model (all constants are in `scripts/mk_gpu_wave.py`): rate r_c = 0.8 x min(G0 AMP steps/s of config c, real `CropSampler` batches/s measured by the CPU smoke kernel); a run costs 40 s init + steps x 1.5 / r_c (two processes share the GPU: factor 1.5; no credit is taken for a second GPU, although the driver gives each process slot its own GPU through `CUDA_VISIBLE_DEVICES=<slot % n_gpu>` when `nvidia-smi` shows more than one) + 75 s per validation pass + 80 s per test pass; kernel overhead 200 s; G1 = overhead + longest-first makespan of its runs on 2 slots; G2 = overhead + 300 s edge stage + 3 rounds (5 seeds on 2 slots) x run cost; G3 = overhead + run (factor 1.0, one process) + 60 s plateau checks.
- Fit rule: planned time must be at most **80 % of the cap**: G1 <= 6,480 s, the WORST candidate's G2 <= 8,640 s (so selection can never pick a config whose 5 seeds do not fit), G3 <= 4,320 s. Search order: E = 15, then 10, then 5; for each E the candidate set {d1,d2,d3,d5,d2w}, then {d1,d2,d3,d5}, then {d1,d2,d3} (d2w, then d5, are dropped by this rule, never d1, d2, d3); the first feasible pair is the wave (a larger E beats a larger candidate set). The control epochs E_ctrl in {15, 10, 5} (steps = E_ctrl x 4,443, one control epoch = 142,149 windows) are the largest that fit the control's own cap. The reduction is the same for all candidates, never for one. If nothing fits (even E = 5 with {d1,d2,d3}, or E_ctrl = 5), the wave stops and the owner is asked: nothing is shrunk silently and the 47.57 target does not move.
- Expected use if everything goes as planned: G0 about 0.1 h, G1 <= 1.8 h, G2 <= 2.4 h, G3 <= 1.2 h (total <= 5.5 h; the rule's ceiling, typical use is lower). The plan output (`protocol/gpu_budget.json`) is committed as **Amendment 4a** before G1 is pushed; it fixes E, the candidate set and E_ctrl, and records the machine actually obtained. The step counts shown in the generated placeholder kernels (E = 15, d1) are not the decision: the rule is.
- **G2 timing gate** (timing only, never metrics, so it adds no selection bias). The planning rates come from synthetic probe numbers and a sampler ceiling measured on an idle machine; in production the crop thread competes with the training loop. After G1 and before G2 is pushed, take the measured `secs` of the chosen config's G1 run (`RUN_STATUS.json`, `runs.g_<cfg>.secs`: same step count, same 2-process layout, includes init and one validation pass) and compute 200 s + 300 s + 3 x (secs + 5 s) (the 5 s swaps the 75 s validation pass for the 80 s test pass). If that exceeds G2's soft deadline (9,600 s) G2 is not pushed and the owner is asked: a G2 that would predictably end with 2-3 seeds cannot carry a claim.

### 4. The paper-recipe control (G3)
Released architecture (`SMPLikeUnet` = smp Unet mobilenet_v2, 6,629,233 parameters, 4 inputs: mag1c, TOA 640/550/460 nm), trained from scratch with the recipe of the released `config.yaml` (experiment f4_hyper_unetsempos1all_all_15ep_R3: optimizer adam, lr 1e-4, batch 32, 128-px windows with overlap 64, weight_mag1c loss weights, BCE pos_weight 1, lr_decay 0.5, lr_patience 4, val_check_interval 0.5, early_stopping_patience 8, max_epochs 15): no jitter, no warmup, constant lr 1e-4 halved by ReduceLROnPlateau(factor 0.5, patience 4) stepped at each half-epoch validation, our class-balanced window sampler, AMP, seed 0, last-step weights. Deliberate differences from the original run, all written here: (a) trained on train minus the validation dates (2,901 tiles), not all of train; (b) the plateau scheduler monitors the loss on 2,048 fixed class-balanced crops of the VALIDATION dates, checked every half epoch (4,443 / 2 = 2,221 steps; patience 4 checks = 2 epochs), not the test loss the public script monitors (Amendment 1/CORRECTED-PROTOCOL item 5); (c) early stopping (patience 8) is not implemented: the control always trains E_ctrl epochs; (d) one seed, AMP. Whether the original run stepped its scheduler per validation or per epoch is not verified here, so the row is labelled "paper-recipe-like", not "reproduction". `train.py`'s `--sched` help text says "4 epochs"; the code and this amendment mean 4 validation checks. The control is one seed and carries no claim; it is the reference for reading the compact rows (what the leader's recipe yields under the corrected protocol with our budget). Its test evaluation is allowed together with validation dates only through the flag `--allow_test_with_val`, which exists for this run and requires `--sched plateau`; nothing is selected on the validation dates in G3.

### 5. Validation split and selection (G1 -> G2)
- Split: whole dates 2019-10-22 and 2019-10-23 of train.csv (`protocol/val_split.json`, `val_ids.csv`: 524 tiles = 280 none + 204 weak + 40 strong, 46 flights; no flight shared with the rest of train or with test), exactly as in Amendments 1-3. The same validation slice has now served two selection stages; with at most 5 candidates and a 2.0-point tolerance the optimism is small but not zero, and it is stated here.
- Rule (Amendments 1, 2, 2b, 2c, 3, verbatim): C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap) on the validation dates at the last step; band = completed runs with C >= best C - 2.0; choose the fewest parameters in the band; ties in size -> highest C (`scripts/select_config.py kernel_outputs/gpu-wide --out protocol/selection_gpu.json`; incomplete runs are excluded). **One clause is added, with its reason, before any GPU number exists**: band members whose parameter count is within 1 % of the smallest band member count as size-equal, and among size-equal members stem stride 2 is preferred over stem stride 1. Reason: stem stride 1 has about 4 x the multiply-accumulates at equal parameters (d5 has 123,429 parameters against 123,649 for d1; d2w 491,945 against 492,197 for d2), so "fewest parameters" alone would prefer the slower model for 0.2 % of parameters, against the edge objective of PLAN.md. For the CPU wave the clause changes nothing (`scripts/select_config.py kernel_outputs/dev` reproduces `protocol/selection.json`; tests/test_select_config.py).
- The result is frozen in **Amendment 4b** (G1 table, band, chosen config, generated G2 script hash, announced TEST-LOG rows), committed before G2 is pushed. G2 uses the chosen config at the same E as G1.

### 6. Seeds and test policy
- Seeds: G1 seed 0 per candidate; G2 seeds 0, 1, 2, 3, 4; G3 seed 0. G0 trains nothing.
- G0 and G1 never mount the STARCOP mirror (their kernel metadata has empty `dataset_sources`, the driver looks for the mirror only if a run asks for `test_eval`, and `train.py` refuses `--test_eval` together with validation dates except for the control's flag). The test split is read only by G2 and G3, and only as follows: **one sealed test evaluation per frozen config and seed**: G2: 5 PyTorch fp32 (one per seed, at the end of its training), then 5 ONNX fp32 and 5 ONNX int8 of the same weights in the edge stage = 15 evaluations; G3: 1 PyTorch fp32. TEST-LOG rows **#22-#26** (G2 PyTorch seeds 0-4), **#27-#31** (ONNX fp32), **#32-#36** (ONNX int8), **#37** (G3) are announced in TEST-LOG.md before anything is pushed and are filled in whatever their outcome. The released-model rows are not repeated (`edge_released` false). Nothing about a test number changes a configuration, a seed list or a step count; a re-run that reads test again is a new row and never replaces the first.

### 7. Reporting
Per-seed and mean +- std rows for all of section 1's columns for every pushed GPU kernel that completed, the G1 validation table (all candidates, including losers and incomplete runs), G0 throughput, the plan, GPU hours actually used per kernel (from `kaggle quota` before and after, and the kernel logs) in `RESOURCES.md` and STATUS.md, and the control row. The results go to `results/gpu_*` and a new section of SUBMISSION.md worded under section 1's claim rule; nothing is uploaded or sent by the builder (weights CC BY-NC 4.0).

### 8. Failures (no silent retry)
- Every failed, cancelled or timed-out kernel is logged at once in STATUS.md and RESOURCES.md with its GPU time (the time counts against the cap), and in TEST-LOG.md if it had read the test split; it stays in every report.
- At most ONE re-push of a kernel, only when the cause is infrastructure (platform error, session preempted, accelerator not available, CUDA architecture mismatch, out-of-memory kill of a process, quota refusal) and not a result; it is announced in STATUS.md before the push; the GPU-hours already used in the wave plus the new cap must stay at or below 8.0; the configuration, seeds and step counts do not change. [Amendment 5 withdraws the NvidiaL4 fallback: it may be a 4-GPU shape billed 4 x; see its item 5.] G0 may be re-pushed once if T4 cannot start; all later kernels then use the shape that worked, recorded in 4a. **Anything else needs the owner's explicit yes in chat**: a retry that would exceed 8.0 GPU-h, a second retry, any retry or extra run motivated by a number, any change of configuration or schedule after 4a/4b.
- A partially completed G1 is used as is (completed runs only, as the CPU rule did); with no completed run there is no GPU selection and G2 is not pushed without the owner. Partially completed G2: each completed seed is reported and logged, n < 5 means no claim. A G3 that fails or hits its cap is reported as "control not completed"; it is not replaced. A device error that appears only on the second process slot (`CUDA_VISIBLE_DEVICES=1`) and not on slot 0 is a code fault, not infrastructure, and needs the owner's yes before any re-push; a G0 case that reports `error` instead of `steps_per_s` is a G0 failure (the planner stops on it).
- If the wave is cut short for budget or quota reasons, the report says which stages did not run; the CPU results and the 47.57 target stand unchanged.

### 9. Pre-flight checks, code state and the CPU smoke
- Gate before each GPU push (STATUS.md runbook): the date is >= 2026-10-10 00:00 UTC; `kaggle quota` shows GPU remaining >= that kernel's cap (the owner's competitions share the quota); `mkdir ../locks/kaggle-gpu` succeeds and `owner` names the kernel and its latest end; a RESOURCES.md row exists; `-t <cap>` is on the push command.
- Code state: `src/starcop_edge` differs from Amendment 3 only in the pip line of `official_check.py` (git diff 24267cc HEAD -- src; hash of src/starcop_edge/*.py is now 54170b5e7b1e2859); new files: `kernels/train_gpu/main.py` (GPU driver: nvidia-smi and early stop if no GPU, run slot -> `CUDA_VISIBLE_DEVICES`, soft deadline, RUN_STATUS.json, optional sampler timing; sha256 e7411f18b69586aa...), `scripts/mk_gpu_wave.py` (generators and planner), the clause in `scripts/select_config.py`, tests. The CPU driver `kernels/train/main.py` and every committed CPU kernel are untouched. Generated scripts (sha256 prefixes): gpu-probe 69783ab6ebcd0613 (regenerated after the first draft of this amendment, still before any push, to log torch/CUDA arch list, device count and whether rasterio, pandas, numpy, onnx, onnxruntime import: a missing module would otherwise show only after hours of training), gpu-wide 84fd7fd81dead3a1, gpu-compact 20a89e6554f4bff1, gpu-control 8cdadfd8265d7cce (placeholders at E = 15 and d1; they are regenerated after 4a/4b).
- `train.py`'s AMP, plateau-scheduler and mnv2 paths were written before Amendment 3 and have never run on a GPU. A private CPU kernel, `bench-gas-gpu-smoke`, runs the same driver, src and run dictionaries (script differs from gpu-wide only in the CFG line and the first comment line) with `device` cpu and 3-4 steps, no test access: all five candidates with validation, mnv2 with the plateau schedule and `--allow_test_with_val`, a run without validation, the soft-deadline termination, and a timed `CropSampler` loop on the real cache. It cannot exercise CUDA, AMP/GradScaler, the `CUDA_VISIBLE_DEVICES` branch (n_gpu is 0 on a CPU kernel), or the edge and test paths (the latter two are unchanged code that ran in the CPU final wave). Smoke v1 ran the previous driver revision (sha256 cf83ff62d49f14c9, `cuda:<n>` device strings); v2 ran the final one. Its numbers are not results and are never used for selection.

---
## Amendment 5 (2026-10-10 ~00:45 UTC by the session clock; the git history orders it). Budgets in BILLED GPU-hours. Written BEFORE any GPU kernel of this track has been pushed; Amendment 4 stays in force except where this amendment replaces it.

### 1. Why (verified, not assumed)
- `bench-small-lm-gpu-probe` was pushed with `machine_shape` `NvidiaL4X1`; the server accepted it without refusal and the kernel ran on **2 x Tesla T4**; `kaggle quota` billed 0.44 GPU-h for 0.27 h of wall time (RESOURCES.md FINDING row 2026-10-10 00:30 UTC, small-lm). The gas builder re-measured it independently while that kernel was running: GPU used 0.54 h at 00:30:29 UTC and 0.63 h at 00:33:01 UTC, i.e. +0.09 h over 152 s of wall time = **2.1 x wall** (quota resolution 0.01 h). So the quota is billed per GPU held, busy or idle, and the previous week's 44.45 h "used" of 30 h is consistent with that.
- Amendment 4 wrote its caps as wall seconds (`-t` 900 / 8,100 / 10,800 / 5,400 = 7.0 wall-h). On a 2 x T4 shape those bill up to 14.0 GPU-h against the gas cap of 8.0 and the account's 30 h. That reading is withdrawn.

### 2. Unit and caps
- Every budget, cap, gate and ledger entry is in **billed GPU-hours = GPUs held x wall hours** (the unit of `kaggle quota`, and of the program caps in PLAN.md). The gas cap is **8.0 billed hours = 7.0 h of kernel caps + 1.0 h retry reserve**. The billing factor b is the number of GPUs the shape actually holds (2 for the T4 shape); `kaggle kernels push -t` takes **wall seconds = floor(billed cap / b)**.
- Caps in billed seconds (sum 25,200 = 7.00 h; G0 rises from 900 to 1,200 and G1 falls from 8,100 to 7,800, because a probe of about 5 wall minutes bills 10 minutes on two GPUs):

| stage | billed cap | hours | `-t` wall s at b = 2 | `-t` at b = 1 | soft deadline (CFG, wall s) b = 2 / b = 1 |
|---|---|---|---|---|---|
| G0 `bench-gas-gpu-probe` | 1,200 | 0.33 | 600 | 1,200 | - |
| G1 `bench-gas-gpu-wide` | 7,800 | 2.17 | 3,900 | 7,800 | 3,300 / 7,200 |
| G2 `bench-gas-gpu-compact` | 10,800 | 3.00 | 5,400 | 10,800 | 4,200 / 9,600 |
| G3 `bench-gas-gpu-control` | 5,400 | 1.50 | 2,700 | 5,400 | 2,100 / 4,800 |

- Planned use (Amendment 4 item 3's 80 % rule, now on billed seconds): G1 <= 6,240, worst-case G2 <= 8,640, G3 <= 4,320 billed seconds, G0 about 0.2 h: at most about 5.7 billed hours, hard ceiling 7.0 by the caps. The ledger, not wall time, is gas's account of spending (item 6).

### 3. Both GPUs are used; single-worker kernels are charged for every GPU
- G1 (5 candidates) and G2 (5 seeds) run P = 2 worker processes from one shared job queue; when the shape holds 2 GPUs each worker slot gets its own GPU through `CUDA_VISIBLE_DEVICES=<slot % n_gpu>` (driver revision `kernels/train_gpu/main.py`, sha256 prefix c1b3117137d1bdde), so no billed GPU idles while jobs wait. The planning model credits this (per-process slowdown 1.15 instead of 1.5 when n_gpu >= P). It is required, not optional: a 2-GPU kernel whose second GPU sits idle is billed double for the same work.
- The G3 control is a single process. On a 2-GPU shape it is charged 2 x its wall time (the "halve the budget for a single-worker run" case): the plan computes its epochs from `b x wall <= 0.8 x 5,400`, which usually means 10 control epochs on 2 x T4 instead of 15. Folding the control into G2's queue would use the idle GPU, but it would couple the control's fate to G2's and is not adopted here.

### 4. Planner and generators
`scripts/mk_gpu_wave.py plan` now takes the billing factor b (default: `n_gpu` from the G0 readback) and applies the fit rule to billed seconds: b x projected wall <= 0.8 x billed cap for G1, for the worst-case G2 candidate and for G3; the search order, epoch tiers {15, 10, 5}, candidate sets, rate model (0.8 x min(probe rate, sampler rate)) and the "no fit -> stop and ask, never shrink silently" rule are as in Amendment 4 item 3. The plan output `protocol/gpu_budget.json` carries b, n_gpu, the wall caps for `-t`, the soft deadlines and `projected_total_billed_h`. Generators write `max_gpus`, `billing_b`, `billed_cap_s`, `wall_cap_s` into each kernel's CFG (`kernels/<dir>/cfg.json`), set `deadline_s = wall cap - margin`, and still never push. The G2 timing gate of Amendment 4 is unchanged, but its limit is G2's `deadline_s` from `kernels/gpu-compact/cfg.json`, not 9,600 s. Illustration with SYNTHETIC probe rates (d1 90, d2 90, d3 55, d5 50, d2w 32, mnv2 38 steps/s) and the measured sampler 61.9 batches/s: on 2 x T4 (b = 2) the plan is E = 5 with {d1, d2, d3, d5} and a 10-epoch control, 3.80 billed h; on a single GPU (b = 1, two workers sharing it) it is E = 10 with the same set and a 15-epoch control, 3.72 billed h. Per unit of work a single-GPU shape is about a third cheaper, so the shape matters more than any other knob.

### 5. Shape request and hardware readback
- Requested in this order: (a) **a single-L4 shape string, only if a readback has verified it** (a RESOURCES.md row showing a kernel that obtained exactly one L4 for that exact string; small-lm is testing this on 2026-10-10); (b) otherwise `NvidiaTeslaT4` (the documented shape, 2 x T4, b = 2). Evidence so far argues against assuming (a): `NvidiaL4X1` was accepted but silently replaced by the default 2 x T4, and the owner's own L4 contract (`NvidiaL4`, 2026-10-05) requests 4 GPUs. Because the server accepts unknown strings without a refusal, "fall back when refused" cannot work: **what counts is the hardware obtained**.
- Readback: G0 stores `env` (device name, n_gpu, torch arch list) in `gpuprobe.json`; the driver prints `nvidia-smi` first. G0 and every GPU kernel **stop at once when more GPUs are visible than `max_gpus`** (the b the caps were computed with), before the cache warm-up, so a mis-sized shape costs seconds, not hours. If fewer GPUs are obtained than budgeted, b is lowered by regenerating with `--billing <n>` before the next push. Amendment 4 item 8 named `NvidiaL4` as the G0 fallback; that is withdrawn (it may be a 4-GPU shape billed 4 x).
- If more than one shape can be probed, the shape is chosen by the plan outputs only, never by model quality: the larger E, then the smaller `projected_total_billed_h`, then `NvidiaTeslaT4`. A G0 on the documented single-GPU shape `NvidiaTeslaP100` (b = 1; billed cap 600 s) is a candidate for that comparison if the coordinator approves it (it is outside the ladder of this item and needs the torch sm_60 check G0 already performs).

### 6. Push gate and ledger
`scripts/gpu_gate.sh <kernel_dir>` must print `GATE: PASS` before every GPU push (it pushes nothing and takes no lock): date after the quota refresh; metadata `is_private`, `enable_gpu`, `machine_shape` set; cfg carries billed cap, wall cap and b with b x wall <= billed cap; no `locks/kaggle-gpu`; `kaggle quota` remaining >= the billed cap (the owner's competitions share the quota); and the gas ledger plus the billed cap <= 7.0 h (<= 8.0 h with `--retry`, item 7). It prints the lock, `-t <wall seconds>` push and watch commands. After each kernel, `scripts/gpu_gate.sh record <stage> <ref> <used_before_h>` appends the billed hours (quota used after minus before) to `results/gpu_ledger.csv`; if a kernel's billed hours exceed its cap by more than 10 %, stop and ask the owner (b was misjudged). Tests: `tests/test_gpu_gate.sh`.

### 7. Retry reserve and failures (replaces the arithmetic of Amendment 4 item 8, nothing else in it)
The reserve is 1.0 billed hour. A kernel may be re-pushed ONCE, only for an infrastructure cause (platform error, preemption, accelerator unavailable, CUDA architecture mismatch, out-of-memory kill, quota refusal, wrong GPU count), announced first in STATUS.md, configuration/seeds/steps unchanged, and only if the ledger plus the new billed cap stays <= 8.0 h (`--retry`). Failed kernels stay in the ledger. Everything else (any other retry, a retry that would pass 8.0 h, any change after 4a/4b) needs the owner's explicit yes.

### 8. What does not change
Target and claim rule; candidates; schedule rule and the tiers; selection rule and the size-equality clause; seeds; the sealed-test policy and TEST-LOG rows #22-#37; the G2 timing gate; the 4a/4b records; reporting; no publication by the builder.

### 9. Code state
`scripts/mk_gpu_wave.py`, `scripts/gpu_gate.sh`, `tests/test_gpu_plan.py`, `tests/test_gpu_gate.sh`, the `max_gpus` guard in `kernels/train_gpu/main.py` and in `kernels/gpu-probe/main.py`; src/starcop_edge unchanged (hash 54170b5e7b1e2859). Generated placeholders (b = 2, NvidiaTeslaT4, E = 15, d1; regenerated after 4a/4b) and their sha256 prefixes are listed in `kernels/VERSIONS.md`. Amendment 4 item 9's statement that the smoke ran the final driver refers to driver revision e7411f18b69586aa; the `max_gpus` guard and Note 1's `edge_runs` option are later revisions, so the CPU smoke is run again on the final revision (CFG carries `max_gpus` 2; its script differs from the GPU kernels only in the first comment line and the CFG line). The guard's abort branch needs more GPUs than a CPU kernel has, and the `edge_runs` line sits in the edge stage, which needs the test mirror; neither can run there.

### Notes to Amendment 5 (2026-10-10 ~01:05 UTC, still before any GPU push)
**Note 1, the control joins G2's kernel (the coordinator's refinement; selection is not reopened).** The G3 control, a single process, would bill b x its wall time with an idle GPU. It is therefore queued as the LAST job of `bench-gas-gpu-compact`'s shared queue: six jobs on two workers, `[seed 0, 1, 2, 3, 4, ctrl_mnv2]`, seeds first so that a soft deadline can only cut the control before a seed. The edge stage (ONNX fp32 + int8) covers the five seeds only (CFG `edge_runs`); the control's single test evaluation happens inside its own process (TEST-LOG row #37 as announced; rows #22-#36 unchanged). Caps: the merged kernel's billed cap is the sum of the two it replaces, 10,800 + 5,400 = **16,200 billed seconds (4.50 h; `-t` 8,100 s at b = 2, 16,200 s at b = 1; soft deadline 6,900 s / 15,000 s)**; the sum of all caps stays 25,200 s (7.00 h) and the reserve 1.00 h. Fit rule: b x projected wall of the merged kernel <= 0.8 x 16,200 = 12,960 billed seconds for the WORST remaining candidate, with the control epochs Ec the largest tier {15, 10, 5} at which every remaining candidate's merged kernel fits (so selection can never pick a config for which no control fits); search order E, then candidate set, then Ec. The G2 timing gate keeps its role and becomes `scripts/g2_timing_gate.sh <cfg>` (the planned control job is scaled by the measured slowdown, never below its plan). If the control is cut or fails while the seeds complete, the row is reported as "control not completed"; the standalone `kernels/gpu-control` (billed cap 5,400 s) stays in the repository and may be run only in the retry reserve under item 7. Nothing in G1, in the selection rule, in the seeds or in the claim rule changes. Illustration with the SYNTHETIC probe rates of item 4 and the measured sampler 61.9 batches/s: b = 2 gives E = 10 with {d1, d2, d3}, Ec = 15, 5.18 billed h (separate kernels would have given E = 5 with a 10-epoch control); b = 1 gives E = 15 with {d1, d2, d3, d5}, Ec = 15, 4.49 billed h (numbers with the 300 s kernel overhead of Note 4).
**Note 2, shape decision (coordinator, 2026-10-10 ~00:55 UTC).** The shape is `NvidiaTeslaT4` (2 x T4, b = 2) with the two-worker queue. Evidence: small-lm's kernels requested with `NvidiaL4X1` and with `NvidiaL4` were both accepted and silently replaced by the account default, 2 x Tesla T4 (env line in RESOURCES.md, torch 2.11.0+cu128, fp16 only); the installed SDK does not know a single-L4 shape; no P100 probe is made (no tensor cores, worse than a T4 for AMP CNN training at the same billed rate). The optional P100 probe of item 5 is withdrawn. Two busy T4s at 2x billing cost the same per useful GPU-hour as one T4 at 1x, so the plan stands as long as both workers stay busy, which Note 1 serves.
**Note 3, ordering and stop rule.** When small-lm releases `locks/kaggle-gpu`, whichever of gas and gauges is ready runs its gate first and takes the lock; the other waits. Gas then carries the wave through as pre-registered (Amendments 4 and 5 and these notes), one background waiter per kernel, every session logged in RESOURCES.md and in `results/gpu_ledger.csv`, and **stops at the 8 billed-hour cap** (the gate refuses beyond 7.0 h, or 8.0 h with `--retry` for a retry that item 7 allows) and reports.
**Note 4 (2026-10-10 ~02:05 UTC, before any GPU push): robustness found while waiting for a GPU session slot.** (a) Kaggle refuses GPU pushes with "Maximum batch GPU session count of 2 reached" while two sessions exist, and QUEUED sessions count: two owner kernels ([names of unrelated kernels redacted]) held both slots from 00:59 UTC; the Kaggle CLI prints that error but exits 0, so the launcher now reads the push output. (b) The page-cache warm-up of the 9 GB cache measured 41 s (smoke v1), 145 s (v4) and the CPU smoke v3 stayed RUNNING for 45+ minutes with no result while a fresh kernel mounted the same cache in 2 s (mount probe): the mount speed varies. Consequences, all before any GPU push: the kernel overhead of the planning model rises from 200 s to 300 s (`K_OVER`; Amendment 4 item 3's "200 s" is replaced, the G2 timing gate uses 300 s + 300 s edge stage); the driver bounds the warm-up (CFG `warm_timeout_s`, default 600 s: a slower mount stops the kernel, because the planned rates assume local-speed reads), escalates the soft deadline from SIGTERM to SIGKILL after 20 s and stops waiting for a process that survives SIGKILL for 90 s (`enforce_deadline`, tested with toy processes in `tests/test_driver_deadline.py`), and rewrites `RUN_STATUS.json` after every finished run. These are scheduling and safety changes in the driver; training, evaluation and scoring code are untouched. Smoke v5 re-runs the final driver on CPU.

---
## Amendment 4a (2026-10-17 19:58 UTC by the session clock). Written after G0 and BEFORE G1 is pushed. No validation or test number of any GPU model exists at this time.

1. G0 `bench-gas-gpu-probe` (script sha256 1b232a22f1a92318): hardware obtained: Tesla T4, n_gpu 2, torch 2.11.0+cu128, cuda 12.8, arch list ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120'], capability [7, 5], host CPUs 4;
   imports: rasterio 1.5.1, pandas 2.3.3, numpy 2.1.3, onnx 1.22.0, onnxruntime MISSING ModuleNotFoundError("No module named 'onnxruntime'"); probe seconds 44.2. Gas ledger so far: gpu-probe gpu-probe: billed 0.05 h (gpus x wall (2 x 92 s)).
   Synthetic throughput (batch 32, 128 x 128 crops, steps/s per process):

   | case | AMP | steps/s | parameters | peak GB |
   |---|---|---|---|---|
   | mnv2_paper | amp 1 | 18.69 | 6629233 | 0.84 |
   | d2_8-128_s2 | amp 1 | 67.89 | 492197 | 0.17 |
   | d1_4-64_s2 | amp 1 | 80.45 | 123649 | 0.09 |
   | d3_16-256_s2 | amp 1 | 49.99 | 1964557 | 0.24 |
   | d5_4-64_s1 | amp 1 | 53.87 | 123429 | 0.17 |
   | d2w_8-128_s1 | amp 1 | 37.68 | 491945 | 0.29 |
   | d3w_16-256_s1 | amp 1 | 23.12 | 1964241 | 0.59 |
   | mnv2_paper | amp 0 | 11.12 | 6629233 | 1.39 |
   | d2_8-128_s2 | amp 0 | 53.1 | 492197 | 0.22 |
   | d1_4-64_s2 | amp 0 | 69.81 | 123649 | 0.11 |
   | d3_16-256_s2 | amp 0 | 29.79 | 1964557 | 0.33 |
   | d5_4-64_s1 | amp 0 | 33.59 | 123429 | 0.27 |
   | d2w_8-128_s1 | amp 0 | 22.74 | 491945 | 0.44 |
   | d3w_16-256_s1 | amp 0 | 12.68 | 1964241 | 0.81 |
2. Plan (`scripts/mk_gpu_wave.py plan`, `protocol/gpu_budget.json` sha256 7315b4d4f9dce682; sampler ceiling 61.9 batches/s from `results/gpu_smoke/sampler_bench.json`; billing factor b = 2, n_gpu 2, per-process slowdown 1.15):
   planning rates (steps/s) {'d1': 49.52, 'd2': 49.52, 'd3': 39.99, 'd5': 43.1, 'd2w': 30.14, 'mnv2': 14.95}; **E = 10 paper epochs = 52,450 steps** for G1 and G2; candidate set ['d1', 'd2', 'd3']; **control epochs Ec = 5** (22,215 steps, plateau check every 2221 steps) inside the G2 kernel;
   projected billed seconds: G1 5,932; worst-case G2 + control 11,640; total 4.88 billed h of the 7.0 h of caps; verdict OK.
   Wall caps for `-t` (billed caps / b; compact = G2 + control merged: 16200 billed s): {'probe': 600, 'wide': 3900, 'compact': 8100, 'control': 2700}; soft deadlines (wall s): {'wide': 3300, 'compact': 6900}.
3. Kernels generated for the plan (not pushed before this amendment is committed): `kernels/gpu-wide` sha256 ba33f8835bee1290 (candidates ['d1', 'd2', 'd3'], 52,450 steps, seed 0, validation dates only, no test mirror, CFG deadline_s 3300).
4. Nothing else changes: Amendments 4 and 5 and their notes apply as written.
5. Notes at the time of writing (facts only; nothing in items 1 to 4 depends on them).
   (a) Cap: there is no "CAP" line from the lead in STATUS.md at the time of this commit (checked immediately before it), so the decision in item 2 is taken at the pre-registered 8 billed hours (caps 1,200 / 7,800 / 16,200 merged billed seconds plus the 1.0 h retry reserve); the lead has asked the owner about a higher cap, and after this amendment is committed the cap stays as it is for this wave.
   (b) G0 history: the first push (19:33:54 UTC) stayed QUEUED for 19 minutes with nothing run (an unrelated owner kernel [name redacted] was QUEUED as well, nothing RUNNING); it was deleted and re-pushed once at 19:54:02 UTC under the coordinator's queue rule of 2026-10-10 (announced in STATUS.md before the re-push, commit 93727d1; 0 GPU-h billed for the first push). The re-push ran at once and ended COMPLETE (kernel time 62 s). This used G0's single re-push; the probe script is the committed one (sha256 1b232a22f1a92318).
   (c) Billing reading for the next wave: `kaggle quota` GPU used 3.36 h immediately before the re-push and 3.38 h after its end (+0.02 h at 0.01 h resolution) for 62 s of kernel time (about 92 s with start-up and teardown) on 2 x Tesla T4; no owner kernel was RUNNING in between (gen12 COMPLETE, gen20 QUEUED, read-only status calls). The ledger books GPUs x wall = 2 x 92 s = 0.05 h next to the raw delta; b stays the GPU count of the readback (Amendment 5 item 2), here 2. A gauges Stage B session of 6,657 s script time moved the counter by 2.41 h including concurrent owner kernels (at most about 1.3 x wall); this is logged, not used.
   (d) Environment for G2: onnxruntime is not installed in the GPU image; G2's edge stage runs `pip install onnxruntime onnxscript` (the generated G2 kernel enables internet), unchanged code. rasterio 1.5.1 and pandas 2.3.3 import, so the test stage can score. torch 2.11.0+cu128 has sm_75 in its arch list (T4 capability 7.5).
   (e) The standalone control kernel was regenerated for the record (`kernels/gpu-control` script sha256 bec580422aa0adaf, Ec = 5, billed cap 5,400 s); it is run only in the retry reserve (Amendment 5 note 1). The AMP rate of the control architecture is low (18.7 steps/s at batch 32), so the pre-registered rule gives Ec = 5: the control is a 5-epoch (22,215-step) paper-recipe-like run, about a third of the paper's 15 epochs, and is labelled with its step count; it carries no claim.

---
## Amendment 4b (2026-10-17 21:03 UTC by the session clock). Written after G1 and BEFORE G2 (with the control) is pushed, and before any test evaluation of a GPU model.

1. G1 `bench-gas-gpu-wide` (validation dates 2019-10-22/23 only, no test access, E = 10 = 52,450 steps, seed 0, last-step weights). Completed runs (C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap)); wall seconds of each run in the last column:

   | candidate | parameters | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC | C | secs |
   |---|---|---|---|---|---|---|---|
   | d1 | 123,649 | 68.78 | 78.06 | 59.79 | 68.65 | 63.92 | 1010 |
   | d2 | 492,197 | 72.63 | 73.37 | 62.63 | 65.61 | 62.24 | 1340 |
   | d3 | 1,964,557 | 65.39 | 73.88 | 56.23 | 61.43 | 61.12 | 1570 |
   Not completed (excluded by the rule): none. Gas ledger so far: gpu-probe gpu-probe: billed 0.05 h (gpus x wall (2 x 92 s)); gpu-wide gpu-wide: billed 1.35 h (gpus x wall (2 x 2427 s)).
2. Rule (Amendment 4 item 5 with its size-equality clause, applied by `scripts/select_config.py`, output `protocol/selection_gpu.json` sha256 9b9ef5ae59277b33): best C = 63.92; band (C >= best - 2.0) = ['g_d1', 'g_d2']; smallest-size set = ['g_d1']; highest C among those -> **g_d1**.
3. Frozen configuration of G2: d1 (see `kernels/gpu-compact/cfg.json`), E = 10 (52,450 steps, batch 32), seeds 0, 1, 2, 3, 4, trained on ALL 3,425 train tiles, last-step weights, AMP; the paper-recipe-like control (5 epochs = 22,215 steps) is the last job of the same queue (Amendment 5 note 1).
   `kernels/gpu-compact/script.py` sha256 6e8268003facabaa, CFG billed cap 16200 s, wall cap 8100 s, soft deadline 6900 s.
4. Test evaluations G2 will make (announced in TEST-LOG.md before this push): rows #22-#26 PyTorch fp32 (seeds 0-4), #27-#31 ONNX fp32, #32-#36 ONNX int8, #37 the control. No test number of any GPU model exists at the time of writing.
5. G2 timing gate (`scripts/g2_timing_gate.sh d1`) output is recorded in STATUS.md before the push; a FAIL means G2 is not pushed.
6. Notes at the time of writing (facts only).
   (a) G1 ran as planned and completed in full: kernel `bench-gas-gpu-wide` pushed 20:17:20 UTC (after repeated "Maximum batch GPU session count of 2" refusals while unrelated owner kernels [names redacted] held the sessions), RUNNING 20:17:52, COMPLETE before 21:02:15 UTC; `RUN_STATUS.json`: three runs rc 0, g_d3 on `CUDA_VISIBLE_DEVICES=0` 1,570 s, g_d2 on device 1 1,340 s, g_d1 on device 1 1,010 s (started when g_d2 ended), kernel time 2,392 s against the projected 2,966 s and the soft deadline of 3,300 s. No run was cut; no queue-rule re-push was needed for G1. The ledger books 1.35 billed h (2 GPUs x 2,427 s); the raw account counter moved 3.70 -> 4.66 h (+0.96 h, 1.4 x wall; it is lumpy and may include owner kernels).
   (b) Selection: the validation composite ranks d1 first (C = 63.92 against 62.24 for d2 and 61.12 for d3), so the band is {d1, d2} and the fewest-parameter member is d1, which is also the CPU-wave choice; the size-equality clause is not used. The validation slice has now served three selection stages (CPU dev wave, this one, and the G1 table); validation numbers are not comparable with test numbers.
   (c) G2 timing gate (`scripts/g2_timing_gate.sh d1`, timing only): "G2+G3 projected wall 4594 s (seed job 1015 s, control job 1964 s, slowdown vs plan x1.00) against the soft deadline 6900 s: PASS". G2 may be pushed.
   (d) Generated G2 kernel: 5 seeds of d1 (52,450 steps each) plus `ctrl_mnv2` (paper-recipe-like control, 22,215 steps = 5 control epochs, plateau check every 2,221 steps, validation dates for the schedule, `allow_test_with_val`) as the last job of the shared queue; edge stage (ONNX fp32 and static int8) for the five seeds only; internet on for the pip install of onnxruntime and onnxscript; datasets `chrisphamda/starcop` and `sankalpsthakur/bench-gas-assets`, kernel source `sankalpsthakur/bench-gas-prep-cache`. TEST-LOG rows #22-#37 apply unchanged (the control is row #37, run inside this kernel). The cap is still the pre-registered 8 billed hours (no "CAP" line in STATUS.md at the time of this commit).

## Amendment 6 (2026-10-18 22:51 UTC). Wave 3 on Colab, validation-gated. Written and committed BEFORE any Colab training run; no number of an E = 15 model exists at this time.

**Disclosure.** This wave was decided AFTER the GPU-wave result (Amendment 4b; weak F1 54.47 +- 2.08; paired against the released mag1c-only run +5.71 [-2.55, +19.20]; claim rule not met). It would be the third evaluation of the d1 architecture on the same 342 test tiles (CPU wave 2026-10-07, GPU wave 2026-10-17). The owner chose on 2026-10-19 to spend free Colab GPU time on it, with a validation gate first and route A for the data (his yes to the Kaggle credential on the Colab VM).

**Planning figure (computed before this amendment from the committed GPU-wave per-tile summaries, with the bootstrap draws of `scripts/aggregate.py`: B = 2,000, `default_rng(0)`):** the paired weak-F1 interval of the five-seed mean against the released mag1c-only run is [-2.55, +19.20] at 95 % (reproduces the published one) and [-3.88, +21.44] at 98.33 % (Bonferroni over three evaluations). Under a pure shift, the seed mean would need about +3.9 points (about 58.3) for the 98.33 % lower bound to clear zero. The CPU-to-GPU change (+11.77 weak F1 for 13.1 x the windows, also batch and AMP) suggests about +1.9 for 1.5 x the windows (rough). The odds of a claim are low, and were known before any run.

### 1. Platform and data
- Colab free tier, one Tesla T4, through the Colab CLI (google-colab-cli 0.7.4). Probe readback 2026-10-18 22:34 UTC (`colab new --gpu T4`, stopped at once): Tesla T4 15,360 MiB, 2 vCPUs, 12.7 GB RAM, 70.3 GB free disk, Python 3.13.15, torch 2.11.0+cu130. Every session records the same (`hw.json`) and stops unless exactly one T4 is visible (`scripts/colab/vm_setup.py`; CFG `max_gpus` 1).
- Route A: the verified train cache of `bench-gas-prep-cache` v1 is downloaded inside the VM with the Kaggle CLI 2.2.4. The credential is the short-lived OAuth access token of the owner's Kaggle login (the refresh token never leaves the Mac), uploaded as a file and deleted by `vm_setup.py` right after the download. The four files must equal the SHA-256 of `results/prep_cache/prep_report.json` (X 5e6f1a9a..., Y d55da93f..., W 0c553912..., meta 14ae7f44...), else the session stops.
- Layout as on Kaggle (`/kaggle/input/notebooks/sankalpsthakur/bench-gas-prep-cache/...`, `/kaggle/working`), so the generated scripts run unchanged. Kernels are generated by `scripts/mk_colab_wave3.py` through `scripts/push_train.py --no-push` from the GPU wave's definitions (`mk_gpu_wave.CANDS`, `COMMON`, `VAL`): same driver (`kernels/train_gpu/main.py`, byte-identical in the generated script), same `src` (unchanged since the GPU wave), one worker, CFG `parallel` 1. `tests/test_colab_wave3.py` pins the configurations. Orchestration: `scripts/colab_wave3.sh` and `scripts/colab/vm_*.py` (session, uploads, setup, one detached job at a time, polling every 300 s, small outputs downloaded, session stopped on exit).

### 2. Stage V: the gate (validation only; no test file exists on the VM)
- One run, `c_d1_e15_val` in `kernels/colab-gate` (script sha256 recorded in the commit): d1, seed 0, E = 15 = 78,675 steps at batch 32, the G1 recipe otherwise unchanged (AdamW lr 2e-3, wd 0, cosine with 200 warm-up steps, jitter 32, AMP, pos_weight 1), trained on train minus the validation dates 2019-10-22/23, final validation on the 524 validation tiles (`eval_every` = steps). It differs from G1's `g_d1` only in the step count.
- Gate statistic: the validation composite C (`scripts/select_config.composite`, FPR rule text10) of this run minus that of G1's `g_d1` (63.92, `protocol/selection_gpu.json`). **The gate passes if and only if the difference is at least +2.0** (the selection tolerance of Amendments 2c and 4). Recorded with it: the differences in validation weak F1, strong F1, tile FPR and AUPRC.
- If the gate fails, wave 3 ends; the test split is not read and Amendment 6a records it. If the gate passes, Amendment 6a records the gate numbers and is committed before any test file is fetched.

### 3. Stage T (only after a passed gate and a committed Amendment 6a)
- Five one-seed jobs `kernels/colab-test-s0` to `s4` (`mk_colab_wave3.py test`): d1, seeds 0-4, E = 15, all 3,425 train tiles, each scored ONCE on the 342 test tiles (PyTorch fp32, last-step weights): **TEST-LOG rows #38-#42, announced here**. No control and no ONNX edge stage; each seed's `model.pt` is kept for a later export.
- Test files: fetched inside the VM only after 6a is committed, from Zenodo (`STARCOP_test.zip`) with `starcop_edge.zen` range reads, every file CRC32-checked against `manifests/zenodo_STARCOP_test.tsv`, written into the mirror layout that `load_test_from_mirror` reads (which checks the CRC again). The fetch code is committed with 6a.
- **Claim rule for wave 3** (stricter than Amendment 4 item 1 because of the third look): "best published result on STARCOP test (paper-text protocol), self-evaluated" only if the five-seed mean weak F1 exceeds 47.57 AND the paired tile-bootstrap interval of the seed-mean difference against the released mag1c-only run at the 98.33 % level (percentiles 0.833 and 99.167, the `aggregate.py` draws) excludes zero. Reported whatever the outcome: all metrics, the 95 % and 98.33 % intervals, both metric variants; the qualifications of SUBMISSION 8.3 (tile FPR above both released runs, one released run, validation not comparable with test) travel with any claim. The GPU-wave (E = 10) result stays reported; wave 3 does not replace it.

### 4. Caps and retries (T4 wall time)
Gate session including setup: 1.5 h. Stage T: 5.0 h in total (about 35-45 min per seed at the probe rates, plus setup). A job lost to a Colab disconnect or preemption is re-run once with the same configuration, seed and steps (an infrastructure retry, logged with its cause); nothing changes after any test number is seen. No publication follows from this wave without a new owner decision.

### Note 1 to Amendment 6 (2026-10-18 22:55 UTC, before the gate is launched; nothing has run)
(a) An unfinished gate run (ended by the soft deadline, by a lost session, or with rc not 0) leaves no validation number. It is re-run once under item 4; if that run is unfinished too, wave 3 ends with the test split unread.
(b) Throughput: the crop-sampler rate on Colab's 2 vCPUs is unmeasured. From two consecutive progress polls the runner projects the job's finish; if the projection exceeds the job cap (5,400 s, the driver's soft deadline), it stops the job as an infrastructure stop. That decision uses throughput only, never a metric, and the stopped run counts as unfinished under (a).
(c) Caps: the 1.5 h of item 4 is the cap of the training job (the driver's soft deadline of 5,400 s); session setup (cache download and hash, about 10-20 min) is counted separately and logged. The runner waits up to 5,700 s so the driver can write its status after its deadline.
(d) Upload paths: the runner confirms that the kernel scripts landed under /content/w3 before it uploads the token, and vm_setup.py removes the token at /root/.kaggle and at /content/root/.kaggle, whichever exists. A failed output download keeps the session alive for a manual retry instead of stopping it.

### Note 2 to Amendment 6 (2026-10-18 22:57 UTC, before the gate is re-launched; no training has run)
The first gate session (22:55 UTC) stopped in setup before any download: on Colab, /kaggle/input is an empty read-only ext4 mount, so the cache directory could not be created there. The hardware readback had passed (one Tesla T4), the token file was removed from the VM, and the session was stopped. A CPU-only probe session (stopped at once) showed that /kaggle and /kaggle/working are writable and that root may bind-mount. vm_setup.py now bind-mounts a writable /content/kaggle_input over the empty /kaggle/input before the download; the driver, the generated scripts and src are unchanged. This is not a retry under item 4 (no job had started).

### Note 3 to Amendment 6 (2026-10-18 23:02 UTC, before the gate is re-launched; no training has run)
The second gate session (22:57-23:02 UTC) also stopped in setup: the bind mount worked and train_W.npy arrived with the right SHA-256, but the 7.18 GB train_X.npy download broke at 6.47 GB ("Connection broken: IncompleteRead", about 25 MB/s), so the hash check stopped the session as designed; the token file was removed and the session stopped. vm_setup.py now downloads the four files one at a time with up to four attempts each, checking the SHA-256 before every attempt. No job had started, so this is not a retry under item 4.

### Note 4 to Amendment 6 (2026-10-18 23:22 UTC, before the gate is re-launched; no training has run)
The third gate session (23:03-23:20 UTC) stopped in setup again: train_Y, train_W and train_meta arrived with the right SHA-256, but train_X.npy (7,182,745,728 B) failed four times (two broken connections at 4.8 and 6.7 GB, two processes killed, rc 137). The Kaggle CLI 2.2.4 holds a whole output file in memory (`out.write(download_response.content)`) and cannot resume. vm_setup.py now lists the signed output URLs through the same Kaggle API and streams each file to disk with `curl -C -` (resume from the last byte, up to eight attempts, a fresh URL each time, the SHA-256 checked once the size is complete). A local check resumed train_meta.csv from 100,000 B to its full size with the recorded SHA-256. No job had started, so this is not a retry under item 4.

### Note 5 to Amendment 6 (2026-10-18 23:44 UTC). The gate job was stopped by the throughput rule; no metric exists.
The fourth session (23:22-23:43 UTC) completed setup (all four cache files with the recorded SHA-256 through the resumable download; token removed) and started the gate job at 23:32 UTC. Training ran at 16.8-17.8 steps/s for the first 3,400 steps, then 10.3; two polls gave 9.2 steps/s, a projected 8,577 s against the 5,400 s cap, so the runner stopped the job at step 6,400 of 78,675 (note 1 b). The driver had written no validation number (log events "start" and "train" only; no val_metrics.json). Under note 1 (a) this run is unfinished and one re-run remains. The slowdown fits page-cache pressure (the 9.0 GB cache in 12.7 GB of RAM, the risk named in the 2026-10-10 Colab design) on top of the 2-vCPU sampler, so a re-run on the same free T4 would very likely be stopped the same way; a probe showed that the free tier refuses an L4. The owner is asked whether the remaining re-run moves to Kaggle (2 x T4, 4 vCPUs, where the GPU wave trained at about 60 steps/s) or wave 3 ends here with the test split unread. No decision depends on any metric.

### Note 6 to Amendment 6 (2026-10-19 16:12 UTC, before any Kaggle push of wave 3; no metric of an E = 15 model exists)
The owner chose on 2026-10-19 to run the remaining gate re-run (note 1 a) on Kaggle instead of the free Colab T4 (note 5). Throughput was the only reason.
(a) Platform: Kaggle's 2 x T4 shape (NvidiaTeslaT4, b = 2), as in the GPU wave, with the same generator definitions (`scripts/mk_colab_wave3.py kaggle-gate|kaggle-test` reuse `gate_run()` and `test_run(s)`), driver and src. Kernels: `kernels/gpu-gate3` (sankalpsthakur/bench-gas-gpu-gate3: the single gate run c_d1_e15_val on one worker; no dataset source, so no test file can be mounted; no internet) and, only after a passed gate and Amendment 6a, `kernels/gpu-test3` (the five seeds on two workers, the test split read from the CRC-verified mirror as in G2). The Colab kernels stay in the repository as the record of notes 1-5.
(b) Budget: wave 3 books its own ledger, `results/gpu_ledger_w3.csv` (GPUs x wall), separate from the closed GPU-wave budget of Amendment 5. Gate cap 6,000 billed s (wall 3,000 s, soft deadline 2,700 s); stage T cap 14,400 billed s (wall 7,200 s, soft deadline 6,600 s); total 5.67 billed h, inside the 7.0 h limit that `scripts/gpu_gate.sh` applies to that ledger. Pushes go through `scripts/gpu_launch.sh` with `LEDGER=results/gpu_ledger_w3.csv` (lock, gate, push with -t wall seconds, watcher). The read-only gate check passed before this note (quota 14.64 h remaining).
(c) Unchanged: the gate rule (dC >= +2.0 against G1's g_d1, `scripts/colab_gate_decision.py`), the claim rule at 98.33 %, TEST-LOG rows #38-#42 (their job column will name bench-gas-gpu-test3), and the retry rule. On Kaggle the driver's soft deadline replaces the Colab runner's throughput stop; this run is the one re-run of note 1 (a), so if it ends unfinished, wave 3 ends with the test split unread.

## Amendment 6a (2026-10-19 17:18 UTC). The gate FAILED; wave 3 ends with the test split unread.
- Gate run `c_d1_e15_val` (bench-gas-gpu-gate3 v1, Kaggle 2 x T4, one worker; pushed 16:34:40 UTC after the session limit freed; rc 0; 78,675 steps in 1,420 s; kernel 1,585 s). Validation on the same 524 tiles as G1 (`results/gpu_runs/gpu-gate3/`).
- `scripts/colab_gate_decision.py` (committed before the run): **verdict FAIL**. Composite C = 61.918 against G1's g_d1 63.925, **dC = -2.007** (the gate needed at least +2.0). Components, E = 15 minus E = 10: weak F1 +0.41 (78.47 vs 78.06), strong F1 -2.09 (66.68 vs 68.78), tile FPR (>10 px) +4.98 (64.77 vs 59.79, worse), AUPRC -1.36 (67.29 vs 68.65).
- Consequences, as pre-registered: no test file was mounted or fetched; TEST-LOG rows #38-#42 are closed as not run; `kernels/gpu-test3` is not generated; the GPU-wave (E = 10) result and its verdict ("not distinguishable from the published row") stand unchanged. A longer schedule did not help this architecture on validation.
- Cost: 0.90 billed GPU-h on the wave-3 ledger (2 x 1,619 s; the account counter moved 15.95 -> 16.76 h). The four Colab sessions used free-tier time only (no compute units).
