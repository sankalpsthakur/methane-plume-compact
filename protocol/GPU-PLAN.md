# GPU wave plan (step 5). Written 2026-10-07 ~10:18 UTC, before any GPU kernel exists. Nothing here has been pushed.

Gate: the account's GPU quota is exhausted until 2026-10-10 00:00 UTC (the owner's own work). No GPU kernel is pushed before then, and only after (a) the lead confirms free quota, (b) `mkdir locks/kaggle-gpu`
with the kernel ref and expected end in `owner`, (c) a row in RESOURCES.md. Gas cap: 8 GPU-hours for the whole week, one GPU session at a time. Kernels are private and named `bench-gas-gpu-*`.
All GPU kernels mount the same train cache as the CPU waves (`bench-gas-prep-cache`, must be COMPLETE) and read test from the CRC-verified mirror, exactly like `kernels/train/main.py`.

## Why a GPU wave
Every CPU number is trained with about 5 % of the paper's budget (8,000 steps x batch 16 = 128k crops vs the paper's 15 epochs = 78,675 steps x batch 32 = 2.5 M crops). The GPU wave asks the one question the
CPU wave cannot: do the compact models clear the 47.57 weak-plume F1 target once they get a paper-equivalent budget. CPU rows stay in the table as the 5 %-budget rows; GPU rows are labelled "paper-equivalent steps".
Neither set is dropped, and the headline is whichever pre-registered row the data supports, stated with its own protocol line.

## Stages (in this order; each stage's rule is fixed now)
| stage | kernel | what | cost model |
|---|---|---|---|
| G0 | `bench-gas-gpu-probe` (kernels/gpu-probe, no inputs) | synthetic-batch throughput of mnv2 (released architecture) and the compact family, AMP on/off | about 0.1 GPU-h |
| G1 wide (dev) | `bench-gas-gpu-wide` | validation-only runs, 1 seed each, 78,675 steps, batch 32, AdamW lr 2e-3 cosine, AMP: the CPU-selected config + **d3** (16..256, stem 2) + **d5** (4..64, stem 1) + **d2w** (8..128, stem 1). d3 and d5 are the Amendment 1 candidates deferred to the GPU; d2w is added here, now, before any GPU number exists | probe-based |
| G2 compact (final) | `bench-gas-gpu-compact` | the config chosen by the rule below, 5 seeds, all of train, last-step weights, test once per seed, ONNX fp32 + int8, same edge pipeline | 5 x steps / probe steps per s |
| G3 control | `bench-gas-gpu-control` | released architecture (mnv2, 6.6 M) with the paper recipe: Adam lr 1e-4, batch 32, windows without jitter, ReduceLROnPlateau x0.5 patience 4 checked twice per epoch, 15 epochs, trained on train minus the validation dates with the plateau on the validation dates (the public code monitors the test loss; we do not), one seed, test once. Only run if at least 2 GPU-h remain after G2 | 66,645 steps / probe steps per s |
Budget check before pushing anything: `python3 scripts/mk_gpu_wave.py plan --probe kernel_outputs/gpu-probe/gpuprobe.json --budget 8` (prints the GPU-h of each stage). If the total exceeds the cap, reduce `--steps`
for G1/G2 equally (never for one candidate only) and say so in the amendment; never lower the 47.57 target.

## Selection rule for G1 (same as the CPU rule, applied to the GPU dev runs)
C = mean(F1_strong_q, F1_weak_q, 100 - FPR_tile_text10, AUPRC_ap) on the validation dates; band = completed runs with C >= best - 2.0; fewest parameters in the band; ties in size -> highest C
(`scripts/select_config.py`, deterministic). The result is frozen in "Amendment 4" before G2 is pushed. The CPU-selected config is part of G1 so that the CPU and GPU choices are comparable.

## What is reported afterwards
- Per-seed and mean +- std (ddof=1) of F1 strong/weak, FPR, AUPRC with the tile-level bootstrap interval, parameters, fp32/int8 size, M4 latency, paired difference to the released checkpoint, for every pushed GPU kernel.
- Every test evaluation goes into TEST-LOG.md (G2: 5 PyTorch + 5 ONNX fp32 + 5 ONNX int8; G3: one). A GPU kernel that fails after reading test is logged too.
- Claim wording: "best published result on STARCOP test, protocol Z, self-evaluated, as of DATE" only if the 5-seed mean weak F1 exceeds 47.57 AND the paired bootstrap interval against the released mag1c-only row excludes zero;
  otherwise the result is reported as "not distinguishable from the published row" or as a negative. Weights are CC BY-NC 4.0 (data license), nothing is uploaded by the builder.
