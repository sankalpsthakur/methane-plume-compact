# STARCOP test results (342 tiles): CPU wave (5 % budget) and GPU wave (13x the crops)

Two sets of rows, neither dropped (Amendment 4 item 1). The first table below and everything down to "GPU wave" is the CPU wave of 2026-10-07 (frozen d1, 8,000 steps x batch 16 = 0.128 M crops, `scripts/finalize.py`, unchanged). The GPU wave of 2026-10-17 (same d1, 52,450 steps x batch 32 = 1.678 M crops, AMP, 5 seeds, Amendments 4a/4b/5) follows below; its generated tables are `results/gpu_final_TABLE.md`. Protocol for every row: strong/weak by emission rate, tile positive above 10 predicted pixels, AUPRC = average precision, last-step weights, p > 0.5, mean +- sample std over seeds [tile-bootstrap 95 % interval of the seed-mean, 2,000 draws].

## CPU wave (2026-10-07)

| model (test, 342 tiles) | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|
| ours, PyTorch fp32, 5-seed mean +- std [tile-bootstrap 95 % of the mean] | 74.73 +- 1.34 [58.0, 85.6] | 42.70 +- 2.30 [21.3, 68.2] | 58.40 +- 9.73 [52.3, 64.6] | 51.29 +- 8.31 [34.4, 67.8] |
| ours, ONNX fp32, 5-seed mean +- std [tile-bootstrap 95 % of the mean] | 74.73 +- 1.34 [58.0, 85.6] | 42.70 +- 2.30 [21.3, 68.2] | 58.40 +- 9.73 [52.3, 64.6] | 51.29 +- 8.31 [34.4, 67.8] |
| ours, ONNX int8 (static QDQ), 5-seed mean +- std [tile-bootstrap 95 % of the mean] | 74.90 +- 1.49 [58.1, 85.7] | 42.89 +- 2.37 [21.3, 68.5] | 57.83 +- 10.96 [51.9, 64.0] | 49.47 +- 7.67 [32.7, 66.2] |
| released mag1c+RGB (run R3), one run (our scorer) | 83.08 | 42.34 | 40.57 | 50.60 |
| released mag1c-only, one run (our scorer) | 66.74 | 48.76 | 36.00 | 40.91 |
| published mag1c+RGB (Sci. Rep. 13:19999 Table 2, 5-run mean +- std) | 81.96 +- 3.71 | 43.42 +- 5.72 | 43.66 +- 7.36 | 51.99 +- 2.76 |
| published mag1c-only (same table) | 74.15 +- 6.10 | 47.57 +- 4.17 | 52.11 +- 10.98 | 49.41 +- 5.49 |

Paired difference (ours - released, same 342 tiles, same bootstrap draws; interval = tile-bootstrap 95 % of the difference of the seed-mean and the single released run):

| ours | reference | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|---|
| pytorch_fp32 | released mag1c+RGB (run R3) | -8.35 [-19.0, -0.3] | +0.36 [-6.3, +9.7] | +17.83 [+12.2, +23.7] | +0.69 [-17.1, +14.1] |
| pytorch_fp32 | released mag1c-only | +7.98 [-1.0, +15.1] | -6.07 [-12.8, +4.6] | +22.40 [+16.6, +28.2] | +10.38 [-7.0, +20.8] |
| onnx_fp32 | released mag1c+RGB (run R3) | -8.35 [-19.0, -0.3] | +0.36 [-6.3, +9.7] | +17.83 [+12.2, +23.7] | +0.69 [-17.1, +14.1] |
| onnx_fp32 | released mag1c-only | +7.98 [-1.0, +15.1] | -6.07 [-12.8, +4.6] | +22.40 [+16.6, +28.2] | +10.38 [-7.0, +20.8] |
| onnx_int8 | released mag1c+RGB (run R3) | -8.18 [-18.8, -0.3] | +0.55 [-6.3, +10.0] | +17.26 [+11.5, +23.1] | -1.13 [-18.7, +12.4] |
| onnx_int8 | released mag1c-only | +8.16 [-0.7, +15.2] | -5.87 [-12.6, +5.0] | +21.83 [+16.1, +27.5] | +8.55 [-8.6, +19.3] |

Per-seed values (PyTorch fp32):

| seed | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|
| 0 | 74.72 | 42.57 | 59.43 | 48.66 |
| 1 | 72.57 | 39.20 | 54.29 | 52.93 |
| 2 | 76.21 | 45.52 | 52.00 | 57.76 |
| 3 | 75.18 | 42.57 | 51.43 | 38.31 |
| 4 | 74.95 | 43.62 | 74.86 | 58.80 |

Size and edge cost:

| model | params | PyTorch state (KB) | ONNX fp32 (KB) | ONNX int8 (KB) | Kaggle EPYC 1-thread median ms fp32 / int8 | M4 1-thread median ms fp32 / int8 |
|---|---|---|---|---|---|---|
| f_d1_s0 | 123,649 | 531 | 506 | 175 | 39 / 61 | 15.65 / 18.24 |
| f_d1_s1 | 123,649 | 531 | 506 | 175 | 38 / 60 | not measured / not measured |
| f_d1_s2 | 123,649 | 531 | 506 | 175 | 40 / 62 | not measured / not measured |
| f_d1_s3 | 123,649 | 531 | 506 | 175 | 38 / 97 | not measured / not measured |
| f_d1_s4 | 123,649 | 531 | 506 | 175 | 38 / 62 | not measured / not measured |
| released_hyperstarcop_mag1c_rgb | 6,629,233 | 26662 | 26497 | 7074 | 334 / 261 | not measured / not measured |

## GPU wave (2026-10-17): frozen d1, E = 10 paper epochs = 52,450 steps x batch 32, AMP, seeds 0-4 (Amendment 4b), and the CPU-vs-GPU summary

Side by side (PyTorch fp32 unless stated; TEST-LOG #5-#9 for the CPU rows, #22-#26 for the GPU rows; ONNX int8 = #32-#36; control = #37):

| model (test, 342 tiles) | training | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|---|
| ours, d1 (123,649 parameters), CPU wave, 5-seed | 8,000 x 16 = 0.128 M crops | 74.73 +- 1.34 [58.0, 85.6] | 42.70 +- 2.30 [21.3, 68.2] | 58.40 +- 9.73 [52.3, 64.6] | 51.29 +- 8.31 [34.4, 67.8] |
| **ours, d1, GPU wave, 5-seed** | 52,450 x 32 = 1.678 M crops | **82.78 +- 2.21 [70.8, 89.3]** | **54.47 +- 2.08 [30.1, 78.2]** | **50.51 +- 1.65 [43.7, 57.1]** | **55.94 +- 8.08 [39.1, 74.9]** |
| ours, d1, GPU wave, ONNX int8 (static QDQ, 175 KB), 5-seed | same weights | 82.85 +- 1.97 [70.8, 89.1] | 55.46 +- 2.57 [31.3, 78.6] | 50.06 +- 2.51 [43.3, 56.6] | 55.50 +- 7.50 [38.3, 74.7] |
| control: released architecture (6,629,233 parameters), paper-recipe-like, 1 run (not a reproduction) | 22,215 x 32 = 0.711 M crops (5 of the paper's 15 epochs) | 71.83 [57.8, 81.5] | 54.42 [45.5, 63.2] | 46.29 [38.8, 54.0] | 56.18 [41.6, 70.9] |
| released mag1c+RGB (run R3), one run, our scorer | paper recipe, 15 epochs | 83.08 | 42.34 | 40.57 | 50.60 |
| released mag1c-only, one run, our scorer | paper recipe, 15 epochs | 66.74 | 48.76 | 36.00 | 40.91 |
| published mag1c+RGB (Sci. Rep. 13:19999 Table 2, 5-run mean +- std) | | 81.96 +- 3.71 | 43.42 +- 5.72 | 43.66 +- 7.36 | 51.99 +- 2.76 |
| published mag1c-only (same table; the target row) | | 74.15 +- 6.10 | **47.57 +- 4.17** | 52.11 +- 10.98 | 49.41 +- 5.49 |

GPU minus CPU, same architecture (13.1x the crops, batch 32 instead of 16, AMP): strong +8.05 (tile-bootstrap intervals CPU [58.0, 85.6], GPU [70.8, 89.3]), weak +11.77 (CPU [21.3, 68.2], GPU [30.1, 78.2]), FPR -7.89 (CPU [52.3, 64.6], GPU [43.7, 57.1]), AUPRC +4.65 (CPU [34.4, 67.8], GPU [39.1, 74.9]). The differences are larger than the seed std of either wave (weak F1: 2.30 and 2.08), but the tile-bootstrap intervals of the two waves overlap on every column and no paired CPU-vs-GPU interval was pre-registered, so they are reported as point estimates and not claimed as a win (protocol rule 3).

**Claim under the pre-registered rule (Amendment 4 item 1).** The 5-seed mean weak F1 54.47 exceeds the target 47.57 by 6.90 points (1.7 published std). The second condition of the rule, a paired tile-bootstrap interval against the released mag1c-only run that excludes zero, is NOT met: +5.71 [-2.5, +19.2]. So the result is reported as "not distinguishable from the published row", and no "best published" claim is made. Against the released mag1c+RGB run R3 the weak-F1 difference is +12.13 [+3.8, +25.3] (that row is not the rule's comparator). Other columns, same scorer: strong F1 vs R3 -0.30 [-5.7, +4.0] and vs released mag1c-only +16.04 [+6.6, +25.6]; AUPRC vs R3 +5.34 [-7.0, +16.8] and vs mag1c-only +15.03 [+2.3, +24.1]; tile-level FPR is WORSE than both released runs under our scorer, +9.94 [+4.5, +15.2] vs R3 and +14.51 [+8.3, +20.6] vs mag1c-only (it is level with the published mag1c-only mean, 52.11 +- 10.98, and above the published mag1c+RGB mean, 43.66 +- 7.36): the weak-F1 gain comes with a worse tile-level false-alarm rate than the released runs.

Official-code variants of the same predictions (label-pixel grouping, 640-pixel tile rule, 16-threshold AUPRC; labelled as in Amendment 1 item 2), GPU wave PyTorch fp32 5-seed mean +- std: F1 strong (px) 83.91 +- 2.15, F1 weak (px) 49.88 +- 2.24, FPR tile (>640 px) 11.31 +- 0.63, AUPRC (trapz16) 56.16 +- 8.24. Same variants, CPU wave: 77.60 +- 1.30, 35.54 +- 2.36, 19.20 +- 1.65, 53.95 +- 3.72. Released mag1c-only: 66.74, 47.51, 9.14, 39.00; released R3: 84.23, 39.02, 12.57, 37.15. (Our scorer equals the official `run_validation` on the released checkpoint: TEST-LOG #4, `results/official_check.json`; the scorer is unchanged, src hash 54170b5e7b1e2859.)

Control row against the released runs (paired, tile bootstrap, one run): weak F1 +5.66 [-7.9, +17.7] vs mag1c-only and +12.08 [-5.8, +24.5] vs R3; strong F1 +5.09 [-17.1, +26.5] vs mag1c-only and -11.25 [-25.2, -0.4] vs R3; FPR +10.29 [+3.4, +17.4] vs mag1c-only and +5.71 [-0.6, +12.4] vs R3. The control trained for 5 of the paper's 15 epochs (the pre-registered rule gave Ec = 5 because its AMP rate on the T4 is 18.7 steps/s) and carries no claim; it shows that the 53.6x smaller d1 matches the released architecture on weak F1 (54.47 vs 54.42) at this budget, is far ahead on strong F1 (82.78 vs 71.83) and has a higher tile-level FPR (50.51 vs 46.29).

Billed GPU time of the wave (Amendment 5 convention, GPUs x wall): G0 0.05 h + G1 1.35 h + G2 + control 2.49 h = 3.89 h of the 8.0 h cap (raw account counter 2.22 h for the same sessions; see RESOURCES.md).

### Generated GPU tables (`results/gpu_final_TABLE.md`; the "Kaggle EPYC" column header is the finalize.py text, in the GPU image the CPU is an Intel Xeon @ 2.00 GHz)

| model (test, 342 tiles) | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|
| ours, PyTorch fp32, 5-seed mean +- std [tile-bootstrap 95 % of the mean] | 82.78 +- 2.21 [70.8, 89.3] | 54.47 +- 2.08 [30.1, 78.2] | 50.51 +- 1.65 [43.7, 57.1] | 55.94 +- 8.08 [39.1, 74.9] |
| ours, ONNX fp32, 5-seed mean +- std [tile-bootstrap 95 % of the mean] | 82.78 +- 2.21 [70.8, 89.3] | 54.47 +- 2.08 [30.1, 78.2] | 50.51 +- 1.65 [43.7, 57.1] | 55.94 +- 8.08 [39.1, 74.9] |
| ours, ONNX int8 (static QDQ), 5-seed mean +- std [tile-bootstrap 95 % of the mean] | 82.85 +- 1.97 [70.8, 89.1] | 55.46 +- 2.57 [31.3, 78.6] | 50.06 +- 2.51 [43.3, 56.6] | 55.50 +- 7.50 [38.3, 74.7] |
| released mag1c+RGB (run R3), one run (our scorer) | 83.08 | 42.34 | 40.57 | 50.60 |
| released mag1c-only, one run (our scorer) | 66.74 | 48.76 | 36.00 | 40.91 |
| published mag1c+RGB (Sci. Rep. 13:19999 Table 2, 5-run mean +- std) | 81.96 +- 3.71 | 43.42 +- 5.72 | 43.66 +- 7.36 | 51.99 +- 2.76 |
| published mag1c-only (same table) | 74.15 +- 6.10 | 47.57 +- 4.17 | 52.11 +- 10.98 | 49.41 +- 5.49 |

Paired difference (ours - released, same 342 tiles, same bootstrap draws; interval = tile-bootstrap 95 % of the difference of the seed-mean and the single released run):

| ours | reference | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|---|
| pytorch_fp32 | released mag1c+RGB (run R3) | -0.30 [-5.7, +4.0] | +12.13 [+3.8, +25.3] | +9.94 [+4.5, +15.2] | +5.34 [-7.0, +16.8] |
| pytorch_fp32 | released mag1c-only | +16.04 [+6.6, +25.6] | +5.71 [-2.5, +19.2] | +14.51 [+8.3, +20.6] | +15.03 [+2.3, +24.1] |
| onnx_fp32 | released mag1c+RGB (run R3) | -0.30 [-5.7, +4.0] | +12.13 [+3.8, +25.3] | +9.94 [+4.5, +15.2] | +5.34 [-7.0, +16.8] |
| onnx_fp32 | released mag1c-only | +16.04 [+6.6, +25.6] | +5.71 [-2.5, +19.2] | +14.51 [+8.3, +20.6] | +15.03 [+2.3, +24.1] |
| onnx_int8 | released mag1c+RGB (run R3) | -0.23 [-5.3, +4.0] | +13.12 [+4.8, +25.9] | +9.49 [+3.9, +14.8] | +4.90 [-7.3, +16.3] |
| onnx_int8 | released mag1c-only | +16.10 [+6.6, +25.9] | +6.70 [-1.6, +19.7] | +14.06 [+7.8, +20.2] | +14.59 [+2.0, +23.6] |

Per-seed values (PyTorch fp32):

| seed | F1 strong | F1 weak | FPR tile (>10 px) | AUPRC |
|---|---|---|---|---|
| 0 | 86.24 | 57.46 | 49.14 | 68.87 |
| 1 | 83.06 | 53.81 | 50.86 | 54.99 |
| 2 | 81.11 | 53.28 | 49.14 | 47.32 |
| 3 | 82.91 | 55.60 | 50.29 | 56.82 |
| 4 | 80.59 | 52.19 | 53.14 | 51.70 |

Size and edge cost:

| model | params | PyTorch state (KB) | ONNX fp32 (KB) | ONNX int8 (KB) | Kaggle EPYC 1-thread median ms fp32 / int8 | M4 1-thread median ms fp32 / int8 |
|---|---|---|---|---|---|---|
| gf_d1_s0 | 123,649 | 532 | 506 | 175 | 33 / 55 | 18.41 / 18.76 |
| gf_d1_s1 | 123,649 | 532 | 506 | 175 | 32 / 58 | not measured / not measured |
| gf_d1_s2 | 123,649 | 532 | 506 | 175 | 34 / 54 | not measured / not measured |
| gf_d1_s3 | 123,649 | 532 | 506 | 175 | 32 / 56 | not measured / not measured |
| gf_d1_s4 | 123,649 | 532 | 506 | 175 | 33 / 55 | not measured / not measured |
| released_hyperstarcop_mag1c_rgb | 6,629,233 | 26662 | 26497 | 7074 | 334 / 261 | not measured / not measured |
