# TEST-LOG: every evaluation on the official STARCOP test split (342 tiles), including failed ones

Rule: no model of ours is tuned on test. Each line is one test evaluation. Times are UTC, 2026-10-07.

| # | time | kernel (version) | model / config | purpose | outcome |
|---|---|---|---|---|---|
| 1 | 03:30 | bench-gas-baseline v1 | released HyperSTARCOP mag1c+RGB | reproduction of the published evaluation | FAILED at startup (mount path assumption); no test data was read, no result |
| 2 | 03:41 | bench-gas-baseline v2 | released HyperSTARCOP mag1c+RGB (run R3), exact float TIFF inputs | reproduction (our scorer) | OK. F1 strong/weak (by qplume) 83.08/42.34, FPR(>10 px) 40.57, AUPRC(AP) 50.60: inside the published mean +- 1 std. The official-code cross-check inside v2 crashed on a torchmetrics shim, not on data |
| 3 | 04:05 | bench-gas-baseline v3 | released mag1c+RGB (again, identical numbers) and released mag1c-only (first time) | reproduction + the row holding the best published weak F1 | OK for our scorer: mag1c-only F1 strong/weak (qplume) 66.74/48.76, FPR(>10 px) 36.00, AUPRC 40.91. The official `run_validation` cross-check ran on all 342 tiles but crashed when saving (output dir not created), so its numbers were lost |
| 4a | 13:41-13:50 UTC (kernel window) | bench-gas-final-ref v1 | released mag1c+RGB, official `run_validation` | cross-check of our scorer | FAILED before any test tile was read by the official code: `ModuleNotFoundError: segmentation_models_pytorch` (the pip line of `official_check.py` lacked it; fixed in the src and re-run as v2). The kernel was otherwise COMPLETE |
| 4 | 13:51-14:05 UTC (kernel window) | bench-gas-final-ref v2 | released mag1c+RGB, official `run_validation` (commit c4789268) | cross-check of our scorer (redo of the official part of #3) | OK. official F1 strong(px)/weak(px) 84.23/39.02, FPR(640 px) 12.57, AUPRC(trapz16) 37.15; difference to our scorer at full precision: 0 on F1 strong/weak and FPR, 3.5e-07 on AUPRC trapz16 (the kernel's own abs_diff 4e-5 is the rounding of the reference values it was given) |
| 5 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s0 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 74.72/42.57, FPR(>10 px) 59.43, AUPRC 48.66 |
| 6 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s1 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 72.57/39.20, FPR(>10 px) 54.29, AUPRC 52.93 |
| 7 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s2 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 76.21/45.52, FPR(>10 px) 52.00, AUPRC 57.76 |
| 8 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s3 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 75.18/42.57, FPR(>10 px) 51.43, AUPRC 38.31 |
| 9 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s4 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 74.95/43.62, FPR(>10 px) 74.86, AUPRC 58.79 |
| 10 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s0 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 74.72/42.57, FPR(>10 px) 59.43, AUPRC 48.66 |
| 11 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s1 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 72.57/39.20, FPR(>10 px) 54.29, AUPRC 52.93 |
| 12 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s2 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 76.21/45.52, FPR(>10 px) 52.00, AUPRC 57.76 |
| 13 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s3 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 75.18/42.57, FPR(>10 px) 51.43, AUPRC 38.31 |
| 14 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s4 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 74.95/43.62, FPR(>10 px) 74.86, AUPRC 58.80 |
| 15 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s0 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 74.99/42.73, FPR(>10 px) 57.71, AUPRC 47.36 |
| 16 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s1 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 72.54/39.33, FPR(>10 px) 56.57, AUPRC 51.05 |
| 17 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s2 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 76.53/45.97, FPR(>10 px) 51.43, AUPRC 55.80 |
| 18 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s3 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 75.69/43.23, FPR(>10 px) 47.43, AUPRC 37.30 |
| 19 | 12:31-13:41 UTC (kernel window) | bench-gas-final v1 | f_d1_s4 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 74.75/43.21, FPR(>10 px) 76.00, AUPRC 55.81 |
| 20 | 13:41-13:50 UTC (kernel window) | bench-gas-final-ref v1 | released mag1c+RGB onnx fp32 (same export as ours) | size, latency and score of the leader under the identical edge pipeline | OK. F1 strong/weak (rate) 83.08/42.35, FPR(>10 px) 40.57, AUPRC 50.60 |
| 21 | 13:41-13:50 UTC (kernel window) | bench-gas-final-ref v1 | released mag1c+RGB onnx int8 (same export as ours) | size, latency and score of the leader under the identical edge pipeline | OK. F1 strong/weak (rate) 83.21/43.08, FPR(>10 px) 37.71, AUPRC 49.31 |

## GPU wave (Amendments 4, 4a, 4b, 5): rows #22-#37, announced on 2026-10-09 before any GPU kernel existed and filled in on 2026-10-17 (times UTC, 2026-10-17; the kernel window is the Kaggle run, push 21:04:30)

Same rules as above. `<cfg>` is the config frozen in Amendment 4b (chosen by the G1 validation rule). `bench-gas-gpu-wide` (G1) and `bench-gas-gpu-probe` (G0) never read the test split. Times are filled in by the lead from the kernel logs.

| # | time | kernel (version) | model / config | purpose | outcome |
|---|---|---|---|---|---|
| 22 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s0 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 86.24/57.46, FPR(>10 px) 49.14, AUPRC 68.87 |
| 23 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s1 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 83.06/53.81, FPR(>10 px) 50.86, AUPRC 54.99 |
| 24 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s2 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 81.11/53.28, FPR(>10 px) 49.14, AUPRC 47.32 |
| 25 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s3 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 82.91/55.60, FPR(>10 px) 50.29, AUPRC 56.82 |
| 26 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s4 PyTorch fp32, last-step weights | final test evaluation (one per seed) | OK. F1 strong/weak (rate) 80.59/52.19, FPR(>10 px) 53.14, AUPRC 51.70 |
| 27 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s0 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 86.24/57.46, FPR(>10 px) 49.14, AUPRC 68.87 |
| 28 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s1 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 83.06/53.81, FPR(>10 px) 50.86, AUPRC 54.99 |
| 29 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s2 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 81.11/53.28, FPR(>10 px) 49.14, AUPRC 47.32 |
| 30 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s3 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 82.91/55.60, FPR(>10 px) 50.29, AUPRC 56.82 |
| 31 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s4 onnx fp32 | edge variant, same test tiles | OK. F1 strong/weak (rate) 80.59/52.19, FPR(>10 px) 53.14, AUPRC 51.70 |
| 32 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s0 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 86.16/58.30, FPR(>10 px) 49.14, AUPRC 67.58 |
| 33 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s1 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 82.86/53.68, FPR(>10 px) 52.00, AUPRC 54.43 |
| 34 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s2 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 81.76/54.71, FPR(>10 px) 49.14, AUPRC 47.70 |
| 35 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s3 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 82.38/58.01, FPR(>10 px) 46.86, AUPRC 56.33 |
| 36 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 | gf_d1_s4 onnx int8 | edge variant, same test tiles | OK. F1 strong/weak (rate) 81.07/52.60, FPR(>10 px) 53.14, AUPRC 51.48 |
| 37 | about 21:08-22:23 UTC (kernel window) | bench-gas-gpu-compact v1 (control job) | ctrl_mnv2 PyTorch fp32, last-step weights (paper-recipe-like control, plateau on the validation dates) | control row for reading the compact results | OK. F1 strong/weak (rate) 71.83/54.42, FPR(>10 px) 46.29, AUPRC 56.18 |

Amendment 4b (2026-10-17 21:0x UTC, before the G2 push): `<cfg>` = **d1**, E = 10 (52,450 steps), seeds 0-4; the control (row #37) runs as the last job inside `bench-gas-gpu-compact` (Amendment 5 note 1), 5 control epochs = 22,215 steps, so row #37's kernel is `bench-gas-gpu-compact`, not `bench-gas-gpu-control`. Rows #22-#37 are otherwise as announced above.
