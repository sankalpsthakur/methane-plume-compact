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

## Wave 3 on Colab (Amendment 6): rows #38-#42, announced on 2026-10-19 before any Colab training run

Rows #38-#42 exist only if the validation gate of Amendment 6 passes and Amendment 6a is committed; until then no test file is fetched. **Outcome (Amendment 6a): the gate failed, so none of these rows ran.** One row per seed (d1, E = 15, seeds 0-4, PyTorch fp32, last-step weights), each scored once.

| # | time | session / job | model / config | purpose | outcome |
|---|---|---|---|---|---|
| 38 | not run | (bench-gas-gpu-test3, never generated) | c_d1_e15_s0 | final test evaluation (one per seed) | NOT RUN: the validation gate failed (Amendment 6a, dC = -2.007); no test file was read |
| 39 | not run | (bench-gas-gpu-test3, never generated) | c_d1_e15_s1 | final test evaluation (one per seed) | NOT RUN: the validation gate failed (Amendment 6a, dC = -2.007); no test file was read |
| 40 | not run | (bench-gas-gpu-test3, never generated) | c_d1_e15_s2 | final test evaluation (one per seed) | NOT RUN: the validation gate failed (Amendment 6a, dC = -2.007); no test file was read |
| 41 | not run | (bench-gas-gpu-test3, never generated) | c_d1_e15_s3 | final test evaluation (one per seed) | NOT RUN: the validation gate failed (Amendment 6a, dC = -2.007); no test file was read |
| 42 | not run | (bench-gas-gpu-test3, never generated) | c_d1_e15_s4 | final test evaluation (one per seed) | NOT RUN: the validation gate failed (Amendment 6a, dC = -2.007); no test file was read |

## WV3 wave (Amendment 7): rows #43-#48, announced on 2026-10-19 (UTC; Amendment 7 commit 506d766 at 20:32 UTC) before any WV3 file was fetched or any WV3 model was trained. **Outcome (Amendment 7b, 2026-10-20 03:43 UTC): the reproduction gate FAILED (control AUPRC 9.43, band [11.08, 15.00]); rows #44-#48 did not run; stage B was never pushed.**

The simulated-WorldView-3 split, scored on the same 342 test tiles. Row #43 is the paper-recipe control (the reproduction gate of Amendment 7 item 6) inside stage A (`bench-gas-gpu-wv3a`); rows #44-#48 are the five seeds of the frozen candidate (chosen by the rule of item 7, same E as G1) inside stage B (`bench-gas-gpu-wv3b`): PyTorch fp32, last-step weights, one sealed evaluation each, no ONNX or int8 rows. Rows #44-#48 exist only if the gate passes and Amendment 7b is committed; otherwise they are closed as "not run". The cache kernel, G0 and the G1 runs never read the test split. Rows are filled by `scripts/testlog_rows_wv3.py` whatever their outcome.

| # | time | kernel (version) | model / config | purpose | outcome |
|---|---|---|---|---|---|
| 43 | see kernel log | bench-gas-gpu-wv3a v1 | ctrl_wv3 PyTorch fp32, last-step weights (paper-recipe control: released architecture, ImageNet encoder, 3 Varon+Sanchez channels, 15 epochs, plateau on the validation dates) | reproduction gate | OK. F1 strong/weak (rate) 17.02/18.18, FPR(>10 px) 39.43, AUPRC(AP) 9.43; reproduction gate FAIL: AUPRC 9.43 outside [11.08, 15.00] (published 13.04 +- 1.96, z -1.84; `protocol/wv3_gate.json`, Amendment 7b); the only test read of this wave |
| 44 | not run | (bench-gas-gpu-wv3b, never pushed) | wf_<cfg>_s0 | final test evaluation (one per seed) | NOT RUN: the reproduction gate failed (Amendment 7 item 6, 7b); no candidate was selected and no test file was read for it |
| 45 | not run | (bench-gas-gpu-wv3b, never pushed) | wf_<cfg>_s1 | final test evaluation (one per seed) | NOT RUN: the reproduction gate failed (Amendment 7 item 6, 7b); no candidate was selected and no test file was read for it |
| 46 | not run | (bench-gas-gpu-wv3b, never pushed) | wf_<cfg>_s2 | final test evaluation (one per seed) | NOT RUN: the reproduction gate failed (Amendment 7 item 6, 7b); no candidate was selected and no test file was read for it |
| 47 | not run | (bench-gas-gpu-wv3b, never pushed) | wf_<cfg>_s3 | final test evaluation (one per seed) | NOT RUN: the reproduction gate failed (Amendment 7 item 6, 7b); no candidate was selected and no test file was read for it |
| 48 | not run | (bench-gas-gpu-wv3b, never pushed) | wf_<cfg>_s4 | final test evaluation (one per seed) | NOT RUN: the reproduction gate failed (Amendment 7 item 6, 7b); no candidate was selected and no test file was read for it |

## Retroactive rows #49-#51: the plume operating-budget audit (another workstream), read the test split on 2026-10-11 UTC; logged retroactively on 2026-11-18 (UTC)

The audit (appendix/operating_budget/) read the full STARCOP test split (all 342 tiles, labels and the source TIFFs) in one private Kaggle CPU kernel, `industrial-plume-budget-cpu` v1 (kernel window 16:51-16:57 UTC on 2026-10-11, 386 s wall, 2 CPU threads; source SHA-256 c763cd2b..., output archive SHA-256 1ae0d9eb...; protocol frozen at source commit 3be30cd before the run). It scored the RELEASED references only, at three calibrated tile-indication budgets (1 %, 5 %, 10 %, threshold calibrated on the mask-negative tiles of the other two acquisition dates) plus one fixed diagnostic threshold. It was run by another workstream, not by this track's scripts: logged retroactively on 2026-11-18; run by another workstream; no model of this track was selected on it. No model of this track (methane-plume-compact included) was scored under it, and none is to be: that would be a further look at the test split. The same kernel's per-tile standard-rule counts for the two learned references equal the per-tile summaries of rows #2 and #3 on all 342 tiles (`appendix/operating_budget/recompute_report.txt`), so the rows add no new information about the released models' pixel predictions; the new content is the calibrated operating-point and localisation view.

| # | time | kernel (version) | model / config | purpose | outcome |
|---|---|---|---|---|---|
| 49 | 2026-10-11 16:51-16:57 UTC (kernel window; see above) | industrial-plume-budget-cpu v1 (other workstream) | released mag1c+RGB checkpoint (6,629,233 parameters), fp32 CPU dense logits, no new training | operating-budget audit: per-date negative-tile indication budget, localised weak recall, weak mean IoU | OK. Fails every bar. 5 % rule: negative-tile indications 13/175, worst date 14.29 %; weak localised recall 36/108 = 33.33 %; weak mean IoU 0.1414 (bars: at most 5 % on every date, at least 80 %, at least 0.25). Weak = 108 tiles in this audit (mask positive and 0 < qplume < 1000), not the 110 of this track |
| 50 | 2026-10-11 16:51-16:57 UTC (same kernel run) | industrial-plume-budget-cpu v1 (other workstream) | released mag1c-only checkpoint (6,628,369 parameters), same settings | same | OK. Fails every bar. 5 % rule: 9/175, worst date 11.43 %; weak localised recall 30/108 = 27.78 %; weak mean IoU 0.1145 |
| 51 | 2026-10-11 16:51-16:57 UTC (same kernel run) | industrial-plume-budget-cpu v1 (other workstream) | raw mag1c (no weights; fixed 200 ppm.m diagnostic threshold, calibrated thresholds otherwise) | same | OK. Fails every bar. 5 % rule: 7/175, worst date 7.14 %; weak localised recall 0/108; weak mean IoU 0.0007 |
