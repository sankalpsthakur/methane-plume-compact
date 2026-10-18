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
