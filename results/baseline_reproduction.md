# Reproduction of the published HyperSTARCOP evaluation (released checkpoints, official test split)

Kernel `bench-gas-baseline` (private CPU kernel; v2 and v3, 2026-10-07). Inputs: the 342 official test tiles read from the Kaggle mirror
with every file CRC32-verified against the Zenodo archive; exact float inputs (no fp16 cache), official normalisation constants
(read from the checkpoint's `normalizer.*` buffers: factors 1750/60/60/60, clip [0, 2]). Models: `isp-uv-es/starcop` on Hugging Face
(CC BY-NC 4.0), loaded strictly into our dependency-free re-implementation (`starcop_edge.models.SMPLikeUnet`), which is bit-identical to
`segmentation_models_pytorch` 0.5.0 on the same weights (max abs logit difference 0.0 on 8 real tiles).

Each checkpoint is ONE training run (the mag1c+RGB file is experiment `f4_hyper_unetsempos1all_all_15ep_R3`, run 3 of 5), so it is compared to the
paper's 5-run mean +- std (Table 2), not to the mean itself. 95% intervals are tile-level bootstrap (2000 draws) and show how little the 342-tile
test split resolves: single-model differences of a few F1 points are inside the sampling noise.

| metric (variant) | released mag1c+RGB | published mag1c+RGB | released mag1c-only | published mag1c-only |
|---|---|---|---|---|
| F1 strong, by emission rate >= 1000 kg/h (paper text) | 83.08 [73.1, 88.2] | 81.96 +- 3.71 | 66.74 [48.0, 78.6] | 74.15 +- 6.10 |
| F1 weak, by emission rate < 1000 kg/h (paper text) | 42.34 [24.6, 64.3] | 43.42 +- 5.72 | 48.76 [30.9, 66.4] | 47.57 +- 4.17 |
| FPR by tile, positive if > 10 predicted px (paper text) | 40.57 [33.0, 47.5] | 43.66 +- 7.36 | 36.00 [28.8, 42.8] | 52.11 +- 10.98 |
| AUPRC, pooled-pixel average precision | 50.60 [25.5, 78.7] | 51.99 +- 2.76 | 40.91 [23.5, 65.4] | 49.41 +- 5.49 |
| *official-code variant* F1 strong, > 1000 label px | 84.23 | (same column) | 66.74 | |
| *official-code variant* F1 weak, <= 1000 label px | 39.02 | | 47.51 | |
| *official-code variant* FPR by tile, > 10*H*W/64^2 = 640 px | 12.57 | | 9.14 | |
| *official-code variant* AUPRC, trapezoid over the 16 thresholds | 37.15 | | 39.00 | |

Outcome: the released mag1c+RGB checkpoint lands inside the published mean +- 1 std on all four headline metrics under the paper-text
variants, so the evaluation is reproduced and the gate to training is passed. The released mag1c-only checkpoint is inside 1.5 std on all
four (weak F1 inside 1 std); it is a single run that happens to be conservative (lower recall, fewer false-positive tiles).

Inconsistencies found while reproducing (reported for the corrected-protocol write-up):
1. The public code's tile rule is `> 10*H*W/64^2` predicted pixels (= 640 px for 512x512), the paper text says "more than 10 active pixels".
   The published FPR (43.7) matches the 10-px rule (40.6 for the released run), not the code (12.6).
2. The public `run_validation` splits strong/weak by label pixel count (> 1000 px); the paper and Zenodo description split by emission rate
   (>= 1000 kg/h). The two groupings move 40 of the 166 plume tiles between groups (7 strong by pixels are weak by rate, 33 weak by pixels are strong
   by rate), and 3 tiles disagree on plume vs no-plume (2 have label pixels but qplume 0; 1 has qplume > 0 but no label pixel).
3. One sentence in the paper's dataset section swaps strong and weak ("lower than 1000 kg/h are labelled strong"); the Metrics section is correct.
4. The paper says Adam lr 0.001; the released config says 1e-4, and the released checkpoint's optimizer state (lr 2.5e-5 after two halvings
   of 1e-4) agrees with the config.
5. In the public training script the validation set is the test set (`self.val_dataset = self.test_dataset`), and the ReduceLROnPlateau schedule
   (the two halvings above) monitors `val_loss`, i.e. test loss. The released model is the last-epoch checkpoint (not the best-test-loss one), so the
   leak is limited to the LR schedule, but a corrected protocol must not do this.
6. The public `run_validation` crashes on recent torchmetrics (`ConfusionMatrix(num_classes=2)` without `task`) and does not create its output
   directory; the official-code cross-check inside our kernel therefore needed a shim and one retry (see TEST-LOG).
