# WV3 bars: what the paper publishes for the simulated-WorldView-3 split, verified, and what the release holds for it

Prepared 2026-10-19/20 for Amendment 7 (`protocol/PREREGISTRATION.md`). Machine-readable copy of the bars: `protocol/wv3_published.json` (the single source for `scripts/wv3_gate_decision.py` and `scripts/wv3_claim.py`).
Inventory behind section 6: `scripts/wv3_manifest_audit.py` -> `results/wv3_manifest_audit.json` (manifests only; no STARCOP data was downloaded, no band of any tile was decoded by the builder).

## 1. Sources and how they were read
| source | what was used | note |
|---|---|---|
| Ruzicka, Mateo-Garcia, Gomez-Chova, Vaughan, Guanter, Markham, *Semantic segmentation of methane plumes with hyperspectral machine learning models*, Sci. Rep. 13:19999 (2023), open access: https://www.nature.com/articles/s41598-023-44918-6 = PMC10656523 | full text, Table 1 (multispectral, WV3), Table 2 (AVIRIS), section "Metrics", "Machine learning models", "Experimental setup" | The Nature page answers a plain fetch with a 303 redirect to an identity-provider cookie flow (not followed), and the PMC page with a browser-verification screen. The text was therefore read from the Europe PMC REST full-text XML of the same article (https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10656523/fullTextXML, fetched 2026-10-19, 159 kB). Quotes below are kept short; everything else is paraphrased |
| Zenodo record 10.5281/zenodo.7863343 (v1.0, CC BY-NC 4.0), REST metadata | file list and zip sizes, description text | metadata only |
| Hugging Face `isp-uv-es/starcop` (API listing, README) | which checkpoints exist | metadata only |
| github.com/spaceml-org/STARCOP README | training instructions, links | documentation only; the source code (GPL-3.0) was not read |
| `manifests/*.tsv`, `data_meta/*.csv`, `results/prep_cache/prep_report.json` (this repository) | data inventory, rates | |

## 2. Table 1 of the paper (WV3 rows), mean +- sd of 5 training runs
| model (input channels, ImageNet-pretrained MobileNetV2 U-Net, 6.6 M parameters) | F1 strong | F1 weak | FPR by tile (lower is better) | AUPRC |
|---|---|---|---|---|
| Baseline, ratio + morphology (Sanchez ratio B8-B8_MLR, threshold 0.05, opening) | 7.44 | 0.5 | 100.0 | N/A |
| Our (Varon): B7-B5, B8-B5, B7-B6 | 30.72 +- 2.87 | 10.35 +- 1.52 | 87.89 +- 4.67 | 11.92 +- 1.35 |
| Our (Sanchez): B7-B7_MLR, B8-B8_MLR, band B1 | 26.59 +- 3.13 | 9.32 +- 1.05 | 94.4 +- 1.30 | 9.96 +- 1.43 |
| Our (Varon+Sanchez): B7-B5, B8-B5, B7-B7_MLR | **31.89 +- 2.44** | **11.04 +- 0.75** | 90.51 +- 4.23 | **13.04 +- 1.96** |

Caption (paraphrased): results of the multispectral models on the test set against the baseline, mean of 5 training runs of each model. The paper does not define the "+-" symbol in the caption; its text speaks of results "within the range of standard deviation", so it is read as the standard deviation over the 5 runs.
Bold = best per column in the paper, except FPR: the lowest FPR (best) is the **Varon** row, 87.89 +- 4.67, not the Varon+Sanchez row (90.51 +- 4.23). The three other best values all belong to the Varon+Sanchez row, which is therefore the comparator of Amendment 7.

## 3. The scout's claims, checked
| scout claim | verdict | detail |
|---|---|---|
| best per column: strong F1 31.89 +- 2.44, weak F1 11.04 +- 0.75, FPR 87.89 +- 4.67, AUPRC 13.04 +- 1.96 | **confirmed** (numbers), with a caveat | the FPR belongs to a different row (Varon) than the other three (Varon+Sanchez, FPR 90.51 +- 4.23). A model cannot "beat all four best-per-column values" with one published row; the gate and the claim use the Varon+Sanchez row, FPR is reported against both |
| deterministic baseline row strong F1 7.44 (weak 0.5, FPR 100: degenerate) | numbers **confirmed**, "degenerate" is the scout's reading | the paper prints 7.44 / 0.5 / 100.0 / N/A and calls nothing degenerate. FPR 100 means every no-plume tile is flagged; with a pixel-level F1 of 7.44 on strong plumes that is what a map that is positive almost everywhere gives. Plausible, not stated |
| published WV3 models used three ratio channels on ImageNet-pretrained inputs | **confirmed** | the paper uses an ImageNet-pretrained MobileNetV2 encoder for MultiSTARCOP, which restricts the input to 3 bands; inputs are the ratio products of the table, min-max normalised with training-set statistics |
| no released WV3 checkpoint exists | **confirmed** | HF `isp-uv-es/starcop` (last modified 2023-12-11) holds `models/hyperstarcop_mag1c_only` and `models/hyperstarcop_mag1c_rgb` (checkpoint + config each) and two table images; its README lists those two models only. The paper's data statement promises "the code and the pre-trained models"; the WV3 ones are not among them |
| `TOA_WV3_SWIR1-8` files are in every test and train archive | **confirmed** | section 6: 3,425 of 3,425 train tiles and 342 of 342 test tiles have all eight files in the Zenodo central directories; none lacks one |
| the Kaggle mirror holds 3,194 of 3,767 **train** tiles | **number right, label wrong** | 3,767 = 3,425 train + 342 test tiles. The mirror holds **2,852 of 3,425 train tiles** (all archives but `STARCOP_train_remaining_part2`, 573 tiles) plus all 342 test tiles = 3,194. Already in `results/prep_cache/prep_report.json` (mirror 2,852, Zenodo 573) |
| part2's 573 tiles need about 4.4 GB of Zenodo range reads for 8 channels | **confirmed** | 4.351 GB of uncompressed WV3 members (4,584 files); the zip holds them at 0.961 of that, so about 4.18 GB are actually read |

## 4. What the paper states about the WV3 models (and what it leaves open)
Stated: U-Net with MobileNetV2 encoder, ImageNet-pretrained for the WV3 models, trained on the 3 ratio channels of the chosen variant; per-tile ratio products (the MLR of the Sanchez ratio is fitted on one tile at a time, regressing the signal band on B1-B2 and B4-B6); min-max normalisation with training-set statistics; Adam, learning rate 0.001, ReduceLROnPlateau factor 0.5 / patience 4, 15 epochs, stated for all training runs; WeightedRandomSampler (oversampling of the plume class); **weighted BCE with plume pixels weighted 15** (no mag1c weighting: that is the HyperSTARCOP loss); random rotations and 128 x 128 windows with 64 px overlap (167,825 training windows); test tiles kept at 512 x 512; 5 runs per row.
Not stated (Amendment 7 fixes each, item 3 and 4): batch size; the outlier rule of the Varon scale factor; intercept and fitting pixels of the MLR; the min-max constants; the validity mask; whether the table's FPR uses the 10-pixel tile rule or the code's 640-pixel rule (for AVIRIS the 10-pixel text rule reproduced the published number, `results/CORRECTED-PROTOCOL.md` item 1); whether the table's AUPRC is the threshold-free average precision or the 16-threshold trapezoid of `run_validation` (for AVIRIS the average precision reproduced the published number, the 16-threshold trapezoid did not: 50.60 against 37.15 for the released mag1c+RGB run). Also open: the learning rate (the text says 1e-3; for AVIRIS the released configs and the checkpoint's optimizer state say 1e-4, `CORRECTED-PROTOCOL.md` item 4); no WV3 config was released.

## 5. Same test tiles and same metric protocol as the AVIRIS evaluation?
Evidence for yes: (i) the dataset has one test split, three acquisition days (18, 21 and 25 October 2019), 342 tiles of 512 x 512, and the WV3 files are simulated from the same AVIRIS-NG tiles (the paper says both model families train on the same samples, only different views of them); `manifests/zenodo_STARCOP_test.tsv` lists the eight `TOA_WV3_SWIR*.tif` for every one of the 342 test tiles; (ii) the paper has one "Metrics" section for all tables (AUPRC threshold free; pixel threshold 0.5; F1 reported separately for strong, emission above 1000 kg/h, and weak plumes; a tile is positive above 10 active pixels; FPR on the no-plume tiles) and its results section sends the reader to that section for the metrics of all three tables; (iii) the labels are the same labelbinary masks.
Caveats: the paper never prints "342" next to Table 1; the caption of Table 2 says "entire test set", that of Table 1 only "the test set" (no sign of a different subset, but not explicit). The sentence of the data section that swaps strong and weak (`CORRECTED-PROTOCOL.md` item 3) applies to both. Without a WV3 checkpoint the choices in section 4 (FPR rule, AUPRC estimator) cannot be verified for Table 1; the reproduction control of Amendment 7 tests the AUPRC estimator empirically.

## 6. Data inventory (manifests only)
| archive | tiles | WV3 files | WV3 bytes (uncompressed) | on the Kaggle mirror | tiles lacking a WV3 band |
|---|---|---|---|---|---|
| STARCOP_test | 342 | 2,736 | 2.714 GB | yes | 0 |
| STARCOP_train_easy | 559 | 4,472 | 4.350 GB | yes | 0 |
| STARCOP_train_remaining_part1 | 574 | 4,592 | 4.462 GB | yes | 0 |
| STARCOP_train_remaining_part2 | 573 | 4,584 | 4.351 GB | **no** (Zenodo only; 4.18 GB as zip bytes) | 0 |
| STARCOP_train_remaining_part3 | 573 | 4,584 | 4.520 GB | yes | 0 |
| STARCOP_train_remaining_part4 | 573 | 4,584 | 4.436 GB | yes | 0 |
| STARCOP_train_remaining_part5 | 573 | 4,584 | 4.546 GB | yes | 0 |
| train total (`data_meta/train.csv`) | 3,425 (each id in exactly one archive) | 27,400 | 26.664 GB; 22.314 GB on the mirror (2,852 tiles, 22,816 files), 4.351 GB Zenodo-only (573 tiles, 4,584 files) | | 0 |
A WV3 band file of the test split is 0.50 to 1.17 MB (mean 0.99 MB; LZW float32 TIFF; the train archives are alike), mirror file sizes equal the Zenodo member sizes for every shared file (0 mismatches). The zips are 0.960-0.961 of the sum of their members.

Cache sizes (float16, 512 x 512, 3,425 train tiles): 8 raw bands 14.37 GB (+ 0.90 GB labels = 15.27 GB; fits the 20 GB output limit of a Kaggle kernel), 3 feature channels 5.39 GB, 6 channels 10.77 GB, 11 channels (8 + 3) 19.75 GB (+ labels 20.65 GB: does **not** fit). Amendment 7 builds the **3-channel feature cache** (6.29 GB with labels), not the 8-band cache: float16 rounding of raw reflectance (relative step 5e-4 to 1e-3) is comparable to the weakest plume signals of the SWIR-7/8 ratios (a few 1e-3 and below; strong plumes reach a few 1e-2), while features stored after the division and scaled about 0 sit far above float16 resolution. This is the builder's choice and is reversible before any push: the 8-band cache would fit (15.27 GB with labels) and costs the same Zenodo time.
Build time (CPU kernel, no GPU quota): the AVIRIS cache run (`prep_report.json`) fetched 17,112 mirror files (9.98 GB) and 3,438 Zenodo files (1.95 GB) in 1,757 s with 275 HTTP 429 answers absorbed: Zenodo-bound at 1.11 MB/s over the whole run. At that rate the 4.18 GB of part2 take about 3,780 s = **63 min**; the 22.3 GB read from the mirror at 600 MB/s (measured, `FEASIBILITY.md`) plus decoding 22,816 LZW tiles and the per-tile features add a few minutes that overlap with the Zenodo fetch. Estimate: **60-80 min** for the whole kernel; an 8-band cache would take the same time (the Zenodo bytes are the same).
