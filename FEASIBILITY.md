# Feasibility gate: STARCOP inside a private Kaggle kernel (no data on the Mac)

Verdict: FEASIBLE on free Kaggle CPU kernels, nothing from STARCOP is downloaded to this Mac. Evidence below; kernel names are
`bench-gas-feasibility` (v3), `bench-gas-baseline`, `bench-gas-prep-train`.

## Data route
| item | finding |
|---|---|
| Official source | Zenodo 10.5281/zenodo.7863343 v1.0, 7 zips (60.8 GB) + 3 CSVs, **CC BY-NC 4.0**. Test: 342 tiles (176 no-plume, 57 strong, 109 weak; 3 dates, 44 flights). Train: 3,425 tiles (1,713 no-plume, 273 strong, 1,439 weak; 19 dates, 256 flights). No flight line or date is shared between train and test. |
| Kaggle mirror | `chrisphamda/starcop` (public, 53.9 GB uncompressed in 63,880 files). Labelled "MIT" on Kaggle; that label is wrong for the data, the governing license is CC BY-NC 4.0 (upstream). Mounts at `/kaggle/input/datasets/chrisphamda/starcop`. |
| Integrity | CRC32 of all 6,840 test-split files read in the kernel: 0 mismatches vs the Zenodo zip central directories (5.9 GB in 9.7 s, 607 MB/s). Every file used for training is CRC-checked while it is read (`bench-gas-prep-train`). Zenodo CRCs were obtained with HTTP range requests (8 MB per archive, `scripts/zenodo_manifest.py`), the data itself was not downloaded. |
| Gap in the mirror | `STARCOP_train_remaining_part2.zip` is missing (0 of 11,460 files): 573 train tiles, 330 of them weak-plume tiles (23% of all weak training tiles). `STARCOP_train_easy` (all 273 strong-plume train tiles) is present, so the mirror alone gives a biased train set. The missing tiles are fetched from Zenodo inside the kernel with HTTP range requests (only the 6 files per tile that mag1c+RGB needs, about 1.6 GB), CRC-checked. Measured Zenodo throughput from inside Kaggle is low (0.9 MB/s per connection, 2.4 MB/s with 8 threads), hence 32 parallel connections. |
| Files needed for the paper's best input (mag1c + RGB) | `mag1c`, `TOA_AVIRIS_640nm/550nm/460nm`, `labelbinary`, `weight_mag1c` (6 of 20 files per tile, about 3 MB of 17 MB). TIFFs are LZW, tiled; mag1c and weight are float64, RGB float32, label uint8. `weight_mag1c = clip(mag1c / 400, 0.1, 1.0)` (checked on 20 tiles, 800k pixels). |
| Cache | float16 `clip(raw/factor, 0, 2)` for the 4 inputs (factors 1750, 60, 60, 60 as in the official normaliser), uint8 labels, uint8 weights: about 9 GB train, written to `/kaggle/working` (20 GB limit) of the private CPU kernel `bench-gas-prep-train`, consumed by later kernels as a kernel source. The test split is never cached: final kernels read it from the CRC-verified mirror. |

## Limits checked
- CPU kernel: Intel Xeon 2.2 GHz, 4 vCPU (2 cores, AVX2, no AVX-512/AMX), 31 GB RAM, `/kaggle/working` 20 GB, internet works from kernels (pip, Zenodo, Hugging Face), torch 2.11 CPU build, rasterio 1.5, onnx 1.22, jax 0.11.
- Mount read 600 MB/s; walking 64k files 20 s; decoding 6 bands of a tile 0.09 s (rasterio).
- Training speed on this CPU (batch 32, 128x128 crops): plain U-Net widths 4..64 (0.12M params) 3.8 steps/s, 8..128 (0.49M) 1.85 steps/s, 12..192 (1.1M) 0.85 steps/s. Two threads are as fast as four (2 physical cores).
- Account limits: 5 concurrent CPU sessions (shared by all tracks; program rule: one session per track); GPU quota of the account is exhausted until 2026-10-10 00:00 UTC (owner's own work); TPU queue did not start for another track within 50 minutes, so TPU is not used.
- Local footprint: official CSVs 1.3 MB, Zenodo CRC manifests 15 MB, released checkpoint 80 MB (for the private assets dataset), code.

## Reproduction of the released model (gate for training)
See `results/baseline_reproduction.md`. The released HyperSTARCOP mag1c+RGB checkpoint (run "R3") loads strictly into our dependency-free
re-implementation (bit-identical to `segmentation_models_pytorch` 0.5.0, max abs logit diff 0.0) and, scored on the 342 official test tiles,
falls inside the published 5-run mean +- 1 std on all four metrics under the paper-text protocol.
