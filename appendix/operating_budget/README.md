# Appendix: plume operating-budget audit of the released references

What this is: a retrospective audit, run by another workstream on 2026-10-11 (UTC), that asks a different question of the
STARCOP test split than the track's headline F1 does: if a detector must raise at most 5 % indications on
mask-negative tiles on every acquisition date, how many weak plumes does it still localise? It scored only the
**released references**: the two released STARCOP checkpoints (HyperSTARCOP mag1c+RGB and mag1c-only) and raw mag1c.
It is an organizer reference audit, not an official benchmark, not a blind confirmation and not a ranking.

**methane-plume-compact (our own model) is NOT scored here and must not be.** The audit read the test split once, on
2026-10-11, before and outside this track's `TEST-LOG.md` (logged retroactively as rows #49-#51). Scoring our model against
these gates would be a new look at the test split; "no third look" stands. Nothing in this appendix was used to choose,
tune or select any model of this track.

## The gates, on the 342 test tiles (44 flights, three dates: 2019-10-18, 2019-10-21, 2019-10-25)

A tile is an *indication* when at least 11 of its pixels score strictly above the threshold (equivalently its 11th-largest
pixel score exceeds the threshold). For each held-out date the threshold is the order statistic of rank
ceil((n + 1)(1 - alpha)) of the tile statistic over the mask-negative tiles (no positive mask pixel) of the **other two** dates
(+infinity if the rank exceeds n). Primary alpha = 0.05; 0.01 and 0.10 are secondary. Rotating dates is cross-fitting
(a later date may calibrate an earlier one): a retrospective transfer diagnostic, not a chronological deployment replay.
The operating gates are:

1. at most 5 % mask-negative tile indications on **every** date;
2. at least 80 % weak *localized* recall: an indication **and** mask IoU of at least 0.1;
3. at least 0.25 mean weak-plume mask IoU.

The thresholds were fixed by the audit's protocol before its run; they are organizer research targets, not physical
acceptance criteria. Failing them says the reference fails; it does not say the gates are impossible.

## Results (recomputed here; every number below is printed by `recompute_report.txt`)

Primary 5 % rule (175 mask-negative tiles: 49 / 70 / 56 per date; 108 weak tiles: 24 / 61 / 23 per date):

| Reference | Negative-tile indications | Worst date | Weak localized recall | Weak mean IoU | Gates |
|---|---:|---:|---:|---:|---|
| released mag1c+RGB | 13/175 (3/49, 10/70, 0/56) | 14.29 % | 36/108 = 33.33 % | 0.1414 | all three failed |
| released mag1c-only | 9/175 (1/49, 8/70, 0/56) | 11.43 % | 30/108 = 27.78 % | 0.1145 | all three failed |
| raw mag1c (no postprocessing) | 7/175 (0/49, 5/70, 2/56) | 7.14 % | 0/108 = 0.00 % | 0.0007 | all three failed |

All four rules (negative indications; worst-date rate; weak localized; weak mean IoU):

| Reference | standard | 1 % | 5 % | 10 % |
|---|---|---|---|---|
| released mag1c+RGB | 71/175; 48.98 %; 85/108; 0.4578 | 4/175; 5.71 %; 7/108; 0.0123 | 13/175; 14.29 %; 36/108; 0.1414 | 20/175; 21.43 %; 63/108; 0.2431 |
| released mag1c-only | 63/175; 51.02 %; 77/108; 0.4053 | 5/175; 7.14 %; 7/108; 0.0166 | 9/175; 11.43 %; 30/108; 0.1145 | 17/175; 16.33 %; 43/108; 0.1838 |
| raw mag1c | 175/175; 100 %; 5/108; 0.0315 | 0/175; 0 %; 0/108; 0 | 7/175; 7.14 %; 0/108; 0.0007 | 18/175; 14.29 %; 1/108; 0.0038 |

"standard" is a diagnostic at the original operating point (logit 0, i.e. p = 0.5, for the learned references; a fixed
200 ppm.m for raw mag1c), not a calibrated budget. No reference clears any budget at 5 % and no reference clears the gates.
The audit also reports flight-bootstrap intervals with the thresholds held fixed; they leave out calibration variability and
site dependence and are not reproduced here (see the audit's report).

Limits that travel with these numbers: the original test labels and earlier scores had been consumed before the audit; the
released checkpoints monitored test loss during training; the three dates are one regional campaign, so 44 flights are not 44
independent sites; mask-negative does not prove gas absence; AVIRIS-NG mag1c is not an ordinary-RGB or point-concentration
sensor. Repetition across rules and dates (4,104 per-tile decision records) is not 4,104 independent trials.

## Two weak-plume definitions (read no number against the model card without its definition)

| | this audit | this track (model card, `results/`, TEST-LOG) |
|---|---|---|
| rule | mask has positive pixels **and** 0 < qplume < 1000 kg/h | mask has positive pixels **and** qplume < 1000 kg/h (paper text) |
| tiles | **108** | **110** |
| tiles left out | the 2 positive-mask tiles with qplume = 0 (emission group unknown) are kept in all-positive accounting only | none: those 2 tiles count as weak |
| strong | 57 tiles (qplume >= 1000) | the same 57 tiles |

The two differing tiles are `ang20191021t160052_r9472_c0_w512_h512` (188 mask pixels, qplume 0) and
`ang20191021t170142_r9984_c0_w512_h512` (125 mask pixels, qplume 0). Because pooled pixel counts are summed over the group,
two tiles can move weak F1 a lot: the released mag1c+RGB reference predicts 32,888 false-positive pixels on the first of them
(and 0 of its 188 mask pixels), the mag1c-only reference 18,611. On **identical predictions** (the standard rule, logit 0)
weak pixel F1 is:

| Reference | audit, 108 tiles | this track, 110 tiles |
|---|---:|---:|
| released mag1c+RGB | 53.73 | 42.34 |
| released mag1c-only | 58.58 | 48.76 |

The track's 42.34 and 48.76 are the values of TEST-LOG rows #2 and #3. The audit's 53.73 and 58.58 are the same pixel counts
under the 108-tile definition. The metrics also differ in their tile rule: the track's FPR uses the official ">640 pixels"
(or ">10 pixels") tile rule on the unfiltered tile set at a fixed p = 0.5, while the audit uses an 11-pixel rule at calibrated
thresholds. Neither tile rule is a physical false-alarm rate.

## What was recomputed, and how

`recompute_operating_budget.py` (numpy and standard library only; Apache-2.0; original code of this repository, no code from
the audit kit) recomputes every number above from the audit's saved per-tile records and does not load a model or read a
test tile. It rebuilds each threshold from the tile statistics (exact rational rank), checks that the saved indication flags
follow from it on all 342 tiles for all 12 reference/rule cells, recomputes the aggregates from the saved pixel counts
(tp, fp, fn, tn at the fold threshold), and compares everything with the audit's published `RESULTS.json`.

Result of the run (`recompute_report.txt`): **0 mismatches**. For the released mag1c+RGB and mag1c-only references the
standard-rule per-tile counts (tp, fp, fn, tn), `n_pos` versus mask pixels, and `qplume` equal the track's
`kernel_outputs/baseline/baseline_test_summaries.npz` and `baseline_mag1c_only_test_summaries.npz` on all 342 tiles (differing tiles: 0);
the 110-tile weak F1 from the track's counts is 42.34 / 48.76 and the 108-tile value 53.73 / 58.58. (Those two `.npz` files
stay in the private repo; the comparison output is committed as text. There is no track-side file for raw mag1c.)

Reproduce from the CSV copy in this folder (about a second, numpy only):

```
python3 recompute_operating_budget.py --tables data-cc-by-nc
```

or from the audit's original output archive (`plume-budget-output.zip` in the kit of the Hub dataset
`sankalpsthakur/plume-operating-budget-audit`), optionally with `--published RESULTS.json` and `--baseline <dir with the track's npz>`.

## Contents and licences (kept apart)

| Path | What | Licence |
|---|---|---|
| `README.md`, `recompute_operating_budget.py`, `recompute_report.txt` | this text, the numpy-only recompute script and its output | Apache-2.0 (the track's own code), text as the repository |
| `data-cc-by-nc/` | `tile_scores.csv` (per reference and tile: id, date, flight, qplume, mask pixel count, tile statistic) and `pixel_counts.csv` (tp, fp, fn, tn and the indication flag per reference, rule, tile): tables derived from STARCOP labels and metadata | **CC BY-NC 4.0**, attribution to Ruzicka et al., STARCOP v1.0, Zenodo 7863343; see `data-cc-by-nc/NOTICE`. Non-commercial use only |
| `gpl/` | **not present.** The audit kit's own code was labelled GPL-3.0; none of it is carried, because the numbers are recomputed by the Apache-2.0 script above | n/a |

No checkpoint, source image or score map is included. The audit's MARS appendix and its publication receipts are not part of this
track and are not carried.

Source and attribution: Ruzicka, V., Mateo-Garcia, G., Gomez-Chova, L., Vaughan, A., Guanter, L., Markham, A., "STARCOP dataset:
Semantic Segmentation of Methane Plumes with Hyperspectral Machine Learning Models", Zenodo, 2023,
https://doi.org/10.5281/zenodo.7863343 (CC BY-NC 4.0). The Kaggle mirror's "MIT" label is not the dataset's licence.
