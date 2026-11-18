#!/usr/bin/env python3
"""WV3 data inventory of the STARCOP release from the committed manifests (Amendment 7 draft). Nothing is downloaded; stdlib only.

Reads manifests/zenodo_<archive>.tsv (name, uncompressed size, crc32, compress type; the official zip central directories), manifests/mirror_files.tsv
(Kaggle mirror chrisphamda/starcop: name, size), data_meta/{train,test}.csv and results/prep_cache/prep_report.json, and writes results/wv3_manifest_audit.json:
  * per archive: tiles, per-band file counts and bytes (uncompressed), tiles lacking one or more of the eight TOA_WV3_SWIR files (Zenodo and mirror);
  * mirror-vs-Zenodo size agreement of every WV3 file the two share;
  * coverage of data_meta/train.csv and test.csv by the archives (which ids live where; which are not on the mirror);
  * cache sizes (N x C x 512 x 512 float16) and the build-time estimate from the measured AVIRIS cache run.
usage: scripts/wv3_manifest_audit.py [--root .] [--out results/wv3_manifest_audit.json]"""
import argparse, csv, json, os, sys

BANDS = [f"TOA_WV3_SWIR{i}.tif" for i in range(1, 9)]
TRAIN_ARCHIVES = ["STARCOP_train_easy"] + [f"STARCOP_train_remaining_part{i}" for i in range(1, 6)]
ARCHIVES = ["STARCOP_test"] + TRAIN_ARCHIVES
# size of each official zip, Zenodo API record 7863343 (v1.0), fetched 2026-10-19 (metadata only): used for the zip-bytes / uncompressed-bytes ratio of a range read
ZIP_BYTES = {"STARCOP_train_easy": 8997773286, "STARCOP_train_remaining_part5": 9417306040, "STARCOP_train_remaining_part1": 9240740614,
             "STARCOP_train_remaining_part3": 9333864104, "STARCOP_test": 5632783317, "STARCOP_train_remaining_part4": 9188310245,
             "STARCOP_train_remaining_part2": 9011870388}
HW = 512
LIST_LIMIT = 20   # tile ids listed per finding; the counts are always exact


def read_zenodo(path):
    """{tile: {file: (size, crc)}} from one manifest; rows that are not tile members (the CSV at the archive root) are returned separately."""
    tiles, other = {}, []
    for line in open(path):
        name, size, crc, _ = line.rstrip("\n").split("\t")
        p = name.split("/")
        if len(p) == 3: tiles.setdefault(p[1], {})[p[2]] = (int(size), crc)
        else: other.append((name, int(size)))
    return tiles, other


def read_mirror(path):
    """{archive: {tile: {file: size}}} and the non-tile rows, from the Kaggle mirror listing (archive folder / archive / tile / file)."""
    arch, other = {}, []
    for line in open(path):
        name, size = line.rstrip("\n").split("\t")
        p = name.split("/")
        if len(p) == 4: arch.setdefault(p[0], {}).setdefault(p[2], {})[p[3]] = int(size)
        else: other.append((name, int(size)))
    return arch, other


def read_ids(path):
    with open(path, newline="") as fh:
        return [r["id"] for r in csv.DictReader(fh)]


def band_stats(tiles, size_of):
    """per-band file count and bytes, tiles with all eight bands, and the tiles lacking any (sorted)."""
    cnt = {b: 0 for b in BANDS}; byt = {b: 0 for b in BANDS}; lacking = []
    for t, files in tiles.items():
        miss = [b for b in BANDS if b not in files]
        if miss: lacking.append((t, miss))
        for b in BANDS:
            if b in files: cnt[b] += 1; byt[b] += size_of(files[b])
    lacking.sort()
    return dict(tiles=len(tiles), tiles_with_all_8=len(tiles) - len(lacking), tiles_lacking=len(lacking), lacking_examples=[dict(tile=t, missing=m) for t, m in lacking[:LIST_LIMIT]],
                files=cnt, bytes=byt, total_files=sum(cnt.values()), total_bytes=sum(byt.values()))


def audit(root):
    man = os.path.join(root, "manifests"); meta = os.path.join(root, "data_meta")
    zen, zen_other = {}, {}
    for a in ARCHIVES: zen[a], zen_other[a] = read_zenodo(os.path.join(man, f"zenodo_{a}.tsv"))
    mir, mir_other = read_mirror(os.path.join(man, "mirror_files.tsv"))
    train_ids, test_ids = read_ids(os.path.join(meta, "train.csv")), read_ids(os.path.join(meta, "test.csv"))
    out = dict(bands=BANDS, archives={})
    for a in ARCHIVES:
        z = band_stats(zen[a], lambda v: v[0])
        m = band_stats(mir[a], lambda v: v) if a in mir else None
        shared = [t for t in zen[a] if a in mir and t in mir[a]]
        mismatch = [(t, b) for t in shared for b in BANDS if b in zen[a][t] and b in mir[a][t] and zen[a][t][b][0] != mir[a][t][b]]
        all_files_z = sum(v[0] for f in zen[a].values() for v in f.values())
        out["archives"][a] = dict(zenodo=z, mirror=m, on_mirror=a in mir, tiles_only_in_zenodo=len(set(zen[a]) - set(mir.get(a, {}))), tiles_only_on_mirror=len(set(mir.get(a, {})) - set(zen[a])),
                                  wv3_size_mismatches_mirror_vs_zenodo=len(mismatch), mismatch_examples=[dict(tile=t, band=b) for t, b in mismatch[:LIST_LIMIT]],
                                  zip_bytes=ZIP_BYTES[a], uncompressed_member_bytes=all_files_z, zip_over_uncompressed=(round(ZIP_BYTES[a] / all_files_z, 4) if all_files_z else None),
                                  wv3_estimated_range_read_bytes=(round(z["total_bytes"] * ZIP_BYTES[a] / all_files_z) if all_files_z else 0))
    where = {t: a for a in TRAIN_ARCHIVES for t in zen[a]}
    dup = [t for a in TRAIN_ARCHIVES for t in zen[a] if sum(t in zen[b] for b in TRAIN_ARCHIVES) > 1]
    tr_where = [where.get(t) for t in train_ids]
    on_mirror = [t for t in train_ids if where.get(t) in mir and t in mir[where[t]]]
    not_on_mirror = [t for t in train_ids if t not in set(on_mirror)]
    nm_arch = {}
    for t in not_on_mirror: nm_arch[where.get(t)] = nm_arch.get(where.get(t), 0) + 1
    train_lacking = [t for t in train_ids if where.get(t) and any(b not in zen[where[t]][t] for b in BANDS)]
    test_lacking = [t for t in test_ids if t not in zen["STARCOP_test"] or any(b not in zen["STARCOP_test"][t] for b in BANDS)]
    zen_wv3_train = sum(zen[where[t]][t][b][0] for t in train_ids if where.get(t) for b in BANDS if b in zen[where[t]][t])
    zen_wv3_test = sum(zen["STARCOP_test"][t][b][0] for t in test_ids if t in zen["STARCOP_test"] for b in BANDS if b in zen["STARCOP_test"][t])
    nm_set = set(not_on_mirror)
    zen_wv3_not_on_mirror = sum(zen[where[t]][t][b][0] for t in nm_set if where.get(t) for b in BANDS if b in zen[where[t]][t])
    zen_wv3_not_on_mirror_zip = sum(zen[where[t]][t][b][0] * ZIP_BYTES[where[t]] / out["archives"][where[t]]["uncompressed_member_bytes"] for t in nm_set if where.get(t) for b in BANDS if b in zen[where[t]][t])
    out["coverage"] = dict(train_csv_tiles=len(train_ids), train_ids_unique=len(set(train_ids)), train_ids_in_no_train_archive=sum(w is None for w in tr_where), train_ids_in_more_than_one_archive=len(set(dup)),
                           train_tiles_by_archive={a: sum(w == a for w in tr_where) for a in TRAIN_ARCHIVES}, archive_tiles_not_in_train_csv={a: len(set(zen[a]) - set(train_ids)) for a in TRAIN_ARCHIVES},
                           train_tiles_on_mirror=len(on_mirror), train_tiles_not_on_mirror=len(not_on_mirror), train_tiles_not_on_mirror_by_archive=nm_arch,
                           train_tiles_lacking_a_wv3_band_in_zenodo=len(train_lacking), test_csv_tiles=len(test_ids), test_tiles_lacking_a_wv3_band_in_zenodo=len(test_lacking),
                           test_tiles_on_mirror=sum(t in mir.get("STARCOP_test", {}) for t in test_ids), tiles_train_plus_test=len(train_ids) + len(test_ids),
                           mirror_tiles_train_plus_test=len(on_mirror) + sum(t in mir.get("STARCOP_test", {}) for t in test_ids),
                           wv3_bytes_train=zen_wv3_train, wv3_bytes_test=zen_wv3_test, wv3_bytes_train_not_on_mirror=zen_wv3_not_on_mirror, wv3_bytes_train_not_on_mirror_zip_estimate=round(zen_wv3_not_on_mirror_zip))
    N = len(train_ids); px = HW * HW
    cache = {f"C{c}": dict(channels=c, bytes_per_tile=c * px * 2, train_bytes=N * c * px * 2, train_GB=round(N * c * px * 2 / 1e9, 2)) for c in (3, 6, 8, 11)}
    cache["labels_uint8"] = dict(train_bytes=N * px, train_GB=round(N * px / 1e9, 2))
    out["cache_sizes_fp16"] = cache
    rp_path = os.path.join(root, "results", "prep_cache", "prep_report.json")
    if os.path.exists(rp_path):
        rp = json.load(open(rp_path))
        zrate = rp["zenodo_bytes_fetched"] / rp["secs"]    # whole-kernel average, Zenodo-bound (429 cool-downs included): a lower bound of the rate
        out["build_estimate"] = dict(aviris_run=dict(secs=rp["secs"], zenodo_bytes=rp["zenodo_bytes_fetched"], zenodo_files=rp["zenodo_fetches"], mirror_bytes=rp["mirror_bytes_fetched"], mirror_files=rp["mirror_fetches"],
                                                  http429=rp["zen_stats"]["http429"], tiles=rp["train_tiles"]), zenodo_rate_B_per_s=round(zrate),
                                     wv3_zenodo_bytes=out["coverage"]["wv3_bytes_train_not_on_mirror_zip_estimate"], wv3_zenodo_files=8 * len(not_on_mirror),
                                     wv3_zenodo_seconds=round(out["coverage"]["wv3_bytes_train_not_on_mirror_zip_estimate"] / zrate),
                                     wv3_mirror_bytes=zen_wv3_train - zen_wv3_not_on_mirror, wv3_mirror_files=8 * len(on_mirror))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--out", default="results/wv3_manifest_audit.json")
    a = ap.parse_args()
    res = audit(a.root)
    path = a.out if os.path.isabs(a.out) else os.path.join(a.root, a.out)
    json.dump(res, open(path, "w"), indent=1)
    c = res["coverage"]
    print(f"train tiles {c['train_csv_tiles']} (on mirror {c['train_tiles_on_mirror']}, not on mirror {c['train_tiles_not_on_mirror']} {c['train_tiles_not_on_mirror_by_archive']}); test tiles {c['test_csv_tiles']}")
    print(f"tiles lacking a WV3 band: train {c['train_tiles_lacking_a_wv3_band_in_zenodo']}, test {c['test_tiles_lacking_a_wv3_band_in_zenodo']}")
    for k, v in res["archives"].items():
        z = v["zenodo"]; m = v["mirror"]
        print(f"{k:34s} zenodo tiles {z['tiles']:5d} all8 {z['tiles_with_all_8']:5d} WV3 files {z['total_files']:6d} {z['total_bytes']/1e9:7.3f} GB | mirror {'-' if m is None else str(m['tiles']) + ' tiles ' + format(m['total_bytes']/1e9, '.3f') + ' GB'} | size mismatches {v['wv3_size_mismatches_mirror_vs_zenodo']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
