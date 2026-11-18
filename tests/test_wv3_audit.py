"""scripts/wv3_manifest_audit.py on tiny synthetic manifests (no data): per-archive counts, a tile lacking a WV3 band, a mirror-vs-Zenodo size mismatch, tiles that are not on the mirror, cache sizes."""
import json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wv3_manifest_audit as A


def test_audit():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "manifests")); os.makedirs(os.path.join(tmp, "data_meta")); os.makedirs(os.path.join(tmp, "results", "prep_cache"))
        z = {a: [] for a in A.ARCHIVES}; mirror = []
        def tile(arch, tid, drop=(), sizes=100, on_mirror=True, mirror_size=None):
            for b in A.BANDS:
                if b in drop: continue
                z[arch].append(f"{arch}/{tid}/{b}\t{sizes}\tdeadbeef\t8")
                if on_mirror: mirror.append(f"{arch}/{arch}/{tid}/{b}\t{mirror_size or sizes}")
            z[arch].append(f"{arch}/{tid}/mag1c.tif\t5\tdeadbeef\t8")
        tile("STARCOP_test", "te0"); tile("STARCOP_test", "te1")
        tile("STARCOP_train_easy", "a0"); tile("STARCOP_train_easy", "a1", drop=("TOA_WV3_SWIR3.tif",)); tile("STARCOP_train_remaining_part1", "b0", mirror_size=101)
        tile("STARCOP_train_remaining_part2", "c0", on_mirror=False); tile("STARCOP_train_remaining_part2", "c1", on_mirror=False)
        for a in A.ARCHIVES: open(os.path.join(tmp, "manifests", f"zenodo_{a}.tsv"), "w").write("\n".join(z[a]) + ("\n" if z[a] else ""))
        open(os.path.join(tmp, "manifests", "mirror_files.tsv"), "w").write("\n".join(mirror) + "\n")
        open(os.path.join(tmp, "data_meta", "train.csv"), "w").write("id,x\na0,1\na1,1\nb0,1\nc0,1\nc1,1\n"); open(os.path.join(tmp, "data_meta", "test.csv"), "w").write("id,x\nte0,1\nte1,1\n")
        json.dump(dict(zenodo_bytes_fetched=1000, secs=10, zenodo_fetches=3, mirror_bytes_fetched=2000, mirror_fetches=6, train_tiles=5, zen_stats=dict(http429=0)), open(os.path.join(tmp, "results", "prep_cache", "prep_report.json"), "w"))
        r = A.audit(tmp); c = r["coverage"]
        assert c["train_csv_tiles"] == 5 and c["train_tiles_on_mirror"] == 3 and c["train_tiles_not_on_mirror"] == 2 and c["train_tiles_not_on_mirror_by_archive"] == {"STARCOP_train_remaining_part2": 2}
        assert c["train_tiles_lacking_a_wv3_band_in_zenodo"] == 1 and c["test_tiles_lacking_a_wv3_band_in_zenodo"] == 0 and c["tiles_train_plus_test"] == 7 and c["mirror_tiles_train_plus_test"] == 5
        e = r["archives"]["STARCOP_train_easy"]; assert e["zenodo"]["tiles"] == 2 and e["zenodo"]["tiles_with_all_8"] == 1 and e["zenodo"]["lacking_examples"] == [dict(tile="a1", missing=["TOA_WV3_SWIR3.tif"])] and e["zenodo"]["total_files"] == 15
        assert r["archives"]["STARCOP_train_remaining_part1"]["wv3_size_mismatches_mirror_vs_zenodo"] == 8 and e["wv3_size_mismatches_mirror_vs_zenodo"] == 0
        p2 = r["archives"]["STARCOP_train_remaining_part2"]; assert p2["on_mirror"] is False and p2["mirror"] is None and p2["zenodo"]["total_bytes"] == 1600 and c["wv3_bytes_train_not_on_mirror"] == 1600
        assert r["cache_sizes_fp16"]["C8"]["train_bytes"] == 5 * 8 * 512 * 512 * 2 and r["cache_sizes_fp16"]["C3"]["channels"] == 3 and r["build_estimate"]["wv3_zenodo_files"] == 16
    finally: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_audit(); print("wv3 manifest audit: counts, lacking tiles, mirror mismatches, cache sizes OK")
