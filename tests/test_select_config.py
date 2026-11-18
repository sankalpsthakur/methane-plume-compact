import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from select_config import select, composite

def R(p, f1s, f1w, fpr, ap): return dict(params=p, C=composite(dict(F1_strong_q=f1s, F1_weak_q=f1w, FPR_tile_text10=fpr, AUPRC_ap=ap)))
def test_rules():
    # d1 outside the band -> among equally small d2/d4/d6 the highest C wins
    runs = {"d1": R(120_000, 70, 30, 50, 40), "d2": R(490_000, 80, 40, 40, 50), "d4": R(490_000, 80, 42, 38, 50), "d6": R(490_000, 79, 41, 40, 49)}
    assert select(runs)[0] == "d4"
    # d1 inside the band -> fewest parameters wins even though its C is lower
    runs["d1"] = R(120_000, 79, 40, 40, 50)
    ch, best, band, cands = select(runs); assert ch == "d1" and cands == ["d1"], (ch, band)
    # exact tie in C and size -> deterministic (first by max over dict order), still a member of the band
    runs = {"a": R(1, 50, 50, 50, 50), "b": R(1, 50, 50, 50, 50)}; assert select(runs)[0] in ("a", "b")
    assert abs(composite(dict(F1_strong_q=80, F1_weak_q=40, FPR_tile_text10=40, AUPRC_ap=50)) - 57.5) < 1e-9
def test_size_equal_clause_amendment4():
    def S(p, c, st): r = R(p, c, c, 100 - c, c); r["stem_stride"] = st; return r
    # d5 (stem 1) has 0.18 % fewer parameters than d1 (stem 2): both in the band -> d1 (stem 2), not d5
    runs = {"d1": S(123_649, 60, 2), "d5": S(123_429, 61, 1), "d2": S(492_197, 62, 2)}
    ch, best, band, cands = select(runs); assert ch == "d1" and cands == ["d1"], (ch, cands)
    # d1 outside the band -> d5 is the smallest remaining size-equal group of one
    runs["d1"] = S(123_649, 50, 2); assert select(runs)[0] == "d5"
    # a 2 % size gap is a real gap: the smaller one wins even if it is stem 1
    runs = {"a": S(100_000, 60, 1), "b": S(102_000, 61, 2)}; assert select(runs)[0] == "a"
    # only stem-1 members size-equal -> highest C among them
    runs = {"a": S(100_000, 60, 1), "b": S(100_500, 61, 1)}; assert select(runs)[0] == "b"

def test_exclude_amendment7():
    """the control (6.6 M parameters, best validation C) shares the stage-A output directory: without --exclude it would set the best C and could even be chosen"""
    import json, shutil, subprocess, tempfile
    from select_config import load_runs
    tmp = tempfile.mkdtemp()
    try:
        def mk(name, params, c):
            d = os.path.join(tmp, name); os.makedirs(d)
            json.dump(dict(F1_strong_q=c, F1_weak_q=c, FPR_tile_text10=100 - c, AUPRC_ap=c, FPR_tile_official640=100 - c), open(os.path.join(d, "val_metrics.json"), "w"))
            open(os.path.join(d, "log.jsonl"), "w").write(json.dumps(dict(event="start", params=params, args=dict(stem_stride=2))) + "\n")
        mk("g_d1", 123649, 60.0); mk("g_d2", 492197, 62.0); mk("g_d3", 1964557, 61.0); mk("ctrl_wv3", 6629233 - 288, 70.0); os.makedirs(os.path.join(tmp, "logs"))
        runs, inc = load_runs(tmp); assert set(runs) == {"g_d1", "g_d2", "g_d3", "ctrl_wv3"}
        assert select(runs)[0] == "ctrl_wv3"                                   # without the exclusion the control wins the highest C of a size-tied band of one
        runs, inc = load_runs(tmp, ("ctrl_wv3",)); assert set(runs) == {"g_d1", "g_d2", "g_d3"} and select(runs)[0] == "g_d1"
        runs, inc = load_runs(tmp, ("ctrl_wv3", "g_d3")); assert set(runs) == {"g_d1", "g_d2"}
        out = os.path.join(tmp, "sel.json"); subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "select_config.py"), tmp, "--exclude", "ctrl_wv3,g_d3", "--out", out], check=True, capture_output=True)
        sel = json.load(open(out)); assert sel["chosen"] == "g_d1" and sel["excluded"] == ["ctrl_wv3", "g_d3"] and set(sel["runs"]) == {"g_d1", "g_d2"}
    finally: shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    test_rules(); test_size_equal_clause_amendment4(); test_exclude_amendment7(); print("select_config: ok")
