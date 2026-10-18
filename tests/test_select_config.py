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

if __name__ == "__main__":
    test_rules(); test_size_equal_clause_amendment4(); print("select_config: ok")
