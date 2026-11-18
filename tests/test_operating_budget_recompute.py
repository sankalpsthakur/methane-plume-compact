# Copyright 2026 Sankalp Thakur. Licensed under the Apache License, Version 2.0.
"""Pins the headline numbers of the operating-budget appendix to its committed CSV tables (numpy only, no data download).
These are the audit's published values; the test fails if the recompute script or the tables drift."""
import math, os, subprocess, sys, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
APPX = os.path.join(ROOT, "appendix", "operating_budget")
sys.path.insert(0, APPX)
import recompute_operating_budget as R


class OperatingBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiles, cls.counts = R.load_tables(os.path.join(APPX, "data-cc-by-nc"))
        cls.m = {(r, k): R.aggregate(cls.tiles[r], cls.counts[(r, k)]) for r in R.REFS for k in R.RULES}

    def test_group_sizes(self):
        m = self.m[("released-mag1c-rgb", "0.05")]
        self.assertEqual((m["n_negative"], m["n_weak"], m["n_weak_program"], m["n_strong"], m["n_positive_unknown_rate"]),
                         (175, 108, 110, 57, 2))

    def test_primary_rule_numbers(self):
        want = {"released-mag1c-rgb": (13, 36, 0.141410), "released-mag1c-only": (9, 30, 0.114492), "raw-mag1c": (7, 0, 0.000705)}
        for r, (neg, loc, iou) in want.items():
            m = self.m[(r, "0.05")]
            self.assertEqual((m["false_indications"], m["weak_localized"]), (neg, loc), r)
            self.assertAlmostEqual(m["weak_mean_iou"], iou, places=6)
            self.assertGreater(m["worst_date_fpr"], 0.05)  # no reference meets the budget

    def test_two_weak_definitions_on_identical_predictions(self):
        for r, f108, f110 in (("released-mag1c-rgb", 53.73, 42.34), ("released-mag1c-only", 58.58, 48.76)):
            m = self.m[(r, "standard")]
            self.assertAlmostEqual(100 * m["weak_pixel_f1_audit108"], f108, places=2)
            self.assertAlmostEqual(100 * m["weak_pixel_f1_program110"], f110, places=2)

    def test_thresholds_follow_the_rule(self):
        for r in R.REFS:
            for k in ("0.01", "0.05", "0.1"):
                th = R.thresholds(self.tiles[r], k)
                flags = [t["tile_score"] > th[t["date"]][0] for t in self.tiles[r]]
                self.assertEqual(flags, [c["indication"] for c in self.counts[(r, k)]], (r, k))

    def test_rank_is_exact(self):
        # ceil((126 + 1) * 0.95) = 121 (floating point would give 120.65 -> 121 too, but 0.9 * 10 style cases must not drift)
        self.assertEqual(R.thresholds(self.tiles["raw-mag1c"], "0.05")["2019-10-18"][1:], (121, 126))
        self.assertEqual(math.ceil((9 + 1) * (1 - R.Fraction("0.1"))), 9)

    def test_cli_runs_clean(self):
        out = subprocess.run([sys.executable, os.path.join(APPX, "recompute_operating_budget.py"), "--tables",
                              os.path.join(APPX, "data-cc-by-nc")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("MISMATCHES: none", out.stdout)


if __name__ == "__main__":
    unittest.main()
