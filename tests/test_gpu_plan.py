import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import mk_gpu_wave as W

def probe(scale):  # synthetic probe: steps/s per case, AMP on
    base = {"d1": 80, "d2": 70, "d3": 50, "d5": 45, "d2w": 30, "mnv2": 35}
    return {f"{W.PROBE_KEY[c]}|bs32|amp1": {"steps_per_s": v * scale} for c, v in base.items()}

def test_caps_are_billed_seconds_and_fit_the_gas_cap():
    assert sum(W.CAPS_Q.values()) == 25200 and sum(W.CAPS_Q.values()) + W.RESERVE_Q == W.GAS_CAP_Q == 28800   # 7.0 h of caps + 1.0 h retry reserve = the 8 h gas cap, all in BILLED seconds
    assert W.CAPS_MERGED == W.CAPS_Q["compact"] + W.CAPS_Q["control"] == 16200                                 # the merged G2+G3 kernel spends exactly the two caps it replaces
    for b in (1, 2, 4):
        assert all(W.wall_cap(k, b) * b <= W.CAPS_Q[k] for k in W.CAPS_Q)                                   # `-t` (wall) x GPUs held never exceeds the billed cap

def test_queue_scheduling_and_merge_saves_the_idle_worker():
    assert W.sched([10] * 5 + [14]) == 34 and W.sched([10, 10, 10]) == 20 and W.sched([5]) == 5
    for scale in (0.5, 1.0, 2.0):
        pr = probe(scale)
        sep = W.t_compact(10, "d1", pr, 60.0, 2) + W.t_control(10, pr, 60.0)
        assert W.t_merged(10, "d1", 10, pr, 60.0, 2) < sep       # one kernel with both workers busy is cheaper than two kernels, the second with an idle GPU

def test_fast_single_gpu_gets_the_paper_schedule_and_all_candidates():
    for merge in (True, False):
        r = W.plan(probe(3.0), 1000.0, b=1, n_gpu=1, merge=merge)
        assert r["verdict"] == "OK" and r["wave"]["epochs"] == 15 and r["wave"]["cands"] == ["d1", "d2", "d3", "d5", "d2w"] and r["control"]["epochs"] == 15

def test_billing_never_helps_a_given_gpu():
    for scale in (0.3, 0.6, 1.0, 2.0, 4.0):
        r1 = W.plan(probe(scale), 1000.0, b=1, n_gpu=1); r2 = W.plan(probe(scale), 1000.0, b=2, n_gpu=2)
        c1 = (r1["control"] or {"epochs": 0})["epochs"]; c2 = (r2["control"] or {"epochs": 0})["epochs"]
        e1 = (r1["wave"] or {"epochs": 0})["epochs"]; e2 = (r2["wave"] or {"epochs": 0})["epochs"]
        assert e2 <= e1 + 5 and c2 <= max(c1, 15)       # sanity bounds only; the strict checks are the billed-cap inequalities below
    assert W.plan(probe(1.0), 1000.0, b=2, n_gpu=2, merge=False)["control"]["epochs"] <= W.plan(probe(1.0), 1000.0, b=1, n_gpu=1, merge=False)["control"]["epochs"]   # a lone control bills b x wall

def test_projections_respect_the_billed_caps():
    for b in (1, 2):
        for merge in (True, False):
            for scale in (0.15, 0.25, 0.4, 0.7, 1.0, 2.0):
                r = W.plan(probe(scale), 60.0, b=b, n_gpu=b, merge=merge)
                if r["verdict"] != "OK": continue
                w = r["wave"]; capc = W.CAPS_MERGED if merge else W.CAPS_Q["compact"]
                assert w["projected_wide_billed_s"] <= W.FRAC * W.CAPS_Q["wide"] + 1 and w["projected_compact_worst_billed_s"] <= W.FRAC * capc + 1
                assert w["epochs"] in W.TIERS_E and w["cands"][:3] == ["d1", "d2", "d3"] and r["control"]["epochs"] in W.TIERS_E
                assert w["soft_deadline_wall_s"]["wide"] == W.wall_cap("wide", b) - W.SOFT_MARGIN["wide"]
                if merge:
                    m = r["merged"]; assert m["billed_cap_s"] == W.CAPS_MERGED and m["wall_cap_s"] == W.CAPS_MERGED // b and set(m["per_candidate"]) == set(w["cands"])
                    assert all(v["billed_s"] <= W.FRAC * W.CAPS_MERGED + 1 for v in m["per_candidate"].values()) and m["control_epochs"] == r["control"]["epochs"]
                else:
                    assert r["control"]["projected_billed_s"] <= W.FRAC * W.CAPS_Q["control"] + 1
                assert r["projected_total_billed_h"] <= r["sum_caps_billed_h"]

def test_sampler_is_the_ceiling():
    slow_sampler = W.plan(probe(10.0), 8.0, b=2, n_gpu=2); fast_sampler = W.plan(probe(10.0), 500.0, b=2, n_gpu=2)
    assert (fast_sampler["wave"] or {"epochs": 0})["epochs"] >= (slow_sampler["wave"] or {"epochs": 0})["epochs"]

def test_no_fit_is_reported_not_hidden():
    r = W.plan(probe(0.01), 1000.0, b=2, n_gpu=2)
    assert r["verdict"].startswith("NO FIT") and r["wave"] is None

def test_monotone_in_gpu_speed():
    prev = 0
    for scale in (0.2, 0.4, 0.8, 1.6, 3.2):
        e = (W.plan(probe(scale), 1000.0, b=2, n_gpu=2)["wave"] or {"epochs": 0})["epochs"]
        assert e >= prev; prev = e

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f()
    print("gpu plan: ok")
