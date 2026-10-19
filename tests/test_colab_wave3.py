"""Amendment 6 (wave 3 on Colab): the generated configurations are the pre-registered ones.
The gate is one validation-only run (d1, seed 0, E = 15) that cannot read the test split; the test stage is five one-seed jobs
(d1, seeds 0-4, E = 15). Both equal the GPU wave's run dictionaries except for the name and the step count, on one T4 (one worker)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import mk_gpu_wave as W
import mk_colab_wave3 as C

E15 = 15 * W.EPOCH_STEPS
DIFF = {"name", "steps", "eval_every"}


def strip(r):
    return {k: v for k, v in r.items() if k not in DIFF}


def test_gate_is_one_validation_run_without_test_access():
    ks = C.configs("gate")
    assert [n for n, _ in ks] == ["colab-gate"]
    cfg = ks[0][1]
    assert len(cfg["runs"]) == 1
    r = cfg["runs"][0]
    assert r["name"] == "c_d1_e15_val" and r["seed"] == 0 and r["steps"] == E15 == 78675 and r["eval_every"] == E15
    assert r["val_dates"] == W.VAL
    g1_d1 = dict(name="g_d1", seed=0, val_dates=W.VAL, eval_every=1, steps=1, **W.COMMON, **W.CANDS["d1"])  # the G1 run, steps aside
    assert strip(r) == strip(g1_d1)
    assert not any(x.get("test_eval") for x in cfg["runs"]) and not cfg.get("need_test")


def test_test_stage_is_five_one_seed_jobs_matching_g2():
    ks = C.configs("test")
    assert [n for n, _ in ks] == [f"colab-test-s{s}" for s in range(5)]
    for s, (_, cfg) in enumerate(ks):
        assert len(cfg["runs"]) == 1
        r = cfg["runs"][0]
        assert r["name"] == f"c_d1_e15_s{s}" and r["seed"] == s and r["steps"] == E15 and r["test_eval"] is True
        assert "val_dates" not in r
        g2 = dict(name=f"gf_d1_s{s}", seed=s, test_eval=True, steps=1, **W.COMMON, **W.CANDS["d1"])
        assert strip(r) == strip(g2)


def test_one_t4_one_worker_no_edge_stage():
    for stage in ("gate", "test"):
        for _, cfg in C.configs(stage):
            assert cfg["gpu"] is True and cfg["max_gpus"] == 1 and cfg["parallel"] == 1
            assert not cfg.get("edge") and "edge_runs" not in cfg
            assert 0 < cfg["deadline_s"] <= 5400



def test_kaggle_configs_reuse_the_same_runs():
    (gname, gcfg), = C.configs("kaggle-gate")
    assert gname == "gpu-gate3" and gcfg["runs"] == C.configs("gate")[0][1]["runs"]
    (tname, tcfg), = C.configs("kaggle-test")
    assert tname == "gpu-test3" and tcfg["runs"] == [cfg["runs"][0] for _, cfg in C.configs("test")]
    for cfg, workers in ((gcfg, 1), (tcfg, 2)):
        assert cfg["machine_shape"] == "NvidiaTeslaT4" and cfg["max_gpus"] == 2 and cfg["billing_b"] == 2 and cfg["parallel"] == workers
        assert cfg["billing_b"] * cfg["wall_cap_s"] <= cfg["billed_cap_s"] and 0 < cfg["deadline_s"] < cfg["wall_cap_s"]
        assert not cfg.get("edge") and "edge_runs" not in cfg
    assert not any(r.get("test_eval") for r in gcfg["runs"]) and not gcfg.get("need_test")
    assert (gcfg["billed_cap_s"] + tcfg["billed_cap_s"]) / 3600 <= 7.0   # wave-3 ledger limit of scripts/gpu_gate.sh


if __name__ == "__main__":
    test_gate_is_one_validation_run_without_test_access(); test_test_stage_is_five_one_seed_jobs_matching_g2(); test_one_t4_one_worker_no_edge_stage()
    test_kaggle_configs_reuse_the_same_runs()
    print("wave 3 configs: all checks passed")
