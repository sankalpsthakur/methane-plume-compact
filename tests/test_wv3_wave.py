"""Amendment 7: the generated WV3 configurations are the pre-registered ones; budget arithmetic; planner; reproduction-gate decision; claim rule. numpy only except for the constants check of the ImageNet block (imports starcop_edge.models)."""
import json, math, os, shutil, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "src"))
import mk_gpu_wave as G
import mk_wv3_wave as V
import wv3_claim as C
import wv3_gate_decision as GD
from starcop_edge import metrics as M

PUB = json.load(open(os.path.join(ROOT, "protocol", "wv3_published.json")))


def test_published_bars_are_table_1():
    r = PUB["rows"]; assert PUB["comparator_row"] == "varon_sanchez" and PUB["primary_metric"] == "AUPRC_ap"
    assert r["varon_sanchez"] == {"F1_strong_q": [31.89, 2.44], "F1_weak_q": [11.04, 0.75], "FPR_tile_text10": [90.51, 4.23], "AUPRC_ap": [13.04, 1.96]}
    assert r["varon"]["FPR_tile_text10"] == [87.89, 4.67] and r["sanchez"]["AUPRC_ap"] == [9.96, 1.43] and r["baseline_ratio_morpho"]["F1_strong_q"][0] == 7.44
    best = {c: max((v[c][0] for k, v in r.items() if k != "baseline_ratio_morpho"), key=lambda x: x) for c in ("F1_strong_q", "F1_weak_q", "AUPRC_ap")}
    assert best == {"F1_strong_q": 31.89, "F1_weak_q": 11.04, "AUPRC_ap": 13.04}
    assert min(v["FPR_tile_text10"][0] for k, v in r.items() if k != "baseline_ratio_morpho") == 87.89


def test_budget_arithmetic():
    assert sum(V.CAPS_Q.values()) == V.TOTAL_Q == 18000 == int(5.0 * 3600)                                   # 5.0 billed GPU-h in total, no reserve
    for k, c in V.CAPS_Q.items(): assert V.wall(k) * V.B <= c
    assert V.TIERS_E == (5, 4, 3) and V.TIERS_E[0] * 5245 == 26225 and V.CTRL_STEPS == 15 * 4443 == 66645 and V.G1_ORDER == ["d3", "d2", "d1"] and V.LEDGER == "results/gpu_ledger_w4.csv"


def test_stage_configs_are_the_preregistered_ones():
    nm, a = V.stage_cfg("a")
    assert nm == "gpu-wv3a" and [r["name"] for r in a["runs"]] == ["ctrl_wv3", "g_d3", "g_d2", "g_d1"] and a["parallel"] == 2 and a["internet"] is True and a["cache_kernel"] == V.CACHE_KERNEL and a["test_cache"] == "/tmp/wv3_test"
    assert a["billed_cap_s"] == 12000 and a["wall_cap_s"] == 6000 and a["billing_b"] == 2 and a["max_gpus"] == 2 and a["deadline_s"] == 5850 and a["machine_shape"] == "NvidiaTeslaT4"
    ctrl = a["runs"][0]
    assert ctrl == dict(name="ctrl_wv3", arch="mnv2", sched="plateau", plateau_every=2221, seed=0, steps=66645, batch=32, lr=1e-4, wd=0.0, jitter=0, widths="8,16,32,64,128", pos_weight=15.0, device="cuda", amp=True,
                        val_dates="2019-10-22,2019-10-23", eval_every=66645, test_eval=True, allow_test_with_val=True, test_kind="wv3", pretrained="imagenet", in_ch=3)
    for r, n in zip(a["runs"][1:], ["d3", "d2", "d1"]):                                                      # G1: the AVIRIS candidate definitions, validation dates only, 5 paper epochs
        exp = dict(name=f"g_{n}", seed=0, val_dates=G.VAL, eval_every=26225, steps=26225, in_ch=3, **G.COMMON, **G.CANDS[n]); assert r == exp and not r.get("test_eval") and r["hardneg"] == 0.0 and r["pos_weight"] == 1.0
    assert sum(bool(r.get("test_eval")) for r in a["runs"]) == 1, "stage A reads the test split once, for the control"
    nb, b = V.stage_cfg("b", "d1")
    assert nb == "gpu-wv3b" and [r["name"] for r in b["runs"]] == [f"wf_d1_s{s}" for s in range(5)] and [r["seed"] for r in b["runs"]] == list(range(5)) and b["parallel"] == 2 and not b.get("internet") and not b.get("edge")
    assert b["billed_cap_s"] == 5400 and b["wall_cap_s"] == 2700 and b["deadline_s"] == 2550 and all(r["test_eval"] and r["test_kind"] == "wv3" and r["steps"] == 26225 and "val_dates" not in r for r in b["runs"])
    assert V.probe_cfg()["billed_cap_s"] == 600 and V.probe_cfg()["wall_cap_s"] == 300
    from starcop_edge.models import IMAGENET_MNV2
    assert V.IMAGENET == IMAGENET_MNV2 and V.IMAGENET["sha256_prefix"] == "b0353104"
    rp = V.prep_cfg(); assert rp["labels_sha256"].startswith("d55da93f") and rp["meta_sha256"].startswith("14ae7f44") and rp["val_dates"] == ["2019-10-22", "2019-10-23"]


def test_committed_kernels_match_the_generator():
    """the committed kernel directories are what the generator makes for the E and the candidate their own runs name (so the regenerated kernels of 7a / 7b keep passing)"""
    for stage, name in (("a", "gpu-wv3a"), ("b", "gpu-wv3b")):
        kd = os.path.join(ROOT, "kernels", name)
        if not os.path.exists(os.path.join(kd, "cfg.json")): print("  (kernel dir", name, "not generated yet: skipped)"); continue
        cfg = json.load(open(os.path.join(kd, "cfg.json"))); meta = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        runs = {r["name"]: r for r in cfg["runs"]}
        epochs, config = ((runs["g_d1"]["steps"] // 5245, "d1") if stage == "a" else (cfg["runs"][0]["steps"] // 5245, cfg["runs"][0]["name"].split("_")[1]))
        assert epochs in V.TIERS_E and cfg == V.stage_cfg(stage, config, epochs)[1], (name, epochs, config)
        assert meta["is_private"] == "true" and meta["enable_gpu"] == "true" and meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_sources"] == [V.CACHE_KERNEL] and meta["dataset_sources"] == ["chrisphamda/starcop", "sankalpsthakur/bench-gas-assets"]
        assert meta["enable_internet"] == ("true" if stage == "a" else "false") and meta["id"] == f"sankalpsthakur/bench-gas-{name}"
        import ast
        line = next(l for l in open(os.path.join(kd, "script.py")).read().splitlines() if l.startswith("CFG = json.loads(")); assert json.loads(ast.literal_eval(line[len("CFG = json.loads("):-1])) == cfg      # the CFG line of the script is the cfg.json
    kd = os.path.join(ROOT, "kernels", "prep-wv3")
    if os.path.exists(os.path.join(kd, "kernel-metadata.json")):
        m = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        assert m["enable_gpu"] == "false" and m["enable_internet"] == "true" and m["kernel_sources"] == ["sankalpsthakur/bench-gas-prep-cache"] and "wv3" in m["id"]
    kd = os.path.join(ROOT, "kernels", "gpu-probe-wv3")
    if os.path.exists(os.path.join(kd, "cfg.json")):
        m = json.load(open(os.path.join(kd, "kernel-metadata.json"))); assert m["enable_gpu"] == "true" and m["enable_internet"] == "true" and m["dataset_sources"] == [] and json.load(open(os.path.join(kd, "cfg.json"))) == V.probe_cfg()


def probe(scale=1.0, scale_mnv2=1.0, imagenet_ok=True):
    base = {"d1": 80.45, "d2": 67.89, "d3": 49.99, "d5": 53.87, "d2w": 37.68, "mnv2": 18.69}                  # the AVIRIS G0 numbers: an illustration only
    p = {f"{G.PROBE_KEY[c]}|bs32|amp1": {"steps_per_s": v * (scale_mnv2 if c == "mnv2" else scale)} for c, v in base.items()}
    p["imagenet_check"] = {"load_ok": imagenet_ok, "hash_ok": imagenet_ok}
    return p


def test_planner():
    r = V.plan(probe(), 61.9); c = r["chosen"]
    assert r["verdict"] == "OK" and c["epochs"] == 5 and c["stage_a"]["fits"] and c["stage_a"]["g1_hidden_under_control"] and c["stage_b"]["feasible_candidates"] == ["d1", "d2"] and c["stage_b"]["infeasible_candidates"] == ["d3"], r["verdict"]
    assert c["stage_a"]["billed_s"] <= 0.95 * 12000 and max(c["stage_b"]["billed_s"][k] for k in ("d1", "d2")) <= 0.95 * 5400 and c["projected_total_billed_h"] <= 5.0
    print("  illustration with the AVIRIS G0 numbers and sampler 61.9: stage A", c["stage_a"]["wall_s"], "s wall,", c["stage_a"]["billed_s"], "billed; stage B d1", c["stage_b"]["wall_s"]["d1"], "s wall; total", c["projected_total_billed_h"], "billed h")
    assert abs(c["rates_steps_per_s"]["control"] - 0.776 * 18.69) < 0.01                                      # the control rate is the AVIRIS measurement (14.51 steps/s), not the sampler-bound formula
    assert V.plan(probe(0.6), 61.9)["chosen"]["epochs"] == 4                                                  # slower compact runs: the next tier (only d1 still fits stage B)
    assert V.plan(probe(), 45.0)["chosen"]["epochs"] == 3                                                     # a slower crop sampler: the third tier
    for p, smp in ((probe(), 35.0), (probe(), 12.0), (probe(1.0, 0.8), 61.9), (probe(1.0, 0.75), 61.9)):    # nothing fits: stop and ask, nothing is shrunk silently
        assert V.plan(p, smp)["verdict"].startswith("NO FIT") and V.plan(p, smp)["chosen"] is None
    assert V.plan(probe(imagenet_ok=False), 61.9)["verdict"].startswith("NO FIT or NO GO")
    p = probe(); del p["imagenet_check"]; assert V.plan(p, 61.9)["verdict"].startswith("NO FIT or NO GO")
    assert V.plan(probe(1.5), 200.0)["verdict"] == "OK"
    bad = probe(); bad["mnv2_paper|bs32|amp1"] = {"error": "boom"}
    try: V.plan(bad, 61.9); assert False
    except SystemExit as e: assert "G0 failure" in str(e)


def write_run(tmp, rc=0, auprc=13.0, with_metrics=True):
    os.makedirs(os.path.join(tmp, "ctrl_wv3"), exist_ok=True)
    json.dump({"runs": {"ctrl_wv3": {"rc": rc, "device": "cuda", "secs": 5000}}}, open(os.path.join(tmp, "RUN_STATUS.json"), "w"))
    if with_metrics: json.dump(dict(AUPRC_ap=auprc, F1_strong_q=30.0, F1_weak_q=10.0, FPR_tile_text10=88.0, AUPRC_trapz16=9.0, F1_strong_px=31.0, F1_weak_px=9.5, FPR_tile_official640=40.0), open(os.path.join(tmp, "ctrl_wv3", "test_metrics.json"), "w"))


def test_gate_decision():
    pub = os.path.join(ROOT, "protocol", "wv3_published.json"); tmp = tempfile.mkdtemp()
    try:
        for auprc, verdict in ((13.04, "PASS"), (11.08, "PASS"), (15.0, "PASS"), (11.07, "FAIL"), (15.01, "FAIL"), (40.0, "FAIL"), (2.0, "FAIL")):
            write_run(tmp, auprc=auprc); r = GD.decide(tmp, pub); assert r["verdict"] == verdict and r["band"] == [11.08, 15.0], (auprc, r)
        write_run(tmp, auprc=13.5); r = GD.decide(tmp, pub); assert r["z"] == round((13.5 - 13.04) / 1.96, 2) and r["columns"]["F1_strong_q"]["z"] == round((30.0 - 31.89) / 2.44, 2) and r["variants_labelled_not_gated"]["AUPRC_trapz16"] == 9.0
        write_run(tmp, rc=1); assert GD.decide(tmp, pub)["verdict"] == "UNFINISHED"
        write_run(tmp, with_metrics=False); os.remove(os.path.join(tmp, "ctrl_wv3", "test_metrics.json")) if os.path.exists(os.path.join(tmp, "ctrl_wv3", "test_metrics.json")) else None; assert GD.decide(tmp, pub)["verdict"] == "UNFINISHED"
        write_run(tmp, auprc=14.0); p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wv3_gate_decision.py"), tmp], capture_output=True, text=True); assert p.returncode == 0 and '"PASS"' in p.stdout
        write_run(tmp, auprc=20.0); p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wv3_gate_decision.py"), tmp], capture_output=True, text=True); assert p.returncode == 1
    finally: shutil.rmtree(tmp, ignore_errors=True)


def test_stage_b_gate_and_testlog_rows():
    import wv3_b_gate as BG
    tmp = tempfile.mkdtemp()
    try:
        def rs(**runs): json.dump({"runs": {f"g_{k}": ({"rc": 0, "secs": v} if v else {"rc": 1, "secs": 10}) for k, v in runs.items()}}, open(os.path.join(tmp, "RUN_STATUS.json"), "w"))
        assert BG.project(680) == 200 + 3 * 680 + (200 - 30) + 2 * (60 - 30) == 2470
        rs(d1=680, d2=700, d3=870); r = BG.gate(tmp, 2550); assert [r[c]["feasible"] for c in ("d3", "d2", "d1")] == [False, True, True] and r["d1"]["projected_wall_s"] == 2470
        cfg = os.path.join(tmp, "cfg.json"); json.dump({"deadline_s": 2550}, open(cfg, "w"))
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wv3_b_gate.py"), tmp, "--cfg", cfg, "--out", os.path.join(tmp, "f.json")], capture_output=True, text=True)
        assert p.returncode == 0 and json.load(open(os.path.join(tmp, "f.json")))["exclude_from_selection"] == "ctrl_wv3,g_d3", p.stdout
        rs(d1=0, d2=700, d3=870); p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wv3_b_gate.py"), tmp, "--cfg", cfg], capture_output=True, text=True); assert p.returncode == 1          # d1 unfinished: stop and ask
        rs(d1=900, d2=700, d3=870); p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wv3_b_gate.py"), tmp, "--cfg", cfg], capture_output=True, text=True); assert p.returncode == 1          # d1 too slow: stop and ask
        # TEST-LOG rows
        write_run(tmp, auprc=13.5)
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "testlog_rows_wv3.py"), "--stage-a", tmp, "--gate-failed"], capture_output=True, text=True).stdout.strip().splitlines()
        assert len(out) == 6 and out[0].startswith("| 43 |") and "AUPRC(AP) 13.50" in out[0] and all(l.startswith(f"| {44 + i} | not run") and "NOT RUN" in l for i, l in enumerate(out[1:]))
        os.makedirs(os.path.join(tmp, "b", "wf_d1_s0")); json.dump(dict(AUPRC_ap=40.0, F1_strong_q=50.0, F1_weak_q=30.0, FPR_tile_text10=20.0), open(os.path.join(tmp, "b", "wf_d1_s0", "test_metrics.json"), "w"))
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "testlog_rows_wv3.py"), "--stage-a", tmp, "--stage-b", os.path.join(tmp, "b")], capture_output=True, text=True).stdout.strip().splitlines()
        assert out[1].startswith("| 44 |") and "OK. F1 strong/weak (rate) 50.00/30.00" in out[1] and "FAILED or missing" in out[2]
    finally: shutil.rmtree(tmp, ignore_errors=True)


def test_driver_argv_is_accepted_by_train_py():
    """the command lines the GPU driver builds for the stage-A and stage-B runs parse in train.py with the intended values (a flag mismatch would otherwise show after the kernel has been pushed)"""
    import ast, argparse
    src = open(os.path.join(ROOT, "kernels", "train_gpu", "main.py")).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "argv")
    from starcop_edge import train
    class Captured(Exception): pass
    orig = argparse.ArgumentParser.parse_args
    def cap(self, args=None, namespace=None): raise Captured(orig(self, args, namespace))
    argparse.ArgumentParser.parse_args = cap
    try:
        for stage, cfg in (("a", V.stage_cfg("a")[1]), ("b", V.stage_cfg("b", "d1")[1]), ("b3", V.stage_cfg("b", "d2", 3)[1])):
            ns_g = {"cache": "/kaggle/input/x/cache", "mirror": "/kaggle/input/m", "assets": "/kaggle/input/a", "PRETRAINED": "/kaggle/working/imagenet_mobilenet_v2.pth", "CFG": cfg}
            exec(compile(ast.Module([fn], []), "driver", "exec"), ns_g)
            for r in cfg["runs"]:
                try: train.main(ns_g["argv"](r, "cuda")); assert False
                except Captured as c: a = c.args[0]
                assert a.in_ch == 3 and a.device == "cuda" and a.steps == r["steps"] and a.seed == r["seed"] and a.amp is True and a.hardneg == 0.0 and a.batch == 32
                if r["name"] == "ctrl_wv3":
                    assert (a.arch, a.sched, a.plateau_every, a.lr, a.pos_weight, a.jitter) == ("mnv2", "plateau", 2221, 1e-4, 15.0, 0) and a.pretrained == ns_g["PRETRAINED"] and a.allow_test_with_val and a.test_eval
                    assert a.val_dates == "2019-10-22,2019-10-23" and a.test_kind == "wv3" and a.test_cache == "/tmp/wv3_test" and a.test_mirror == "/kaggle/input/m" and a.assets == "/kaggle/input/a"
                elif r["name"].startswith("g_"):
                    assert a.arch == "edge" and a.sched == "cosine" and not a.test_eval and not a.pretrained and a.val_dates == "2019-10-22,2019-10-23" and a.lr == 2e-3 and a.pos_weight == 1.0 and a.stem_stride == 2
                else:
                    assert a.test_eval and a.test_kind == "wv3" and a.test_cache == "/tmp/wv3_test" and not a.val_dates and a.arch == "edge" and a.lr == 2e-3 and a.pos_weight == 1.0
    finally: argparse.ArgumentParser.parse_args = orig


def test_t_quantiles():
    assert abs(C.t_ppf(0.975, 4) - 2.776445) < 1e-5 and abs(C.t_ppf(0.9875, 4) - 3.495406) < 1e-5 and abs(C.t_ppf(0.975, 2) - 4.302653) < 1e-5 and abs(C.t_ppf(0.975, 30) - 2.04227) < 1e-4


def seed_files(tmp, signals, n_tiles=50, seed=0):
    """per-tile summaries of 5 'models' on synthetic 64 x 64 tiles: plume tiles (strong and weak) and empty tiles; logit = noise + signal * label"""
    rng = np.random.default_rng(seed); labs, q = [], []
    for i in range(n_tiles):
        lab = np.zeros((64, 64), np.uint8)
        if i % 3 != 0: s = 6 + (i % 4) * 6; lab[10:10 + s, 12:12 + s] = 1
        labs.append(lab); q.append(0.0 if i % 3 == 0 else (1500.0 if i % 2 else 300.0))
    paths = []
    for k, sig in enumerate(signals):
        r2 = np.random.default_rng(100 + k); sums = [M.tile_summary(r2.normal(-2.0, 2.0, (64, 64)).astype(np.float32) + sig * l, l) for l in labs]
        S = M.stack_summaries(sums); p = os.path.join(tmp, f"s{k}.npz"); np.savez_compressed(p, **S, qplume=np.array(q), ids=np.array([f"t{i}" for i in range(n_tiles)])); paths.append(p)
    return paths


def test_claim_rule():
    tmp = tempfile.mkdtemp()
    try:
        paths = seed_files(tmp, [4.0, 4.2, 3.9, 4.1, 4.0])
        pub = json.loads(json.dumps(PUB)); passed = {"verdict": "PASS"}
        r = C.analyse(paths, pub, passed, B=200, chunk=100)
        m = r["metrics"]["AUPRC_ap"]; assert r["n_seeds"] == 5 and 30 < m["mean"] < 100 and m["boot_lo"] < m["mean"] < m["boot_hi"] and m["t_lo"] < m["mean"] and abs(r["t_quantile"] - 3.495406) < 1e-5
        assert r["boot_percentiles"] == [1.25, 98.75] and r["claim"]["claim"] is True and all(r["claim"]["conditions"].values()), r["claim"]
        assert r["claim"]["wording"].endswith(", as of " + __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d")) and C.analyse(paths, pub, passed, B=100, date="2026-11-02")["claim"]["wording"].endswith("self-evaluated, as of 2026-11-02")      # PLAN rule 5: 'as of DATE'
        assert set(r["metrics"]) >= {"AUPRC_ap", "F1_strong_q", "F1_weak_q", "FPR_tile_text10", "AUPRC_trapz16"}
        # (a) the gate: FAIL or not supplied -> no claim, whatever the numbers
        assert C.analyse(paths, pub, {"verdict": "FAIL"}, B=200)["claim"]["claim"] is False and C.analyse(paths, pub, None, B=200)["claim"]["claim"] is False
        # (b) four seeds -> no claim
        r4 = C.analyse(paths[:4], pub, passed, B=200); assert r4["claim"]["conditions"]["b_all_5_seeds"] is False and r4["claim"]["claim"] is False
        # (c) a margin below one published sd is not a win: put the published mean just under our mean
        pm = m["mean"]; p2 = json.loads(json.dumps(PUB)); p2["rows"]["varon_sanchez"]["AUPRC_ap"] = [pm - 0.5 * 1.96, 1.96]
        c2 = C.analyse(paths, p2, passed, B=200)["claim"]; assert c2["conditions"]["c_margin_over_published_sd"] is False and c2["claim"] is False and c2["wording"] == "not distinguishable from the published row"
        # (d), (e): margin above one sd but the intervals reach the published mean
        p3 = json.loads(json.dumps(PUB)); p3["rows"]["varon_sanchez"]["AUPRC_ap"] = [pm - 2.0 * 1.0, 1.0]
        c3 = C.analyse(paths, p3, passed, B=200)["claim"]; assert c3["conditions"]["c_margin_over_published_sd"] is True
        lo = min(m["boot_lo"], m["t_lo"]); p4 = json.loads(json.dumps(PUB)); p4["rows"]["varon_sanchez"]["AUPRC_ap"] = [lo + 0.01, 0.001]
        c4 = C.analyse(paths, p4, passed, B=200)["claim"]; assert c4["claim"] is False and not (c4["conditions"]["d_lower_tile_bootstrap_above_published_mean"] and c4["conditions"]["e_lower_t_bound_above_published_mean"])
        # below the published mean -> a negative result
        p5 = json.loads(json.dumps(PUB)); p5["rows"]["varon_sanchez"]["AUPRC_ap"] = [pm + 5.0, 1.0]; assert C.analyse(paths, p5, passed, B=200)["claim"]["wording"].startswith("below the published row")
        # the script end to end
        gate = os.path.join(tmp, "gate.json"); json.dump(passed, open(gate, "w")); out = os.path.join(tmp, "claim.json")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wv3_claim.py"), out, "--seeds"] + paths + ["--gate", gate, "--B", "100"], capture_output=True, text=True)
        assert p.returncode == 0 and "CLAIM" in p.stdout and json.load(open(out))["claim"]["claim"] is True, p.stdout + p.stderr
    finally: shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f()
    print("wv3 wave: configs, budget, planner, gate decision, claim rule: all checks passed")
