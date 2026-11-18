#!/usr/bin/env python3
"""WV3 wave generator and budget planner (protocol/PREREGISTRATION.md Amendment 7). NOTHING is ever pushed from here: the lead pushes with `scripts/gpu_launch.sh` (GPU) or `kaggle kernels push` (CPU) after reviewing the commit.
Unit: BILLED GPU-seconds (GPUs held x wall seconds, Amendment 5); the whole wave has 5.0 billed hours (18,000 s) in total, no retry reserve, on its own ledger results/gpu_ledger_w4.csv
(`LEDGER=results/gpu_ledger_w4.csv GAS_CAP_H=5.0 RESERVE_H=0.0 scripts/gpu_launch.sh kernels/<dir>`). `-t` takes wall seconds = billed cap / 2 (the 2 x Tesla T4 shape).
  prep     : CPU kernel kernels/prep-wv3 (WV3 feature cache; no GPU)          mk_wv3_wave.py prep
  probe    : G0, kernels/gpu-probe-wv3 (synthetic throughput, hardware, ImageNet-weights check)
  plan     : G0 readback + sampler rate on the real cache -> fit of stage A and stage B  mk_wv3_wave.py plan --probe gpuprobe.json --sampler sampler_bench.json [--out protocol/wv3_budget.json]
  stage-a  : kernels/gpu-wv3a, the paper-recipe control (reproduction gate, one worker) beside the G1 validation runs d3, d2, d1 (the other worker)   mk_wv3_wave.py stage-a [--epochs E]
  stage-b  : kernels/gpu-wv3b, G2: five seeds of the frozen candidate, test read once per seed   mk_wv3_wave.py stage-b --config d1 [--epochs E]
E (paper epochs of the compact runs) is the largest of 5, 4, 3 that the plan fits into the caps; the default of the generators is 5 (the placeholder committed with Amendment 7, regenerated after the plan if the plan chooses another tier)."""
import argparse, json, math, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mk_gpu_wave as G
root = G.root

B = 2                                                    # billing factor of the 2 x Tesla T4 shape (Amendment 5)
TIERS_E = (5, 4, 3)                                      # paper-epoch equivalents (1 = 5,245 steps of batch 32) of the compact runs; the plan takes the largest tier that fits the caps (Amendment 7 item 9)
CTRL_EPOCHS = 15                                         # the paper's schedule
CTRL_STEPS = CTRL_EPOCHS * G.CTRL_EPOCH_STEPS            # 66,645 = 15 x 4,443 steps on train minus the validation dates
CAPS_Q = dict(probe=600, a=12000, b=5400)                # billed seconds; sum = 18,000 = 5.0 h
TOTAL_Q = 18000
SOFT_MARGIN = dict(a=150, b=150)                         # wall seconds between the soft deadline and the platform's -t
FRAC = 0.95                                              # plan to at most 95 % of a cap (no retry reserve in this wave; the soft deadline keeps what finished)
G1_ORDER = ["d3", "d2", "d1"]                            # queue order of the validation runs (longest first)
PROBE_BILLED_S = 180                                     # planning figure for G0 (the AVIRIS G0 billed 184 s)
# Planning model: the constants of scripts/mk_gpu_wave.py (Amendments 4/5) for sampler-bound compact runs (0.8 x min(G0 rate, sampler rate), divided by the per-process slowdown rho = 1.15 when both workers run: the AVIRIS seeds
# measured 43.0 steps/s against 43.06 planned), recalibrated where the AVIRIS GPU wave measured otherwise:
#  - GPU-bound control: 22,200 steps of the AVIRIS control (released architecture, batch 32, AMP, plateau checks included) took 1,529.7 s beside a busy second worker = 14.51 steps/s = 0.776 x the G0 rate of 18.69 (planned 0.8 / 1.15 = 0.696)
#  - validation pass of 524 tiles: 9-15 s measured (30 s planned); kernel overhead 42-140 s measured (200 s planned)
#  - the WV3 test pass computes the features of 342 tiles (9 files each, about 150 s) once per kernel: the first run pays it (200 s seeds, 220 s control), later runs read the kernel-local cache (60 s)
K_OVER, EVAL_S, CTRL_EFF, CTRL_TEST_S, SEED_TEST_FIRST_S, SEED_TEST_CACHED_S = 200, 30, 0.776, 220, 200, 60
CACHE_KERNEL = "sankalpsthakur/bench-gas-prep-wv3"
TEST_CACHE = "/tmp/wv3_test"                             # kernel-local directory of the computed test features (not /kaggle/working)
IMAGENET = dict(url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth", sha256_prefix="b0353104")
LEDGER = "results/gpu_ledger_w4.csv"


def wall(stage): return CAPS_Q[stage] // B


def control_run():
    """the paper-recipe control (Amendment 7 item 4): released architecture, ImageNet-pretrained encoder, 3 Varon+Sanchez channels, Adam lr 1e-4, batch 32, pos_weight 15, no mag1c weights, plateau x0.5 / patience 4 checked every half epoch
    on the validation dates, 15 epochs on train minus the validation dates, seed 0, last-step weights, test scored once"""
    return dict(name="ctrl_wv3", arch="mnv2", sched="plateau", plateau_every=G.CTRL_EPOCH_STEPS // 2, seed=0, steps=CTRL_STEPS, batch=32, lr=1e-4, wd=0.0, jitter=0, widths="8,16,32,64,128", pos_weight=15.0, device="cuda", amp=True,
                val_dates=G.VAL, eval_every=CTRL_STEPS, test_eval=True, allow_test_with_val=True, test_kind="wv3", pretrained="imagenet", in_ch=3)


def g1_runs(epochs):
    steps = epochs * G.EPOCH_STEPS
    return [dict(name=f"g_{n}", seed=0, val_dates=G.VAL, eval_every=steps, steps=steps, in_ch=3, **G.COMMON, **G.CANDS[n]) for n in G1_ORDER]


def seed_runs(cfg, epochs):
    return [dict(name=f"wf_{cfg}_s{s}", seed=s, test_eval=True, test_kind="wv3", steps=epochs * G.EPOCH_STEPS, in_ch=3, **G.COMMON, **G.CANDS[cfg]) for s in range(5)]


def stage_cfg(stage, config="d1", epochs=TIERS_E[0]):
    base = dict(gpu=True, machine_shape=G.SHAPE, max_gpus=B, billing_b=B, parallel=2, threads=2, cache_kernel=CACHE_KERNEL, warm_timeout_s=600, test_cache=TEST_CACHE)
    if stage == "a":
        return "gpu-wv3a", dict(base, billed_cap_s=CAPS_Q["a"], wall_cap_s=wall("a"), deadline_s=wall("a") - SOFT_MARGIN["a"], internet=True, imagenet=IMAGENET, runs=[control_run()] + g1_runs(epochs))
    if stage == "b":
        if config not in G.CANDS: raise SystemExit(f"unknown config {config!r}")
        return "gpu-wv3b", dict(base, billed_cap_s=CAPS_Q["b"], wall_cap_s=wall("b"), deadline_s=wall("b") - SOFT_MARGIN["b"], runs=seed_runs(config, epochs))
    raise SystemExit("stage a|b")


def probe_cfg():
    return dict(machine_shape=G.SHAPE, max_gpus=B, billing_b=B, billed_cap_s=CAPS_Q["probe"], wall_cap_s=wall("probe"), imagenet=IMAGENET)


def prep_cfg():
    rp = json.load(open(root / "results" / "prep_cache" / "prep_report.json"))
    return dict(split="train", labels_sha256=rp["sha256"]["train_Y.npy"], meta_sha256=rp["sha256"]["train_meta.csv"], val_dates=G.VAL.split(","), zen_conc=12, zen_workers=16, mirror_workers=8, diag_tiles=300, min_auroc=0.55, bench=True)


def _kernel_dir(name, cfg, meta, kernel_main=None):
    kd = root / "kernels" / name; kd.mkdir(parents=True, exist_ok=True)
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run([sys.executable, str(root / "scripts" / "make_kernel.py"), str(kd), "--cfg", json.dumps(cfg)], check=True)
    (kd / "cfg.json").write_text(json.dumps(cfg, indent=1))
    return kd


def gen_probe():
    cfg = probe_cfg(); meta = G.meta_for("gpu-probe-wv3", cfg, False, True)
    kd = _kernel_dir("gpu-probe-wv3", cfg, meta)
    print(f"generated {kd.relative_to(root)} (NOT pushed): billed cap {CAPS_Q['probe']} s, push with -t {wall('probe')}")


def gen_prep():
    cfg = prep_cfg()
    meta = {"id": "sankalpsthakur/bench-gas-prep-wv3", "title": "bench-gas-prep-wv3", "code_file": "script.py", "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": "", "dataset_sources": ["chrisphamda/starcop", "sankalpsthakur/bench-gas-assets"], "competition_sources": [], "kernel_sources": ["sankalpsthakur/bench-gas-prep-cache"], "model_sources": []}
    kd = root / "kernels" / "prep-wv3"; (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run([sys.executable, str(root / "scripts" / "make_kernel.py"), str(kd), "--cfg", json.dumps(cfg)], check=True)
    print(f"generated {kd.relative_to(root)} (CPU kernel, NOT pushed; no GPU quota)")


def gen_stage(stage, config, epochs):
    name, cfg = stage_cfg(stage, config, epochs)
    subprocess.run([sys.executable, str(root / "scripts" / "push_train.py"), name, json.dumps(cfg), "--no-push"], check=True)
    print(f"generated kernels/{name} (NOT pushed): E = {epochs}, billed cap {cfg['billed_cap_s']} s, push with -t {cfg['wall_cap_s']} (wall seconds), soft deadline {cfg['deadline_s']} s;", [(r['name'], r['steps']) for r in cfg['runs']])


# ---- budget model (wall seconds of each stage; billed = B x wall)
def probe_rate(probe, key):
    rec = probe.get(f"{G.PROBE_KEY[key]}|bs32|amp1", {})
    if "steps_per_s" not in rec: raise SystemExit(f"G0 case {G.PROBE_KEY[key]}|bs32|amp1 has no steps_per_s ({rec.get('error', 'missing')}): treat as a G0 failure, do not plan around it")
    return rec["steps_per_s"]


def plan_tier(probe, sampler, epochs):
    rho = G.RHO_PINNED; steps = epochs * G.EPOCH_STEPS
    eff = {c: G.EFF * min(probe_rate(probe, c), sampler) / rho for c in G1_ORDER}                       # effective steps/s of a compact run with both workers busy
    ctrl_rate = min(CTRL_EFF * probe_rate(probe, "mnv2"), G.EFF * sampler / rho)
    ctrl = G.INIT_S + CTRL_STEPS / ctrl_rate + EVAL_S + CTRL_TEST_S
    g1 = {c: G.INIT_S + steps / eff[c] + EVAL_S for c in G1_ORDER}
    chain = sum(g1.values())                                                                             # the second worker runs d3, d2, d1 one after the other while the first trains the control
    a_wall = K_OVER + max(ctrl, chain)
    seed = {c: G.INIT_S + steps / eff[c] for c in G1_ORDER}                                              # training part of a seed job; the test pass is added below
    b_wall = {c: K_OVER + math.ceil(5 / G.P) * seed[c] + SEED_TEST_FIRST_S + (math.ceil(5 / G.P) - 1) * SEED_TEST_CACHED_S for c in G1_ORDER}   # five seeds on two workers = three rounds; the first round computes the test features; no edge stage
    feasible = [c for c in G1_ORDER[::-1] if B * b_wall[c] <= FRAC * CAPS_Q["b"]]
    worst_b = max(B * b_wall[c] for c in feasible) if feasible else None
    return dict(epochs=epochs, steps=steps, rates_steps_per_s=dict({c: round(v, 2) for c, v in eff.items()}, control=round(ctrl_rate, 2)),
                stage_a=dict(control_job_s=round(ctrl), g1_job_s={c: round(v) for c, v in g1.items()}, g1_chain_s=round(chain), wall_s=round(a_wall), billed_s=round(B * a_wall), fit_limit_billed_s=FRAC * CAPS_Q["a"],
                             fits=bool(B * a_wall <= FRAC * CAPS_Q["a"]), g1_hidden_under_control=bool(chain <= ctrl), soft_deadline_wall_s=wall("a") - SOFT_MARGIN["a"]),
                stage_b=dict(seed_job_s={c: round(v) for c, v in seed.items()}, wall_s={c: round(v) for c, v in b_wall.items()}, billed_s={c: round(B * v) for c, v in b_wall.items()}, fit_limit_billed_s=FRAC * CAPS_Q["b"],
                             feasible_candidates=feasible, infeasible_candidates=[c for c in G1_ORDER if c not in feasible], soft_deadline_wall_s=wall("b") - SOFT_MARGIN["b"]),
                projected_total_billed_h=(round((PROBE_BILLED_S + B * a_wall + worst_b) / 3600, 2) if worst_b else None))


def plan(probe, sampler):
    ic = probe.get("imagenet_check") or {}
    out = dict(model=dict(FRAC=FRAC, EFF=G.EFF, RHO=G.RHO_PINNED, K_OVER=K_OVER, INIT_S=G.INIT_S, EVAL_S=EVAL_S, CTRL_EFF=CTRL_EFF, CTRL_TEST_S=CTRL_TEST_S, SEED_TEST_FIRST_S=SEED_TEST_FIRST_S, SEED_TEST_CACHED_S=SEED_TEST_CACHED_S, P=G.P),
               billing_b=B, tiers=list(TIERS_E), control_epochs=CTRL_EPOCHS, control_steps=CTRL_STEPS, caps_billed_s=CAPS_Q, caps_wall_s={k: wall(k) for k in CAPS_Q}, total_cap_billed_h=TOTAL_Q / 3600, sampler_batches_per_s=sampler,
               imagenet_check=dict(load_ok=bool(ic.get("load_ok")), hash_ok=bool(ic.get("hash_ok"))), tiers_tried={})
    chosen = None
    for e in TIERS_E:
        t = plan_tier(probe, sampler, e); out["tiers_tried"][str(e)] = t
        if t["stage_a"]["fits"] and "d1" in t["stage_b"]["feasible_candidates"] and (t["projected_total_billed_h"] or 99) <= TOTAL_Q / 3600:
            chosen = t; break
    ok = bool(chosen) and out["imagenet_check"]["load_ok"] and out["imagenet_check"]["hash_ok"]
    out["chosen"] = chosen; out["verdict"] = "OK" if ok else "NO FIT or NO GO: stop and ask the owner (nothing is shrunk silently): chosen tier %s, ImageNet weights %s" % (chosen and chosen["epochs"], out["imagenet_check"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["prep", "probe", "plan", "stage-a", "stage-b"]); ap.add_argument("--config", default="d1")
    ap.add_argument("--epochs", type=int, default=TIERS_E[0], help="stage-a / stage-b: paper-epoch equivalents of the compact runs (the tier chosen by the plan)"); ap.add_argument("--probe", default=""); ap.add_argument("--sampler", default="", help="sampler_bench.json of the prep-wv3 kernel, or a number (batches/s)"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.what == "prep": gen_prep()
    elif a.what == "probe": gen_probe()
    elif a.what in ("stage-a", "stage-b"):
        if a.epochs not in TIERS_E: raise SystemExit(f"--epochs must be one of {TIERS_E}")
        gen_stage(a.what[-1], a.config, a.epochs)
    else:
        pr = json.load(open(a.probe))
        if "aborted" in pr: raise SystemExit("G0 aborted: " + pr["aborted"])
        try: sampler = float(a.sampler)
        except ValueError: sampler = float(json.load(open(a.sampler))["batch32_batches_per_s"])
        res = plan(pr, sampler); res["probe_device"] = pr.get("device"); res["probe_torch"] = pr.get("torch"); res["machine_shape"] = G.SHAPE
        print(json.dumps(res, indent=1))
        if a.out: json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)
        sys.exit(0 if res["verdict"] == "OK" else 3)
