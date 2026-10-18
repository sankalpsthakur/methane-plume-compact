#!/usr/bin/env python3
"""GPU wave generator and budget planner (protocol/PREREGISTRATION.md Amendments 4 and 5; protocol/GPU-PLAN.md). NOTHING is ever pushed from here: GPU kernels are pushed by hand
with `kaggle kernels push -p kernels/<dir> -t <wall seconds>` after `scripts/gpu_gate.sh <dir>` passes (GPU day runbook in STATUS.md).
AMENDMENT 5 UNIT: the caps are BILLED GPU-seconds (GPUs held x wall seconds = the unit of `kaggle quota`); the quota is billed per GPU (2 x T4 held = 2 billed hours per wall hour,
verified 2026-10-10: RESOURCES.md FINDING row and a direct quota-vs-wall measurement). `-t` takes WALL seconds = cap // b, b = billed GPUs (the billing factor).
  probe   : G0 throughput/hardware probe kernel (synthetic data)        mk_gpu_wave.py probe [--shape S] [--billing b]
  plan    : probe + sampler rate -> epochs E, candidate set, control epochs, projected billed seconds vs the caps  (writes protocol/gpu_budget.json)
            mk_gpu_wave.py plan --probe kernel_outputs/gpu-probe/gpuprobe.json --sampler results/gpu_smoke/sampler_bench.json [--billing b] [--out protocol/gpu_budget.json]
  wide    : G1 dev runs on the validation dates, 1 seed each, E paper-epochs, no test access     mk_gpu_wave.py wide --epochs E --cands d1,d2,d3 [--shape S] [--billing b]
  compact : G2 final wave of ONE frozen config: seeds 0..4 on all of train, test once per seed, ONNX fp32/int8 edge export     mk_gpu_wave.py compact --epochs E --config dN [--shape S] [--billing b]
  control : G3 paper-recipe mnv2 control (Adam lr 1e-4, batch 32, plateau x0.5 / patience 4 checked twice per epoch, no jitter) on train minus the validation dates   mk_gpu_wave.py control --epochs Ec [--shape S] [--billing b]
  smoke   : CPU twin of the GPU wave (private CPU kernel, no GPU quota)"""
import argparse, json, math, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parent.parent
VAL = "2019-10-22,2019-10-23"
SHAPE = "NvidiaTeslaT4"                       # documented shape; on this account it is 2 x Tesla T4, billed 2x. Other strings were silently replaced by that default (small-lm FINDING 2026-10-10)
BILLING = 2                                   # default billing factor b for the generators: billed GPUs of the default shape; set from the G0 readback (n_gpu) for anything else
EPOCH_STEPS = 5245                            # one paper epoch at batch 32 = 167,840 windows = 5,245 steps (15 epochs = 78,675 steps)
CTRL_EPOCH_STEPS = 4443                       # control trains on train minus the validation dates: 2,901 tiles x 49 windows = 142,149 windows = 4,443 steps/epoch at batch 32
COMMON = dict(batch=32, lr=2e-3, wd=0.0, jitter=32, device="cuda", amp=True)
CANDS = {  # d1,d2 as in the CPU waves (Amendment 2); d3,d5 from Amendment 1; d2w = d2 widths with stem stride 1 (Amendment 4)
    "d1": dict(widths="4,8,16,32,64", stem_stride=2, pos_weight=1.0, hardneg=0.0), "d2": dict(widths="8,16,32,64,128", stem_stride=2, pos_weight=1.0, hardneg=0.0),
    "d3": dict(widths="16,32,64,128,256", stem_stride=2, pos_weight=1.0, hardneg=0.0), "d5": dict(widths="4,8,16,32,64", stem_stride=1, pos_weight=1.0, hardneg=0.0),
    "d2w": dict(widths="8,16,32,64,128", stem_stride=1, pos_weight=1.0, hardneg=0.0)}
COST_ORDER = ["d2w", "d5", "d3", "d2", "d1"]   # most expensive first
PROBE_KEY = {"d1": "d1_4-64_s2", "d2": "d2_8-128_s2", "d3": "d3_16-256_s2", "d5": "d5_4-64_s1", "d2w": "d2w_8-128_s1", "mnv2": "mnv2_paper"}
# ---- budget model (Amendments 4 and 5). Caps are BILLED GPU-seconds; they sum to 25,200 = 7.00 billed hours, leaving 1.00 billed hour of the 8 h gas cap for one retry (Amendment 5 item 7).
CAPS_Q = dict(probe=1200, wide=7800, compact=10800, control=5400)
RESERVE_Q = 3600
GAS_CAP_Q = sum(CAPS_Q.values()) + RESERVE_Q                 # 28,800 billed s = 8.00 h
SOFT_MARGIN = dict(wide=600, compact=1200, control=600)      # wall seconds: CFG deadline_s = wall cap - margin (the compact margin leaves room for the ONNX/int8 edge stage)
FRAC, EFF, K_OVER, P = 0.8, 0.8, 300, 2                      # plan to <= 80 % of a billed cap; real rate = 0.8 x min(probe rate, sampler rate); 2 worker processes; kernel overhead 200 s
RHO_SHARED, RHO_PINNED = 1.5, 1.15                           # per-process slowdown: 2 processes share ONE GPU (1.5) / each has its own GPU through CUDA_VISIBLE_DEVICES (1.15: host CPU contention)
EVAL_S, TEST_S, INIT_S, EDGE_S, PLATEAU_S = 75, 80, 40, 300, 60
TIERS_E = (15, 10, 5)                                        # paper-epoch equivalents of the G1/G2 schedule; control epochs are chosen separately from the same list
CAND_SETS = (["d1", "d2", "d3", "d5", "d2w"], ["d1", "d2", "d3", "d5"], ["d1", "d2", "d3"])   # d2w, then d5, are dropped (most expensive first) when a set does not fit; d1, d2 and the wider d3 are never dropped


def wall_cap(stage, b): return CAPS_Q[stage] // b             # wall seconds for `kaggle kernels push -t`
def rho(n_gpu): return RHO_PINNED if n_gpu >= P else RHO_SHARED


def rate(probe, sampler, c):
    rec = probe.get(f"{PROBE_KEY[c]}|bs32|amp1", {})
    if "steps_per_s" not in rec:
        raise SystemExit(f"G0 case {PROBE_KEY[c]}|bs32|amp1 has no steps_per_s ({rec.get('error', 'missing')}): treat as a G0 failure (Amendment 4 item 8), do not plan around it")
    return EFF * min(rec["steps_per_s"], sampler)


def job_s(steps, r, val, test, rh):
    return INIT_S + steps * rh / r + (EVAL_S if val else 0) + (TEST_S if test else 0)


def lpt(durations, p=P):
    slots = [0.0] * p
    for d in sorted(durations, reverse=True):
        slots[slots.index(min(slots))] += d
    return max(slots)


# wall seconds of each stage; the billed seconds are b x wall (every GPU held is billed, busy or idle: the single control run bills b x its wall time)
def t_wide(E, cands, probe, sampler, n_gpu):
    return K_OVER + lpt([job_s(E * EPOCH_STEPS, rate(probe, sampler, c), True, False, rho(n_gpu)) for c in cands])


def t_compact(E, c, probe, sampler, n_gpu):
    return K_OVER + EDGE_S + math.ceil(5 / P) * job_s(E * EPOCH_STEPS, rate(probe, sampler, c), False, True, rho(n_gpu))


def t_control(Ec, probe, sampler):
    return K_OVER + job_s(Ec * CTRL_EPOCH_STEPS, rate(probe, sampler, "mnv2"), True, True, 1.0) + PLATEAU_S


CAPS_MERGED = CAPS_Q["compact"] + CAPS_Q["control"]          # Amendment 5 note 1: the control runs in G2's kernel, as the last job of the shared queue: 16,200 billed s


def sched(durations, p=P):
    """List scheduling in queue order: the driver starts the next queued run on the first free worker slot."""
    slots = [0.0] * p
    for d in durations:
        slots[slots.index(min(slots))] += d
    return max(slots)


def merged_jobs(E, c, Ec, probe, sampler, n_gpu):
    seed = job_s(E * EPOCH_STEPS, rate(probe, sampler, c), False, True, rho(n_gpu))
    ctrl = job_s(Ec * CTRL_EPOCH_STEPS, rate(probe, sampler, "mnv2"), True, True, rho(n_gpu)) + PLATEAU_S
    return seed, ctrl


def t_merged(E, c, Ec, probe, sampler, n_gpu):
    seed, ctrl = merged_jobs(E, c, Ec, probe, sampler, n_gpu)
    return K_OVER + EDGE_S + sched([seed] * 5 + [ctrl])           # queue order: the 5 seeds first, the control last, so a soft deadline cuts the control before any seed


def plan(probe, sampler, b=BILLING, n_gpu=None, merge=True):
    n_gpu = b if n_gpu is None else n_gpu
    out = dict(model=dict(FRAC=FRAC, EFF=EFF, RHO=rho(n_gpu), K=K_OVER, P=P, EVAL_S=EVAL_S, TEST_S=TEST_S, INIT_S=INIT_S, EDGE_S=EDGE_S, PLATEAU_S=PLATEAU_S), billing_b=b, n_gpu=n_gpu,
               caps_billed_s=CAPS_Q, caps_wall_s={k: wall_cap(k, b) for k in CAPS_Q}, sampler_batches_per_s=sampler, machine_shape=SHAPE)
    out["rates_steps_per_s"] = {c: round(rate(probe, sampler, c), 2) for c in list(CANDS) + ["mnv2"]}
    out["wave"] = None; out["merged"] = None
    for E in TIERS_E:
        for S in CAND_SETS:
            tw = t_wide(E, S, probe, sampler, n_gpu)
            if merge:   # G2 + G3 in one kernel: the control epochs Ec are the largest tier at which EVERY remaining candidate's merged kernel fits (so selection can never pick a config without a control)
                ec = next((Ec for Ec in TIERS_E if all(b * t_merged(E, c, Ec, probe, sampler, n_gpu) <= FRAC * CAPS_MERGED for c in S)), None)
                if b * tw > FRAC * CAPS_Q["wide"] or ec is None: continue
                tc = {c: t_merged(E, c, ec, probe, sampler, n_gpu) for c in S}
                out["merged"] = dict(control_epochs=ec, control_steps=ec * CTRL_EPOCH_STEPS, plateau_every=CTRL_EPOCH_STEPS // 2, billed_cap_s=CAPS_MERGED, wall_cap_s=CAPS_MERGED // b,
                                     soft_deadline_wall_s=CAPS_MERGED // b - SOFT_MARGIN["compact"], per_candidate={c: dict(seed_job_s=round(merged_jobs(E, c, ec, probe, sampler, n_gpu)[0]), control_job_s=round(merged_jobs(E, c, ec, probe, sampler, n_gpu)[1]),
                                     wall_s=round(tc[c]), billed_s=round(b * tc[c])) for c in S})
                capc = CAPS_MERGED
            else:
                tc = {c: t_compact(E, c, probe, sampler, n_gpu) for c in S}; capc = CAPS_Q["compact"]
                if not (b * tw <= FRAC * CAPS_Q["wide"] and b * max(tc.values()) <= FRAC * capc): continue
            if True:
                out["wave"] = dict(epochs=E, steps=E * EPOCH_STEPS, cands=S, projected_wide_wall_s=round(tw), projected_wide_billed_s=round(b * tw), projected_compact_worst_wall_s=round(max(tc.values())),
                                   projected_compact_worst_billed_s=round(b * max(tc.values())), projected_compact_wall_s={c: round(v) for c, v in tc.items()},
                                   soft_deadline_wall_s=dict(wide=wall_cap("wide", b) - SOFT_MARGIN["wide"], compact=(CAPS_MERGED // b if merge else wall_cap("compact", b)) - SOFT_MARGIN["compact"]))
                break
        if out["wave"]: break
    out["control"] = None
    if merge and out["merged"]:
        m = out["merged"]; out["control"] = dict(epochs=m["control_epochs"], steps=m["control_steps"], plateau_every=m["plateau_every"], merged_into="compact", soft_deadline_wall_s=m["soft_deadline_wall_s"])
    elif not merge:
        for Ec in TIERS_E:
            tc = t_control(Ec, probe, sampler)
            if b * tc <= FRAC * CAPS_Q["control"]:
                out["control"] = dict(epochs=Ec, steps=Ec * CTRL_EPOCH_STEPS, plateau_every=CTRL_EPOCH_STEPS // 2, projected_wall_s=round(tc), projected_billed_s=round(b * tc),
                                      soft_deadline_wall_s=wall_cap("control", b) - SOFT_MARGIN["control"]); break
    out["sum_caps_billed_h"] = round(sum(CAPS_Q.values()) / 3600, 2); out["gas_cap_billed_h"] = round(GAS_CAP_Q / 3600, 2); out["merge_control_into_g2"] = merge
    if out["wave"] and out["control"]:
        out["projected_total_billed_h"] = round((out["wave"]["projected_wide_billed_s"] + out["wave"]["projected_compact_worst_billed_s"] + (0 if merge else out["control"]["projected_billed_s"])) / 3600, 2)
    out["verdict"] = "OK" if out["wave"] and out["control"] else "NO FIT: stop and ask the owner (do not shrink silently)"
    return out


def control_run(steps):
    return dict(name="ctrl_mnv2", arch="mnv2", sched="plateau", plateau_every=CTRL_EPOCH_STEPS // 2, seed=0, steps=steps, batch=32, lr=1e-4, wd=0.0, jitter=0, widths="8,16,32,64,128", device="cuda", amp=True,
                val_dates=VAL, eval_every=steps, test_eval=True, allow_test_with_val=True)


def meta_for(name, cfg, with_test, internet):
    return {"id": f"sankalpsthakur/bench-gas-{name}", "title": f"bench-gas-{name}", "code_file": "script.py", "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_tpu": "false", "enable_internet": "true" if internet else "false", "machine_shape": cfg["machine_shape"], "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}


def billing_fields(stage, a):
    b = a.billing; wc = wall_cap(stage, b)
    return dict(machine_shape=a.shape, max_gpus=b, billing_b=b, billed_cap_s=CAPS_Q[stage], wall_cap_s=wc), wc


def gen(stage, a):
    if stage == "probe":
        bf, wc = billing_fields("probe", a); cfg = dict(bf)
        kd = root / "kernels" / "gpu-probe"; (kd / "kernel-metadata.json").write_text(json.dumps(meta_for("gpu-probe", cfg, False, False), indent=1))
        subprocess.run([sys.executable, str(root / "scripts/make_kernel.py"), str(kd), "--cfg", json.dumps(cfg)], check=True); (kd / "cfg.json").write_text(json.dumps(cfg, indent=1))
        print(f"generated kernels/gpu-probe (NOT pushed): billed cap {CAPS_Q['probe']} s, push with -t {wc}", json.dumps(cfg)); return
    if stage == "wide":
        cands = a.cands.split(","); assert all(c in CANDS for c in cands), cands
        steps = a.steps or a.epochs * EPOCH_STEPS
        order = [c for c in COST_ORDER if c in cands]   # longest first (the LPT order assumed by the budget model)
        runs = [dict(name=f"g_{n}", seed=0, val_dates=VAL, eval_every=steps, steps=steps, **COMMON, **CANDS[n]) for n in order]   # eval_every = steps: final validation only (no mid-run evals)
        bf, wc = billing_fields("wide", a); cfg = dict(gpu=True, **bf, parallel=P, threads=2, deadline_s=wc - SOFT_MARGIN["wide"], runs=runs); kname = "gpu-wide"
    elif stage == "compact":
        steps = a.steps or a.epochs * EPOCH_STEPS
        runs = [dict(name=f"gf_{a.config}_s{s}", seed=s, test_eval=True, steps=steps, **COMMON, **CANDS[a.config]) for s in range(5)]
        bf, wc = billing_fields("compact", a); extra = {}
        if a.with_control:   # Amendment 5 note 1: the paper-recipe control is the LAST job of the same queue, so the second worker never idles while it trains; the edge stage still covers the 5 seeds only
            runs = runs + [control_run(a.with_control * CTRL_EPOCH_STEPS)]; extra = dict(edge_runs=[r["name"] for r in runs[:5]])
            bf = dict(bf, billed_cap_s=CAPS_MERGED, wall_cap_s=CAPS_MERGED // a.billing); wc = CAPS_MERGED // a.billing
        cfg = dict(gpu=True, **bf, parallel=P, threads=2, edge=True, edge_released=False, deadline_s=wc - SOFT_MARGIN["compact"], **extra, runs=runs); kname = "gpu-compact"
    elif stage == "smoke":  # CPU twin of the GPU day (a private CPU kernel; Amendment 4 item 9): the SAME driver, src and run dictionaries, device cpu, 3-4 steps, no test access, no edge stage
        wide = [dict(name=f"s_{n}", seed=0, val_dates=VAL, val_max=48, eval_every=2, steps=3, **{**COMMON, "device": "cpu"}, **CANDS[n]) for n in COST_ORDER]
        ctrl = dict(name="s_ctrl", arch="mnv2", sched="plateau", plateau_every=2, seed=0, steps=4, batch=32, lr=1e-4, wd=0.0, jitter=0, widths="8,16,32,64,128", device="cpu", amp=True,
                    val_dates=VAL, val_max=48, eval_every=4, allow_test_with_val=True)
        fin = dict(name="s_final", seed=0, steps=3, **{**COMMON, "device": "cpu"}, **CANDS["d1"])
        dl = dict(name="s_deadline", seed=0, steps=100000, **{**COMMON, "device": "cpu"}, **CANDS["d1"])   # never finishes: the soft deadline must terminate it and keep the others
        cfg = dict(gpu=False, driver="train_gpu", bench_sampler=True, max_gpus=a.billing, parallel=3, threads=2, deadline_s=a.deadline, runs=[dl] + wide + [ctrl, fin]); kname = "gpu-smoke"; steps = 0; runs = cfg["runs"]; wc = 0
    else:
        steps = a.steps or a.epochs * CTRL_EPOCH_STEPS
        runs = [control_run(steps)]
        bf, wc = billing_fields("control", a); cfg = dict(gpu=True, **bf, parallel=1, threads=2, deadline_s=wc - SOFT_MARGIN["control"], runs=runs); kname = "gpu-control"
    if cfg.get("deadline_s", 1) <= 0: raise SystemExit(f"wall cap {wc} s is not above the soft margin: billing factor {a.billing} too large for this stage")
    subprocess.run([sys.executable, str(root / "scripts/push_train.py"), kname, json.dumps(cfg), "--no-push"], check=True)
    print("generated kernels/%s (NOT pushed):" % kname, json.dumps({k: v for k, v in cfg.items() if k != "runs"}), [(r["name"], r["steps"]) for r in runs])
    if stage != "smoke": print(f"billed cap {cfg.get('billed_cap_s', CAPS_Q.get(stage))} s = {cfg.get('billed_cap_s', CAPS_Q.get(stage))/3600:.2f} h; push with -t {wc} (wall seconds), after scripts/gpu_gate.sh kernels/{kname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["plan", "probe", "wide", "compact", "control", "smoke"])
    ap.add_argument("--epochs", type=int, default=15, help="paper-epoch equivalents: steps = epochs x 5,245 (wide/compact) or x 4,443 (control)"); ap.add_argument("--steps", type=int, default=0, help="explicit step count (overrides --epochs)")
    ap.add_argument("--cands", default="d1,d2,d3,d5,d2w"); ap.add_argument("--config", default="d1")
    ap.add_argument("--with-control", type=int, default=0, help="compact only: control epochs Ec; the control joins G2's queue as its last job (Amendment 5 note 1), cap = G2 + G3 caps")
    ap.add_argument("--separate-control", action="store_true", help="plan only: plan G3 as its own kernel (Amendment 5 item 3) instead of the merged G2+G3 kernel")
    ap.add_argument("--shape", default=SHAPE, help="machine_shape written into the metadata (NvidiaTeslaT4 default = 2 x T4 billed 2x; a single-GPU string only after a verified readback, Amendment 5 item 5)")
    ap.add_argument("--billing", type=float, default=None, help="billing factor b = billed GPUs of the shape actually obtained (default: 2 for generators, the probe's n_gpu for plan)")
    ap.add_argument("--n-gpu", type=int, default=None, help="plan only: GPUs obtained (default: from the probe env)")
    ap.add_argument("--deadline", type=int, default=720, help="smoke only: soft deadline (s)")
    ap.add_argument("--probe", default=""); ap.add_argument("--sampler", default="", help="sampler_bench.json of the CPU smoke kernel, or a number (batches/s)"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.stage == "plan":
        pr = json.load(open(a.probe))
        if "aborted" in pr: raise SystemExit("G0 aborted: " + pr["aborted"])
        try: sampler = float(a.sampler)
        except ValueError: sampler = float(json.load(open(a.sampler))["batch32_batches_per_s"])
        n_gpu = a.n_gpu or int(pr.get("env", {}).get("n_gpu", BILLING)); b = int(a.billing) if a.billing else n_gpu
        res = plan(pr, sampler, b, n_gpu, merge=not a.separate_control); res["probe_device"] = pr.get("device"); res["probe_torch"] = pr.get("torch"); res["machine_shape"] = a.shape
        print(json.dumps(res, indent=1))
        if a.out: json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)
        sys.exit(0 if res["verdict"] == "OK" else 3)
    a.billing = int(a.billing) if a.billing else BILLING
    gen(a.stage, a)
