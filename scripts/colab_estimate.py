#!/usr/bin/env python3
"""Colab alternative (STATUS.md, section 'Colab alternative'): the Amendment 5 planner model converted to ONE T4 and ONE worker process (b = 1, P = 1, per-process slowdown 1.0).
Nothing here touches Kaggle or Colab; stdlib only, run through bin/guard.sh. The probe rates are SYNTHETIC placeholders unless --probe points to a real gpuprobe.json; a Colab run of the same probe replaces them.
usage: colab_estimate.py [--probe gpuprobe.json] [--sampler 61.9]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import mk_gpu_wave as W
ap = argparse.ArgumentParser(); ap.add_argument("--probe", default=""); ap.add_argument("--sampler", type=float, default=61.9)
a = ap.parse_args()
SYN = {"d1": 90, "d2": 90, "d3": 55, "d5": 50, "d2w": 32, "mnv2": 38}
probe = json.load(open(a.probe)) if a.probe else {f"{W.PROBE_KEY[c]}|bs32|amp1": {"steps_per_s": v} for c, v in SYN.items()}
src = "real probe " + a.probe if a.probe else "SYNTHETIC probe rates " + str(SYN)
RHO = 1.0   # one process alone on one GPU


def seed_job(E, c): return W.job_s(E * W.EPOCH_STEPS, W.rate(probe, a.sampler, c), False, True, RHO)
def g1_job(E, c): return W.job_s(E * W.EPOCH_STEPS, W.rate(probe, a.sampler, c), True, False, RHO)
def ctrl_job(Ec): return W.job_s(Ec * W.CTRL_EPOCH_STEPS, W.rate(probe, a.sampler, "mnv2"), True, True, RHO) + W.PLATEAU_S
def t_g1(E, S): return W.K_OVER + sum(g1_job(E, c) for c in S)
def t_g2(E, c, Ec): return W.K_OVER + W.EDGE_S + 5 * seed_job(E, c) + ctrl_job(Ec)       # one queue, one worker: 5 seeds, then the control


def plan1():
    cap_g1, cap_g2 = W.FRAC * W.CAPS_Q["wide"], W.FRAC * (W.CAPS_Q["compact"] + W.CAPS_Q["control"])    # the Amendment 5 caps, now in T4-wall seconds (b = 1)
    for E in W.TIERS_E:
        for S in W.CAND_SETS:
            if t_g1(E, S) > cap_g1: continue
            ec = next((Ec for Ec in W.TIERS_E if all(t_g2(E, c, Ec) <= cap_g2 for c in S)), None)
            if ec: return E, S, ec
    return None


h = lambda s: f"{s/3600:.2f} h"
print(f"{src}; sampler ceiling {a.sampler} batches/s; planning rate = 0.8 x min(probe, sampler) steps/s: " + ", ".join(f"{c} {W.rate(probe, a.sampler, c):.1f}" for c in list(W.CANDS) + ["mnv2"]))
print("\nPer-job wall time on one T4, one worker (s; includes 40 s init, 75 s validation pass for G1 runs / 80 s test pass for G2 seeds; control = validation + test + 60 s plateau checks):")
print("| steps | " + " | ".join(f"E={E} ({E * W.EPOCH_STEPS:,})" for E in W.TIERS_E) + " |\n|---|" + "---|" * len(W.TIERS_E))
for c in ["d1", "d2", "d3", "d5", "d2w"]:
    print(f"| {c} G1 run / G2 seed | " + " | ".join(f"{g1_job(E, c):,.0f} / {seed_job(E, c):,.0f} ({g1_job(E, c)/60:.0f} / {seed_job(E, c)/60:.0f} min)" for E in W.TIERS_E) + " |")
print("| control mnv2 | " + " | ".join(f"Ec={E} ({E * W.CTRL_EPOCH_STEPS:,} steps): {ctrl_job(E):,.0f} ({ctrl_job(E)/60:.0f} min)" for E in W.TIERS_E) + " |")
p = plan1()
print("\nConverted plan (Amendment 5 search order and 80 % rule, caps = wide 7,800 s and G2 + control 16,200 s of T4 wall time):")
if not p: print("NO FIT")
else:
    E, S, ec = p
    g1 = t_g1(E, S); g2w = {c: t_g2(E, c, ec) for c in S}; worst = max(g2w.values())
    print(f"E = {E} ({E * W.EPOCH_STEPS:,} steps), candidates {S}, control {ec} epochs ({ec * W.CTRL_EPOCH_STEPS:,} steps); G1 {g1:,.0f} s = {h(g1)}; G2 + control per chosen config: " + ", ".join(f"{c} {g2w[c]:,.0f} s = {h(g2w[c])}" for c in S) + f"; worst-case total with G0 (~600 s) = {h(600 + g1 + worst)} of T4 time")
    for c in S: print(f"  if {c} is chosen: total {h(600 + g1 + g2w[c])} = G0 0.17 h + G1 {h(g1)} + G2 + control {h(g2w[c])}")
print("\nCompare, same inputs on Kaggle's 2 x T4 shape with two workers (Amendment 5): see STATUS.md / `mk_gpu_wave.py plan` (billed hours = 2 x wall).")
