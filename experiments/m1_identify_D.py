"""Milestone 1: is dissipation identifiable from ordinary closed-loop motion, and does parity separation
protect it from conservative / actuator model error? Baseline = naive fit that trusts the model's conservative part."""
import numpy as np
import torch
from _common import parser, save, plt
from tracearrow.systems import NLinkArm
from tracearrow.model import model_from_arm, AnalyticCore, DissipativeHead, TRACEModel
from tracearrow.data import MultisineRef, Simulator, rollout
from tracearrow.control import ComputedTorque
from tracearrow.timereversal import estimate_dissipation

ap = parser(__doc__)
args = ap.parse_args()
B, K, seeds = (4, 300, 2) if args.quick else (8, 800, args.seeds or 3)
dt = 0.01
arm = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=[0.15, 0.10], dc=[0.05, 0.04])
tdv, tdc = arm.dv, arm.dc

# name -> (model builder, gauge scale s: estimates are divided by s because (sM, sV, s*gain, s*F) is dynamically equivalent)
def nonuniform():
    head = DissipativeHead(2, eps=arm.eps); head.set_values(dv=arm.dv, dc=arm.dc, gain=torch.ones(2))
    return TRACEModel(AnalyticCore(arm.replace(masses=[1.0, 1.1])), head)

cases = {
    "exact model": (lambda: model_from_arm(arm), 1.0),
    "gravity x0.8": (lambda: model_from_arm(arm, grav_scale=0.8), 1.0),
    "actuator gain x0.8": (lambda: model_from_arm(arm, gain0=0.8), 1.0),
    "uniform inertia x1.2 (gauge-corrected)": (lambda: model_from_arm(arm, mass_scale=1.2), 1.2),
    "mass+grav+gain combined (gauge-corrected)": (lambda: model_from_arm(arm, mass_scale=1.15, grav_scale=0.9, gain0=0.85), 1.15),
    "NON-uniform inertia (link-2 mass +37%)": (nonuniform, 1.0),
}
res = {c: {"naive": [], "parity": []} for c in cases}
for s in range(seeds):
    ref = MultisineRef(2, B, amp=0.6, seed=s)
    traj = rollout(Simulator(arm, torch.zeros(B, 2), torch.zeros(B, 2), dt), ComputedTorque(model_from_arm(arm), ref), K)
    for c, (build, gs) in cases.items():
        for lab, par in (("naive", False), ("parity", True)):
            e = estimate_dissipation(build(), traj, parity=par)
            err_v = ((e["dv"] / gs - tdv).abs() / tdv).mean().item() * 100
            err_c = ((e["dc"] / gs - tdc).abs() / tdc).mean().item() * 100
            res[c][lab].append((err_v, err_c))

summary = {c: {lab: dict(dv_err_pct=float(np.mean([x[0] for x in v])), dc_err_pct=float(np.mean([x[1] for x in v])))
               for lab, v in d.items()} for c, d in res.items()}
print(f"{'case':46s} {'naive dv/dc err %':>20s} {'parity dv/dc err %':>22s}")
for c, d in summary.items():
    print(f"{c:46s} {d['naive']['dv_err_pct']:9.1f}/{d['naive']['dc_err_pct']:<9.1f} {d['parity']['dv_err_pct']:11.1f}/{d['parity']['dc_err_pct']:<9.1f}")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(cases)); w = 0.38
ax.bar(x - w / 2, [summary[c]["naive"]["dc_err_pct"] for c in cases], w, label="naive")
ax.bar(x + w / 2, [summary[c]["parity"]["dc_err_pct"] for c in cases], w, label="parity-separated (TRACE)")
ax.set_xticks(x); ax.set_xticklabels([c.split(" (")[0] for c in cases], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Coulomb-friction error (%)"); ax.legend(); ax.set_title("M1: identifying dissipation under model mismatch")
save(args.out, "m1_identify_D", dict(config=dict(B=B, K=K, seeds=seeds, dt=dt), summary=summary), fig)
