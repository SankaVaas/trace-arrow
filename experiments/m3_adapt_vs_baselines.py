"""Milestone 3: closed-loop fault injection on a 2-link arm. Methods: NoAdapt, SGD-head (same params/budget),
TRACE (parity-separated RLS on the head). Also compares drift alarms: time-reversal sigma vs plain residual energy.
Scenarios: 'friction' (wear: dv x4, dc x3) | 'gain' (actuator gain x0.7) | 'payload' (link-2 mass x2, a CONSERVATIVE fault)."""
import numpy as np
import torch
from _common import parser, save, plt
from tracearrow.systems import NLinkArm
from tracearrow.model import model_from_arm
from tracearrow.data import MultisineRef, Simulator
from tracearrow.control import ComputedTorque
from tracearrow.adapt import NoAdapt, TraceAdapter, SgdHeadAdapter
from tracearrow.detect import window_stats, EwmaAlarm
from tracearrow.timereversal import localize_refit

ap = parser(__doc__)
args = ap.parse_args()
T_END, T_FAULT, SEEDS = (8.0, 4.0, args.seeds or 1) if args.quick else (14.0, 6.0, args.seeds or 3)
T_CAL = T_FAULT - 1.0
dt, U, W, L = 0.01, 10, 25, 2e-4          # control step, adapt stride, alarm window, process noise
arm0 = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=[0.15, 0.10], dc=[0.05, 0.04])
FAULTS = {
    "friction": lambda a: a.replace(dv=a.dv * 4, dc=a.dc * 3),
    "gain": lambda a: a.replace(gain=a.gain * 0.7),
    "payload": lambda a: a.replace(masses=[a.m[0].item(), a.m[1].item() * 2]),
}


def window(buf, a, b):
    return dict(q=torch.cat(buf["q"][a:b + 1], 0).unsqueeze(0), p=torch.cat(buf["p"][a:b + 1], 0).unsqueeze(0),
                qd=torch.cat(buf["qd"][a:b + 1], 0).unsqueeze(0), u=torch.cat(buf["u"][a:b], 0).unsqueeze(0), dt=dt)


def run(method, scenario, seed):
    model = model_from_arm(arm0)
    adapter = dict(none=NoAdapt, sgd=SgdHeadAdapter, trace=TraceAdapter)[method](model)
    ref = MultisineRef(2, 1, amp=0.6, seed=seed)
    ctrl = ComputedTorque(model, ref)
    q0, qd0, _ = ref(0.0)
    sim = Simulator(arm0, q0, arm0.mass(q0).matmul(qd0.unsqueeze(-1)).squeeze(-1), dt, substeps=5, L=L, seed=100 + seed)
    K, kf = int(T_END / dt), int(T_FAULT / dt)
    buf = dict(q=[sim.q.clone()], p=[sim.p.clone()], qd=[arm0.qdot(sim.q, sim.p)], u=[])
    err, stats, params = [], [], []
    snap = None
    for k in range(K):
        if k == kf:
            sim.sys = FAULTS[scenario](arm0)
        u = ctrl(k, sim.t, sim.q, sim.p)
        sim.step(u)
        buf["u"].append(u.clone()); buf["q"].append(sim.q.clone()); buf["p"].append(sim.p.clone())
        buf["qd"].append(sim.sys.qdot(sim.q, sim.p))
        err.append((sim.q - ref(sim.t)[0]).norm().item())
        if (k + 1) % U == 0:
            adapter.update(window(buf, k + 1 - U, k + 1))
        if (k + 1) % W == 0:
            s = window_stats(model, window(buf, k + 1 - W, k + 1), L)
            stats.append((sim.t, s["sigma"].item(), s["resid"].item()))
        if method == "trace" and snap is None and sim.t >= T_CAL:
            snap = adapter.snapshot()
        if (k + 1) % 50 == 0:
            params.append((sim.t, *model.head.dv.tolist(), *model.head.dc.tolist(), *model.head.gain.tolist()))
    a0, b0 = int((T_FAULT + 0.5) / dt), int((T_FAULT + 3.5) / dt)
    loc = localize_refit(model if method == "none" else model_from_arm(arm0), window(buf, a0, min(b0, K))) if method == "none" else None
    return dict(err=np.array(err), stats=np.array(stats), params=np.array(params), loc=loc)


def rms(e, a, b):
    return float(np.sqrt(np.mean(e[int(a / dt):int(b / dt)] ** 2)))


results = {}
for sc in FAULTS:
    results[sc] = {}
    for m in ("none", "sgd", "trace"):
        runs = [run(m, sc, s) for s in range(SEEDS)]
        results[sc][m] = runs
    print(f"[{sc}] done")

summary = {}
print(f"\n{'scenario':9s} {'method':6s} {'RMS err pre':>12s} {'post (1.5s+)':>13s}")
for sc, d in results.items():
    summary[sc] = {}
    for m, runs in d.items():
        pre = np.mean([rms(r["err"], 1.0, T_FAULT) for r in runs]); post = np.mean([rms(r["err"], T_FAULT + 1.5, T_END) for r in runs])
        summary[sc][m] = dict(rms_pre=pre, rms_post=post)
        print(f"{sc:9s} {m:6s} {pre:12.4f} {post:13.4f}")

print("\nDrift alarm on the no-adaptation run (EWMA, calibrated on healthy windows; delay in seconds, nan = missed):")
det = {}
for sc, runs in {s: d["none"] for s, d in results.items()}.items():
    det[sc] = {}
    for name, col in (("sigma", 1), ("resid", 2)):
        delays, false_alarms = [], 0
        for r in runs:
            st = r["stats"]; t = st[:, 0]
            cal = st[(t > 1.0) & (t <= T_CAL), col]
            al = EwmaAlarm.calibrate(cal, lam=0.3, k=5.0)
            fired = [tt for tt, x in zip(t[t > 1.0], st[t > 1.0, col]) if al.update(x)]
            false_alarms += sum(1 for tt in fired if tt < T_FAULT)
            after = [tt for tt in fired if tt >= T_FAULT]
            delays.append(after[0] - T_FAULT if after else float("nan"))
        det[sc][name] = dict(delays=delays, false_alarms=false_alarms)
        print(f"  {sc:9s} {name:6s} delay {np.round(delays, 2).tolist()}  false alarms before fault: {false_alarms}")

print("\nFault localisation: fraction of the healthy model's residual explained by refitting each parity class")
print("(held-out R^2 on post-fault data, healthy model): friction -> odd; actuator-gain and payload both -> gain (gauge-equivalent, not separable)")
loc = {sc: [r["loc"] for r in d["none"]] for sc, d in results.items()}
for sc, ls in loc.items():
    print(f"  {sc:9s} odd={np.mean([l['odd'] for l in ls]):.3f}  gain={np.mean([l['gain'] for l in ls]):.3f}  even={np.mean([l['even'] for l in ls]):.3f}")

fig, axs = plt.subplots(2, 3, figsize=(13, 6))
for j, sc in enumerate(FAULTS):
    ax = axs[0, j]
    for m, c in (("none", "gray"), ("sgd", "tab:orange"), ("trace", "tab:blue")):
        e = np.mean([np.convolve(r["err"], np.ones(25) / 25, "same") for r in results[sc][m]], 0)
        ax.plot(np.arange(len(e)) * dt, e, label=m, color=c)
    ax.axvline(T_FAULT, color="r", ls=":"); ax.set_title(f"fault: {sc}"); ax.set_ylim(0, None)
    if j == 0:
        ax.set_ylabel("tracking error |q-q_ref|"); ax.legend()
    ax = axs[1, j]
    for name, col, c in (("sigma", 1, "tab:green"), ("resid", 2, "tab:purple")):
        st = results[sc]["none"][0]["stats"]; z = (st[:, col] - st[(st[:, 0] > 1) & (st[:, 0] <= T_CAL), col].mean()) / (st[(st[:, 0] > 1) & (st[:, 0] <= T_CAL), col].std() + 1e-12)
        ax.plot(st[:, 0], z, label=name, color=c)
    ax.axvline(T_FAULT, color="r", ls=":"); ax.set_xlabel("time (s)")
    if j == 0:
        ax.set_ylabel("healthy-standardised stat"); ax.legend()
save(args.out, "m3_adapt_vs_baselines",
     dict(config=dict(T_END=T_END, T_FAULT=T_FAULT, seeds=SEEDS, dt=dt, L=L), tracking=summary, detection=det, localisation=loc), fig)
