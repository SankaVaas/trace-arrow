"""Milestone 2: calibration of the fluctuation statistic E[exp(-sigma)] (== 1 for a correct model).
Sim windows come from a known initial density and known protocol, with process noise L. Friction includes Coulomb
(non-FDT) terms, so this also shows the identity itself does not need fluctuation-dissipation consistency."""
import numpy as np
import torch
from _common import parser, save, plt
from tracearrow.systems import NLinkArm
from tracearrow.model import model_from_arm
from tracearrow.calib import ft_windows
from tracearrow.timereversal import sigma, ft_statistic

ap = parser(__doc__)
args = ap.parse_args()
B = 4000 if args.quick else 20000
dt, N, L = 0.004, 100, 0.05
arm = NLinkArm([1.0], [0.5], dv=0.2, dc=0.1)
traj, init = ft_windows(arm, B=B, N=N, dt=dt, L=L, seed=1)

out = {"dissipation_scale": {}, "noise_scale": {}}
for s in (0.5, 0.8, 1.0, 1.25, 1.5, 2.0):
    m = model_from_arm(arm, dv0=arm.dv * s, dc0=arm.dc * s)
    with torch.no_grad():
        out["dissipation_scale"][s] = ft_statistic(sigma(m, traj, L, init))
for s in (0.5, 0.8, 1.0, 1.25, 2.0):
    m = model_from_arm(arm)
    with torch.no_grad():
        out["noise_scale"][s] = ft_statistic(sigma(m, traj, L * s, init))
for k, d in out.items():
    print(k)
    for s, r in d.items():
        flag = "ok" if r["lo"] <= 1 <= r["hi"] else "VIOLATED"
        print(f"  x{s:<5} E[exp(-sigma)] = {r['mean']:8.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]  {flag}")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, (k, d), lab in zip(axs, out.items(), ("model dissipation scale (D, Coulomb+viscous)", "model noise scale (L)")):
    xs = list(d); ax.errorbar(xs, [d[s]["mean"] for s in xs], yerr=[[d[s]["mean"] - d[s]["lo"] for s in xs], [d[s]["hi"] - d[s]["mean"] for s in xs]], fmt="o-")
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel("E[exp(-sigma)]")
fig.suptitle("M2: fluctuation statistic is 1 only for the correct model")
save(args.out, "m2_ft_calibration", dict(config=dict(B=B, N=N, dt=dt, L=L), **{k: {str(s): v for s, v in d.items()} for k, d in out.items()}), fig)
