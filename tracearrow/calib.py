"""Fluctuation-statistic calibration in simulation: windows drawn from a KNOWN initial density, driven by a KNOWN
random protocol, with process noise L. For a correct model  E[exp(-sigma)] = 1  (path-probability normalisation),
with or without fluctuation-dissipation consistency."""
import math
import torch
from .data import Simulator
from .timereversal import GaussianInit


def ft_windows(arm, B=20000, N=100, dt=0.004, L=0.05, substeps=4, u_amp=0.3, u_freq=1.5,
               sq=0.5, sp=0.3, seed=0):
    n = arm.n
    g = torch.Generator().manual_seed(seed)
    q0, p0 = sq * torch.randn(B, n, generator=g), sp * torch.randn(B, n, generator=g)
    phi = 2 * math.pi * torch.rand(B, n, generator=g)
    sim = Simulator(arm, q0, p0, dt, substeps=substeps, L=L, seed=seed + 1)
    qs, ps, us = [q0], [p0], []
    for k in range(N):
        u = u_amp * torch.sin(2 * math.pi * u_freq * sim.t + phi)
        sim.step(u)
        qs.append(sim.q.clone()); ps.append(sim.p.clone()); us.append(u)
    traj = dict(q=torch.stack(qs, 1), p=torch.stack(ps, 1), u=torch.stack(us, 1), dt=dt)
    return traj, GaussianInit(0.0, sq, 0.0, sp)
