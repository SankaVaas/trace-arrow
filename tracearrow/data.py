"""Simulation: batched stepping (RK4 deterministic, Euler-Maruyama with process noise), references, rollouts."""
import math
import torch
from .integrators import rk4_step


class Simulator:
    """Zero-order-hold control over dt, `substeps` internal steps. L = process-noise matrix (diag, n,) or None.
    Noise enters momentum: dp = ... + sqrt(2 L dt) xi  (underdamped Langevin)."""

    def __init__(self, system, q0, p0, dt, substeps=5, L=None, seed=0):
        self.sys, self.dt, self.substeps = system, dt, substeps
        self.q, self.p = q0.clone(), p0.clone()
        self.L = None if L is None else torch.as_tensor(L, dtype=torch.get_default_dtype())
        self.gen = torch.Generator().manual_seed(seed)
        self.t = 0.0

    def step(self, u):
        h = self.dt / self.substeps
        with torch.no_grad():
            for _ in range(self.substeps):
                if self.L is None:
                    self.q, self.p = rk4_step(self.sys, self.q, self.p, u, h)
                else:
                    qd, pd = self.sys.drift(self.q, self.p, u)
                    xi = torch.randn(self.p.shape, generator=self.gen)
                    self.q, self.p = self.q + h * qd, self.p + h * pd + torch.sqrt(2 * self.L * h) * xi
        self.t += self.dt


class MultisineRef:
    """Smooth random reference q_ref(t) = sum_k a_k sin(w_k t + phi_k), per batch element and joint."""

    def __init__(self, n, B=1, amp=0.5, nfreq=3, fmin=0.3, fmax=1.2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = 2 * math.pi * (fmin + (fmax - fmin) * torch.rand(B, n, nfreq, generator=g))
        self.phi = 2 * math.pi * torch.rand(B, n, nfreq, generator=g)
        self.a = amp / nfreq * torch.ones(B, n, nfreq)

    def __call__(self, t):
        ph = self.w * t + self.phi
        q = (self.a * torch.sin(ph)).sum(-1)
        qd = (self.a * self.w * torch.cos(ph)).sum(-1)
        qdd = -(self.a * self.w ** 2 * torch.sin(ph)).sum(-1)
        return q, qd, qdd


def rollout(sim, controller, K):
    """Run K control steps. controller(k, t, q, p) -> u. Returns traj dict: q,p (B,K+1,n); u (B,K,n)."""
    qs, ps, us = [sim.q.clone()], [sim.p.clone()], []
    for k in range(K):
        u = controller(k, sim.t, sim.q, sim.p)
        sim.step(u)
        us.append(u.clone()); qs.append(sim.q.clone()); ps.append(sim.p.clone())
    q, p = torch.stack(qs, 1), torch.stack(ps, 1)
    with torch.no_grad():
        B, K1, n = q.shape
        qd = sim.sys.qdot(q.reshape(-1, n), p.reshape(-1, n)).reshape(B, K1, n)
    return dict(q=q, p=p, qd=qd, u=torch.stack(us, 1), dt=sim.dt)


def slice_traj(traj, a, b):
    """Sub-window [a, b) in control steps."""
    out = dict(q=traj["q"][:, a:b + 1], p=traj["p"][:, a:b + 1], u=traj["u"][:, a:b], dt=traj["dt"])
    if "qd" in traj:
        out["qd"] = traj["qd"][:, a:b + 1]
    return out


def concat_traj(trajs):
    out = dict(q=torch.cat([t["q"] for t in trajs]), p=torch.cat([t["p"] for t in trajs]),
               u=torch.cat([t["u"] for t in trajs]), dt=trajs[0]["dt"])
    if all("qd" in t for t in trajs):
        out["qd"] = torch.cat([t["qd"] for t in trajs])
    return out
