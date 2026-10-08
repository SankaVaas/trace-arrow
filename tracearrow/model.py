"""Split-structure port-Hamiltonian model: (core: M(q), V(q)) + (head: dissipation, actuator gain)."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .physics import joint_matrix, hamiltonian, grad_q_H, friction_force, input_force


def inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    return x + torch.log(-torch.expm1(-x))


class AnalyticCore(nn.Module):
    """Conservative core from a known arm, optionally mis-specified (for robustness tests)."""

    def __init__(self, arm, mass_scale=1.0, grav_scale=1.0):
        super().__init__()
        self.arm = arm.replace(masses=arm.m * mass_scale, g=arm.g * grav_scale)

    def mass(self, q):
        return self.arm.mass(q)

    def potential(self, q):
        return self.arm.potential(q)


class NeuralCore(nn.Module):
    """Learned core. M(q) = L L^T + eps I via Cholesky (SPD by construction); V(q) scalar MLP."""

    def __init__(self, n, hidden=64):
        super().__init__()
        self.n = n
        nl = n * (n + 1) // 2
        self.register_buffer("tril", torch.tril_indices(n, n))
        self.mass_net = nn.Sequential(nn.Linear(2 * n, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, nl))
        self.pot_net = nn.Sequential(nn.Linear(2 * n, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        with torch.no_grad():  # start near M = 0.3 I
            self.mass_net[-1].weight.mul_(0.1)
            self.mass_net[-1].bias.zero_()
            k = 0
            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        self.mass_net[-1].bias[k] = float(inv_softplus(0.5))
                    k += 1

    def _feat(self, q):
        return torch.cat([torch.sin(q), torch.cos(q)], -1)

    def mass(self, q):
        out = self.mass_net(self._feat(q))
        L = torch.zeros(*q.shape[:-1], self.n, self.n, dtype=q.dtype)
        L[..., self.tril[0], self.tril[1]] = out
        d = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-2
        L = torch.tril(L, -1) + torch.diag_embed(d)
        return L @ L.transpose(-1, -2) + 1e-3 * torch.eye(self.n)

    def potential(self, q):
        return self.pot_net(self._feat(q)).squeeze(-1)


class DissipativeHead(nn.Module):
    """Joint dissipation (viscous + smoothed Coulomb, both >= 0) and actuator gain: the only part adapted online."""

    def __init__(self, n, eps=0.05, dv0=0.05, dc0=0.02, gain0=1.0):
        super().__init__()
        self.n, self.eps = n, eps
        self.register_buffer("T", joint_matrix(n))
        full = lambda x: inv_softplus(torch.full((n,), float(x)))
        self.raw_dv, self.raw_dc, self.raw_gain = nn.Parameter(full(dv0)), nn.Parameter(full(dc0)), nn.Parameter(full(gain0))

    dv = property(lambda s: F.softplus(s.raw_dv))
    dc = property(lambda s: F.softplus(s.raw_dc))
    gain = property(lambda s: F.softplus(s.raw_gain))

    def set_values(self, dv=None, dc=None, gain=None):
        with torch.no_grad():
            for raw, val in ((self.raw_dv, dv), (self.raw_dc, dc), (self.raw_gain, gain)):
                if val is not None:
                    raw.copy_(inv_softplus(torch.as_tensor(val).clamp_min(1e-6)))

    def friction_force(self, qd):
        return friction_force(self.T, self.dv, self.dc, self.eps, qd)

    def input_force(self, u):
        return input_force(self.T, self.gain, u)


class TRACEModel(nn.Module):
    def __init__(self, core, head):
        super().__init__()
        self.core, self.head = core, head
        self.n = head.n

    @property
    def T(self):
        return self.head.T

    def mass(self, q):
        return self.core.mass(q)

    def potential(self, q):
        return self.core.potential(q)

    def hamiltonian(self, q, p):
        return hamiltonian(self.mass, self.potential, q, p)[0]

    def qdot(self, q, p):
        return hamiltonian(self.mass, self.potential, q, p)[1]

    def grad_q_H(self, q, p):
        return grad_q_H(self.mass, self.potential, q, p)

    def friction_force(self, qd):
        return self.head.friction_force(qd)

    def input_force(self, u):
        return self.head.input_force(u)

    def drift(self, q, p, u):
        qd = self.qdot(q, p)
        return qd, -self.grad_q_H(q, p) - self.friction_force(qd) + self.input_force(u)

    def dissipative_parameters(self):
        return list(self.head.parameters())


def model_from_arm(arm, mass_scale=1.0, grav_scale=1.0, dv0=None, dc0=None, gain0=1.0):
    """Convenience: analytic (possibly mis-specified) core + head initialised at the given values."""
    head = DissipativeHead(arm.n, eps=arm.eps, dv0=0.05, dc0=0.02, gain0=gain0)
    head.set_values(dv=dv0 if dv0 is not None else arm.dv, dc=dc0 if dc0 is not None else arm.dc, gain=gain0 * torch.ones(arm.n))
    return TRACEModel(AnalyticCore(arm, mass_scale, grav_scale), head)
