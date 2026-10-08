"""Ground-truth plant: planar n-link arm (point masses at link ends), hanging-down convention."""
import torch
from .physics import joint_matrix, hamiltonian, grad_q_H, friction_force, input_force


def _vec(x, n):
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    return x.expand(n).clone() if x.ndim == 0 else x.clone()


class NLinkArm:
    def __init__(self, masses, lengths, g=9.81, dv=0.1, dc=0.05, eps=0.05, gain=1.0):
        self.m = torch.as_tensor(masses, dtype=torch.get_default_dtype())
        self.l = torch.as_tensor(lengths, dtype=torch.get_default_dtype())
        self.n = len(self.m)
        self.g, self.eps = g, eps
        self.dv, self.dc, self.gain = _vec(dv, self.n), _vec(dc, self.n), _vec(gain, self.n)
        self.T = joint_matrix(self.n)
        self.suffix = torch.flip(torch.cumsum(torch.flip(self.m, [0]), 0), [0])
        idx = torch.arange(self.n)
        self._S = self.suffix[torch.maximum(idx[:, None], idx[None, :])]
        self._LL = self.l[:, None] * self.l[None, :]

    def replace(self, **kw):
        a = dict(masses=self.m, lengths=self.l, g=self.g, dv=self.dv, dc=self.dc, eps=self.eps, gain=self.gain)
        a.update(kw)
        return NLinkArm(**a)

    def mass(self, q):
        return self._LL * self._S * torch.cos(q.unsqueeze(-1) - q.unsqueeze(-2))

    def potential(self, q):
        return -self.g * (self.suffix * self.l * torch.cos(q)).sum(-1)

    def hamiltonian(self, q, p):
        return hamiltonian(self.mass, self.potential, q, p)[0]

    def qdot(self, q, p):
        return hamiltonian(self.mass, self.potential, q, p)[1]

    def grad_q_H(self, q, p):
        return grad_q_H(self.mass, self.potential, q, p)

    def friction_force(self, qd):
        return friction_force(self.T, self.dv, self.dc, self.eps, qd)

    def input_force(self, u):
        return input_force(self.T, self.gain, u)

    def drift(self, q, p, u):
        qd = self.qdot(q, p)
        pd = -self.grad_q_H(q, p) - self.friction_force(qd) + self.input_force(u)
        return qd, pd
