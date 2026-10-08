"""Shared mechanics helpers. Generalized coords q (absolute link angles), momenta p = M(q) qdot.

Dynamics (port-Hamiltonian, Rayleigh dissipation in velocity):
    qdot = dH/dp = M^-1 p
    pdot = -dH/dq - T^T f(T qdot) + T^T (gain * u)
with joint velocities w = T qdot and joint friction f(w) = dv*w + dc*tanh(w/eps).
"""
import torch


def joint_matrix(n):
    """T maps absolute-angle velocities to relative joint velocities: w = T qd."""
    return torch.eye(n) - torch.diag(torch.ones(n - 1), -1)


def solve_qdot(M, p):
    return torch.linalg.solve(M, p.unsqueeze(-1)).squeeze(-1)


def hamiltonian(mass_fn, pot_fn, q, p):
    qd = solve_qdot(mass_fn(q), p)
    return 0.5 * (p * qd).sum(-1) + pot_fn(q), qd


def grad_q_H(mass_fn, pot_fn, q, p):
    """dH/dq via autograd. Keeps the graph only if the caller is tracking gradients (training)."""
    create = torch.is_grad_enabled()
    with torch.enable_grad():
        qq = q.detach().requires_grad_(True)
        H, _ = hamiltonian(mass_fn, pot_fn, qq, p)
        (g,) = torch.autograd.grad(H.sum(), qq, create_graph=create)
    return g


def friction_force(T, dv, dc, eps, qd):
    w = qd @ T.T
    return (dv * w + dc * torch.tanh(w / eps)) @ T


def input_force(T, gain, u):
    return (gain * u) @ T
