"""Integrators. `sys_` needs: grad_q_H, qdot, friction_force, input_force, drift."""
import torch


def euler_step(sys_, q, p, u, h):
    qd, pd = sys_.drift(q, p, u)
    return q + h * qd, p + h * pd


def rk4_step(sys_, q, p, u, h):
    k1 = sys_.drift(q, p, u)
    k2 = sys_.drift(q + 0.5 * h * k1[0], p + 0.5 * h * k1[1], u)
    k3 = sys_.drift(q + 0.5 * h * k2[0], p + 0.5 * h * k2[1], u)
    k4 = sys_.drift(q + h * k3[0], p + h * k3[1], u)
    return (q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            p + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def _dissipative_half(sys_, q, p, u, h, iters):
    """Sub-flow  pdot = -F(qdot(q,p)) + Q_u  at fixed q, implicit midpoint via fixed point."""
    Qu = sys_.input_force(u)
    pn = p.clone()
    for _ in range(iters):
        qd = sys_.qdot(q, 0.5 * (p + pn))
        pn = p + h * (-sys_.friction_force(qd) + Qu)
    return pn


def strang_step(sys_, q, p, u, h, iters=8):
    """Strang splitting: dissipation/input half-step, symplectic implicit-midpoint conservative
    step, dissipation/input half-step. Second order; conservative part is exactly symplectic."""
    p = _dissipative_half(sys_, q, p, u, 0.5 * h, iters)
    qn, pn = q.clone(), p.clone()
    for _ in range(iters):
        qm, pm = 0.5 * (q + qn), 0.5 * (p + pn)
        qn = q + h * sys_.qdot(qm, pm)
        pn = p - h * sys_.grad_q_H(qm, pm)
    p = _dissipative_half(sys_, qn, pn, u, 0.5 * h, iters)
    return qn, p
