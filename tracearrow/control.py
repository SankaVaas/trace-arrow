"""Model-based tracking controller. The input force enters quadratically, so the optimal command is closed form:
    u* = argmin ||u A - r||_W^2 + lam ||u||^2   (A = diag(gain) T, r = required generalized input force)."""
import torch


class ComputedTorque:
    def __init__(self, model, ref, kp=100.0, kd=20.0, lam=1e-8, W=None):
        self.model, self.ref, self.kp, self.kd, self.lam, self.W = model, ref, kp, kd, lam, W

    def __call__(self, k, t, q, p):
        m = self.model
        with torch.no_grad():
            qr, qdr, qddr = self.ref(t)
            qd = m.qdot(q, p)
            qdd_des = qddr + self.kd * (qdr - qd) + self.kp * (qr - q)
            e = 1e-6
            Mdot_qd = ((m.mass(q + e * qd) - m.mass(q - e * qd)) / (2 * e) @ qd.unsqueeze(-1)).squeeze(-1)
            pdot_des = Mdot_qd + (m.mass(q) @ qdd_des.unsqueeze(-1)).squeeze(-1)
            r = pdot_des + m.grad_q_H(q, p) + m.friction_force(qd)
            A = m.head.gain.unsqueeze(-1) * m.T            # Qu = u @ A
            W = torch.ones(m.n) if self.W is None else self.W
            AW = A * W
            rhs = AW @ r.T
            u = torch.linalg.solve(AW @ A.T + self.lam * torch.eye(m.n), rhs).T
        return u
