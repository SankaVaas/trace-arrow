"""Drift statistics and EWMA alarm, calibrated on healthy data (empirical null)."""
import math
import torch
from .timereversal import step_terms


def window_stats(model, traj, L):
    """Per-window statistics under the current model:
       sigma  = path entropy-production-like statistic  S_r - S_f  (time-reversal contrast)
       resid  = mean forward OM residual energy          (conventional prediction-error baseline)"""
    with torch.no_grad():
        a, qd, _ = step_terms(model, traj)
        Ff = model.friction_force(qd)
        L = torch.as_tensor(L, dtype=a.dtype)
        sigma = -traj["dt"] * (a * Ff / L).sum((-1, -2))
        resid = (((a + Ff) ** 2) / L).mean((-1, -2))
    return dict(sigma=sigma, resid=resid)


class EwmaAlarm:
    def __init__(self, mu0, sd0, lam=0.3, k=4.0, two_sided=True):
        self.mu0, self.sd0, self.lam, self.k, self.two = mu0, max(sd0, 1e-12), lam, k, two_sided
        self.z = mu0
        self.limit = k * self.sd0 * math.sqrt(lam / (2 - lam))

    def update(self, x):
        self.z = self.lam * x + (1 - self.lam) * self.z
        dev = self.z - self.mu0
        return (abs(dev) if self.two else dev) > self.limit

    @classmethod
    def calibrate(cls, samples, **kw):
        s = torch.as_tensor(samples)
        return cls(s.mean().item(), s.std().item() if len(s) > 1 else 1.0, **kw)
