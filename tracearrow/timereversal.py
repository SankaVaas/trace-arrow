"""Time-reversal machinery.

Observed discrete step (midpoint rule):   a = dp/dt + dH/dq(qbar,pbar) - T^T(gain u)
Model friction F_theta is odd under (q,p,t)->(q,-p,-t); the conservative part is even. So
    forward OM residual   e_f = a + F_theta(qdot)
    reversed OM residual  e_r = a - F_theta(qdot)
    sigma_path = S_r - S_f = -dt * sum a^T L^-1 F_theta        with S = 1/4 sum dt e^T L^-1 e
Trajectory layout: q,p (B,K+1,n); u (B,K,n); dt scalar.
"""
import torch


def reverse_traj(traj):
    out = dict(q=traj["q"].flip(1), p=-traj["p"].flip(1), u=traj["u"].flip(1), dt=traj["dt"])
    if "qd" in traj:
        out["qd"] = -traj["qd"].flip(1)
    return out


def momentum(model, traj):
    """Sensor-realistic: if joint velocities were measured, p = M_model(q) qd (so inertia error propagates);
    otherwise use the stored momentum."""
    if "qd" in traj:
        return (model.mass(traj["q"]) @ traj["qd"].unsqueeze(-1)).squeeze(-1)
    return traj["p"]


def step_terms(model, traj, gain=None):
    q, p, u, dt = traj["q"], momentum(model, traj), traj["u"], traj["dt"]
    qm, pm = 0.5 * (q[:, 1:] + q[:, :-1]), 0.5 * (p[:, 1:] + p[:, :-1])
    B, K, n = qm.shape
    gH = model.grad_q_H(qm.reshape(-1, n), pm.reshape(-1, n)).reshape(B, K, n)
    qd = model.qdot(qm.reshape(-1, n), pm.reshape(-1, n)).reshape(B, K, n)
    Qu = model.input_force(u) if gain is None else (gain * u) @ model.T
    a = (p[:, 1:] - p[:, :-1]) / dt + gH - Qu
    return a, qd, qm


def residuals(model, traj):
    a, qd, _ = step_terms(model, traj)
    Ff = model.friction_force(qd)
    return a + Ff, a - Ff


def om_action(e, L, dt):
    return 0.25 * dt * (e * e / L).sum((-1, -2))


def sigma(model, traj, L, init=None):
    """Model-implied entropy-production-like statistic per window: S_r - S_f (+ boundary log-density ratio)."""
    a, qd, _ = step_terms(model, traj)
    Ff = model.friction_force(qd)
    L = torch.as_tensor(L, dtype=a.dtype)
    s = -traj["dt"] * (a * Ff / L).sum((-1, -2))
    return s + (init.log_ratio(traj) if init is not None else 0.0)


class GaussianInit:
    """Known initial density rho0(q,p) (independent Gaussians). Reversed-process density is rho0(q,-p)."""

    def __init__(self, mq, sq, mp, sp):
        self.mq, self.sq, self.mp, self.sp = [torch.as_tensor(x, dtype=torch.get_default_dtype()) for x in (mq, sq, mp, sp)]

    def logpdf(self, q, p):
        return (-0.5 * ((q - self.mq) / self.sq) ** 2 - 0.5 * ((p - self.mp) / self.sp) ** 2).sum(-1)

    def log_ratio(self, traj):
        q0, p0 = traj["q"][:, 0], traj["p"][:, 0]
        qN, pN = traj["q"][:, -1], traj["p"][:, -1]
        return self.logpdf(q0, p0) - self.logpdf(qN, -pN)


def ft_statistic(sig, n_boot=2000, seed=0):
    """Mean exp(-sigma) with bootstrap 95% CI (== 1 for a correct model, any dissipation law)."""
    w = torch.exp(-sig)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(w), (n_boot, len(w)), generator=g)
    bm = w[idx].mean(1)
    return dict(mean=w.mean().item(), lo=torch.quantile(bm, 0.025).item(), hi=torch.quantile(bm, 0.975).item(),
                sigma_mean=sig.mean().item(), sigma_std=sig.std().item())


# ---------- parity-separated regression: odd features -> dissipation, even features -> conservative error ----------
def even_features(q, qd):
    """Functions even under p -> -p that can absorb conservative-model error (n=1: 6 terms, n=2: 9, n=3: 16)."""
    n = q.shape[-1]
    f = [torch.ones_like(q[..., :1]), torch.sin(q), torch.cos(q)]
    for j in range(n):
        for k in range(j, n):
            pr = qd[..., j:j + 1] * qd[..., k:k + 1]
            f.append(pr * torch.cos(q[..., j:j + 1] - q[..., k:k + 1]))
            if k > j:
                f.append(pr * torch.sin(q[..., j:j + 1] - q[..., k:k + 1]))
    return torch.cat(f, -1)


def design(model, traj, gain_ref=None, parity=True):
    """Return X (S*n, P), y (S*n), index dict. a = -F(w) + T^T((g-g_ref)u) + even(q,qd)."""
    a, qd, qm = step_terms(model, traj, gain=gain_ref)
    B, K, n = a.shape
    T, eps = model.T, model.head.eps
    w = qd @ T.T
    cols, idx = [], {}
    def add(name, c):
        idx.setdefault(name, []).append(len(cols)); cols.append(c)
    for j in range(n):
        add("dv", -w[..., j:j + 1] * T[j])
    for j in range(n):
        add("dc", -torch.tanh(w[..., j:j + 1] / eps) * T[j])
    if parity:
        for j in range(n):
            add("dgain", traj["u"][..., j:j + 1] * T[j])
        phi = even_features(qm, qd)
        for i in range(n):
            for f in range(phi.shape[-1]):
                c = torch.zeros(B, K, n)
                c[..., i] = phi[..., f]
                add("even", c)
    X = torch.stack(cols, -1).reshape(-1, len(cols))
    return X, a.reshape(-1), idx


def _lstsq(X, y, ridge=1e-9):
    A = X.T @ X
    A = A + ridge * A.diagonal().mean() * torch.eye(A.shape[0])
    return torch.linalg.solve(A, X.T @ y)


def estimate_dissipation(model, traj, parity=True):
    """parity=True: joint fit of odd (dissipation) + even (conservative/gain error) terms.
    parity=False (naive baseline): fit dissipation only, trusting the model's conservative part and gain."""
    X, y, idx = design(model, traj, parity=parity)
    beta = _lstsq(X, y)
    out = dict(dv=beta[idx["dv"]], dc=beta[idx["dc"]])
    if parity:
        out["dgain"] = beta[idx["dgain"]]
        out["even_norm"] = beta[idx["even"]].norm().item()
    return out


class RLS:
    """Recursive least squares with forgetting (batch update)."""

    def __init__(self, dim, forget=0.98, p0=1e2, theta0=None):
        self.lam, self.P = forget, p0 * torch.eye(dim)
        self.theta = torch.zeros(dim) if theta0 is None else theta0.clone()

    def update(self, X, y):
        Px = self.P @ X.T
        S = self.lam * torch.eye(X.shape[0]) + X @ Px
        Kg = torch.linalg.solve(S, Px.T).T
        self.theta = self.theta + Kg @ (y - X @ self.theta)
        self.P = (self.P - Kg @ Px.T) / self.lam
        self.P = 0.5 * (self.P + self.P.T)


def localize_refit(model, traj):
    """Which parity class explains the model's residual?  y0 = a + F_theta is what the current model fails to explain.
    Fit on the first half of the window, score held-out R^2 on the second half, for three hypotheses:
      odd (dissipation dv,dc) | gain (actuator gain) | even (conservative error, even-in-velocity terms)."""
    X, _, idx = design(model, traj, parity=True)
    a, qd, _ = step_terms(model, traj)
    B, K, n = a.shape
    y0 = (a + model.friction_force(qd))
    h = K // 2
    tr = torch.zeros(B, K, n, dtype=torch.bool); tr[:, :h] = True
    tr, te = tr.reshape(-1), ~tr.reshape(-1)
    y0 = y0.reshape(-1)
    base = (y0[te] ** 2).sum().clamp_min(1e-30)
    out = {}
    for name, cols in (("odd", idx["dv"] + idx["dc"]), ("gain", idx["dgain"]), ("even", idx["even"])):
        Xs = X[:, cols]
        beta = _lstsq(Xs[tr], y0[tr], ridge=1e-6)
        out[name] = max(0.0, (1 - ((y0[te] - Xs[te] @ beta) ** 2).sum() / base).item())
    return out
