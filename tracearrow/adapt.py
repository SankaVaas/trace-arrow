"""Online adapters for the dissipative head (the conservative core stays frozen)."""
import torch
from .timereversal import design, step_terms, RLS


class NoAdapt:
    def __init__(self, model):
        self.model = model

    def update(self, traj):
        pass


class TraceAdapter:
    """Recursive parity-separated regression. Odd features update (dv, dc); even features (incl. u) absorb
    conservative / actuator error so it cannot alias into dissipation. Only the head is written back."""

    def __init__(self, model, forget=0.995, p0=1e2, p0_even=1.0):
        self.model = model
        self.gain_ref = model.head.gain.detach().clone()
        n = model.n
        d = torch.zeros(1, 2, n)  # dummy traj to size the design matrix
        dummy = dict(q=torch.zeros(1, 2, n), p=torch.zeros(1, 2, n), u=torch.zeros(1, 1, n), dt=0.01)
        _, _, self.idx = design(model, dummy, gain_ref=self.gain_ref)
        P = sum(len(v) for v in self.idx.values())
        th0 = torch.zeros(P)
        th0[self.idx["dv"]] = model.head.dv.detach()
        th0[self.idx["dc"]] = model.head.dc.detach()
        self.rls = RLS(P, forget=forget, p0=p0, theta0=th0)
        d = torch.full((P,), p0)
        d[self.idx["even"]] = p0_even  # tight prior on nuisance (conservative) terms: less variance in healthy operation
        self.rls.P = torch.diag(d)

    def update(self, traj):
        with torch.no_grad():
            X, y, _ = design(self.model, traj, gain_ref=self.gain_ref)
            self.rls.update(X, y)
            th = self.rls.theta
            self.model.head.set_values(dv=th[self.idx["dv"]].clamp_min(1e-4), dc=th[self.idx["dc"]].clamp_min(1e-4),
                                       gain=(self.gain_ref + th[self.idx["dgain"]]).clamp_min(0.3))

    def snapshot(self):
        th = self.rls.theta
        return dict(odd=torch.cat([th[self.idx["dv"]], th[self.idx["dc"]]]).clone(), even=th[self.idx["even"]].clone(),
                    gain=th[self.idx["dgain"]].clone())

    @staticmethod
    def localize(base, now):
        """Compare current estimate to a healthy snapshot: dissipative-type vs conservative-type change."""
        odd = (now["odd"] - base["odd"]).norm() / base["odd"].norm().clamp_min(1e-9)
        even = (now["even"] - base["even"]).norm()
        return dict(odd_shift=odd.item(), even_shift=even.item(), gain_shift=(now["gain"] - base["gain"]).norm().item())


class SgdHeadAdapter:
    """Baseline: plain SGD/Adam on the head (dissipation + gain) minimising the forward OM residual.
    Same parameters and per-update step budget as TRACE, but no parity separation."""

    def __init__(self, model, lr=0.02, steps=5):
        self.model, self.steps = model, steps
        self.opt = torch.optim.Adam(model.dissipative_parameters(), lr=lr)

    def update(self, traj):
        for _ in range(self.steps):
            a, qd, _ = step_terms(self.model, traj)
            loss = ((a + self.model.friction_force(qd)) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
