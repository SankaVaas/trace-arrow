import torch
from tracearrow.systems import NLinkArm
from tracearrow.model import NeuralCore, DissipativeHead, TRACEModel, model_from_arm
from tracearrow.timereversal import step_terms


def test_neural_core_mass_spd_and_symmetric():
    core = NeuralCore(3, 16)
    M = core.mass(torch.randn(100, 3) * 3)
    assert torch.allclose(M, M.transpose(-1, -2))
    assert torch.linalg.eigvalsh(M).min() > 0


def test_head_positive_and_set_values_roundtrip():
    h = DissipativeHead(2)
    h.set_values(dv=torch.tensor([0.3, 0.1]), dc=torch.tensor([0.02, 0.2]))
    assert torch.allclose(h.dv, torch.tensor([0.3, 0.1])) and (h.dc > 0).all()
    with torch.no_grad():
        h.raw_dv.fill_(-50.0)
    assert (h.dv > 0).all()


def test_analytic_core_matches_plant():
    arm = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=[0.1, 0.2], dc=[0.03, 0.04])
    m = model_from_arm(arm)
    q, p, u = torch.randn(5, 2), torch.randn(5, 2), torch.randn(5, 2)
    for a, b in zip(arm.drift(q, p, u), m.drift(q, p, u)):
        assert torch.allclose(a, b, atol=1e-10)


def test_gradients_reach_core_and_head():
    m = TRACEModel(NeuralCore(1, 8), DissipativeHead(1))
    n = 4
    traj = dict(q=torch.randn(2, n + 1, 1), p=torch.randn(2, n + 1, 1), u=torch.randn(2, n, 1), dt=0.01)
    a, qd, _ = step_terms(m, traj)
    ((a + m.friction_force(qd)) ** 2).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters())
