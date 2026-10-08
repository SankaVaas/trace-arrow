import torch
from tracearrow.systems import NLinkArm
from tracearrow.model import model_from_arm
from tracearrow.data import MultisineRef, Simulator, rollout
from tracearrow.control import ComputedTorque
from tracearrow.calib import ft_windows
from tracearrow.timereversal import (reverse_traj, residuals, step_terms, estimate_dissipation, sigma, ft_statistic,
                                     localize_refit)
from tracearrow.adapt import TraceAdapter
from tracearrow.data import slice_traj

ARM = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=[0.15, 0.10], dc=[0.05, 0.04])


def _traj(B=4, K=500, seed=0):
    ref = MultisineRef(2, B, amp=0.6, seed=seed)
    return rollout(Simulator(ARM, torch.zeros(B, 2), torch.zeros(B, 2), 0.01), ComputedTorque(model_from_arm(ARM), ref), K), ref


def test_reverse_is_involution():
    tr, _ = _traj(2, 20)
    r = reverse_traj(reverse_traj(tr))
    for k in ("q", "p", "u", "qd"):
        assert torch.allclose(r[k], tr[k])


def test_exact_model_has_small_forward_residual_and_odd_reverse_residual():
    tr, _ = _traj()
    m = model_from_arm(ARM)
    ef, er = residuals(m, tr)
    a, qd, _ = step_terms(m, tr)
    assert ef.abs().mean() < 2e-3 and ef.abs().max() < 3e-2   # midpoint-rule error only (largest near zero-velocity Coulomb kink)
    assert torch.allclose(er - ef, -2 * m.friction_force(qd), atol=1e-10)  # reversed residual flips the dissipative term


def test_reversed_trajectory_residual_matches_formula():
    tr, _ = _traj(2, 50)
    m = model_from_arm(ARM)
    _, er = residuals(m, tr)
    ef_rev, _ = residuals(m, reverse_traj(tr))        # forward residual evaluated on the reversed path
    assert torch.allclose(ef_rev, er.flip(1), atol=1e-8)


def test_estimator_recovers_dissipation_and_beats_naive_under_mismatch():
    tr, _ = _traj()
    exact = estimate_dissipation(model_from_arm(ARM), tr)
    assert ((exact["dv"] - ARM.dv).abs() / ARM.dv).max() < 0.02 and ((exact["dc"] - ARM.dc).abs() / ARM.dc).max() < 0.05
    bad = model_from_arm(ARM, grav_scale=0.8, gain0=0.8)
    par, naive = estimate_dissipation(bad, tr), estimate_dissipation(bad, tr, parity=False)
    e = lambda r: ((r["dc"] - ARM.dc).abs() / ARM.dc).mean().item()
    assert e(par) < 0.05 and e(naive) > 5 * e(par)


def test_uniform_inertia_error_is_a_gauge():
    tr, _ = _traj()
    s = 1.2
    r = estimate_dissipation(model_from_arm(ARM, mass_scale=s), tr)
    assert ((r["dv"] / s - ARM.dv).abs() / ARM.dv).max() < 0.03
    assert torch.allclose(r["dgain"], torch.full((2,), s - 1), atol=0.02)


def test_fluctuation_statistic_is_one_for_true_model_and_not_for_wrong_dissipation():
    arm = NLinkArm([1.0], [0.5], dv=0.2, dc=0.1)
    traj, init = ft_windows(arm, B=6000, N=100, dt=0.004, L=0.05, seed=3)
    good = ft_statistic(sigma(model_from_arm(arm), traj, 0.05, init))
    assert good["lo"] <= 1 <= good["hi"]
    wrong = ft_statistic(sigma(model_from_arm(arm, dv0=arm.dv * 2, dc0=arm.dc * 2), traj, 0.05, init))
    assert wrong["lo"] > 1.5


def test_trace_adapter_converges_from_wrong_head():
    tr, _ = _traj(1, 400)
    m = model_from_arm(ARM, dv0=ARM.dv * 0.3, dc0=ARM.dc * 3)
    ad = TraceAdapter(m)
    for k in range(0, 300, 10):
        ad.update(slice_traj(tr, k, k + 10))
    assert ((m.head.dv - ARM.dv).abs() / ARM.dv).max() < 0.1 and ((m.head.dc - ARM.dc).abs() / ARM.dc).max() < 0.15


def test_localisation_flags_dissipative_fault_as_odd():
    ref = MultisineRef(2, 4, amp=0.6, seed=5)
    worn = ARM.replace(dv=ARM.dv * 4, dc=ARM.dc * 3)
    tr = rollout(Simulator(worn, torch.zeros(4, 2), torch.zeros(4, 2), 0.01), ComputedTorque(model_from_arm(ARM), ref), 600)
    r = localize_refit(model_from_arm(ARM), tr)
    assert r["odd"] > 0.5 and r["odd"] > r["even"]
