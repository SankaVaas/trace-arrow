import numpy as np
import torch
from scipy.integrate import solve_ivp
from tracearrow.systems import NLinkArm
from tracearrow.integrators import euler_step, rk4_step, strang_step


def run(arm, step, q, p, u, h, n):
    with torch.no_grad():
        for _ in range(n):
            q, p = step(arm, q, p, u, h)
    return q, p


def test_mass_matrix_spd():
    arm = NLinkArm([1.0, 0.8, 0.5], [0.5, 0.4, 0.3])
    q = torch.randn(200, 3) * 2
    assert torch.linalg.eigvalsh(arm.mass(q)).min() > 0


def test_energy_conserved_without_friction():
    arm = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=0.0, dc=0.0)
    q, p = torch.tensor([[0.8, -0.4]]), torch.tensor([[0.1, 0.2]])
    H0 = arm.hamiltonian(q, p)
    q, p = run(arm, rk4_step, q, p, torch.zeros(1, 2), 0.005, 2000)
    assert abs((arm.hamiltonian(q, p) - H0).item()) < 1e-6


def test_friction_dissipates_monotonically():
    arm = NLinkArm([1.0, 0.8], [0.5, 0.4])
    q, p, u = torch.tensor([[0.8, -0.4]]), torch.tensor([[0.1, 0.2]]), torch.zeros(1, 2)
    Hs = []
    for _ in range(200):
        q, p = run(arm, rk4_step, q, p, u, 0.005, 5)
        Hs.append(arm.hamiltonian(q, p).item())
    assert np.all(np.diff(Hs) <= 1e-9)


def test_matches_scipy_reference():
    arm = NLinkArm([1.0], [0.5], dv=0.2, dc=0.1)
    def f(t, x):
        q, p = torch.tensor([[x[0]]]), torch.tensor([[x[1]]])
        with torch.no_grad():
            qd, pd = arm.drift(q, p, torch.zeros(1, 1))
        return [qd.item(), pd.item()]
    ref = solve_ivp(f, [0, 3], [1.0, 0.0], method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
    q, p = run(arm, rk4_step, torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.zeros(1, 1), 0.003, 1000)
    assert abs(q.item() - ref[0]) < 1e-6 and abs(p.item() - ref[1]) < 1e-6


def test_strang_bounded_energy_vs_euler():
    arm = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=0.0, dc=0.0)
    q0, p0, u = torch.tensor([[0.8, -0.4]]), torch.tensor([[0.1, 0.2]]), torch.zeros(1, 2)
    H0 = arm.hamiltonian(q0, p0).item()
    de = abs(arm.hamiltonian(*run(arm, euler_step, q0, p0, u, 0.01, 1000)).item() - H0)
    ds = abs(arm.hamiltonian(*run(arm, strang_step, q0, p0, u, 0.01, 1000)).item() - H0)
    assert ds < 1e-3 < de


def test_strang_dissipates_and_is_second_order():
    arm = NLinkArm([1.0], [0.5], dv=0.3, dc=0.1)
    q0, p0, u = torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.zeros(1, 1)
    ref = run(arm, rk4_step, q0, p0, u, 0.0005, 2000)
    errs = []
    for h, n in ((0.02, 50), (0.01, 100)):
        q, p = run(arm, strang_step, q0, p0, u, h, n)
        errs.append(abs(q.item() - ref[0].item()) + abs(p.item() - ref[1].item()))
    assert errs[0] / errs[1] > 3.0  # ~4 for second order
    assert arm.hamiltonian(*run(arm, strang_step, q0, p0, u, 0.01, 100)).item() < arm.hamiltonian(q0, p0).item()
