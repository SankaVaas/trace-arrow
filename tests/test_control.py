import torch
from tracearrow.systems import NLinkArm
from tracearrow.model import model_from_arm
from tracearrow.data import MultisineRef, Simulator, rollout
from tracearrow.control import ComputedTorque


def test_computed_torque_tracks_with_exact_model():
    arm = NLinkArm([1.0, 0.8], [0.5, 0.4])
    ref = MultisineRef(2, 3, amp=0.6, seed=0)
    tr = rollout(Simulator(arm, torch.zeros(3, 2), torch.zeros(3, 2), 0.01), ComputedTorque(model_from_arm(arm), ref), 400)
    qref = torch.stack([ref(k * 0.01)[0] for k in range(1, 401)], 1)
    assert (tr["q"][:, 100:] - torch.cat([qref[:, 99:-1], qref[:, -1:]], 1)).abs().max() < 0.2
    assert (tr["q"][:, 1:] - qref).pow(2).mean().sqrt() < 0.05


def test_controller_degrades_with_wrong_model():
    arm = NLinkArm([1.0, 0.8], [0.5, 0.4], dv=[0.6, 0.4], dc=[0.2, 0.2])
    ref = MultisineRef(2, 3, amp=0.6, seed=0)
    def err(m):
        tr = rollout(Simulator(arm, torch.zeros(3, 2), torch.zeros(3, 2), 0.01), ComputedTorque(m, ref), 400)
        qref = torch.stack([ref(k * 0.01)[0] for k in range(1, 401)], 1)
        return (tr["q"][:, 1:] - qref).pow(2).mean().sqrt().item()
    assert err(model_from_arm(arm)) < err(model_from_arm(arm, dv0=arm.dv * 0, dc0=arm.dc * 0))
