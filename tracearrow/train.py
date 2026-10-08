"""Offline fit of core and/or head by one-step OM residual (derivative matching)."""
import torch
from .timereversal import step_terms


def pretrain(model, traj, epochs=300, lr=3e-3, train_core=True, train_head=True, verbose=False):
    params = (list(model.core.parameters()) if train_core else []) + (list(model.head.parameters()) if train_head else [])
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for ep in range(epochs):
        a, qd, _ = step_terms(model, traj)
        loss = ((a + model.friction_force(qd)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
        if verbose and ep % 50 == 0:
            print(ep, loss.item())
    return hist
