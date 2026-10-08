# trace-arrow

**TRACE: Time-Reversal Asymmetry Control & Estimation.** A robot's own motion is split by time-reversal parity:
the conservative part of the dynamics is even, dissipation is odd. TRACE uses that split to (1) identify dissipation
and actuator error from ordinary closed-loop data, (2) test a model with a fluctuation statistic, and (3) tell
dissipative faults (wear) from non-dissipative ones. CPU only; the full suite runs on a laptop or free Colab.

## Model
State (q, p), p = M(q) q̇. `qdot = ∂H/∂p`, `pdot = -∂H/∂q - Tᵀ f(T q̇) + Tᵀ(gain·u)`, joint friction
`f(w) = dv·w + dc·tanh(w/ε)`. The model is split: **core** (M(q) via Cholesky, V(q); analytic or neural) and
**head** (dv, dc ≥ 0, actuator gain; the only part adapted online).

Midpoint residual `a = Δp/dt + ∂H/∂q − Tᵀ(gain·u)`; forward `e_f = a + F`, reversed `e_r = a − F`;
`σ = S_r − S_f = −dt Σ aᵀL⁻¹F` (+ boundary term from a known initial density).

## Layout
```
tracearrow/   physics, systems (n-link arm), model, integrators (RK4, Strang-split), data (sim, refs, rollouts),
              control (closed-form quadratic-in-u tracking), timereversal (residuals, sigma, FT stat, parity
              regression, localisation), adapt (TRACE RLS, SGD baseline), detect (EWMA alarm), calib, train
experiments/  m1_identify_D.py  m2_ft_calibration.py  m3_adapt_vs_baselines.py   (all accept --quick)
tests/        20 tests: physics, integrators vs scipy, estimator, FT identity, adapter, controller
notebooks/    TRACE_colab.ipynb
results/      JSON + PNG outputs, logs
```

## Run
```
pip install -e . && python -m pytest -q tests
python experiments/m1_identify_D.py            # ~1 min
python experiments/m2_ft_calibration.py        # ~1 min
python experiments/m3_adapt_vs_baselines.py    # ~6 min (3 seeds); --quick for a smoke test
```
Colab: open `notebooks/TRACE_colab.ipynb`, set your GitHub URL, run all.

## Results so far (simulation, noise-free sensing except process noise)
**M1, identifiability.** Parity-separated regression recovers dv/dc to 0.2%/0.5% under gravity error, actuator-gain
error and uniform inertia error; the naive fit that trusts the model is off by up to 17%/76%. It does **not** fix
non-uniform inertia error (25%/50%, worse than naive on dv).
**Gauge.** A uniform inertia error s is not an error: (sM, sV, s·gain, s·F) is dynamically equivalent to the truth, so
estimates are correct up to that scale (verified: dv/s exact, gain shift = s−1).
**M2, fluctuation statistic.** For the correct model E[exp(−σ)] = 0.98 (95% CI 0.93–1.04), *including Coulomb friction
that violates fluctuation-dissipation*; the identity needs a correct path model, not FDT (FDT only gives σ the reading
heat/T). Wrong dissipation gives 0.86 (0.5×), 1.33 (1.25×), 14.4 (2×). Caveats: σ sees only the ratio D/L, so it cannot
separate a friction change from a noise change; it is far more sensitive to over- than under-estimated dissipation
(wear is the weak direction), so M3 alarms on the mean shift of σ instead.
**M3, closed-loop faults (3 seeds).**
| fault | RMS err no-adapt | SGD head | TRACE |
|---|---|---|---|
| friction ×4/×3 | 0.099 | **0.016** | 0.065 |
| actuator gain ×0.7 | 0.046 | **0.008** | 0.016 |
| payload (link-2 mass ×2) | 0.151 | 0.144 | 0.136 |

* **TRACE adaptation currently loses to the plain SGD head baseline** (slower, 24-parameter recursive fit).
* **The σ alarm is slower than a plain residual-energy alarm** (0.75–6 s vs 0.25–0.5 s), no false alarms for either.
* **Localisation works for one distinction:** friction faults are explained by the odd class (held-out R² 0.94 vs 0.22),
  gain and payload faults are not (odd 0.00). Gain vs payload are not separable (gauge-equivalent).

## Honest status and next steps
The parity decomposition is a real, tested mechanism for robust dissipation estimation and for flagging dissipative
faults. The adaptation-speed and detection-delay claims are not supported yet. Next: (1) lower-variance adaptation
(parity-regularised SGD, adaptive forgetting); (2) alarm on the odd-class residual R² instead of σ; (3) observation
noise and filtered velocities; (4) neural core (`NeuralCore`, `train.pretrain`) end-to-end; (5) MLP-residual and
selective-Bayesian-adaptation baselines (arXiv 2606.09640); (6) thermal-port backup idea.
Not covered: observation noise, stiction/backlash, hardware.
