"""Shared synthetic-epidemic ground truth and a one-shot recovery helper.

Used by the robustness experiments (noise_sweep, sparsity_sweep) so they all
recover the *same* known beta(t) and report error the same way -- on the
identifiable window where infectious prevalence is non-negligible.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import seir
from src.pinn import EpidemicPINN, Params, initial_fractions_from_cases
from src.train import set_seed, train, predict

SIGMA, GAMMA, RHO = 0.33, 0.20, 1.0
T = 160.0
STATE0 = [0.9999, 6e-5, 4e-5, 0.0]


def true_beta(t):
    """Hidden transmission rate: a lockdown drop (R0 2.5 -> 1.1) during the wave."""
    return 0.50 - 0.28 / (1 + np.exp(-(t - 45) / 8))


def ground_truth(n=200):
    """Return (t, cases_frac_true, beta_true, identifiable_mask) for the scenario."""
    t = np.linspace(0, T, n)
    states = seir.integrate_seir(STATE0, true_beta, SIGMA, GAMMA, t)
    cases = RHO * SIGMA * states[:, 1]
    identifiable = states[:, 2] > 0.02 * states[:, 2].max()
    return t, cases, true_beta(t), identifiable


def recover(t_obs, cases_obs, seed, adam_iters=2000, lbfgs_iters=400):
    """Fit the model to (t_obs, cases_obs) and return beta evaluated on a dense grid.

    Returns (t_dense, beta_pred_dense) so callers can compare against true_beta on a
    common grid regardless of how the observations were subsampled.
    """
    set_seed(seed)
    e0, i0 = initial_fractions_from_cases(cases_obs, SIGMA, GAMMA)
    params = Params(sigma=SIGMA, gamma=GAMMA, rho=RHO, beta_max=1.0, w_smooth=1e-3, n_grid=100)
    model = EpidemicPINN(t_obs, params, e0=e0, i0=i0)
    train(model, t_obs, cases_obs, adam_iters=adam_iters, lbfgs_iters=lbfgs_iters, verbose=False)

    t_dense = np.linspace(0, T, 200)
    _, beta_dense, _ = predict(model, t_dense)
    return t_dense, beta_dense


def rel_l2(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
