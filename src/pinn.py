"""The epidemic PINN: a beta(t) network trained through a differentiable SEIR ODE.

The model has two kinds of unknown:
  * the transmission-rate network beta_phi(t) (a ``BetaNet``);
  * the initial infected fractions E0, I0 (learnable; S0 = 1 - E0 - I0, R0 = 0),
    which anchor the otherwise-unobserved start of the latent trajectory.

A forward pass evaluates beta on the integration grid, integrates SEIR with
``integrate_seir_torch``, and reads modelled reported cases off the E compartment.
Because the ODE is integrated exactly, the SEIR physics holds by construction --
so the loss is just a (relative) data-fit term plus a smoothness prior on beta(t).
No data-vs-residual weight to tune.

sigma and gamma are *fixed* (learning them jointly with beta is non-identifiable);
only beta(t) and the two initial fractions are recovered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .model import BetaNet
from . import seir


@dataclass
class Params:
    """Fixed epidemiological constants and loss settings for one fit."""

    sigma: float = 0.33          # E->I rate (latent ~3 days; pre-symptomatic spread)
    gamma: float = 0.20          # I->R rate (infectious ~5 days; Millevoi 2023)
    rho: float = 1.0             # reporting fraction (constant; biases level not shape)
    beta_max: float = 1.5        # upper bound for beta(t) (R0 up to 7.5 at gamma=0.2)
    w_smooth: float = 1e-3       # weight on the dbeta/dtau smoothness prior
    n_grid: int = 200            # SEIR integration grid resolution over the window


class EpidemicPINN(nn.Module):
    """beta(t) network + learnable initial fractions, fit through a SEIR integrator.

    Args:
        t: (n,) array of observation times in days (0..T), strictly increasing.
        params: fixed constants and loss settings.
        e0, i0: initial guesses for the exposed/infectious fractions at t[0].
    """

    # initial E0, I0 are each bounded to (0, MAX_INIT) via a sigmoid, so the
    # learnable start can never drive S0 = 1 - E0 - I0 negative (which would make
    # the ODE diverge). 0.1 leaves ample headroom for any realistic first wave.
    MAX_INIT = 0.1

    def __init__(self, t, params: Params, e0, i0):
        super().__init__()
        self.p = params
        self.beta = BetaNet(beta_min=0.0, beta_max=params.beta_max)
        # store the sigmoid pre-images of the initial-fraction guesses
        self.raw_e0 = nn.Parameter(torch.tensor(_logit(e0, self.MAX_INIT), dtype=torch.float64))
        self.raw_i0 = nn.Parameter(torch.tensor(_logit(i0, self.MAX_INIT), dtype=torch.float64))
        # uniform integration grid spanning the observation window
        self.t_grid = np.linspace(float(t[0]), float(t[-1]), params.n_grid)
        self.T = float(t[-1] - t[0])
        tau = (self.t_grid - self.t_grid[0]) / self.T
        self.register_buffer("tau_grid", torch.tensor(tau.reshape(-1, 1), dtype=torch.float64))

    # -- forward --
    def rollout(self):
        """Integrate SEIR on the grid; return (states (G,4), beta_grid (G,))."""
        beta_grid = self.beta(self.tau_grid).squeeze(1)
        E0 = self.MAX_INIT * torch.sigmoid(self.raw_e0)
        I0 = self.MAX_INIT * torch.sigmoid(self.raw_i0)
        S0 = 1.0 - E0 - I0
        R0 = torch.zeros((), dtype=torch.float64)
        state0 = torch.stack([S0, E0, I0, R0])
        states = seir.integrate_seir_torch(state0, beta_grid, self.p.sigma, self.p.gamma, self.t_grid)
        return states, beta_grid

    def predict_on(self, t_eval):
        """Modelled reported case fraction, R_eff, beta and states at times ``t_eval``.

        Integration happens on the internal grid; outputs are linearly interpolated
        to ``t_eval`` (the observation/plotting times).
        """
        states, beta_grid = self.rollout()
        E = states[:, 1]
        cases = seir.incidence(E, self.p.sigma, self.p.rho)
        reff = seir.r_eff(beta_grid, self.p.gamma, states[:, 0])
        return _interp(t_eval, self.t_grid, cases), \
            _interp(t_eval, self.t_grid, reff), \
            _interp(t_eval, self.t_grid, beta_grid), states

    # -- loss --
    def loss(self, t_data, cases_frac):
        """Relative case-fit MSE + smoothness prior on beta(t).

        ``t_data`` and ``cases_frac`` are 1-D tensors of the observations.
        """
        states, beta_grid = self.rollout()
        cases_grid = seir.incidence(states[:, 1], self.p.sigma, self.p.rho)
        cases_pred = _interp(t_data, self.t_grid, cases_grid)

        scale = torch.mean(cases_frac ** 2) + 1e-18
        l_data = torch.mean((cases_pred - cases_frac) ** 2) / scale

        # smoothness: penalise dbeta/dtau (finite-difference on the grid is enough)
        dbeta = (beta_grid[1:] - beta_grid[:-1]) * self.p.n_grid
        l_smooth = torch.mean(dbeta ** 2)

        total = l_data + self.p.w_smooth * l_smooth
        return total, {"data": l_data.item(), "smooth": l_smooth.item()}


def _logit(value, scale):
    """Inverse of ``scale * sigmoid(.)`` -- the pre-image of a bounded fraction."""
    frac = min(max(value / scale, 1e-6), 1 - 1e-6)
    return float(np.log(frac / (1 - frac)))


def _interp(x, xp, fp):
    """Differentiable 1-D linear interpolation of (xp, fp) at points x (torch)."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    xp_t = torch.as_tensor(np.asarray(xp), dtype=torch.float64)
    idx = torch.clamp(torch.searchsorted(xp_t, x, right=True), 1, len(xp_t) - 1)
    x0, x1 = xp_t[idx - 1], xp_t[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    w = (x - x0) / (x1 - x0)
    return f0 + w * (f1 - f0)


def initial_fractions_from_cases(cases_frac, sigma, gamma):
    """First-guess (E0, I0) infected fractions from the initial observed case rate.

    At early growth the infectious prevalence I ~ incidence/gamma and the exposed
    pool E ~ incidence/sigma. Only needs to be approximately right -- E0, I0 are
    then refined as learnable parameters during the fit.
    """
    inc0 = float(cases_frac[0])
    return inc0 / sigma, inc0 / gamma
