"""Training loop: Adam for global search, then L-BFGS to polish.

Standard recipe. Everything runs CPU + float64: the models are tiny and float64
is numerically clean for the ODE rollout and L-BFGS (Apple MPS lacks float64).
"""
from __future__ import annotations

import numpy as np
import torch

from .pinn import EpidemicPINN


def set_seed(seed):
    """Seed numpy + torch so every run is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(arr):
    return torch.as_tensor(np.asarray(arr), dtype=torch.float64)


def train(model: EpidemicPINN, t_data, cases_frac,
          adam_iters=4000, lbfgs_iters=500, lr=5e-3, log_every=500, verbose=True):
    """Fit ``model`` to observed cases. Returns the loss-component history."""
    torch.set_default_dtype(torch.float64)
    model.double()
    t = _to_tensor(t_data)
    cases = _to_tensor(cases_frac)
    history = []

    # stage 1: Adam (global search) with mild exponential lr decay
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=0.999)
    for it in range(adam_iters):
        adam.zero_grad()
        total, parts = model.loss(t, cases)
        total.backward()
        adam.step()
        sched.step()
        history.append(parts)
        if verbose and it % log_every == 0:
            print(f"[adam {it:5d}] total={total.item():.3e} "
                  + " ".join(f"{k}={v:.2e}" for k, v in parts.items()))

    # stage 2: L-BFGS (polish to a tight minimum)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=lbfgs_iters, max_eval=lbfgs_iters,
        history_size=50, tolerance_grad=1e-10, tolerance_change=1e-12,
        line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        total, _ = model.loss(t, cases)
        total.backward()
        return total

    lbfgs.step(closure)
    total, parts = model.loss(t, cases)
    history.append(parts)
    if verbose:
        print(f"[lbfgs done] total={total.item():.3e} "
              + " ".join(f"{k}={v:.2e}" for k, v in parts.items()))
    return history


@torch.no_grad()
def predict(model: EpidemicPINN, t_eval):
    """Evaluate the trained model. Returns (states, beta, r_eff) as numpy arrays.

    ``states`` is interpolated onto ``t_eval`` for plotting alongside the data.
    """
    cases, reff, beta, states_grid = model.predict_on(t_eval)
    # interpolate full state columns onto t_eval for convenience
    from .pinn import _interp
    cols = [_interp(t_eval, model.t_grid, states_grid[:, j]).numpy() for j in range(4)]
    states = np.stack(cols, axis=1)
    return states, beta.numpy(), reff.numpy()
