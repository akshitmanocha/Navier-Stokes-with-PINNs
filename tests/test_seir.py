"""Unit tests for the SEIR physics -- the project's correctness gate.

The central test (``test_torch_integrator_matches_numpy``) closes the loop between
the two integrators: the differentiable torch RK4 used for training must reproduce
the numpy RK4 ground-truth solver to high precision on the same beta(t). If they
agree, backprop is happening through a correct forward model.
"""
import numpy as np
import torch

from src import seir

torch.set_default_dtype(torch.float64)

SIGMA, GAMMA = 0.33, 0.20


def _beta_fn(t):
    """A smooth, clearly time-varying beta(t): a 'lockdown' drop and recovery."""
    return 0.6 - 0.3 * np.tanh((t - 60.0) / 15.0) + 0.1 * np.sin(t / 30.0)


def test_numpy_integrator_conserves_population():
    """S + E + I + R must stay = 1 (the ODE only moves mass between compartments)."""
    t = np.linspace(0, 180, 1000)
    states = seir.integrate_seir([0.99, 0.005, 0.005, 0.0], _beta_fn, SIGMA, GAMMA, t)
    assert np.allclose(states.sum(axis=1), 1.0, atol=1e-8)


def test_torch_integrator_matches_numpy():
    """The differentiable torch RK4 must reproduce the numpy RK4 trajectory.

    Same beta(t), same initial state, same grid -> the two integrators should agree
    to within their (second-order, midpoint-beta) discretisation error. This proves
    the model the PINN backprops through is the SEIR system we tested in numpy.
    """
    t = np.linspace(0, 180, 400)
    state0 = [0.99, 0.005, 0.005, 0.0]
    np_states = seir.integrate_seir(state0, _beta_fn, SIGMA, GAMMA, t)

    beta_grid = torch.tensor(_beta_fn(t))
    state0_t = torch.tensor(state0)
    torch_states = seir.integrate_seir_torch(state0_t, beta_grid, SIGMA, GAMMA, t).numpy()

    assert np.max(np.abs(torch_states - np_states)) < 1e-4


def test_torch_integrator_is_differentiable():
    """Gradients must flow from the trajectory back to beta -- the whole point."""
    t = np.linspace(0, 120, 200)
    beta_grid = torch.full((len(t),), 0.4, requires_grad=True)
    state0 = torch.tensor([0.99, 0.006, 0.004, 0.0])
    states = seir.integrate_seir_torch(state0, beta_grid, SIGMA, GAMMA, t)
    states[:, 1].sum().backward()           # d(total exposed)/d(beta)
    assert beta_grid.grad is not None
    assert torch.isfinite(beta_grid.grad).all()
    assert beta_grid.grad.abs().sum() > 0    # beta genuinely affects the trajectory


def test_r0_and_reff_formulas():
    """R0 = beta/gamma; R_eff = R0 * S."""
    beta = torch.tensor([0.4])
    assert torch.allclose(seir.r0(beta, GAMMA), torch.tensor([2.0]))
    assert torch.allclose(seir.r_eff(beta, GAMMA, torch.tensor([0.5])), torch.tensor([1.0]))


def test_incidence_scaling():
    """Incidence = rho*sigma*E; linear in rho."""
    E = torch.tensor([0.05])
    assert torch.allclose(seir.incidence(E, SIGMA, rho=0.5),
                          0.5 * seir.incidence(E, SIGMA, rho=1.0))
