"""SEIR epidemic model: the physics of the project.

The model recovers a time-varying transmission rate beta(t) by training a neural
network *through* a differentiable SEIR simulator (a neural-ODE / differentiable-
simulator approach). Two integrators live here, sharing one right-hand side:

  * ``integrate_seir`` -- a plain-numpy RK4 solver. Given a known beta(t) it
    produces an exact trajectory. Used to generate synthetic ground-truth data and
    as the test oracle.
  * ``integrate_seir_torch`` -- the same RK4 scheme in torch, differentiable
    end-to-end. The PINN backpropagates the case-fit loss through it into the
    beta network. Because the ODE is integrated exactly, the SEIR physics holds by
    construction -- there is no separate residual term to weight.

Compartments are fractions of the population N (S + E + I + R = 1):

    dS/dt = - beta(t) S I
    dE/dt = + beta(t) S I - sigma E
    dI/dt = + sigma E     - gamma I
    dR/dt = + gamma I

beta, sigma, gamma are per-day rates. (S I is already S I / N once S, I are
fractions, so N drops out of the normalised system.)
"""
from __future__ import annotations

import numpy as np
import torch


# --- shared right-hand side -------------------------------------------------


def _rhs(S, E, I, R, beta, sigma, gamma):
    """SEIR derivatives. Works for numpy scalars/arrays or torch tensors alike."""
    new_infections = beta * S * I
    e_to_i = sigma * E
    i_to_r = gamma * I
    return (-new_infections,
            new_infections - e_to_i,
            e_to_i - i_to_r,
            i_to_r)


# --- ground-truth forward model (numpy) ------------------------------------


def integrate_seir(state0, beta_fn, sigma, gamma, t):
    """Integrate the SEIR ODE on grid ``t`` with classical RK4 (numpy).

    ``beta_fn`` maps a time to the transmission rate, so this handles a
    time-varying beta(t). Returns an array of shape (len(t), 4): the (S,E,I,R)
    trajectory. RK4 (not Euler) so the trajectory is accurate enough to be a
    test oracle and to generate synthetic data.
    """
    states = np.empty((len(t), 4))
    states[0] = state0
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        tk = t[k]
        S, E, I, R = states[k]
        k1 = _rhs(S, E, I, R, beta_fn(tk), sigma, gamma)
        k2 = _rhs(*(np.array([S, E, I, R]) + 0.5 * dt * np.array(k1)), beta_fn(tk + 0.5 * dt), sigma, gamma)
        k3 = _rhs(*(np.array([S, E, I, R]) + 0.5 * dt * np.array(k2)), beta_fn(tk + 0.5 * dt), sigma, gamma)
        k4 = _rhs(*(np.array([S, E, I, R]) + dt * np.array(k3)), beta_fn(tk + dt), sigma, gamma)
        states[k + 1] = np.array([S, E, I, R]) + (dt / 6.0) * (
            np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4))
    return states


# --- differentiable forward model (torch) ----------------------------------


def integrate_seir_torch(state0, beta_grid, sigma, gamma, t):
    """Differentiable RK4 integration of SEIR (torch), returning the trajectory.

    The PINN trains by backpropagating through this. To keep the rollout cheap and
    fully differentiable, beta is supplied pre-evaluated on the integration grid as
    ``beta_grid`` (shape (len(t),)); RK4's midpoints use the average of adjacent
    grid values, which is second-order accurate and avoids extra network calls.

    Args:
        state0:    length-4 tensor (S, E, I, R) at t[0], differentiable.
        beta_grid: (len(t),) tensor of beta evaluated at each ``t``.
        sigma, gamma: fixed rates.
        t:         (len(t),) numpy array of times (assumed increasing).

    Returns:
        (len(t), 4) tensor trajectory, differentiable wrt state0 and beta_grid.
    """
    S, E, I, R = state0[0], state0[1], state0[2], state0[3]
    traj = [torch.stack([S, E, I, R])]
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        b0 = beta_grid[k]
        bm = 0.5 * (beta_grid[k] + beta_grid[k + 1])   # midpoint beta
        b1 = beta_grid[k + 1]
        k1 = _rhs(S, E, I, R, b0, sigma, gamma)
        k2 = _rhs(S + 0.5 * dt * k1[0], E + 0.5 * dt * k1[1], I + 0.5 * dt * k1[2], R + 0.5 * dt * k1[3], bm, sigma, gamma)
        k3 = _rhs(S + 0.5 * dt * k2[0], E + 0.5 * dt * k2[1], I + 0.5 * dt * k2[2], R + 0.5 * dt * k2[3], bm, sigma, gamma)
        k4 = _rhs(S + dt * k3[0], E + dt * k3[1], I + dt * k3[2], R + dt * k3[3], b1, sigma, gamma)
        S = S + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        E = E + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        I = I + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        R = R + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        traj.append(torch.stack([S, E, I, R]))
    return torch.stack(traj)


# --- reported quantities ----------------------------------------------------


def r0(beta, gamma):
    """Basic reproduction number R0(t) = beta(t) / gamma."""
    return beta / gamma


def r_eff(beta, gamma, S):
    """Effective reproduction number R_eff(t) = (beta/gamma) * S.

    S is the susceptible *fraction*. This is the interpretable curve that crosses
    1.0 at epidemic turning points and is compared against OWID's reproduction_rate.
    """
    return (beta / gamma) * S


def incidence(E, sigma, rho):
    """Observation operator: modelled reported new-case fraction per day.

    New symptomatic cases flow out of E at rate sigma; only a fraction rho is
    reported/ascertained. ``E`` is the exposed fraction; multiply the result by the
    population N for counts.
    """
    return rho * sigma * E
