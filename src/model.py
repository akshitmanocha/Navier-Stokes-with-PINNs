"""The transmission-rate network beta_phi(t).

The project's core idea: instead of fitting a single constant transmission rate,
recover a free *function* beta(t) as a small neural network of time, and read
R0(t) = beta(t)/gamma off it. The state trajectory (S, E, I, R) is not a network
-- it is produced by integrating the SEIR ODE with this beta (see
``src.seir.integrate_seir_torch``), so the physics holds by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim, out_dim, width, depth):
    """A plain tanh MLP. ``depth`` = number of hidden layers."""
    layers = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


class BetaNet(nn.Module):
    """tau in [0,1] -> beta(t), bounded to (beta_min, beta_max).

    The sigmoid output transform is a *hard* positivity/boundedness constraint
    (more robust than penalising negative outputs). Kept deliberately
    low-capacity: a small network is itself a smoothness prior on beta(t).
    """

    def __init__(self, beta_min=0.0, beta_max=1.5, width=16, depth=2):
        super().__init__()
        self.net = _mlp(1, 1, width, depth)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, tau):
        span = self.beta_max - self.beta_min
        return self.beta_min + span * torch.sigmoid(self.net(tau))
