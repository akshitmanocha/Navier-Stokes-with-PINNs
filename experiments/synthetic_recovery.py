"""Day-4 controlled validation: can the model recover a *known* beta(t)?

We generate synthetic SEIR data with a beta(t) we choose, train on (noisy)
reported cases only, and check that we recover the hidden beta(t) it was never
shown. This is the ground-truth proof of correctness, done before touching real
data: on real data the true beta is unknown, so recovery can only be *verified*
here.

The honest framing -- and the project's central identifiability point -- is that
beta(t) is only recoverable where the infectious prevalence I(t) is
non-negligible: where I ~ 0, the term beta*S*I vanishes and the data carry no
information about beta. We therefore report error on that identifiable window and
shade the rest.

Run:  python -m experiments.synthetic_recovery
Writes: figures/synthetic_recovery.png, results/synthetic_recovery.txt
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import seir
from src.pinn import EpidemicPINN, Params
from src.train import set_seed, train, predict

SIGMA, GAMMA, RHO = 0.33, 0.20, 1.0
T = 160.0          # days
N = 1_000_000.0    # population
NOISE = 0.05       # 5% multiplicative gaussian noise on observed cases
SEED = 0
STATE0 = [0.9999, 6e-5, 4e-5, 0.0]   # small seed, low attack rate (like real first waves)


def true_beta(t):
    """Hidden transmission rate: a sharp 'lockdown' drop during the active wave.

    R0 falls from 2.5 to 1.1 as the lockdown takes effect around day 45 -- the
    canonical COVID story (an intervention flattens a growing wave). The drop
    happens while infectious prevalence is high, the regime where beta is
    identifiable from cases alone.
    """
    return 0.50 - 0.28 / (1 + np.exp(-(t - 45) / 8))


def rel_l2(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    set_seed(SEED)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- generate synthetic ground truth ---
    t = np.linspace(0, T, 200)
    states_true = seir.integrate_seir(STATE0, true_beta, SIGMA, GAMMA, t)
    cases_frac_true = RHO * SIGMA * states_true[:, 1]
    beta_true_vals = true_beta(t)
    r0_true = beta_true_vals / GAMMA

    # identifiable window: where infectious prevalence is non-negligible
    I_true = states_true[:, 2]
    identifiable = I_true > 0.02 * I_true.max()

    # observe noisy reported cases (the only thing the model sees)
    rng = np.random.default_rng(SEED)
    cases_obs = np.clip(cases_frac_true * (1 + NOISE * rng.standard_normal(len(t))), 0, None)

    # --- fit ---
    params = Params(sigma=SIGMA, gamma=GAMMA, rho=RHO, beta_max=1.0, w_smooth=1e-3)
    model = EpidemicPINN(t, params, e0=STATE0[1], i0=STATE0[2])
    train(model, t, cases_obs, adam_iters=3000, lbfgs_iters=500)

    # --- evaluate recovery ---
    _, beta_pred, _ = predict(model, t)
    cases_pred, _, _, _ = model.predict_on(t)
    cases_pred = cases_pred.detach().numpy()
    r0_pred = beta_pred / GAMMA

    err_beta_ident = rel_l2(beta_pred, beta_true_vals, identifiable)
    err_beta_full = rel_l2(beta_pred, beta_true_vals)
    err_cases = rel_l2(cases_pred, cases_frac_true)

    summary = (
        f"Synthetic beta(t) recovery  (noise={NOISE:.0%}, seed={SEED})\n"
        f"  relative L2 error, recovered beta(t), identifiable window: {err_beta_ident:.4f}\n"
        f"  relative L2 error, recovered beta(t), full window:         {err_beta_full:.4f}\n"
        f"  relative L2 error, fitted cases:                           {err_cases:.4f}\n"
        f"  identifiable fraction of window (I > 2% of peak):          {identifiable.mean():.2f}\n")
    print(summary)
    with open("results/synthetic_recovery.txt", "w") as f:
        f.write(summary)

    _plot(t, r0_true, r0_pred, identifiable, cases_obs, cases_frac_true, cases_pred,
          err_beta_ident, err_cases, N)
    return err_beta_ident


def _plot(t, r0_true, r0_pred, identifiable, cases_obs, cases_true, cases_pred,
          err_beta, err_cases, N):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, r0_true, "k-", lw=2.5, label="true R0(t) (hidden)")
    ax1.plot(t, r0_pred, "tab:red", lw=2, ls="--", label="recovered R0(t)")
    ax1.axhline(1.0, color="gray", ls=":", lw=1, label="R0 = 1")
    ax1.fill_between(t, *ax1.get_ylim(), where=~identifiable, color="gray", alpha=0.12,
                     label="not identifiable (I $\\approx$ 0)")
    ax1.set_xlabel("day"); ax1.set_ylabel("reproduction number R0(t)")
    ax1.set_title(f"Recovery of a hidden R0(t)\nrel-L2 = {err_beta:.3f} on the identifiable window")
    ax1.legend(loc="upper right", fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(t, cases_obs * N, "o", ms=2.5, color="gray", alpha=0.45,
             label=f"observed cases (+{0.05:.0%} noise)")
    ax2.plot(t, cases_true * N, "k-", lw=1.5, label="true cases")
    ax2.plot(t, cases_pred * N, "tab:red", lw=2, ls="--", label="model fit")
    ax2.set_xlabel("day"); ax2.set_ylabel("reported new cases / day")
    ax2.set_title(f"Case fit  (rel-L2 = {err_cases:.3f})")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("Controlled validation: the model recovers a transmission rate it never saw", fontsize=13)
    fig.tight_layout()
    fig.savefig("figures/synthetic_recovery.png", dpi=140)
    print("Wrote figures/synthetic_recovery.png")


if __name__ == "__main__":
    main()
