"""Classical baseline: a rolling-window growth-rate estimate of R0(t).

The standard non-PINN way to get a time-varying reproduction number from cases is
to estimate the local exponential growth rate r(t) in a sliding window and map it
to R0 through the SEIR characteristic equation. For SEIR a growth rate r implies

    R0 = (1 + r/sigma) (1 + r/gamma)

(the dominant eigenvalue relation: a case grows through the E and I stages, each
contributing a (1 + r/rate) factor). We fit r(t) by regressing log(cases) on time
in a centred window, convert to R0(t), and compare against the PINN recovery on
the same synthetic ground truth.

This shows what the PINN buys over a classical method: the growth-rate baseline is
noisy, undefined where cases are flat/zero, and not constrained to be ODE-
consistent, whereas the PINN gives a smooth, mechanistic, ODE-consistent beta(t).

Run:  python -m experiments.baseline_comparison
Writes: figures/baseline_comparison.png, results/baseline_comparison.txt
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._synthetic import (ground_truth, recover, rel_l2, true_beta,
                                     SIGMA, GAMMA)

WINDOW = 14   # days in the rolling growth-rate window
NOISE = 0.05
SEED = 0


def growth_rate_r0(t, cases, window):
    """Rolling-window exponential growth rate -> R0 via the SEIR relation.

    Returns R0_hat aligned to ``t`` (NaN where the window has too few positive
    cases to fit a slope).
    """
    log_c = np.full(len(t), np.nan)
    pos = cases > 0
    log_c[pos] = np.log(cases[pos])
    r = np.full(len(t), np.nan)
    half = window // 2
    for k in range(len(t)):
        lo, hi = max(0, k - half), min(len(t), k + half + 1)
        sl = slice(lo, hi)
        good = np.isfinite(log_c[sl])
        if good.sum() >= 5:
            tt, yy = t[sl][good], log_c[sl][good]
            r[k] = np.polyfit(tt, yy, 1)[0]   # slope of log(cases) = growth rate
    r0 = (1 + r / SIGMA) * (1 + r / GAMMA)
    return r0


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    t, cases_true, beta_true, identifiable = ground_truth()
    r0_true = beta_true / GAMMA

    rng = np.random.default_rng(SEED)
    obs = np.clip(cases_true * (1 + NOISE * rng.standard_normal(len(t))), 0, None)

    # classical baseline
    r0_baseline = growth_rate_r0(t, obs, WINDOW)

    # PINN recovery on the same data
    t_dense, beta_pinn = recover(t, obs, SEED)
    r0_pinn = np.interp(t, t_dense, beta_pinn) / GAMMA

    # errors on the identifiable window (baseline may be NaN -> count only finite)
    base_ok = identifiable & np.isfinite(r0_baseline)
    err_base = rel_l2(r0_baseline[base_ok], r0_true[base_ok])
    err_pinn = rel_l2(r0_pinn[identifiable], r0_true[identifiable])

    summary = (
        f"PINN vs classical growth-rate baseline (noise={NOISE:.0%}, seed={SEED})\n"
        f"  R0(t) rel-L2, classical baseline (identifiable window): {err_base:.4f}\n"
        f"  R0(t) rel-L2, PINN              (identifiable window): {err_pinn:.4f}\n"
        f"  baseline undefined on {np.sum(identifiable & ~np.isfinite(r0_baseline))} "
        f"of {identifiable.sum()} identifiable days\n")
    print(summary)
    with open("results/baseline_comparison.txt", "w") as f:
        f.write(summary)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, r0_true, "k-", lw=2.5, label="true R0(t)")
    ax.plot(t, r0_pinn, "tab:red", lw=2.2, ls="--", label=f"PINN  (rel-L2 {err_pinn:.3f})")
    ax.plot(t, r0_baseline, "tab:green", lw=1.5, marker=".", ms=4, alpha=0.8,
            label=f"classical growth-rate  (rel-L2 {err_base:.3f})")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.fill_between(t, *ax.get_ylim(), where=~identifiable, color="gray", alpha=0.12,
                    label="not identifiable (I $\\approx$ 0)")
    ax.set_xlabel("day"); ax.set_ylabel("reproduction number R0(t)")
    ax.set_title("PINN vs a classical growth-rate baseline\n"
                 "the PINN is smooth, ODE-consistent, and defined everywhere; the baseline is noisy and gappy")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/baseline_comparison.png", dpi=140)
    print("Wrote figures/baseline_comparison.png")


if __name__ == "__main__":
    main()
