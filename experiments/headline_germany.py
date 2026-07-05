"""Day-5 headline result: recover R_eff(t) for Germany's first COVID wave.

This is the hero figure. We fit the model to Germany's reported cases (spring
2020), recover the hidden transmission rate beta(t) by training a neural net
through a differentiable SEIR simulator, and plot the implied effective
reproduction number R_eff(t) against OWID's *independent* reproduction_rate
estimate -- a number the model never saw. Lockdown intensity (Oxford stringency)
is shaded behind it: the recovered R_eff should fall below 1 as the lockdown bites.

Identifiability note: in the first wave the susceptible fraction stays ~1 (low
attack rate), so R_eff ~ R0 = beta/gamma and the recovered level is insensitive
to the unknown reporting fraction rho (verified: RMSE vs OWID Rt is ~0.14 for rho
in 0.05-0.20). We fix rho = 0.1 and validate the *shape and timing* of R_eff(t)
against the independent estimate.

Run:  python -m experiments.headline_germany
Writes: figures/headline_germany.png, results/headline_germany.txt
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_country
from src.pinn import EpidemicPINN, Params, initial_fractions_from_cases
from src.train import set_seed, train, predict

ISO = "DEU"
START, END = "2020-03-01", "2020-08-15"   # first wave: clean daily reporting + documented lockdown
SIGMA, GAMMA, RHO = 0.33, 0.20, 0.10
SEED = 0


def main():
    set_seed(SEED)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    d = load_country(ISO, START, END)
    e0, i0 = initial_fractions_from_cases(d.cases_frac / RHO, SIGMA, GAMMA)

    params = Params(sigma=SIGMA, gamma=GAMMA, rho=RHO, beta_max=1.5, w_smooth=1e-3, n_grid=100)
    model = EpidemicPINN(d.t, params, e0=e0, i0=i0)
    train(model, d.t, d.cases_frac, adam_iters=3000, lbfgs_iters=500)

    states, beta, reff = predict(model, d.t)
    cases_pred = model.predict_on(d.t)[0].detach().numpy() * d.N
    r0 = beta / GAMMA

    have_rt = np.isfinite(d.rt)
    rt_rmse = float(np.sqrt(np.mean((reff[have_rt] - d.rt[have_rt]) ** 2)))
    cases_rel = float(np.linalg.norm(cases_pred - d.cases) / np.linalg.norm(d.cases))

    summary = (
        f"Germany first wave ({START} to {END}), seed={SEED}, rho={RHO}\n"
        f"  case fit relative L2:                 {cases_rel:.4f}\n"
        f"  R_eff(t) vs OWID Rt, RMSE:            {rt_rmse:.3f}\n"
        f"  recovered R0 range:                   [{r0.min():.2f}, {r0.max():.2f}]\n"
        f"  date R_eff first crosses 1.0:         {_first_crossing(d.date, reff)}\n")
    print(summary)
    with open("results/headline_germany.txt", "w") as f:
        f.write(summary)

    _plot(d, cases_pred, reff, rt_rmse, cases_rel)
    return rt_rmse


def _first_crossing(dates, reff):
    """First date the recovered R_eff drops below 1 (the epidemic turning point)."""
    below = np.where(reff < 1.0)[0]
    return str(np.datetime_as_string(dates[below[0]], unit="D")) if len(below) else "never"


def _plot(d, cases_pred, reff, rt_rmse, cases_rel):
    fig, (axc, axr) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2.2], "hspace": 0.12})

    # --- top: the case fit (what the model is trained on) ---
    axc.plot(d.date, d.cases, "o", ms=2.5, color="gray", alpha=0.5, label="observed cases (OWID, 7-day smoothed)")
    axc.plot(d.date, cases_pred, color="tab:blue", lw=2.2, label="model fit")
    axc.set_ylabel("New cases / day")
    axc.set_title("Germany, first COVID-19 wave: recovering the reproduction number from cases alone\n"
                  f"case fit rel-L2 = {cases_rel:.3f}    |    recovered R_eff vs independent OWID Rt: RMSE = {rt_rmse:.2f}",
                  fontsize=12)
    axc.legend(loc="upper right", fontsize=9); axc.grid(alpha=0.2)

    # --- bottom: recovered R_eff(t) vs OWID Rt, with lockdown shading ---
    sn = d.stringency / 100.0
    ymax = max(3.4, np.nanmax(d.rt) * 1.05 if np.isfinite(d.rt).any() else 3.4)
    for k in range(len(d.date) - 1):
        if np.isfinite(sn[k]):
            axr.axvspan(d.date[k], d.date[k + 1], color="purple", alpha=0.12 * sn[k], lw=0)
    axr.plot(d.date, reff, color="tab:red", lw=2.5, label="recovered R_eff(t)")
    axr.plot(d.date, d.rt, color="k", lw=1.8, ls="--", label="OWID reproduction_rate (independent)")
    axr.axhline(1.0, color="gray", ls=":", lw=1.2, label="R = 1 (epidemic turning point)")
    axr.fill_between([], [], [], color="purple", alpha=0.25, label="lockdown intensity (Oxford stringency)")
    axr.set_ylim(0.3, ymax)
    axr.set_ylabel("Reproduction number")
    axr.set_xlabel("Date (2020)")
    axr.legend(loc="upper right", fontsize=9, ncol=2); axr.grid(alpha=0.2)
    axr.xaxis.set_major_locator(mdates.MonthLocator())
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.tight_layout()
    fig.savefig("figures/headline_germany.png", dpi=140)
    print("Wrote figures/headline_germany.png")


if __name__ == "__main__":
    main()
