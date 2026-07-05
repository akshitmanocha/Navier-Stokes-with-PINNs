"""Day-9 robustness experiment: how sparse can the observations be?

Real reporting is often weekly, not daily. We subsample the synthetic cases to
every k days (k = 1, 3, 7, 14), recover beta(t), and measure the relative-L2 error
on the identifiable window, averaged over seeds. This finds where recovery breaks
as data thins out -- directly relevant to coarse real-world reporting.

A fixed 5% noise is applied at every sparsity so the only varying factor is the
observation interval.

Run:  python -m experiments.sparsity_sweep
Writes: figures/sparsity_sweep.png, results/sparsity_sweep.csv
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._synthetic import ground_truth, recover, rel_l2, true_beta

INTERVALS = [1, 3, 7, 14]   # observe every k days
SEEDS = [0, 1, 2, 3, 4]
NOISE = 0.05


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    t, cases_true, _, identifiable = ground_truth()
    means, stds = [], []
    rows = ["interval_days,n_obs,mean_rel_l2,std_rel_l2"]

    for k in INTERVALS:
        idx = np.arange(0, len(t), k)
        errs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            obs_full = np.clip(cases_true * (1 + NOISE * rng.standard_normal(len(t))), 0, None)
            t_obs, obs = t[idx], obs_full[idx]
            t_dense, beta_dense = recover(t_obs, obs, seed)
            ident_dense = np.interp(t_dense, t, identifiable.astype(float)) > 0.5
            errs.append(rel_l2(beta_dense, true_beta(t_dense), ident_dense))
        means.append(np.mean(errs)); stds.append(np.std(errs))
        rows.append(f"{k},{len(idx)},{np.mean(errs):.4f},{np.std(errs):.4f}")
        print(f"every {k:2d} days ({len(idx):3d} obs): beta rel-L2 = {np.mean(errs):.3f} +/- {np.std(errs):.3f}")

    with open("results/sparsity_sweep.csv", "w") as f:
        f.write("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(INTERVALS, means, yerr=stds, marker="o", capsize=4, lw=2, color="tab:blue")
    ax.set_xlabel("observation interval (days between reports)")
    ax.set_ylabel("recovered beta(t) relative L2 error\n(identifiable window)")
    ax.set_title(f"Data-sparsity robustness of beta(t) recovery\n(5% noise, mean $\\pm$ std over {len(SEEDS)} seeds)")
    ax.set_xticks(INTERVALS); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("figures/sparsity_sweep.png", dpi=140)
    print("Wrote figures/sparsity_sweep.png")


if __name__ == "__main__":
    main()
