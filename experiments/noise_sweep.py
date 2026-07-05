"""Day-8 robustness experiment: how does noise degrade beta(t) recovery?

We add increasing multiplicative Gaussian noise to the synthetic reported cases
and measure the relative-L2 error of the recovered beta(t) on the identifiable
window, averaged over several seeds (error bars = std). A graceful, monotone
degradation is the signature of a well-posed, stable inverse method.

Run:  python -m experiments.noise_sweep
Writes: figures/noise_sweep.png, results/noise_sweep.csv
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._synthetic import ground_truth, recover, rel_l2, true_beta

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    t, cases_true, _, identifiable = ground_truth()
    means, stds = [], []
    rows = ["noise,mean_rel_l2,std_rel_l2"]

    for noise in NOISE_LEVELS:
        errs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            obs = cases_true if noise == 0 else np.clip(
                cases_true * (1 + noise * rng.standard_normal(len(t))), 0, None)
            t_dense, beta_dense = recover(t, obs, seed)
            # compare on the identifiable window, resampled to the dense grid
            ident_dense = np.interp(t_dense, t, identifiable.astype(float)) > 0.5
            errs.append(rel_l2(beta_dense, true_beta(t_dense), ident_dense))
        means.append(np.mean(errs)); stds.append(np.std(errs))
        rows.append(f"{noise},{np.mean(errs):.4f},{np.std(errs):.4f}")
        print(f"noise={noise:.0%}: beta rel-L2 = {np.mean(errs):.3f} +/- {np.std(errs):.3f}")

    with open("results/noise_sweep.csv", "w") as f:
        f.write("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar([n * 100 for n in NOISE_LEVELS], means, yerr=stds,
                marker="o", capsize=4, lw=2, color="tab:red")
    ax.set_xlabel("observation noise (%)")
    ax.set_ylabel("recovered beta(t) relative L2 error\n(identifiable window)")
    ax.set_title(f"Noise robustness of beta(t) recovery\n(mean $\\pm$ std over {len(SEEDS)} seeds)")
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("figures/noise_sweep.png", dpi=140)
    print("Wrote figures/noise_sweep.png")


if __name__ == "__main__":
    main()
