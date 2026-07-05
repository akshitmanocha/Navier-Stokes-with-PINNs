"""One-command reproduction of every result and figure in the project.

Runs (in order): the data overview, the synthetic-recovery validation, the
Germany headline, the classical-baseline comparison, and the noise/sparsity
sweeps. Each writes its figures to figures/ and its metrics to results/.

Usage:
    python run_all.py            # everything
    python run_all.py --quick    # skip the multi-seed sweeps (the slow part)

Everything runs on CPU + float64 in a few minutes (the sweeps dominate).
"""
import argparse
import importlib


# (module, human label). The sweeps are last so --quick can drop them.
STEPS = [
    ("experiments.explore_data", "Data overview figures"),
    ("experiments.synthetic_recovery", "Synthetic beta(t) recovery (validation)"),
    ("experiments.headline_germany", "Germany headline: R_eff(t) vs OWID Rt"),
    ("experiments.baseline_comparison", "PINN vs classical growth-rate baseline"),
]
SWEEPS = [
    ("experiments.noise_sweep", "Noise-robustness sweep"),
    ("experiments.sparsity_sweep", "Data-sparsity sweep"),
]


def run(steps):
    for module, label in steps:
        print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
        mod = importlib.import_module(module)
        if hasattr(mod, "main"):
            mod.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the multi-seed sweeps")
    args = ap.parse_args()
    run(STEPS if args.quick else STEPS + SWEEPS)
    print("\nAll done. See figures/ and results/.")
