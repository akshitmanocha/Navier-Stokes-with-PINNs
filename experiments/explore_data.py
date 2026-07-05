"""
Exploratory visualization of the OWID COVID-19 data for the Epidemic-PINN project.

Produces four figures in figures/:
  1. data_overview_germany.png   - the "hero preview": cases + Rt + lockdown stringency for one country
  2. data_multicountry_cases.png - smoothed new cases across countries (log scale)
  3. data_rt_vs_stringency.png   - independent Rt estimate vs lockdown intensity (the inverse-problem target)
  4. data_quality_raw_vs_smooth.png - why we must use the smoothed series (raw is spiky)

Run:  python -m experiments.explore_data
Data is read from the per-country slices in data/cache/<ISO>.csv
(regenerate them with: python data/make_slices.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CACHE = "data/cache"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ISO code -> display name. These slices were cached from the OWID megafile.
COUNTRIES = {
    "DEU": "Germany",
    "ITA": "Italy",
    "GBR": "United Kingdom",
    "USA": "United States",
    "KOR": "South Korea",
}
HERO_ISO = "DEU"  # headline country for the project (clean daily reporting + clear policy timing)


def load(iso):
    df = pd.read_csv(f"{CACHE}/{iso}.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def fig_overview(iso):
    """Three stacked panels sharing the x-axis, each signal on its own scale.

    This is the 'what the PINN must explain' picture:
      - top:    the case curve  -> the data the PINN is trained on
      - middle: reproduction_rate -> the independent target our recovered R0(t) is compared to
      - bottom: stringency_index  -> the real lockdown timing we will annotate
    Putting Rt on its own axis (instead of sharing the 0-100 stringency scale) is essential:
    Rt only ranges ~0.5-3, so it was invisible when crushed against the case/stringency scale.
    """
    df = load(iso)
    name = COUNTRIES[iso]
    # focus on the daily-reporting era where all three signals are dense
    df = df[(df.date >= "2020-02-15") & (df.date <= "2022-12-31")]

    fig, (axc, axr, axs) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.4], "hspace": 0.12})

    # --- panel 1: cases (the training data) ---
    axc.fill_between(df.date, df.new_cases_smoothed, color="tab:blue", alpha=0.20)
    axc.plot(df.date, df.new_cases_smoothed, color="tab:blue", lw=1.5)
    axc.set_ylabel("New cases / day\n(7-day smoothed)", color="tab:blue")
    axc.tick_params(axis="y", labelcolor="tab:blue")
    axc.set_title(f"{name}: the inverse problem the PINN solves\n"
                  "train on cases (top) -> recover R0(t) and compare to the independent Rt (middle); "
                  "lockdown timing (bottom) should explain the dips",
                  fontsize=12)
    axc.grid(True, alpha=0.2)

    # --- panel 2: Rt on its own scale (the recovery target) ---
    axr.plot(df.date, df.reproduction_rate, color="tab:red", lw=1.8,
             label="Reproduction rate Rt (OWID, independent estimate)")
    axr.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6, label="Rt = 1 (epidemic turning point)")
    axr.fill_between(df.date, 1.0, df.reproduction_rate,
                     where=df.reproduction_rate >= 1.0, color="tab:red", alpha=0.12)
    axr.fill_between(df.date, 1.0, df.reproduction_rate,
                     where=df.reproduction_rate < 1.0, color="tab:green", alpha=0.12)
    axr.set_ylabel("Rt")
    axr.legend(loc="upper right", fontsize=9, framealpha=0.9)
    axr.grid(True, alpha=0.2)

    # --- panel 3: stringency (the policy annotation source) ---
    axs.fill_between(df.date, df.stringency_index, color="gray", alpha=0.35)
    axs.plot(df.date, df.stringency_index, color="dimgray", lw=1.2)
    axs.set_ylabel("Stringency\n(0-100)")
    axs.set_ylim(0, 100)
    axs.set_xlabel("Date")
    axs.grid(True, alpha=0.2)

    axs.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.tight_layout()
    path = f"{OUT}/data_overview_{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_multicountry():
    """Smoothed new cases per million across countries (log scale) to compare wave timing."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for iso, name in COUNTRIES.items():
        df = load(iso)
        df = df[(df.date >= "2020-02-15") & (df.date <= "2023-06-30")]
        per_m = df.new_cases_smoothed / df.population * 1e6
        ax.plot(df.date, per_m, lw=1.5, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("New cases per million per day (7-day smoothed, log scale)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(fontsize=10, ncol=2, loc="lower center")
    ax.set_title("COVID-19 waves across countries (per-capita, log scale): different timing, same method applies",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_ylim(bottom=1)
    fig.tight_layout()
    path = f"{OUT}/data_multicountry_cases.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_rt_vs_stringency():
    """Rt vs stringency for the hero country: visually, more lockdown -> lower Rt (with lag)."""
    df = load(HERO_ISO)
    name = COUNTRIES[HERO_ISO]
    df = df[(df.date >= "2020-03-01") & (df.date <= "2022-12-31")].dropna(
        subset=["reproduction_rate", "stringency_index"])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(df.stringency_index, df.reproduction_rate,
                    c=mdates.date2num(df.date), cmap="viridis", s=10, alpha=0.7)
    ax.axhline(1.0, color="tab:red", ls="--", lw=1, alpha=0.7, label="Rt = 1 (epidemic turning point)")
    ax.set_xlabel("Oxford stringency index (0 = no measures, 100 = strictest lockdown)")
    ax.set_ylabel("Reproduction rate Rt (OWID estimate)")
    ax.set_title(f"{name}: lockdown intensity vs reproduction number\n"
                 "(the relationship the PINN-recovered R0(t) should reflect)")
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax)
    loc = mdates.AutoDateLocator()
    cbar.ax.yaxis.set_major_locator(loc)
    cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    cbar.set_label("Date")
    fig.tight_layout()
    path = f"{OUT}/data_rt_vs_stringency.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_quality():
    """Raw vs smoothed cases: motivates using new_cases_smoothed (weekend/weekly reporting spikes)."""
    df = load(HERO_ISO)
    name = COUNTRIES[HERO_ISO]
    # a window where weekly reporting artifacts are visible
    df = df[(df.date >= "2022-01-01") & (df.date <= "2022-06-30")]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(df.date, df.new_cases, color="gray", lw=0.9, alpha=0.7, label="Raw new_cases (spiky)")
    ax.plot(df.date, df.new_cases_smoothed, color="tab:blue", lw=2.2, label="new_cases_smoothed (7-day)")
    ax.set_ylabel("New cases per day")
    ax.set_xlabel("Date (2022 H1)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.set_title(f"{name}: why we train on the smoothed series\n"
                 "raw daily counts have weekend/weekly reporting artifacts and NaN'd negative corrections")
    fig.tight_layout()
    path = f"{OUT}/data_quality_raw_vs_smooth.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    paths = [
        fig_overview(HERO_ISO),
        fig_multicountry(),
        fig_rt_vs_stringency(),
        fig_quality(),
    ]
    print("Wrote figures:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
