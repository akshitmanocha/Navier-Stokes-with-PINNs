"""Regenerate the cached per-country OWID slices in data/cache/<ISO>.csv.

The committed slices are small (~100 KB each) and frozen, so the project runs
offline and identically for any reviewer. This script documents exactly how they
were produced from the full Our World in Data COVID-19 megafile and lets anyone
rebuild or extend them. The OWID dataset stopped updating 2024-08-19, so the
result is a fixed, citable snapshot.

The 94 MB megafile is intentionally NOT committed. This script downloads it to a
temp location, slices out the columns and countries we use, and writes the slices.

Usage:
    python data/make_slices.py
"""
import os

import pandas as pd

MEGAFILE_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
CACHE = os.path.dirname(os.path.abspath(__file__)) + "/cache"

# ISO codes to cache (clean daily reporting + documented lockdown timing).
COUNTRIES = ["DEU", "ITA", "GBR", "USA", "KOR"]

# Columns the project uses (see src/data.py and experiments/explore_data.py).
COLUMNS = ["iso_code", "location", "date", "total_cases", "new_cases",
           "new_cases_smoothed", "new_deaths_smoothed", "reproduction_rate",
           "stringency_index", "population"]


def main(source=MEGAFILE_URL):
    os.makedirs(CACHE, exist_ok=True)
    print(f"Reading megafile from {source} (this is the large download) ...")
    df = pd.read_csv(source, usecols=COLUMNS)

    for iso in COUNTRIES:
        sub = df[df.iso_code == iso].sort_values("date")
        out = f"{CACHE}/{iso}.csv"
        sub.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(sub)} rows, {sub.date.min()} -> {sub.date.max()})")
    print("Done.")


if __name__ == "__main__":
    main()
