"""Load and prepare an OWID country slice for the PINN.

The OWID COVID dataset is frozen (last updated 2024-08-19), so a cached
per-country CSV in ``data/cache/<ISO>.csv`` is a fixed, reproducible snapshot.
The slices are produced by ``data/make_slices.py``; this module turns one slice
into clean, scaled training arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CACHE = "data/cache"


@dataclass
class EpidemicData:
    """A clean, model-ready window of one country's epidemic.

    Attributes:
        date:     (n,) datetime64 -- calendar dates (for plotting/annotation).
        t:        (n,) days since window start, 0..T.
        tau:      (n,) scaled time in [0, 1].
        cases:    (n,) reported new cases per day (7-day smoothed, raw counts).
        cases_frac:(n,) cases divided by population N (the scale the PINN fits).
        rt:       (n,) OWID reproduction_rate (validation target; may have NaNs).
        stringency:(n,) Oxford stringency index 0-100 (lockdown annotation).
        N:        population.
        T:        window length in days.
        name:     country display name.
    """

    date: np.ndarray
    t: np.ndarray
    tau: np.ndarray
    cases: np.ndarray
    cases_frac: np.ndarray
    rt: np.ndarray
    stringency: np.ndarray
    N: float
    T: float
    name: str


def load_country(iso, start, end, cache=CACHE):
    """Load one country slice, clean it, clip to [start, end], and scale.

    Steps mirror the data-quality notes in the plan:
      * use ``new_cases_smoothed`` (raw daily counts are spiky);
      * drop leading/trailing rows where smoothed cases are NaN (OWID NaN's
        negative corrections and the pre-/post-epidemic tails);
      * interpolate any interior NaN gaps in the case series;
      * build days-since-start, scaled time tau in [0,1], and the case fraction.
    """
    df = pd.read_csv(f"{cache}/{iso}.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df.date >= pd.Timestamp(start)) & (df.date <= pd.Timestamp(end))]

    # Trim to the span where the training signal (smoothed cases) actually exists,
    # then fill any interior reporting gaps by time interpolation.
    valid = df.new_cases_smoothed.notna()
    df = df.loc[valid.idxmax(): valid[::-1].idxmax()].copy()
    df["new_cases_smoothed"] = df.new_cases_smoothed.interpolate().clip(lower=0.0)

    name = df.location.iloc[0]
    N = float(df.population.iloc[0])
    days = (df.date - df.date.iloc[0]).dt.days.to_numpy().astype(float)
    T = float(days[-1])

    cases = df.new_cases_smoothed.to_numpy()
    return EpidemicData(
        date=df.date.to_numpy(),
        t=days,
        tau=days / T,
        cases=cases,
        cases_frac=cases / N,
        rt=df.reproduction_rate.to_numpy(),
        stringency=df.stringency_index.to_numpy(),
        N=N,
        T=T,
        name=name,
    )
