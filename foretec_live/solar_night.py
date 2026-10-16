"""Physical night clamp for solar forecasts.

Models with no physical prior (linear baselines, most foundation models)
happily forecast solar generation in the middle of the night: measured over
issues 2026-10-11..15, `lin_cov` averaged 8.5-21% of daily peak while the sun
was below the horizon, and 22 of 35 BE models exceeded 1%. On a chart that
reads as sunrise happening hours too early.

Generation before sunrise is not a modelling opinion, it is impossible, so we
zero it for every model alike — point and quantiles. Verified lossless where
the actuals are trustworthy: across 19-31 days of metered data, BE, NL and DE
record **exactly 0.00%** of peak whenever the sun is below -3 deg everywhere
in the zone. (FR, DK1 and DK2 actuals do carry a night baseline of 1-3% of
peak - that is a defect in those feeds, tracked separately; the clamp is
still applied there so every model is treated identically.)

Threshold: the highest solar elevation at any of the zone's capacity-weighted
solar centroids. -3 deg (not 0) keeps a margin for the centroid approximation
and for the sites that lie east of the easternmost centroid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ELEV_FLOOR_DEG = -3.0


def max_solar_elevation(points: list[tuple[float, float]], index: pd.DatetimeIndex) -> np.ndarray:
    """Highest solar elevation (deg) over `points` at each UTC timestamp."""
    idx = pd.DatetimeIndex(index)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    best = np.full(len(idx), -90.0)
    for lat, lon in points:
        ha = np.radians(15.0 * (hour + lon / 15.0 - 12.0))
        el = np.degrees(np.arcsin(np.sin(np.radians(lat)) * np.sin(decl)
                                  + np.cos(np.radians(lat)) * np.cos(decl) * np.cos(ha)))
        best = np.maximum(best, el)
    return best


def solar_points(centroids: pd.DataFrame, zone: str) -> list[tuple[float, float]]:
    """Capacity-weighted solar centroids of a zone; falls back to all of its points."""
    z = centroids[centroids["zone"] == zone]
    s = z[z["bucket"].astype(str).str.contains("solar", case=False, na=False)]
    use = s if not s.empty else z
    return list(zip(use["latitude"].astype(float), use["longitude"].astype(float)))


def clamp_night(df: pd.DataFrame, points: list[tuple[float, float]],
                elev_floor_deg: float = ELEV_FLOOR_DEG) -> tuple[pd.DataFrame, int]:
    """Zero point and quantile columns where the sun is below the horizon everywhere.

    Returns the frame and the number of quarter-hours actually changed.
    """
    if not points or "delivery_utc" not in df.columns or df.empty:
        return df, 0
    night = max_solar_elevation(points, pd.DatetimeIndex(df["delivery_utc"])) < elev_floor_deg
    if not night.any():
        return df, 0
    cols = [c for c in (["point"] + [c for c in df.columns if c.startswith("q")]) if c in df.columns]
    if not cols:
        return df, 0
    block = df.loc[night, cols]
    changed = int((block.fillna(0.0).to_numpy() != 0.0).any(axis=1).sum())
    df.loc[night, cols] = 0.0
    return df, changed
