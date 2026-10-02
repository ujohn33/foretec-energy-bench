"""Covariate selection shared by covariate models. Needs only numpy and pandas.

Column names are '<zone>.<bucket>.<what>.<variable>', e.g. 'BE.wind_offshore.ens_mean.wind_speed_100m',
'NL.solar.icon_eu.global_tilted_irradiance', 'FR.load_fc.entsoe.load_mw', 'fuel.ccgt.srmc.eur_mwh'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ("BE", "NL", "FR")


def load_covariates(req: dict) -> pd.DataFrame:
    cov = pd.read_parquet(req["covariates"])
    cov.index = pd.to_datetime(cov.index)
    return cov


def full_features(cols, zone: str, target: str) -> list[str]:
    """Everything relevant to one series: every NWP model, the ensemble statistics and disagreement."""
    if target == "wind":
        pre = [f"{zone}.wind_onshore.", f"{zone}.wind_offshore."]
    elif target == "solar":
        pre = [f"{zone}.solar."]
    else:  # price: own load, residual-load drivers in all coupled zones, fuel
        pre = [f"{zone}.load.", f"{zone}.load_fc.", "fuel."]
        pre += [f"{z}.{b}.ens_" for z in ZONES for b in ("wind_onshore", "wind_offshore", "solar")]
        pre += [f"{z}.load_fc." for z in ZONES if z != zone]
    return [c for c in cols if any(c.startswith(p) for p in pre)]


def compact_features(cols, zone: str, target: str) -> list[str]:
    """A handful of ensemble means and spreads, for models that take few covariates."""
    if target == "wind":
        want = [f"{zone}.wind_onshore.ens_mean.wind_speed_100m", f"{zone}.wind_offshore.ens_mean.wind_speed_100m",
                f"{zone}.wind_onshore.ens_std.wind_speed_100m", f"{zone}.wind_offshore.ens_std.wind_speed_100m"]
    elif target == "solar":
        want = [f"{zone}.solar.ens_mean.global_tilted_irradiance", f"{zone}.solar.ens_std.global_tilted_irradiance",
                f"{zone}.solar.ens_mean.cloud_cover", f"{zone}.solar.ens_mean.temperature_2m"]
    else:
        want = [f"{zone}.load_fc.entsoe.load_mw", f"{zone}.load.ens_mean.temperature_2m", "fuel.ccgt.srmc.eur_mwh"]
        want += [f"{z}.wind_onshore.ens_mean.wind_speed_100m" for z in ZONES]
        want += [f"{z}.wind_offshore.ens_mean.wind_speed_100m" for z in ZONES]
        want += [f"{z}.solar.ens_mean.global_tilted_irradiance" for z in ZONES]
    return [c for c in want if c in cols]


def calendar(index: pd.DatetimeIndex, tz: str = "Europe/Brussels") -> pd.DataFrame:
    loc = index.tz_localize("UTC").tz_convert(tz)
    qh = loc.hour * 4 + loc.minute // 15
    return pd.DataFrame({"cal_qh_sin": np.sin(2 * np.pi * qh / 96), "cal_qh_cos": np.cos(2 * np.pi * qh / 96),
                         "cal_dow": loc.dayofweek, "cal_weekend": (loc.dayofweek >= 5).astype(int)}, index=index)


def future_index(last: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(last + pd.Timedelta("15min") * np.arange(1, steps + 1))
