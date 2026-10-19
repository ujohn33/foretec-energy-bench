"""Covariate selection shared by covariate models. Needs only numpy and pandas.

Column names are '<zone>.<bucket>.<what>.<variable>', e.g. 'BE.wind_offshore.ens_mean.wind_speed_100m',
'NL.solar.icon_eu.global_tilted_irradiance', 'FR.load_fc.entsoe.load_mw', 'fuel.ccgt.srmc.eur_mwh'.

The ENTSO-E day-ahead load forecast ('<zone>.load_fc.entsoe.load_mw') is a price input only, never a load input (since
issue 16 Oct 2026): a load model would be learning the TSO's forecast, the benchmark it is meant to beat, and for NL it
is a different quantity from the scored load (TenneT's is net of embedded solar, NED's load is gross; their correlation
was -0.45 in Jul-Sep 2026). A live run of an earlier issue day (a re-run) keeps the rule of its day and gives load
models the load forecast; backtests always follow the current rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ("BE", "NL", "FR")   # fallback; the zones are read from the covariate columns
LOAD_FC_FOR_LOAD_THROUGH = "2026-10-15"   # last live issue day whose load models got the ENTSO-E load forecast
_REQ: dict = {}                            # issue day and live flag of the request, set by load_covariates


def _load_fc_for_load() -> bool:
    return bool(_REQ.get("live")) and "" < _REQ.get("issue_date", "") <= LOAD_FC_FOR_LOAD_THROUGH


def zones_of(cols) -> list[str]:
    """Zones present in the covariates: those with weather buckets."""
    return sorted({c.split(".")[0] for c in cols if c.count(".") == 3 and c.split(".")[1] in ("wind_onshore", "solar")})


def load_covariates(req: dict) -> pd.DataFrame:
    _REQ.update(issue_date=str(req.get("issue_date") or ""), live=bool(req.get("live")))
    cov = pd.read_parquet(req["covariates"])
    cov.index = pd.to_datetime(cov.index)
    return cov


def full_features(cols, zone: str, target: str) -> list[str]:
    """Everything relevant to one series: every NWP model, the ensemble statistics and disagreement."""
    if target == "wind":
        pre = [f"{zone}.wind_onshore.", f"{zone}.wind_offshore."]
    elif target == "solar":
        pre = [f"{zone}.solar."]
    elif target == "load":   # weather at the load centres (and the TSO's load forecast in re-runs of live days to 15 Oct)
        pre = [f"{zone}.load."] + ([f"{zone}.load_fc."] if _load_fc_for_load() else [])
    else:  # price: own load, residual-load drivers in all coupled zones, fuel
        pre = [f"{zone}.load.", f"{zone}.load_fc.", "fuel."]
        zs = zones_of(cols) or list(ZONES)
        pre += [f"{z}.{b}.ens_" for z in zs for b in ("wind_onshore", "wind_offshore", "solar")]
        pre += [f"{z}.load_fc." for z in zs if z != zone]
    return [c for c in cols if any(c.startswith(p) for p in pre)]


def compact_features(cols, zone: str, target: str) -> list[str]:
    """A handful of ensemble means and spreads, for models that take few covariates."""
    if target == "wind":
        want = [f"{zone}.wind_onshore.ens_mean.wind_speed_100m", f"{zone}.wind_offshore.ens_mean.wind_speed_100m",
                f"{zone}.wind_onshore.ens_std.wind_speed_100m", f"{zone}.wind_offshore.ens_std.wind_speed_100m"]
    elif target == "solar":
        want = [f"{zone}.solar.ens_mean.global_tilted_irradiance", f"{zone}.solar.ens_std.global_tilted_irradiance",
                f"{zone}.solar.ens_mean.cloud_cover", f"{zone}.solar.ens_mean.temperature_2m"]
    elif target == "load":
        want = ([f"{zone}.load_fc.entsoe.load_mw"] if _load_fc_for_load() else []) + [
            f"{zone}.load.ens_mean.temperature_2m", f"{zone}.load.ens_mean.shortwave_radiation",
            f"{zone}.load.ens_mean.cloud_cover"]
    else:
        want = [f"{zone}.load_fc.entsoe.load_mw", f"{zone}.load.ens_mean.temperature_2m", "fuel.ccgt.srmc.eur_mwh"]
        zs = zones_of(cols) or list(ZONES)
        want += [f"{z}.wind_onshore.ens_mean.wind_speed_100m" for z in zs]
        want += [f"{z}.wind_offshore.ens_mean.wind_speed_100m" for z in zs]
        want += [f"{z}.solar.ens_mean.global_tilted_irradiance" for z in zs]
    return [c for c in want if c in cols]


def calendar(index: pd.DatetimeIndex, tz: str = "Europe/Brussels") -> pd.DataFrame:
    loc = index.tz_localize("UTC").tz_convert(tz)
    qh = loc.hour * 4 + loc.minute // 15
    return pd.DataFrame({"cal_qh_sin": np.sin(2 * np.pi * qh / 96), "cal_qh_cos": np.cos(2 * np.pi * qh / 96),
                         "cal_dow": loc.dayofweek, "cal_weekend": (loc.dayofweek >= 5).astype(int)}, index=index)


def future_index(last: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(last + pd.Timedelta("15min") * np.arange(1, steps + 1))


def series_frames(req: dict, cov: pd.DataFrame, inp: pd.DataFrame, key: str):
    """(t_hist, t_fut, past covariates [T x k], future covariates [H x k], names) for one series, gap-filled."""
    zone, target = key.split("_")
    t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
    t_fut = future_index(t_hist[-1], req["horizon"][key])
    cols = compact_features(cov.columns, zone, target)
    c = cov.reindex(t_hist.append(t_fut))[cols].interpolate(limit_direction="both").fillna(0.0)
    return t_hist, t_fut, c.loc[t_hist].to_numpy(np.float32), c.loc[t_fut].to_numpy(np.float32), cols


def nonneg(key: str, point, q):
    """Generation cannot be negative; prices can."""
    if key.endswith("price"):
        return point, q
    return np.clip(point, 0, None), (None if q is None else np.clip(q, 0, None))
