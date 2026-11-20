"""Simons (Abel AI Lab, November 2026), zero-shot. Env: simons.

Simons-297M, the largest of the four released sizes, through Abel's package (pinned in scripts/install_model_envs.sh;
code Apache-2.0, weights OpenMDW-1.1) and its forecast_series call. It returns nine quantiles, 0.1 ... 0.9, the
benchmark's levels; the point forecast is the median. Without covariates, series with the same horizon are forecast in
one batch.

With covariates=True each series gets the compact covariate set (fl_cov.compact_features: NWP ensemble means/spreads,
ENTSO-E load forecast where the target gets it, fuel) and the four calendar columns, as known-future covariates over
the history and the delivery day. Simons reads at most 16 covariates, so the compact set is cut to its first 12, in
fl_cov's order (own zone first; only price has more). The package fits an in-context ridge regression on them (kept if
a backtest on the history favours it) and reads them as covariate rows, one series per pass (a third of the memory of
one batch).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fl_cov import calendar, compact_features, future_index, load_covariates, nonneg
from fl_io import read_request, write_output

REPO, REVISION = "abel-lab/Simons-297M", "335dc22e4ce26ca0a8bc567de9e93dbb1bd2ce3a"


def covariate_frame(req: dict, cov: pd.DataFrame, inp: pd.DataFrame, key: str) -> pd.DataFrame:
    """[history + horizon, k] known-future covariates of one series."""
    zone, target = key.split("_")
    t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
    index = t_hist.append(future_index(t_hist[-1], req["horizon"][key]))
    cols = compact_features(cov.columns, zone, target)[:12]
    values = cov.reindex(index)[cols].interpolate(limit_direction="both")
    return pd.concat([values, calendar(index)], axis=1)


def run(covariates: bool = False) -> None:
    from simons import QUANTILES, Simons, forecast_series
    req, series = read_request()
    if tuple(req["quantiles"]) != tuple(QUANTILES):
        raise SystemExit(f"Simons forecasts the quantiles {QUANTILES}, the request asks for {req['quantiles']}")
    model = Simons.from_pretrained(REPO, revision=REVISION)
    cov = load_covariates(req) if covariates else None
    inp = pd.read_parquet(req["input"]) if covariates else None
    by_horizon = {}
    for key in series:
        by_horizon.setdefault(req["horizon"][key], []).append(key)
    results = {}
    for h, keys in by_horizon.items():
        extra = {"future_covariates": [covariate_frame(req, cov, inp, k) for k in keys], "batch_size": 1} if covariates else {}
        for key, q in zip(keys, forecast_series(model, [series[k] for k in keys], h, **extra)):
            results[key] = nonneg(key, q[:, 4], np.sort(q, axis=1))   # (h, 9): 0.1 ... 0.9; sorted in case a release stops ordering them
    write_output(req, results)
