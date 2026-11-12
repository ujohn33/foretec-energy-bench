"""Simons (Abel AI Lab; code Apache-2.0, weights OpenMDW-1.1), zero-shot, from the target's own history or with
known-future covariates. Env: simons.

Four sizes (25M, 60M, 178M, 297M) through the simons package (pinned in scripts/install_model_envs.sh). The quantile
head returns exactly the benchmark's nine levels (0.1 ... 0.9); the point forecast is the median. Without covariates,
series with the same horizon are forecast in one batch. The package keeps forecasts of a never-negative history at or
above zero.

With covariates=True each series gets the compact covariate set (fl_cov.compact_features: NWP ensemble means and
spreads; for prices the zone's ENTSO-E load forecast, temperature, fuel cost and the zones' wind) and
the four calendar columns, over the history and the delivery day. Simons reads at most 16 covariates, so the compact
set is cut to its first 12 (only price has more). It fits an in-context ridge regression on them (kept if a backtest
on the history favours it) and reads them as extra rows, one series per pass (a third of the memory of one batch).
Weights pinned per size below.
"""
import pandas as pd

from fl_cov import calendar, compact_features, future_index, load_covariates
from fl_io import read_request, write_output

REVISIONS = {"abel-lab/Simons-25M": "ea871af33d47bae63a695b2def374ed7a8839953",
             "abel-lab/Simons-60M": "7e637d0bd8e51fb4b7b8ef17f4b636d020a92cc7",
             "abel-lab/Simons-178M": "b36d6364647d75a89a17703d9bfb2109b13c2427",
             "abel-lab/Simons-297M": "4a5ab590fdcfbbcc630737188cdec1ba9e069a5a"}


def covariate_frame(req: dict, cov: pd.DataFrame, inp: pd.DataFrame, key: str) -> pd.DataFrame:
    """[history + horizon, k] known-future covariates of one series."""
    zone, target = key.split("_")
    t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
    index = t_hist.append(future_index(t_hist[-1], req["horizon"][key]))
    cols = compact_features(cov.columns, zone, target)[:12]
    values = cov.reindex(index)[cols].interpolate(limit_direction="both")
    return pd.concat([values, calendar(index)], axis=1)


def run(repo: str, covariates: bool) -> None:
    from simons import QUANTILES, Simons, forecast_series
    req, series = read_request()
    if tuple(req["quantiles"]) != tuple(QUANTILES):
        raise SystemExit(f"Simons forecasts the quantiles {QUANTILES}, the request asks for {req['quantiles']}")
    model = Simons.from_pretrained(repo, revision=REVISIONS[repo])
    cov = load_covariates(req) if covariates else None
    inp = pd.read_parquet(req["input"]) if covariates else None
    by_horizon = {}
    for key in series:
        by_horizon.setdefault(req["horizon"][key], []).append(key)
    results = {}
    for h, keys in by_horizon.items():
        extra = {"future_covariates": [covariate_frame(req, cov, inp, k) for k in keys], "batch_size": 1} if covariates else {}
        for key, q in zip(keys, forecast_series(model, [series[k] for k in keys], h, **extra)):
            results[key] = (q[:, 4], q)                                  # (h, 9): 0.1 ... 0.9
    write_output(req, results)
