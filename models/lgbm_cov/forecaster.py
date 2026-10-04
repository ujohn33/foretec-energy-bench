"""LightGBM on covariates, per series. Wind and solar: weather -> output. Price: + yesterday's price at the same quarter-hour."""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from fl_cov import calendar, full_features, future_index, load_covariates
from fl_io import read_request, write_output

req, series = read_request()
cov = load_covariates(req)
inp = pd.read_parquet(req["input"])
PARAMS = dict(n_estimators=200, learning_rate=0.06, num_leaves=15, min_child_samples=20, max_bin=63,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=2, verbose=-1)

results = {}
for key, y in series.items():
    t0 = time.time()
    zone, target = key.split("_")
    t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
    t_fut = future_index(t_hist[-1], req["horizon"][key])
    cols = full_features(cov.columns, zone, target)
    X_all = cov.reindex(t_hist.append(t_fut))[cols].join(calendar(t_hist.append(t_fut)))
    if target == "price":   # D-1 prices are known before the gate: same quarter-hour one day earlier
        ys = pd.Series(y, index=t_hist)
        X_all["price_lag_1d"] = ys.reindex(X_all.index - pd.Timedelta("1D")).values
    X_hist, X_fut = X_all.loc[t_hist], X_all.loc[t_fut]
    ok = X_hist.notna().mean(axis=1) > 0.5
    point = lgb.LGBMRegressor(**PARAMS).fit(X_hist[ok], y[ok.values]).predict(X_fut)
    q = np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=a, **PARAMS).fit(X_hist[ok], y[ok.values]).predict(X_fut)
                         for a in req["quantiles"]])
    if target != "price":   # generation cannot be negative
        point, q = np.clip(point, 0, None), np.clip(q, 0, None)
    results[key] = (point, np.sort(q, axis=1))
    print(f"{key}: {X_hist.shape[1]} features, {int(ok.sum())} rows, {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
write_output(req, results)
