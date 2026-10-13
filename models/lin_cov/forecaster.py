"""Ridge regression on the lgbm_cov inputs, per series: the linear twin of lgbm_cov."""
import sys
import time

import numpy as np
import pandas as pd
from fl_cov import calendar, full_features, future_index, load_covariates
from fl_io import read_request, write_output
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

req, series = read_request()
cov = load_covariates(req)
inp = pd.read_parquet(req["input"])
HOLDOUT = 7 * 96          # residuals of the last 7 days give the intervals


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))


results = {}
for key, y in series.items():
    t0 = time.time()
    zone, target = key.split("_")
    t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
    t_fut = future_index(t_hist[-1], req["horizon"][key])
    cols = full_features(cov.columns, zone, target)
    X_all = cov.reindex(t_hist.append(t_fut))[cols].join(calendar(t_hist.append(t_fut)))
    X_all = X_all.join(pd.get_dummies(X_all.pop("cal_dow"), prefix="dow", dtype=float))
    if target == "price":   # D-1 prices are known before the gate: same quarter-hour one day earlier
        ys = pd.Series(y, index=t_hist)
        X_all["price_lag_1d"] = ys.reindex(X_all.index - pd.Timedelta("1D")).values
    X_all = X_all.loc[:, X_all.loc[t_hist].notna().mean() > 0.5]           # drop covariates mostly missing
    X_all = X_all.interpolate(limit_direction="both").fillna(X_all.loc[t_hist].median())
    X_hist, X_fut = X_all.loc[t_hist], X_all.loc[t_fut]
    ok = ~np.isnan(y)
    Xh, yh = X_hist[ok], y[ok]
    m = ridge().fit(Xh.iloc[:-HOLDOUT], yh[:-HOLDOUT])
    res = yh[-HOLDOUT:] - m.predict(Xh.iloc[-HOLDOUT:])
    hour_h = Xh.index[-HOLDOUT:].tz_localize("UTC").tz_convert("Europe/Brussels").hour
    qtab = np.array([np.quantile(res[hour_h == h], req["quantiles"]) if (hour_h == h).sum() >= 7 else np.quantile(res, req["quantiles"])
                     for h in range(24)])
    m = ridge().fit(Xh, yh)
    point = m.predict(X_fut)
    q = point[:, None] + qtab[X_fut.index.tz_localize("UTC").tz_convert("Europe/Brussels").hour]
    if target != "price":   # generation and load cannot be negative
        point, q = np.clip(point, 0, None), np.clip(q, 0, None)
    results[key] = (point, np.sort(q, axis=1))
    print(f"{key}: {X_hist.shape[1]} features, {int(ok.sum())} rows, {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
write_output(req, results)
