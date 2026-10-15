"""Tabular deep learning on the lgbm_cov inputs: RealMLP and TabM (pytabkit), trained on the 28-day window. Env: tabnn.

Same features as lgbm_cov (fl_cov.full_features: the NWP ensemble at the zone's centroids, ensemble statistics, ENTSO-E
load forecast, fuel; calendar; for price yesterday's price), so these differ from lgbm_cov only in the regressor.

They are too slow to train inside the live run (TabM takes minutes per series on the server's CPUs), so live they are
trained in the morning, outside the gate window: `foretec-live pretrain` (07:20 CET) trains on the window of
delivery days up to yesterday, from yesterday's frozen covariates, and saves the model; the 11:30 run only predicts,
with today's covariates. If the morning model is missing, the series is reported as not forecast instead of training
at the gate. Backtests train inside the run (no gate there).

Intervals: the model is first fitted without the last 7 days, whose residuals per local hour give the quantiles, then
refitted on the whole window.
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fl_cov import calendar, full_features, future_index, load_covariates, nonneg
from fl_io import read_request, write_output

warnings.filterwarnings("ignore")
HOLDOUT = 7 * 96
TZ = "Europe/Brussels"


def make(kind: str):
    if kind == "realmlp":
        from pytabkit import RealMLP_TD_Regressor
        return RealMLP_TD_Regressor(random_state=0, verbosity=0, n_threads=4, device="cpu", n_epochs=64)
    from pytabkit import TabM_D_Regressor
    return TabM_D_Regressor(random_state=0, verbosity=0, n_threads=4, device="cpu", tabm_k=8, patience=8)


def design(cov: pd.DataFrame, index: pd.DatetimeIndex, zone: str, target: str, y: pd.Series) -> pd.DataFrame:
    X = cov.reindex(index)[full_features(cov.columns, zone, target)].join(calendar(index))
    dow = X.pop("cal_dow")
    for d in range(7):
        X[f"dow{d}"] = (dow == d).astype(float)
    if target == "price":   # D-1 prices are known before the gate: the same quarter-hour one day earlier
        X["price_lag_1d"] = y.reindex(index - pd.Timedelta("1D")).to_numpy()
    return X


def train_series(kind: str, X: pd.DataFrame, y: np.ndarray, quantiles) -> dict:
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    X = X.loc[:, X.notna().mean() > 0.5]
    fill = X.median()
    Xa = X.fillna(fill).to_numpy(np.float32)
    m = make(kind).fit(Xa[:-HOLDOUT], y[:-HOLDOUT].astype(np.float32))
    res = y[-HOLDOUT:] - m.predict(Xa[-HOLDOUT:])
    hour = X.index[-HOLDOUT:].tz_localize("UTC").tz_convert(TZ).hour
    qtab = np.array([np.quantile(res[hour == h], quantiles) if (hour == h).sum() >= 7 else np.quantile(res, quantiles)
                     for h in range(24)])
    return {"model": make(kind).fit(Xa, y.astype(np.float32)), "cols": list(X.columns), "fill": fill, "qtab": qtab}


def predict(art: dict, X: pd.DataFrame):
    Xp = X.reindex(columns=art["cols"]).fillna(art["fill"]).to_numpy(np.float32)
    point = art["model"].predict(Xp)
    hour = X.index.tz_localize("UTC").tz_convert(TZ).hour.to_numpy()
    return point, point[:, None] + art["qtab"][hour]


def main():
    import yaml
    req, series = read_request()
    spec = yaml.safe_load((Path.cwd() / "model.yaml").read_text())
    kind = spec["regressor"]
    cov = load_covariates(req)
    inp = pd.read_parquet(req["input"])
    mode = req.get("mode", "forecast")
    art_dir = Path(req["artifact_dir"]) if req.get("artifact_dir") else None
    art_file = art_dir / f"{req['issue_date']}.pkl" if art_dir else None
    ys = {k: pd.Series(v, index=pd.DatetimeIndex(inp.loc[inp["series"] == k, "time_utc"].sort_values())) for k, v in series.items()}

    if mode == "train":   # morning: the window of delivery days up to yesterday (today's issue day not included)
        day0 = pd.Timestamp(req["issue_date"]).tz_localize(TZ).tz_convert("UTC").tz_localize(None)
        arts = {}
        for key, y in ys.items():
            zone, target = key.split("_")
            t = time.time()
            idx = cov.index[(cov.index < day0) & cov.index.isin(y.index)]
            X = design(cov, idx, zone, target, y)
            arts[key] = train_series(kind, X, y.reindex(idx).to_numpy(float), req["quantiles"])
            print(f"{spec['name']} {key}: trained on {len(idx)} rows ({time.time() - t:.0f}s)", file=sys.stderr, flush=True)
        art_dir.mkdir(parents=True, exist_ok=True)
        tmp = art_file.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(arts))
        tmp.replace(art_file)
        for old in sorted(art_dir.glob("*.pkl"))[:-10]:   # keep the last ten days
            old.unlink()
        return

    arts = pickle.loads(art_file.read_bytes()) if art_file is not None and art_file.exists() else {}
    results, errors = {}, {}
    for key, y in ys.items():
        zone, target = key.split("_")
        t_fut = future_index(y.index[-1], req["horizon"][key])
        try:
            if key in arts:
                art = arts[key]
            elif req.get("live"):   # never train at the gate
                errors[key] = "not trained this morning (foretec-live pretrain); not trained inside the live run"
                continue
            else:                   # backtest: train on the window up to the cut-off, as the morning job would
                X = design(cov, y.index, zone, target, y)
                art = train_series(kind, X, y.to_numpy(float), req["quantiles"])
            point, q = predict(art, design(cov, t_fut, zone, target, y))
            results[key] = nonneg(key, point, np.sort(q, axis=1))
        except Exception as e:  # noqa: BLE001 - one series failing must not sink the others
            errors[key] = f"{type(e).__name__}: {e}"[:200]
    write_output(req, results, errors)


if __name__ == "__main__":
    main()
