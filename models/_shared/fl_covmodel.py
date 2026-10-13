"""Covariate models trained on two years and retrained monthly: lgbm_cov (LightGBM) and lin_cov (ridge).

Same inputs as before (fl_cov.full_features: the NWP ensemble at the zone's centroids, ensemble statistics,
the ENTSO-E load forecast, fuel and calendar; for price also yesterday's price), now trained on the
`train_days` delivery days up to the end of the previous month instead of the 28-day window.

  training  data/static/cov_history (foretec_live/covariates_history.py): per delivery day the covariates as
            the live builder lays them out; IFS from its true pre-cut-off run, ICON-EU and GFS from the
            previous-runs archive at the matching lead time. Built on first use if a month is missing.
  forecast  the live covariates of the request, i.e. only runs published before the cut-off, as always.
  intervals lgbm: one LightGBM per quantile. ridge: residuals of the last `holdout_days` per local hour,
            as in lin_lags.

Needs the main environment (env: _self).
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fl_cov import calendar, full_features, future_index, load_covariates
from fl_lagmodel import history, train_end_for

LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, max_bin=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=4, verbose=-1)


def design(cov: pd.DataFrame, index: pd.DatetimeIndex, zone: str, target: str, y: pd.Series, cols=None) -> pd.DataFrame:
    """Feature matrix for `index`: the series' covariates, calendar, and for price yesterday's price."""
    cols = cols if cols is not None else full_features(cov.columns, zone, target)
    X = cov.reindex(index).reindex(columns=cols).join(calendar(index))
    if target == "price":   # D-1 prices are known before the gate: same quarter-hour one day earlier
        X["price_lag_1d"] = y.reindex(index - pd.Timedelta("1D")).to_numpy()
    return X


def _ridge():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))


def _dummies(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    dow = X.pop("cal_dow")
    for d in range(7):
        X[f"dow{d}"] = (dow == d).astype(float)
    return X


def fit_series(kind: str, X: pd.DataFrame, y: np.ndarray, quantiles, holdout_days: int, tz: str) -> dict:
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    if kind == "lgbm":
        import lightgbm as lgb
        point = lgb.LGBMRegressor(**LGBM).fit(X, y)
        qs = [lgb.LGBMRegressor(objective="quantile", alpha=a, **LGBM).fit(X, y) for a in quantiles]
        return {"kind": kind, "cols": list(X.columns), "point": point, "quantiles": qs, "rows": int(len(y))}
    X = _dummies(X)
    X = X.loc[:, X.isna().mean() <= 0.2]                    # drop covariates missing on more than a fifth of the days
    keep = X.notna().all(axis=1).to_numpy()
    X, y = X[keep], y[keep]
    fill = X.median()
    day = X.index.tz_localize("UTC").tz_convert(tz).normalize()
    hold = (day > day.max() - pd.Timedelta(days=holdout_days)).to_numpy()
    m = _ridge().fit(X[~hold], y[~hold])
    res = y[hold] - m.predict(X[hold])
    hour = X.index[hold].tz_localize("UTC").tz_convert(tz).hour
    qtab = np.array([np.quantile(res[hour == h], quantiles) if (hour == h).sum() > 20 else np.quantile(res, quantiles)
                     for h in range(24)])
    return {"kind": kind, "cols": list(X.columns), "point": _ridge().fit(X, y), "qtab": qtab, "fill": fill,
            "rows": int(len(y)), "holdout_mae": float(np.mean(np.abs(res)))}


def predict(art: dict, X: pd.DataFrame, tz: str):
    if art["kind"] == "lgbm":
        X = X.reindex(columns=art["cols"])
        return art["point"].predict(X), np.column_stack([m.predict(X) for m in art["quantiles"]])
    X = _dummies(X).reindex(columns=art["cols"]).fillna(art["fill"])
    point = art["point"].predict(X)
    hour = X.index.tz_localize("UTC").tz_convert(tz).hour.to_numpy()
    return point, point[:, None] + art["qtab"][hour]


def artifact(cfg, spec: dict, issue_date: dt.date, keys: list[str], quantiles) -> dict:
    from foretec_live.covariates_history import build_history, load_history

    name, kind = spec["name"], spec.get("regressor", "lgbm")
    days, holdout = int(spec.get("train_days", 730)), int(spec.get("holdout_days", 120))
    end_day = train_end_for(issue_date)
    assert end_day < issue_date
    d = Path(cfg["_root"]) / "data" / "models" / name
    f = d / f"{end_day}.pkl"
    art = pickle.loads(f.read_bytes()) if f.exists() else {"train_end": str(end_day), "series": {}}
    todo = [k for k in keys if k not in art["series"]]
    if todo:
        start = end_day - dt.timedelta(days=days - 1)
        build_history(cfg, start, end_day)            # no-op for months already built
        cov = load_history(cfg, start, end_day)
    for key in todo:
        t = time.time()
        zone, target = key.split("_")
        y = history(cfg, zone, target, end_day, days)
        X = design(cov, cov.index, zone, target, y)
        art["series"][key] = fit_series(kind, X, y.reindex(cov.index).to_numpy(dtype=float), quantiles, holdout, cfg["timezone"])
        art["series"][key]["seconds"] = round(time.time() - t, 1)
        print(f"{name} {key}: trained through {end_day} on {art['series'][key]['rows']} rows, "
              f"{len(art['series'][key]['cols'])} features ({time.time() - t:.0f}s)", file=sys.stderr, flush=True)
    if todo:
        d.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(art))
        tmp.replace(f)
        meta = {k: {x: v.get(x) for x in ("rows", "seconds", "holdout_mae")} | {"features": len(v["cols"])} for k, v in art["series"].items()}
        (d / f"{end_day}.json").write_text(json.dumps({"train_end": str(end_day), "train_days": days, "series": meta}, indent=1))
    return art


def main():
    import yaml
    from fl_io import read_request, write_output
    from foretec_live.config import load_config

    req, series = read_request()
    cfg = load_config()
    spec = yaml.safe_load((Path.cwd() / "model.yaml").read_text())
    art = artifact(cfg, spec, dt.date.fromisoformat(req["issue_date"]), list(series), req["quantiles"])
    cov = load_covariates(req)                     # the live covariates: only runs published before the cut-off
    inp = pd.read_parquet(req["input"])
    results, errors = {}, {}
    for key, y in series.items():
        zone, target = key.split("_")
        try:
            t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
            t_fut = future_index(t_hist[-1], req["horizon"][key])
            X = design(cov, t_fut, zone, target, pd.Series(y, index=t_hist), cols=None)
            point, q = predict(art["series"][key], X, cfg["timezone"])
            if target != "price":   # generation and load cannot be negative
                point, q = np.clip(point, 0, None), np.clip(q, 0, None)
            results[key] = (point, np.sort(q, axis=1))
        except Exception as e:  # noqa: BLE001 - one series failing must not sink the others
            errors[key] = f"{type(e).__name__}: {e}"[:200]
    write_output(req, results, errors)


def warm(model_dir: str, issue_date: str):
    import yaml
    from foretec_live.config import load_config
    from foretec_live.registry import ModelSpec

    cfg = load_config()
    raw = yaml.safe_load((Path(model_dir) / "model.yaml").read_text())
    spec = ModelSpec(name=raw["name"], folder=Path(model_dir), raw=raw, targets=raw.get("targets"), zones=raw.get("zones"))
    keys = [f"{z}_{t}" for z in cfg["zones"] for t in cfg["targets"] if spec.applies(z, t)]
    artifact(cfg, raw, dt.date.fromisoformat(issue_date), keys, cfg["quantiles"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "warm":
        warm(sys.argv[2], sys.argv[3])
    else:
        main()
