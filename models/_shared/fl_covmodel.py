"""Covariate models trained on two years and retrained monthly: lgbm_cov (LightGBM) and lin_cov (ridge).

Same inputs as before (fl_cov.full_features: the NWP ensemble at the zone's centroids, ensemble statistics,
the ENTSO-E load forecast, fuel and calendar; for price also yesterday's price), now trained on the
`train_days` delivery days up to the end of the previous month instead of the 28-day window, weighted towards
recent days (half-life `half_life_days`, 120) and with the series' recent level as features (levels()), so a
model trained across two years of capacity growth and price regimes still knows today's level.

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


def levels(y: pd.Series, index: pd.DatetimeIndex, tz: str = "Europe/Brussels") -> pd.DataFrame:
    """The series' recent level as known at the cut-off of each row's issue day: mean of the 7 complete days and
    maximum of the 28 complete days before it (days D-8..D-2 and D-29..D-2 for delivery day D). Two years of
    history span capacity growth (solar +20-30 % a year) and price regimes that trees cannot extrapolate; these
    tell the model where the level is now."""
    day = y.index.tz_localize("UTC").tz_convert(tz).normalize().tz_localize(None)
    dm, dx = y.groupby(day).mean(), y.groupby(day).max()
    full = pd.date_range(dm.index.min(), dm.index.max() + pd.Timedelta(days=3), freq="D")
    dm, dx = dm.reindex(full), dx.reindex(full)
    lvl = dm.rolling(7, min_periods=5).mean().shift(2)
    cap = dx.rolling(28, min_periods=20).max().shift(2)
    rows = index.tz_localize("UTC").tz_convert(tz).normalize().tz_localize(None)
    return pd.DataFrame({"lvl_7d": lvl.reindex(rows).to_numpy(), "max_28d": cap.reindex(rows).to_numpy()}, index=index)


def design(cov: pd.DataFrame, index: pd.DatetimeIndex, zone: str, target: str, y: pd.Series, cols=None) -> pd.DataFrame:
    """Feature matrix for `index`: the series' covariates, calendar, its recent level, and for price yesterday's price."""
    cols = cols if cols is not None else full_features(cov.columns, zone, target)
    X = cov.reindex(index).reindex(columns=cols).join(calendar(index)).join(levels(y, index))
    if target == "price":   # D-1 prices are known before the gate: same quarter-hour one day earlier
        X["price_lag_1d"] = y.reindex(index - pd.Timedelta("1D")).to_numpy()
    return X


def recency_weights(index: pd.DatetimeIndex, half_life_days: float) -> np.ndarray:
    """Training weights halving every half_life_days back from the last training day."""
    age = (index.max() - index) / pd.Timedelta(days=1)
    return np.power(0.5, np.asarray(age, dtype=float) / half_life_days)


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


def fit_series(kind: str, X: pd.DataFrame, y: np.ndarray, quantiles, holdout_days: int, tz: str, half_life: float = 120) -> dict:
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    if kind == "lgbm":
        import lightgbm as lgb
        w = recency_weights(X.index, half_life)
        point = lgb.LGBMRegressor(**LGBM).fit(X, y, sample_weight=w)
        qs = [lgb.LGBMRegressor(objective="quantile", alpha=a, **LGBM).fit(X, y, sample_weight=w) for a in quantiles]
        return {"kind": kind, "cols": list(X.columns), "point": point, "quantiles": qs, "rows": int(len(y))}
    X = _dummies(X)
    X = X.loc[:, X.isna().mean() <= 0.2]                    # drop covariates missing on more than a fifth of the days
    keep = X.notna().all(axis=1).to_numpy()
    X, y = X[keep], y[keep]
    fill = X.median()
    day = X.index.tz_localize("UTC").tz_convert(tz).normalize()
    hold = np.asarray(day > day.max() - pd.Timedelta(days=holdout_days))
    w = recency_weights(X.index, half_life)
    m = _ridge().fit(X[~hold], y[~hold], ridgecv__sample_weight=w[~hold])
    res = y[hold] - m.predict(X[hold])
    hour = X.index[hold].tz_localize("UTC").tz_convert(tz).hour
    qtab = np.array([np.quantile(res[hour == h], quantiles) if (hour == h).sum() > 20 else np.quantile(res, quantiles)
                     for h in range(24)])
    return {"kind": kind, "cols": list(X.columns), "point": _ridge().fit(X, y, ridgecv__sample_weight=w), "qtab": qtab, "fill": fill,
            "rows": int(len(y)), "holdout_mae": float(np.mean(np.abs(res)))}


def predict(art: dict, X: pd.DataFrame, tz: str):
    if art["kind"] == "lgbm":
        X = X.reindex(columns=art["cols"])
        return art["point"].predict(X), np.column_stack([m.predict(X) for m in art["quantiles"]])
    X = _dummies(X).reindex(columns=art["cols"]).fillna(art["fill"])
    point = art["point"].predict(X)
    hour = X.index.tz_localize("UTC").tz_convert(tz).hour.to_numpy()
    return point, point[:, None] + art["qtab"][hour]


def end_for(spec: dict, key: str, issue_date: dt.date) -> dt.date:
    """Last training day: the day before the issue day for targets retrained daily (retrain: {target: daily}),
    otherwise the end of the previous month."""
    target = key.split("_")[1]
    return issue_date - dt.timedelta(days=1) if (spec.get("retrain") or {}).get(target) == "daily" else train_end_for(issue_date)


def scale_kind(cfg, spec: dict, key: str) -> str | None:
    """scale: {target: capacity}: the target is divided by installed capacity (data/static/capacity/<key>.parquet,
    scripts/research/build_capacity.py), or, without a capacity series, by a proxy: the highest output in the 90 days
    up to two days before (known at the cut-off). Removes capacity growth the trees cannot extrapolate."""
    if (spec.get("scale") or {}).get(key.split("_")[1]) != "capacity":
        return None
    return "capacity" if (Path(cfg["_root"]) / "data" / "static" / "capacity" / f"{key}.parquet").exists() else "proxy"


def scale_rows(cfg, key: str, kind: str, y: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    tz = cfg["timezone"]
    if kind == "capacity":
        daily = pd.read_parquet(Path(cfg["_root"]) / "data" / "static" / "capacity" / f"{key}.parquet")["mw"].shift(1)
    else:
        day = y.index.tz_localize("UTC").tz_convert(tz).normalize().tz_localize(None)
        mx = y.groupby(day).max()
        mx = mx.reindex(pd.date_range(mx.index.min(), mx.index.max() + pd.Timedelta(days=3), freq="D"))
        daily = mx.rolling(90, min_periods=60).max().shift(2)
    rows = index.tz_localize("UTC").tz_convert(tz).normalize().tz_localize(None)
    daily = daily[~daily.index.duplicated(keep="last")]
    daily = daily.reindex(daily.index.union(rows.unique())).ffill()   # past the series' end: the last known capacity
    return daily.reindex(rows).to_numpy(dtype=float)


def latest_trained(cfg, spec: dict, key: str, end_day: dt.date):
    """The most recent model of `key` trained through end_day or earlier (for a live run that must not train)."""
    d = Path(cfg["_root"]) / "data" / "models" / spec["name"]
    for f in sorted(d.glob("*.pkl"), reverse=True) if d.exists() else []:
        if dt.date.fromisoformat(f.stem) <= end_day:
            art = pickle.loads(f.read_bytes())
            if key in art["series"]:
                return f.stem, art["series"][key]
    return None, None


def artifact(cfg, spec: dict, issue_date: dt.date, keys: list[str], quantiles, train: bool = True) -> dict:
    """The trained series for this issue day: {"series": {key: model}}, each from the file of its training end.
    train=False (a live run before the gate): never train; a model not trained ahead (nightly catch-up) falls back
    to the latest one trained earlier."""
    out = {"series": {}}
    ends = {}
    for k in keys:
        ends.setdefault(end_for(spec, k, issue_date), []).append(k)
    for end_day, ks in ends.items():
        if train:
            out["series"].update(_artifact_through(cfg, spec, end_day, ks, quantiles)["series"])
            continue
        for k in ks:
            used, model = latest_trained(cfg, spec, k, end_day)
            if model is not None:
                out["series"][k] = model
                if used != str(end_day):
                    print(f"{spec['name']} {k}: not trained through {end_day} ahead of the run, using {used}", file=sys.stderr, flush=True)
    if (spec.get("retrain") or {}):   # daily files pile up: keep 40 days
        d = Path(cfg["_root"]) / "data" / "models" / spec["name"]
        for f in d.glob("*.pkl") if d.exists() else []:
            day = dt.date.fromisoformat(f.stem)
            month_end = (day + dt.timedelta(days=1)).day == 1
            if not month_end and day < issue_date - dt.timedelta(days=40):
                f.unlink()
                f.with_suffix(".json").unlink(missing_ok=True)
    return out


def _artifact_through(cfg, spec: dict, end_day: dt.date, keys: list[str], quantiles) -> dict:
    from foretec_live.covariates_history import build_history, load_history

    name, kind = spec["name"], spec.get("regressor", "lgbm")
    days, holdout = int(spec.get("train_days", 730)), int(spec.get("holdout_days", 120))
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
        yy = y.reindex(cov.index).to_numpy(dtype=float)
        sk = scale_kind(cfg, spec, key)
        if sk:
            yy = yy / scale_rows(cfg, key, sk, y, cov.index)
            yy[~np.isfinite(yy)] = np.nan
        art["series"][key] = fit_series(kind, X, yy, quantiles, holdout, cfg["timezone"], float(spec.get("half_life_days", 120)))
        art["series"][key]["scale"] = sk
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
    art = artifact(cfg, spec, dt.date.fromisoformat(req["issue_date"]), list(series), req["quantiles"], train=not req.get("live"))
    cov = load_covariates(req)                     # the live covariates: only runs published before the cut-off
    inp = pd.read_parquet(req["input"])
    results, errors = {}, {}
    for key, y in series.items():
        zone, target = key.split("_")
        try:
            t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
            t_fut = future_index(t_hist[-1], req["horizon"][key])
            ys = pd.Series(y, index=t_hist)
            X = design(cov, t_fut, zone, target, ys, cols=None)
            if key not in art["series"]:
                raise RuntimeError("no trained model (the nightly catch-up trains them)")
            point, q = predict(art["series"][key], X, cfg["timezone"])
            sk = art["series"][key].get("scale")
            if sk:   # trained on capacity factor: back to MW (the proxy needs 90 days: long history + this snapshot)
                yl = ys if sk == "capacity" else pd.concat([history(cfg, zone, target, t_hist[0].date(), 120), ys])
                yl = yl[~yl.index.duplicated(keep="last")].sort_index()
                sc = scale_rows(cfg, key, sk, yl, t_fut)
                point, q = point * sc, q * sc[:, None]
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
