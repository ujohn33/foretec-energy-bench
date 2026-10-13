"""Day-ahead lag models trained on long history and retrained monthly: lgbm_lags (LightGBM) and lin_lags (ridge).

Both use the same pipeline, so they differ only in the regressor (the classical linear benchmark next to the GBM):

  rows      one per quarter-hour from the history cut-off to the end of delivery day D, for every training day:
            exactly the task of an 11:30 run, with exactly the history public at that cut-off
  features  the same quarter-hour 1..7 days earlier (lag 1 only where it is before the cut-off), the latest
            same-quarter-hour value, the last value and its deviation from yesterday fading over the horizon,
            the 1 h and 24 h means before the cut-off, steps ahead, time-of-day Fourier terms, day of week
  training  the `train_days` delivery days up to the end of the previous month (train_end), so a model never sees
            the month it forecasts. One artifact per train_end in data/models/<model>/, trained on first use
            (the nightly catch-up warms it on the 1st). History: data/static/climatology_history.parquet,
            extended through train_end from the source chain.
  intervals the point model is fitted without the last `holdout_days`, its residuals there give quantiles per
            local hour of day, then it is refitted on everything. Same method for both regressors.

Needs the main environment (env: _self): it reads the history through foretec_live.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STEP = pd.Timedelta("15min")
DAY = 96


def train_end_for(issue_date: dt.date) -> dt.date:
    """Last day of the month before the issue date's month."""
    return issue_date.replace(day=1) - dt.timedelta(days=1)


# ---------------------------------------------------------------- history
def history(cfg, zone: str, target: str, end_day: dt.date, days: int) -> pd.Series:
    """15-min series (UTC-naive index) covering `days` + 8 days before the end of local day end_day."""
    from foretec_live.data import get_source
    from foretec_live.timeutil import delivery_index

    end = delivery_index(end_day - dt.timedelta(days=1), cfg)[-1] + STEP        # end of local end_day (exclusive)
    start = (end - pd.Timedelta(days=days + 8)).floor("D")
    root = Path(cfg["_root"]) / "data" / "static"
    parts = []
    f = root / "climatology_history.parquet"
    if f.exists():
        h = pd.read_parquet(f, filters=[("zone", "==", zone), ("target", "==", target)])
        parts.append(h.set_index("time_utc")["value"])
    ext = root / "history_ext" / f"{zone}_{target}.parquet"   # what the static file lacks, fetched once
    if ext.exists():
        parts.append(pd.read_parquet(ext)["value"])
    s = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    s = s[~s.index.duplicated(keep="last")]
    have_to = s.index.max() + STEP if len(s) else start
    have_from = s.index.min() if len(s) else end
    need = [(a, b) for a, b in ((start, min(have_from, end)), (max(have_to, start), end)) if a < b]
    if need:
        src = get_source(cfg)
        got = [src.fetch(zone, target, a, b) for a, b in need]
        got = [g for g in got if g is not None and len(g)]
        if got:
            new = pd.concat(got).rename("value")
            old = pd.read_parquet(ext)["value"] if ext.exists() else pd.Series(dtype=float)
            both = pd.concat([old, new]).sort_index()
            ext.parent.mkdir(parents=True, exist_ok=True)
            both[~both.index.duplicated(keep="last")].rename("value").to_frame().to_parquet(ext)
            s = pd.concat([s, new]).sort_index()
            s = s[~s.index.duplicated(keep="last")]
    s = s[(s.index >= start) & (s.index < end)]
    return s.reindex(pd.date_range(start, end - STEP, freq=STEP))


# ---------------------------------------------------------------- features
LIN = ["same_qh", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "last", "dev_decay", "mean_1h", "mean_24h",
       "h_frac"] + [f"tod_{f}{k}" for k in (1, 2, 3) for f in ("s", "c")] + [f"dow{d}" for d in range(7)]
GBM = ["lag1", "same_qh", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "last", "dev_decay", "mean_1h",
       "mean_24h", "h", "tod_s1", "tod_c1", "qh", "dow"]


def features(vals: np.ndarray, t0: pd.Timestamp, last_pos: int, n_steps: int, tz: str) -> pd.DataFrame:
    """Rows for the n_steps quarter-hours after position last_pos (the last value public at the cut-off).
    vals: the series on a regular 15-min grid starting at t0 (NaN = missing)."""
    pos = last_pos + np.arange(1, n_steps + 1)
    get = lambda p: np.where((p >= 0) & (p < len(vals)), vals[np.clip(p, 0, len(vals) - 1)], np.nan)
    lags = {k: get(pos - k * DAY) for k in range(1, 8)}
    lag1 = np.where(pos - DAY <= last_pos, lags[1], np.nan)          # yesterday's value only if already public
    same_qh = np.where(np.isnan(lag1), lags[2], lag1)
    last = vals[last_pos]
    h = np.arange(1, n_steps + 1)
    times = pd.DatetimeIndex(t0 + pd.to_timedelta(pos * 15, unit="min"))
    loc = times.tz_localize("UTC").tz_convert(tz)
    qh = (loc.hour * 4 + loc.minute // 15).to_numpy()
    out = {"lag1": lag1, "same_qh": same_qh, **{f"lag{k}": lags[k] for k in range(2, 8)},
           "last": np.full(n_steps, last), "dev_decay": (last - get(np.array([last_pos - DAY]))[0]) * np.exp(-h / 12.0),
           "mean_1h": np.full(n_steps, np.nanmean(vals[max(0, last_pos - 3):last_pos + 1])),
           "mean_24h": np.full(n_steps, np.nanmean(vals[max(0, last_pos - DAY + 1):last_pos + 1])),
           "h": h, "h_frac": h / 150.0, "qh": qh, "dow": loc.dayofweek.to_numpy()}
    for k in (1, 2, 3):
        out[f"tod_s{k}"], out[f"tod_c{k}"] = np.sin(2 * np.pi * k * qh / DAY), np.cos(2 * np.pi * k * qh / DAY)
    for d in range(7):
        out[f"dow{d}"] = (out["dow"] == d).astype(float)
    return pd.DataFrame(out, index=times)


def training_rows(cfg, s: pd.Series, target: str, end_day: dt.date, days: int):
    """Stacked rows for every delivery day D in the window: the run of issue day D-1, as it would have been."""
    from foretec_live.timeutil import availability_cutoff, delivery_index

    vals, t0 = s.to_numpy(dtype=float), s.index[0]
    frames, ys, day_of = [], [], []
    for i in range(days):
        D = end_day - dt.timedelta(days=days - 1 - i)
        issue = D - dt.timedelta(days=1)
        last_t = availability_cutoff(issue, target, cfg) - STEP
        end_t = delivery_index(issue, cfg)[-1]
        last_pos, n = int((last_t - t0) / STEP), int((end_t - last_t) / STEP)
        if last_pos - 7 * DAY < 0:
            continue
        X = features(vals, t0, last_pos, n, cfg["timezone"])
        y = vals[last_pos + 1:last_pos + 1 + n]
        if len(y) < n:
            continue
        frames.append(X)
        ys.append(y)
        day_of += [i] * n
    return pd.concat(frames), np.concatenate(ys), np.array(day_of)


# ---------------------------------------------------------------- fitting
def make_regressor(kind: str, params: dict):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**{"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                                    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "n_jobs": 4,
                                    "verbose": -1, **params})
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15)))


def fit_series(kind: str, params: dict, X: pd.DataFrame, y: np.ndarray, day_of: np.ndarray, holdout_days: int,
               quantiles: list[float], tz: str) -> dict:
    cols = GBM if kind == "lgbm" else LIN
    X = X[cols]
    ok = ~np.isnan(y) if kind == "lgbm" else ~np.isnan(y) & X.notna().all(axis=1).to_numpy()
    X, y, day_of = X[ok], y[ok], day_of[ok]
    fill = X.median()                                    # for the rare missing feature at prediction time
    cut = day_of.max() - holdout_days
    fit, hold = day_of <= cut, day_of > cut
    m = make_regressor(kind, params).fit(X[fit], y[fit])
    res = y[hold] - m.predict(X[hold])
    hour = X.index[hold].tz_localize("UTC").tz_convert(tz).hour
    qtab = pd.DataFrame({h: np.quantile(res[hour == h], quantiles) if (hour == h).sum() > 20 else np.quantile(res, quantiles)
                         for h in range(24)}).T                                   # 24 hours x Q
    holdout_mae = float(np.mean(np.abs(res)))
    m = make_regressor(kind, params).fit(X, y)          # final model on every training day
    return {"model": m, "cols": cols, "fill": fill, "qtab": qtab.to_numpy(), "rows": int(len(y)), "holdout_mae": holdout_mae}


def artifact(cfg, spec: dict, issue_date: dt.date, keys: list[str], quantiles: list[float]) -> dict:
    """The model trained through train_end_for(issue_date), trained now if it is not there yet."""
    name, kind = spec["name"], spec.get("regressor", "lgbm")
    days, holdout = int(spec.get("train_days", 730)), int(spec.get("holdout_days", 120))
    end_day = train_end_for(issue_date)
    assert end_day < issue_date
    d = Path(cfg["_root"]) / "data" / "models" / name
    f = d / f"{end_day}.pkl"
    art = pickle.loads(f.read_bytes()) if f.exists() else {"train_end": str(end_day), "series": {}}
    todo = [k for k in keys if k not in art["series"]]
    for key in todo:
        t = time.time()
        zone, target = key.split("_")
        s = history(cfg, zone, target, end_day, days)
        X, y, day_of = training_rows(cfg, s, target, end_day, days)
        art["series"][key] = fit_series(kind, spec.get("params") or {}, X, y, day_of, holdout, quantiles, cfg["timezone"])
        art["series"][key]["seconds"] = round(time.time() - t, 1)
        print(f"{name} {key}: trained through {end_day} on {art['series'][key]['rows']} rows "
              f"({time.time() - t:.0f}s, holdout MAE {art['series'][key]['holdout_mae']:.1f})", file=sys.stderr, flush=True)
    if todo:
        d.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(art))
        tmp.replace(f)
        meta = {k: {x: v[x] for x in ("rows", "holdout_mae", "seconds")} for k, v in art["series"].items()}
        (d / f"{end_day}.json").write_text(json.dumps({"train_end": str(end_day), "train_days": days, "series": meta}, indent=1))
    return art


def forecast(art_series: dict, y_hist: np.ndarray, t_hist: pd.DatetimeIndex, steps: int, quantiles, tz: str, key: str):
    X = features(y_hist.astype(float), t_hist[0], len(y_hist) - 1, steps, tz)[art_series["cols"]]
    X = X.fillna(art_series["fill"])
    point = art_series["model"].predict(X)
    hour = X.index.tz_localize("UTC").tz_convert(tz).hour.to_numpy()
    q = point[:, None] + art_series["qtab"][hour]
    if not key.endswith("price"):     # generation and load cannot be negative
        point, q = np.clip(point, 0, None), np.clip(q, 0, None)
    return point, np.sort(q, axis=1)


def main():
    import yaml
    from fl_io import read_request, write_output
    from foretec_live.config import load_config

    req, series = read_request()
    cfg = load_config()
    spec = yaml.safe_load((Path.cwd() / "model.yaml").read_text())
    issue = dt.date.fromisoformat(req["issue_date"])
    art = artifact(cfg, spec, issue, list(series), req["quantiles"])
    inp = pd.read_parquet(req["input"])
    results, errors = {}, {}
    for key, y in series.items():
        t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
        try:
            results[key] = forecast(art["series"][key], y, t_hist, req["horizon"][key], req["quantiles"], cfg["timezone"], key)
        except Exception as e:  # noqa: BLE001 - one series failing must not sink the others
            errors[key] = f"{type(e).__name__}: {e}"[:200]
    write_output(req, results, errors)


def warm(model_dir: str, issue_date: str):
    """Train this month's artifact ahead of the run (nightly catch-up)."""
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
