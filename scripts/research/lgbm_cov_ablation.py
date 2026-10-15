"""Why is lgbm_cov_2y not clearly better than lgbm_cov? Ablation on the backtest (offline; the bench is untouched).

Test: issue days 14, 16, ..., 30 Sep 2026, all series; each day forecast from its own frozen pre-cut-off covariates
(data/covariates/<issue>.parquet), scored like the bench: Rel. MAE pooled per series against naive_weekly (from the
backtest scores), DK load on hourly means.

A  window      LightGBM (point) retrained for every test day on the N delivery days before it, N = 28, 90, 180,
               365, 730, uniform weights; plus 730 with recency weights (half-life 120 d) = lgbm_cov_2y's recipe.
               Same features (fl_covmodel.design) and parameters throughout.
B  weather     a 150-day window trained on (a) the true pre-cut-off ICON-EU/GFS runs or (b) the previous-runs
               approximation used for training history; IFS is a true run in both.
C  parameters  random search per target on a validation period inside the training data (the 60 days before
               the first test day), never on the test days; the best set is then evaluated like A.

Run on the server:  FORETEC_HOME=... PYTHONPATH=models/_shared python scripts/research/lgbm_cov_ablation.py
Writes results/research/lgbm_cov_ablation_<part>.csv.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fl_covmodel import LGBM, design, recency_weights
from fl_lagmodel import history
from foretec_live.config import load_config, paths
from foretec_live.covariates import centroids, fetch_run, select_run, _aggregate, _ensemble_stats, load_forecast
from foretec_live.covariates_history import _add_fuel, load_history
from foretec_live.timeutil import delivery_index

cfg = load_config()
ROOT = Path(cfg["_root"])
OUT = ROOT / "results" / "research"
OUT.mkdir(parents=True, exist_ok=True)
TEST_DAYS = [dt.date(2026, 9, 14) + dt.timedelta(days=2 * k) for k in range(9)]
KEYS = [f"{z}_{t}" for z in cfg["zones"] for t in cfg["targets"]]
HOURLY = set((cfg.get("score_resolution") or {}).keys())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pause_for_gate():
    """Never compete with the live run (10:45-12:15 CET/CEST)."""
    while True:
        h = int(pd.Timestamp.now(tz="Europe/Brussels").strftime("%H%M"))
        if not 1045 <= h <= 1215:
            return
        time.sleep(60)


# ---------------------------------------------------------------- data
log("loading history and covariates")
EARLIEST = TEST_DAYS[0] - dt.timedelta(days=731)
COV = load_history(cfg, EARLIEST, TEST_DAYS[-1] - dt.timedelta(days=1))
Y = {k: history(cfg, *k.split("_"), TEST_DAYS[-1] - dt.timedelta(days=1), 760) for k in KEYS}
COV_DAY = pd.Series(COV.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date, index=COV.index)

scores = pd.read_parquet(paths(cfg, "results/backtest")["scores"])
BASE = scores[scores.model == cfg["baseline"]].set_index(["issue_date", "zone", "target"])["mae"]


def actual(issue: dt.date, key: str) -> pd.Series:
    d = issue + dt.timedelta(days=1)
    f = ROOT / "data" / "actuals" / str(d) / f"{key}.parquet"
    return pd.read_parquet(f)["value"].reindex(delivery_index(issue, cfg)) if f.exists() else None


def mae(pred: pd.Series, y: pd.Series, key: str) -> float:
    if key in HOURLY:
        pred, y = pred.resample("1h").mean(), y.resample("1h").mean()
    ok = pred.notna() & y.notna()
    return float((pred[ok] - y[ok]).abs().mean())


def train_rows(cov: pd.DataFrame, cov_day: pd.Series, first: dt.date, last: dt.date) -> pd.DataFrame:
    return cov[(cov_day >= first) & (cov_day <= last)]


CAP_DIR = ROOT / "data" / "static" / "capacity"


def scale_series(key: str, kind: str | None) -> pd.Series | None:
    """Per local delivery day: what wind/solar output is divided by. proxy = the highest output in days D-91..D-2
    (known at the cut-off); capacity = installed capacity on D-1 (data/static/capacity, build_capacity.py)."""
    if kind is None:
        return None
    if kind == "proxy":
        y = Y[key]
        daily = y.groupby(y.index.tz_localize("UTC").tz_convert(cfg["timezone"]).normalize().tz_localize(None)).max()
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(days=3), freq="D"))
        return daily.rolling(90, min_periods=60).max().shift(2)
    f = CAP_DIR / f"{key}.parquet"
    return pd.read_parquet(f)["mw"].shift(1) if f.exists() else None


def per_row(scale: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    days = index.tz_localize("UTC").tz_convert(cfg["timezone"]).normalize().tz_localize(None)
    return scale.reindex(days).to_numpy(dtype=float)


_MODELS: dict = {}


def fit_predict(key: str, cov_train: pd.DataFrame, issue: dt.date, params: dict, half_life: float | None,
                scale_kind: str | None = None, cache_key=None) -> pd.Series:
    zone, target = key.split("_")
    y = Y[key]
    sc = scale_series(key, scale_kind)
    X = design(cov_train, cov_train.index, zone, target, y)
    if cache_key is not None and cache_key in _MODELS:
        m = _MODELS[cache_key]
    else:
        yy = y.reindex(cov_train.index).to_numpy(dtype=float)
        if sc is not None:
            yy = yy / per_row(sc, cov_train.index)
        ok = ~np.isnan(yy) & np.isfinite(yy)
        w = recency_weights(X.index[ok], half_life) if half_life else None
        m = lgb.LGBMRegressor(**params).fit(X[ok], yy[ok], sample_weight=w)
        if cache_key is not None:
            _MODELS[cache_key] = m
    live = pd.read_parquet(ROOT / "data" / "covariates" / f"{issue}.parquet")
    live.index = pd.to_datetime(live.index)
    idx = delivery_index(issue, cfg)
    Xt = design(live, idx, zone, target, y, cols=[c for c in X.columns if c not in ("cal_qh_sin", "cal_qh_cos", "cal_dow",
                                                                                     "cal_weekend", "lvl_7d", "max_28d", "price_lag_1d")])
    Xt = Xt.reindex(columns=X.columns)
    p = m.predict(Xt)
    if sc is not None:
        p = p * per_row(sc, idx)
    if target != "price":
        p = np.clip(p, 0, None)
    return pd.Series(p, index=idx)


def evaluate(name: str, cov: pd.DataFrame, cov_day: pd.Series, window: int, params: dict, half_life=None, keys=KEYS,
             days=TEST_DAYS, train_end: dt.date | None = None, scale_kind: str | None = None) -> list[dict]:
    """train_end: fixed last training day (monthly retraining: one model per series for every test day);
    otherwise retrained for every test day on the `window` days before it."""
    rows = []
    for issue in days:
        pause_for_gate()
        last = train_end or issue - dt.timedelta(days=1)
        tr = train_rows(cov, cov_day, last - dt.timedelta(days=window - 1), last)
        for key in keys:
            y = actual(issue, key)
            zone, target = key.split("_")
            base = BASE.get((str(issue), zone, target))
            if y is None or base is None or y.notna().mean() < 0.95:
                continue
            if scale_kind and scale_series(key, scale_kind) is None:
                continue
            p = fit_predict(key, tr, issue, params, half_life, scale_kind, cache_key=(name, key) if train_end else None)
            rows.append({"config": name, "issue_date": str(issue), "key": key, "zone": zone, "target": target,
                         "mae": mae(p, y, key), "mae_base": float(base), "train_rows": len(tr)})
        log(name, issue, "done")
    return rows


def summary(df: pd.DataFrame) -> pd.DataFrame:
    per = df.groupby(["config", "key", "target"]).apply(lambda d: d.mae.sum() / d.mae_base.sum(), include_groups=False).rename("rel").reset_index()
    t = per.pivot_table(index="config", columns="target", values="rel", aggfunc="mean")
    t["mean"] = per.groupby("config").rel.mean()
    t["median"] = per.groupby("config").rel.median()
    t["gmean"] = per.groupby("config").rel.apply(lambda x: np.exp(np.log(x).mean()))
    return t.round(3)


part = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---------------------------------------------------------------- A: window
if part in ("A", "all"):
    rows = []
    for w in (28, 90, 180, 365, 730):
        rows += evaluate(f"window_{w}", COV, COV_DAY, w, LGBM)
    rows += evaluate("window_730_recency120", COV, COV_DAY, 730, LGBM, half_life=120)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lgbm_cov_ablation_A.csv", index=False)
    log("A: training window\n" + summary(df).to_string())

# ---------------------------------------------------------------- B: training weather
if part in ("B", "all"):
    # true pre-cut-off runs for every model, delivery days in the 150 days before the last test day
    first = TEST_DAYS[0] - dt.timedelta(days=150)
    f_true = OUT / "cov_true_runs.parquet"
    if f_true.exists():
        TRUE = pd.read_parquet(f_true)
    else:
        log("building true-run covariates", first, "..", TEST_DAYS[-1])
        c = centroids(cfg)
        models = list(cfg["covariates"]["nwp"]["models"])
        slices = []
        d = first
        while d < TEST_DAYS[-1]:
            pause_for_gate()
            idx = delivery_index(d - dt.timedelta(days=1), cfg)
            wide = []
            for model in models:
                run = select_run(model, d - dt.timedelta(days=1), cfg)
                cycle = pd.Timedelta(hours=int(cfg["covariates"]["nwp"]["models"][model]["cycle_hours"]))
                for back in range(3):
                    try:
                        wide.append(_aggregate(fetch_run(model, run - back * cycle, cfg, c), c, model).reindex(idx))
                        break
                    except Exception as e:  # noqa: BLE001
                        log("run missing", model, run - back * cycle, str(e)[:80])
            w = pd.concat(wide, axis=1)
            w = pd.concat([w, _ensemble_stats(w, models)], axis=1)
            for zone in cfg["zones"]:
                w[f"{zone}.load_fc.entsoe.load_mw"] = load_forecast(zone, d, cfg).reindex(idx).values
            slices.append(w)
            d += dt.timedelta(days=1)
        TRUE = _add_fuel(pd.concat(slices).sort_index(), cfg)
        TRUE.index.name = "time_utc"
        TRUE.to_parquet(f_true)
    TRUE_DAY = pd.Series(TRUE.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date, index=TRUE.index)
    rows = evaluate("window_150_true_runs", TRUE, TRUE_DAY, 150, LGBM)
    rows += evaluate("window_150_previous_runs", COV, COV_DAY, 150, LGBM)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lgbm_cov_ablation_B.csv", index=False)
    log("B: training weather\n" + summary(df).to_string())

# ---------------------------------------------------------------- C: hyperparameters
if part in ("C", "all"):
    rng = np.random.default_rng(42)
    space = {"num_leaves": [15, 31, 63, 127, 255], "learning_rate": [0.02, 0.03, 0.05, 0.08], "min_child_samples": [20, 50, 100, 200, 400],
             "colsample_bytree": [0.3, 0.5, 0.7, 0.9], "subsample": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 5.0, 20.0],
             "max_bin": [63, 127, 255]}
    N_TRIALS = int(os.environ.get("ABLATION_TRIALS", 8))
    # validation like daily retraining: weekly folds right before the test days, each trained on the 730 days before
    # it (a single fit validated over many weeks would reward parameters that age well, i.e. monthly retraining)
    folds = [(TEST_DAYS[0] - dt.timedelta(days=14), TEST_DAYS[0] - dt.timedelta(days=8)),
             (TEST_DAYS[0] - dt.timedelta(days=7), TEST_DAYS[0] - dt.timedelta(days=1))]
    best = {}
    trials = []
    for target in cfg["targets"]:
        keys = [k for k in KEYS if k.endswith(target)]
        cands = [dict(LGBM)] + [{**LGBM, **{p: rng.choice(v).item() for p, v in space.items()}, "n_estimators": 1500}
                                for _ in range(N_TRIALS)]
        for i, prm in enumerate(cands):
            pause_for_gate()
            errs, iters = [], []
            for v_start, v_end in folds:
                tr = train_rows(COV, COV_DAY, v_start - dt.timedelta(days=730), v_start - dt.timedelta(days=1))
                va = train_rows(COV, COV_DAY, v_start, v_end)
                for key in keys:
                    zone, _ = key.split("_")
                    y = Y[key]
                    X, Xv = design(tr, tr.index, zone, target, y), design(va, va.index, zone, target, y)
                    yy, yv = y.reindex(tr.index).to_numpy(float), y.reindex(va.index).to_numpy(float)
                    ok, okv = ~np.isnan(yy), ~np.isnan(yv)
                    w = recency_weights(X.index[ok], 120)
                    m = lgb.LGBMRegressor(**prm)
                    if i == 0:   # the current parameters, as they are
                        m.fit(X[ok], yy[ok], sample_weight=w)
                    else:
                        m.fit(X[ok], yy[ok], sample_weight=w, eval_set=[(Xv[okv], yv[okv])], eval_metric="l1",
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                        iters.append(m.best_iteration_)
                    base = np.nanmean(np.abs(yv[okv] - y.reindex(va.index - pd.Timedelta("7D")).to_numpy(float)[okv]))
                    errs.append(np.mean(np.abs(m.predict(Xv[okv]) - yv[okv])) / base)
            score = float(np.mean(errs))
            rec = {"target": target, "trial": i, "val_rel_mae": score, "n_estimators": int(np.median(iters)) if iters else prm["n_estimators"],
                   **{k: prm.get(k) for k in space}}
            trials.append(rec)
            log("C", target, i, round(score, 4), rec)
            if target not in best or score < best[target]["val_rel_mae"]:
                best[target] = rec
    pd.DataFrame(trials).to_csv(OUT / "lgbm_cov_ablation_C_trials.csv", index=False)
    (OUT / "lgbm_cov_ablation_C_best.json").write_text(json.dumps(best, indent=1, default=str))
    rows = []
    for target, b in best.items():
        prm = {**LGBM, **{k: b[k] for k in space if b.get(k) is not None}, "n_estimators": int(b["n_estimators"])}
        keys = [k for k in KEYS if k.endswith(target)]
        rows += evaluate("window_730_recency120_tuned", COV, COV_DAY, 730, prm, half_life=120, keys=keys)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lgbm_cov_ablation_C.csv", index=False)
    log("C: tuned parameters\n" + summary(df).to_string())
    log("best parameters per target:\n" + json.dumps(best, indent=1, default=str))

# ---------------------------------------------------------------- D: monthly retraining, with and without recency
MONTHLY_END = TEST_DAYS[0].replace(day=1) - dt.timedelta(days=1)     # 31 Aug: what a monthly model has in September
if part in ("D", "all"):
    rows = evaluate("monthly_730_uniform", COV, COV_DAY, 730, LGBM, train_end=MONTHLY_END)
    rows += evaluate("monthly_730_recency120", COV, COV_DAY, 730, LGBM, half_life=120, train_end=MONTHLY_END)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lgbm_cov_ablation_D.csv", index=False)
    log("D: monthly retraining (compare A's window_730 / window_730_recency120, retrained daily)\n" + summary(df).to_string())

# ---------------------------------------------------------------- E: capacity-scaled wind and solar targets
if part in ("E", "all"):
    keys = [k for k in KEYS if k.endswith(("_wind", "_solar"))]
    rows = []
    for retrain, end in (("daily", None), ("monthly", MONTHLY_END)):
        for kind in (None, "proxy", "capacity"):
            rows += evaluate(f"{retrain}_{kind or 'unscaled'}", COV, COV_DAY, 730, LGBM, half_life=120, keys=keys,
                             train_end=end, scale_kind=kind)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lgbm_cov_ablation_E.csv", index=False)
    log("E: all wind/solar series (capacity configs only where a capacity series exists)\n" + summary(df).to_string())
    have_cap = sorted(df.loc[df.config.str.endswith("capacity"), "key"].unique())
    log("E: the series with real capacity only " + str(have_cap) + "\n" + summary(df[df.key.isin(have_cap)]).to_string())

# ---------------------------------------------------------------- F: monthly model + correction from recent errors
if part in ("F", "all"):
    rows = []
    for issue in TEST_DAYS:
        pause_for_gate()
        live = pd.read_parquet(ROOT / "data" / "covariates" / f"{issue}.parquet")   # true pre-cut-off runs, past days too
        live.index = pd.to_datetime(live.index)
        live_day = pd.Series(live.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date, index=live.index)
        tr = train_rows(COV, COV_DAY, MONTHLY_END - dt.timedelta(days=729), MONTHLY_END)
        for key in KEYS:
            y_act = actual(issue, key)
            zone, target = key.split("_")
            base = BASE.get((str(issue), zone, target))
            if y_act is None or base is None or y_act.notna().mean() < 0.95:
                continue
            p = fit_predict(key, tr, issue, LGBM, 120, None, cache_key=("monthly_730_recency120", key))
            m = _MODELS[("monthly_730_recency120", key)]
            # the model's errors on the 7 delivery days before the issue day (their actuals are public at the cut-off)
            past = live[(live_day >= issue - dt.timedelta(days=7)) & (live_day <= issue - dt.timedelta(days=1))]
            X = design(tr, tr.index, zone, target, Y[key])
            Xp = design(past, past.index, zone, target, Y[key], cols=[c for c in X.columns if c not in (
                "cal_qh_sin", "cal_qh_cos", "cal_dow", "cal_weekend", "lvl_7d", "max_28d", "price_lag_1d")]).reindex(columns=X.columns)
            res = pd.Series(Y[key].reindex(past.index).to_numpy(float) - m.predict(Xp), index=past.index).dropna()
            rday = pd.Series(res.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date, index=res.index)
            last3 = res[rday >= issue - dt.timedelta(days=3)].mean()
            hour = res.index.tz_localize("UTC").tz_convert(cfg["timezone"]).hour
            by_hour = res.groupby(hour).mean()
            th = p.index.tz_localize("UTC").tz_convert(cfg["timezone"]).hour
            clip = (lambda v: v) if target == "price" else (lambda v: v.clip(lower=0))
            for name, q in (("monthly", p), ("monthly_bias3d", clip(p + last3)),
                            ("monthly_hourbias7d", clip(p + by_hour.reindex(th).fillna(0).to_numpy()))):
                rows.append({"config": name, "issue_date": str(issue), "key": key, "zone": zone, "target": target,
                             "mae": mae(q, y_act, key), "mae_base": float(base), "train_rows": len(tr)})
        log("F", issue, "done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lgbm_cov_ablation_F.csv", index=False)
    log("F: monthly model + correction from recent errors (compare A window_730_recency120, retrained daily)\n" + summary(df).to_string())
