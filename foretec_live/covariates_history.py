"""Two years of covariates for training, in exactly the live layout (the columns build_covariates writes).

Scoring never uses these: live and backtest days always get build_covariates, i.e. only runs published before
their cut-off. These frames are only what models train on, one slice per delivery day D:

  ecmwf_ifs     the true run of issue day D-1 (Open-Meteo single runs, archive from 2024-03-14), the run the
                live model would have used. Same cache and fallback as the live builder.
  icon_eu,      the Open-Meteo previous-runs archive (from 2024-02-17): the single-runs archive only starts on
  gfs_seamless  2026-04-02. `<var>_previous_dayN` is the forecast made N days before the valid time, so for
                every quarter-hour we take the N closest to the lead time the live run has at the cut-off
                (ICON-EU 06 UTC: 16-40 h over day D, so N=1 for most of the day; GFS 00 UTC: 22-46 h, N=1 then 2).
                Training values are therefore close to, not identical with, what the model sees live; the size
                of the difference is measured where both archives overlap (validate_previous_runs).
  load_fc       ENTSO-E A65, the same cached day files as the live builder.
  fuel          the cached TTF and EUA quotes known at each day's cut-off; the cache is topped up within the
                oilpriceapi quota, so the oldest days may still lack fuel (NaN).

Output: data/static/cov_history/<YYYY-MM>.parquet, one frame per month (time_utc x columns).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .covariates import (BUCKET_VARS, _aggregate, _centroid_hash, _ensemble_stats, cached_per_zone, centroids, fetch_run,
                         load_forecast, select_run)
from .timeutil import as_date, delivery_index, issue_timestamp

log = logging.getLogger(__name__)
PREV_HOST = "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
PREV_MODELS = ("icon_eu", "gfs_seamless")
VARIABLES = sorted({v for vs in BUCKET_VARS.values() for v in vs})


def _months(a: dt.date, b: dt.date) -> list[dt.date]:
    out, m = [], a.replace(day=1)
    while m <= b:
        out.append(m)
        m = (m + dt.timedelta(days=32)).replace(day=1)
    return out


# ---------------------------------------------------------------- previous runs (ICON-EU, GFS)
def fetch_previous_month(model: str, month: dt.date, cfg, c: pd.DataFrame) -> pd.DataFrame:
    """All centroids, all variables, previous_day1 and previous_day2, one calendar month (UTC). Cached per zone
    once the month is complete."""
    path = lambda cz: Path(cfg["_root"]) / "data" / "cache" / "nwp_prev" / _centroid_hash(cz) / model / f"{month:%Y-%m}.parquet"
    return cached_per_zone(c, path, lambda cm: _request_previous_month(model, month, cfg, cm))


def _request_previous_month(model: str, month: dt.date, cfg, c: pd.DataFrame) -> pd.DataFrame:
    end = (month + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    end_req = min(end, dt.date.today() + dt.timedelta(days=7))   # the API refuses end dates far in the future
    ncfg = cfg["covariates"]["nwp"]
    params = {"latitude": ",".join(f"{x:.4f}" for x in c["latitude"]), "longitude": ",".join(f"{x:.4f}" for x in c["longitude"]),
              "models": model, "minutely_15": ",".join(f"{v}_previous_day{n}" for v in VARIABLES for n in (1, 2)),
              "start_date": str(month), "end_date": str(end_req), "timezone": "GMT", "tilt": ncfg.get("tilt", 35), "azimuth": 0}
    key = os.environ.get("OPENMETEOKEY")
    host = PREV_HOST if key else PREV_HOST.replace("customer-", "")
    if key:
        params["apikey"] = key
    why = ""
    for attempt in range(5):
        try:
            r = requests.get(host, params=params, timeout=300)
            if r.status_code == 200:
                js = r.json()
                js = js if isinstance(js, list) else [js]
                if len(js) == len(c):
                    break
                why = f"{len(js)} of {len(c)} locations"
            else:
                why = f"HTTP {r.status_code} {r.text[:150]}"
        except (requests.RequestException, ValueError) as e:
            why = f"{type(e).__name__}: {e}"
        log.warning("previous runs %s %s (try %d): %s", model, month, attempt + 1, why)
        time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(f"previous runs {model} {month:%Y-%m}: {why}"[:300])
    frames = []
    for i, loc in enumerate(js):
        m = loc["minutely_15"]
        d = pd.DataFrame({f"{v}@{n}": pd.to_numeric(pd.Series(m.get(f"{v}_previous_day{n}")), errors="coerce")
                          for v in VARIABLES for n in (1, 2)})
        d["time_utc"] = pd.to_datetime(m["time"])
        d["centroid"] = i
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.attrs = {"cache": end < dt.date.today() - dt.timedelta(days=3)}   # a month still filling up is not cached
    return out


def previous_runs_day(model: str, delivery: dt.date, cfg, prev: pd.DataFrame) -> pd.DataFrame:
    """The live-run-shaped frame (variables, time_utc, centroid) for delivery day D, each quarter-hour from the
    previous_dayN whose lead is closest to the live run's lead at the cut-off of issue day D-1."""
    idx = delivery_index(delivery - dt.timedelta(days=1), cfg)
    run = select_run(model, delivery - dt.timedelta(days=1), cfg)
    g = prev[prev["time_utc"].isin(idx)]
    lead_h = (g["time_utc"] - run).dt.total_seconds() / 3600
    n = np.clip(np.rint(lead_h / 24), 1, 2).astype(int)
    out = pd.DataFrame({"time_utc": g["time_utc"].to_numpy(), "centroid": g["centroid"].to_numpy()})
    for v in VARIABLES:
        out[v] = np.where(n == 1, g[f"{v}@1"], g[f"{v}@2"])
    return out


# ---------------------------------------------------------------- assemble
def build_day(delivery: dt.date, cfg, c: pd.DataFrame, prev: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list]:
    issue = delivery - dt.timedelta(days=1)
    idx = delivery_index(issue, cfg)
    models = list(cfg["covariates"]["nwp"]["models"])
    wide, used = [], []
    for model in models:
        raw = None
        if model in PREV_MODELS:
            raw = previous_runs_day(model, delivery, cfg, prev[model])
            used.append({"model": model, "source": "previous_runs (lead-matched)"})
        else:
            run = select_run(model, issue, cfg)
            cycle = pd.Timedelta(hours=int(cfg["covariates"]["nwp"]["models"][model]["cycle_hours"]))
            for back in range(3):   # like the live builder: an earlier, equally admissible run if the latest fails
                try:
                    raw = fetch_run(model, run - back * cycle, cfg, c)
                    used.append({"model": model, "source": "single_run", "run_utc": str(run - back * cycle)})
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("NWP %s %s: %s", model, run - back * cycle, str(e)[:120])
        if raw is None or raw.empty:
            used.append({"model": model, "source": "missing"})
            continue
        wide.append(_aggregate(raw, c, model).reindex(idx))
    w = pd.concat(wide, axis=1) if wide else pd.DataFrame(index=idx)
    w = pd.concat([w, _ensemble_stats(w, models)], axis=1)
    for zone in cfg["zones"]:
        w[f"{zone}.load_fc.entsoe.load_mw"] = load_forecast(zone, delivery, cfg).reindex(idx).values
    return w, used


def _add_fuel(cov: pd.DataFrame, cfg) -> pd.DataFrame:
    cache = Path(cfg["_root"]) / "data" / "cache" / "fuel_quotes.parquet"   # the cache only: no quota spent here
    if not cache.exists():
        return cov
    fuel = pd.read_parquet(cache)
    fuel["available_utc"] = pd.to_datetime(fuel["time_utc"]).dt.normalize() + pd.Timedelta(days=1)   # as in fuel_quotes
    delivery = cov.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date
    for code, name in (("TTF_EUR", "fuel.gas.ttf.eur_mwh"), ("CARBON_EUR", "fuel.carbon.eua.eur_t")):
        q = fuel[fuel["code"] == code].set_index("available_utc")["price"].sort_index()
        known = {}
        for d in set(delivery):
            k = q[q.index <= issue_timestamp(d - dt.timedelta(days=1), cfg)]
            # a quote older than 10 days is not what a trader would use: leave the day without fuel
            known[d] = k.iloc[-1] if len(k) and k.index[-1] > issue_timestamp(d - dt.timedelta(days=11), cfg) else np.nan
        cov[name] = [known[d] for d in delivery]
    cov["fuel.ccgt.srmc.eur_mwh"] = cov["fuel.gas.ttf.eur_mwh"] / 0.52 + cov["fuel.carbon.eua.eur_t"] * 0.37
    return cov


def extend_fuel(cfg, start: dt.date, max_calls: int = 40) -> int:
    """Fill the fuel quote cache backwards to `start`, 28-day chunks, at most max_calls requests (free-plan quota)."""
    key = os.environ.get("OIL_PRICE_KEY")
    cache = Path(cfg["_root"]) / "data" / "cache" / "fuel_quotes.parquet"
    if not key or not cache.exists():
        return 0
    have = pd.read_parquet(cache)
    calls, rows = 0, []
    for code in ("TTF_EUR", "CARBON_EUR"):
        first = pd.to_datetime(have.loc[have["code"] == code, "time_utc"]).min()
        cur = (first if pd.notna(first) else pd.Timestamp.now()).normalize()
        while cur.date() > start and calls < max_calls:
            a = cur - pd.Timedelta(days=28)
            calls += 1
            try:
                r = requests.get("https://api.oilpriceapi.com/v1/prices/historical", headers={"Authorization": f"Token {key}"},
                                 params={"by_code": code, "start": f"{a:%Y-%m-%d}", "end": f"{cur:%Y-%m-%d}"}, timeout=90)
                r.raise_for_status()
                data = r.json().get("data") or {}
                got = [{"time_utc": pd.Timestamp(x["created_at"]).tz_convert("UTC").tz_localize(None), "code": code,
                        "price": float(x["price"])} for x in (data.get("prices") if isinstance(data, dict) else data) or []]
            except Exception as e:  # quota or outage: keep what we have, the next night continues
                log.warning("oilpriceapi %s %s: %s", code, f"{a:%Y-%m-%d}", str(e)[:150])
                break
            rows += got
            cur = a
            time.sleep(0.25)
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["time_utc", "code"], keep="last")
        have.sort_values("time_utc").to_parquet(cache, index=False)
    return calls


def build_history(cfg, start: dt.date, end: dt.date, workers: int = 3) -> list[Path]:
    """Monthly training frames for delivery days start..end (inclusive); months already complete are kept."""
    start, end = as_date(start), as_date(end)
    c = centroids(cfg)
    out_dir = Path(cfg["_root"]) / "data" / "static" / "cov_history"
    out_dir.mkdir(parents=True, exist_ok=True)
    days = [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]
    # IFS runs first, a few at a time (each is one cached request)
    ifs = [m for m in cfg["covariates"]["nwp"]["models"] if m not in PREV_MODELS]
    todo = [(m, select_run(m, d - dt.timedelta(days=1), cfg)) for m in ifs for d in days]
    def _get(mr):
        try:
            fetch_run(mr[0], mr[1], cfg, c)
        except Exception as e:  # noqa: BLE001 - build_day falls back to earlier runs
            log.warning("prefetch %s %s: %s", mr[0], mr[1], str(e)[:120])
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_get, todo))
    # the ENTSO-E load forecasts too (one cached request per zone and day; the slow part when done one by one)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda zd: load_forecast(zd[0], zd[1], cfg), [(z, d) for z in cfg["zones"] for d in days]))
    written = []
    for month in _months(start, end):
        f = out_dir / f"{month:%Y-%m}.parquet"
        mdays = [d for d in days if d.replace(day=1) == month]
        if f.exists():
            have = set(pd.read_parquet(f, columns=[]).index.tz_localize("UTC").tz_convert(cfg["timezone"]).date)
            if all(d in have for d in mdays):
                continue
        # previous-runs months covering this month's delivery days (UTC days can spill into the next month)
        prev = {}
        for model in PREV_MODELS:
            need = sorted({m for d in mdays for m in _months(d - dt.timedelta(days=1), d + dt.timedelta(days=1))})
            prev[model] = pd.concat([fetch_previous_month(model, m, cfg, c) for m in need], ignore_index=True)
        slices, meta = [], {}
        for d in mdays:
            w, used = build_day(d, cfg, c, prev)
            slices.append(w)
            meta[str(d)] = used
        cov = pd.concat(slices).sort_index()
        cov = cov[~cov.index.duplicated(keep="last")]
        cov.index.name = "time_utc"
        cov = _add_fuel(cov, cfg)
        cov.to_parquet(f)
        f.with_suffix(".json").write_text(json.dumps({"built_utc": str(pd.Timestamp.now(tz="UTC")), "days": meta}, indent=0))
        written.append(f)
        log.info("covariate history %s: %d days, %d columns", f"{month:%Y-%m}", len(mdays), cov.shape[1])
    return written


def load_history(cfg, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Training covariates for delivery days start..end from the monthly frames."""
    d = Path(cfg["_root"]) / "data" / "static" / "cov_history"
    frames = [pd.read_parquet(d / f"{m:%Y-%m}.parquet") for m in _months(start, end) if (d / f"{m:%Y-%m}.parquet").exists()]
    cov = pd.concat(frames).sort_index()
    day = cov.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date
    return cov[(day >= start) & (day <= end)]


def validate_previous_runs(cfg, days: list[dt.date]) -> pd.DataFrame:
    """Where both archives exist: lead-matched previous runs against the true live run, per model and variable.
    Returns mean absolute difference and the mean value, per zone/bucket/variable, over the given delivery days."""
    c = centroids(cfg)
    rows = []
    for model in PREV_MODELS:
        need = sorted({m for d in days for m in _months(d - dt.timedelta(days=1), d + dt.timedelta(days=1))})
        prev = pd.concat([fetch_previous_month(model, m, cfg, c) for m in need], ignore_index=True)
        for d in days:
            idx = delivery_index(d - dt.timedelta(days=1), cfg)
            a = _aggregate(previous_runs_day(model, d, cfg, prev), c, model).reindex(idx)
            b = _aggregate(fetch_run(model, select_run(model, d - dt.timedelta(days=1), cfg), cfg, c), c, model).reindex(idx)
            for col in a.columns:
                rows.append({"model": model, "column": col, "day": d, "abs_diff": float((a[col] - b[col]).abs().mean()),
                             "mean": float(b[col].abs().mean())})
    return pd.DataFrame(rows)
