"""Covariates: what a forecaster could know at the cut-off, frozen per issue day.

NWP ensemble: ICON-EU, GFS and ECMWF IFS from the Open-Meteo single-runs API, each run picked as the
latest initialisation published before the cut-off (init <= cut-off - lag). Values are fetched at
capacity-weighted centroids (geo/centroids.csv, weighted k-means as in the predico models) and
aggregated per zone and bucket (wind_onshore, wind_offshore, solar, load). Across the models we add
the ensemble mean, std, min, max, range, coefficient of variation and pairwise differences.

Training rows get the same vintage as the forecast: values for delivery day D' come from the runs
that were available at the cut-off of issue day D'-1. So a model trained on the history window sees
features with the same lead time and the same NWP models as on the day it forecasts.

Also: ENTSO-E day-ahead total load forecast (A65, must be published 2 h before the 12:00 gate)
and, when OIL_PRICE_KEY is set, TTF gas and EUA carbon (last quote before the cut-off).

Output per issue day: data/covariates/<issue_date>.parquet (time_utc x columns "<zone>.<bucket>.<what>.<variable>")
and data/covariates/<issue_date>.json (which runs, documents and quotes were used, and when they were fetched).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import paths
from .timeutil import FREQ, as_date, delivery_index, issue_timestamp

log = logging.getLogger(__name__)

BUCKET_VARS = {
    "wind_onshore": ["wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"],
    "wind_offshore": ["wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"],
    "solar": ["global_tilted_irradiance", "shortwave_radiation", "direct_normal_irradiance",
              "diffuse_radiation", "cloud_cover", "temperature_2m"],
    "load": ["temperature_2m", "cloud_cover", "shortwave_radiation"],
}
IRRADIANCE = {"global_tilted_irradiance", "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"}
WIND = {"wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"}


# ---------------------------------------------------------------- centroids
def centroids(cfg) -> pd.DataFrame:
    f = Path(cfg["_root"]) / cfg["covariates"]["centroids"]
    c = pd.read_csv(f)
    return c[c["zone"].isin(cfg["zones"])].reset_index(drop=True)


def _centroid_hash(c: pd.DataFrame) -> str:
    return hashlib.sha1(c[["zone", "bucket", "cluster_id", "latitude", "longitude"]].to_csv(index=False).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- NWP runs
def select_run(model: str, issue_date, cfg) -> pd.Timestamp:
    """Latest initialisation of `model` published before the cut-off of `issue_date` (UTC-naive)."""
    m = cfg["covariates"]["nwp"]["models"][model]
    latest_init = issue_timestamp(issue_date, cfg) - pd.Timedelta(hours=float(m["lag_hours"]))
    return latest_init.floor(f"{int(m['cycle_hours'])}h")


def _nwp_cache(cfg, c: pd.DataFrame, model: str, run: pd.Timestamp) -> Path:
    return Path(cfg["_root"]) / "data" / "cache" / "nwp" / _centroid_hash(c) / model / f"{run:%Y%m%dT%H}.parquet"


def fetch_run(model: str, run: pd.Timestamp, cfg, c: pd.DataFrame) -> pd.DataFrame:
    """All centroids for one model run, long format. Cached forever: a run never changes."""
    f = _nwp_cache(cfg, c, model, run)
    if f.exists():
        return pd.read_parquet(f)
    ncfg = cfg["covariates"]["nwp"]
    variables = sorted({v for vs in BUCKET_VARS.values() for v in vs})
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in c["latitude"]),
        "longitude": ",".join(f"{x:.4f}" for x in c["longitude"]),
        "models": model, "run": f"{run:%Y-%m-%dT%H:%M}", "minutely_15": ",".join(variables),
        "forecast_hours": int(ncfg.get("forecast_hours", 60)), "tilt": ncfg.get("tilt", 35), "azimuth": 0,
        "timezone": "GMT",
    }
    key = os.environ.get("OPENMETEOKEY")
    host = ncfg["host"] if key else ncfg["host"].replace("customer-", "")
    if key:
        params["apikey"] = key
    for attempt in range(5):
        r = requests.get(host, params=params, timeout=120)
        if r.status_code == 200:
            break
        log.warning("open-meteo %s %s: HTTP %s %s", model, run, r.status_code, r.text[:200])
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    frames = []
    for i, loc in enumerate(js):
        m = loc["minutely_15"]
        d = pd.DataFrame({v: pd.to_numeric(pd.Series(m.get(v)), errors="coerce") for v in variables})
        d["time_utc"] = pd.to_datetime(m["time"])
        d["centroid"] = i
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.attrs = {}
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f, index=False)
    (f.with_suffix(".json")).write_text(json.dumps({"model": model, "run_utc": str(run), "fetched_utc": str(pd.Timestamp.now(tz="UTC"))}))
    return out


def _aggregate(raw: pd.DataFrame, c: pd.DataFrame, model: str) -> pd.DataFrame:
    """Capacity-weighted mean per zone and bucket: columns '<zone>.<bucket>.<model>.<var>'."""
    raw = raw.merge(c[["zone", "bucket", "weight"]], left_on="centroid", right_index=True)
    cols = {}
    for (zone, bucket), g in raw.groupby(["zone", "bucket"]):
        w = g["weight"]
        for v in BUCKET_VARS[bucket]:
            num = (g[v] * w).groupby(g["time_utc"]).sum(min_count=1)
            den = w.where(g[v].notna()).groupby(g["time_utc"]).sum().replace(0, np.nan)
            cols[f"{zone}.{bucket}.{model}.{v}"] = num / den
    return pd.DataFrame(cols).sort_index()


def _ensemble_stats(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Predico-style disagreement features across NWP models for every zone/bucket/variable."""
    out = {}
    groups = {}
    for col in wide.columns:
        zone, bucket, model, var = col.split(".")
        groups.setdefault((zone, bucket, var), {})[model] = wide[col]
    for (zone, bucket, var), members in groups.items():
        block = pd.DataFrame(members)[[m for m in models if m in members]]
        p = f"{zone}.{bucket}"
        out[f"{p}.ens_mean.{var}"] = block.mean(axis=1)
        out[f"{p}.ens_std.{var}"] = block.std(axis=1).fillna(0.0)
        out[f"{p}.ens_min.{var}"] = block.min(axis=1)
        out[f"{p}.ens_max.{var}"] = block.max(axis=1)
        out[f"{p}.ens_range.{var}"] = out[f"{p}.ens_max.{var}"] - out[f"{p}.ens_min.{var}"]
        if var in IRRADIANCE or var in WIND:
            out[f"{p}.ens_cv.{var}"] = (out[f"{p}.ens_std.{var}"] / out[f"{p}.ens_mean.{var}"].replace(0, np.nan)).fillna(0.0)
        for a, b in combinations(block.columns, 2):
            out[f"{p}.diff_{a}_{b}.{var}"] = block[a] - block[b]
    return pd.DataFrame(out)


# ---------------------------------------------------------------- ENTSO-E load forecast
def _parse_entsoe(xml: str) -> pd.Series:
    """GL_MarketDocument -> UTC-naive series. Handles A03 curves (a missing position repeats the previous one)."""
    xml = re.sub(r' xmlns="[^"]+"', "", xml, count=1)
    root = ET.fromstring(xml)
    parts = []
    for period in root.iter("Period"):
        start = pd.Timestamp(period.find("timeInterval/start").text).tz_convert("UTC").tz_localize(None)
        end = pd.Timestamp(period.find("timeInterval/end").text).tz_convert("UTC").tz_localize(None)
        res = pd.Timedelta(period.find("resolution").text)
        n = int((end - start) / res)
        vals = np.full(n, np.nan)
        for pt in period.iter("Point"):
            vals[int(pt.find("position").text) - 1] = float(pt.find("quantity").text)
        parts.append(pd.Series(vals, index=pd.date_range(start, periods=n, freq=res)).ffill())
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")].resample(FREQ).ffill()


def load_forecast(zone: str, delivery: dt.date, cfg) -> pd.Series:
    cache = Path(cfg["_root"]) / "data" / "cache" / "entsoe_load_fc" / f"{zone}_{delivery}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["value"]
    token = os.environ.get("ENTSOE_TOKEN")
    if not token:
        return pd.Series(dtype=float)
    idx = delivery_index(delivery - dt.timedelta(days=1), cfg)
    q = {"securityToken": token, "documentType": "A65", "processType": "A01",
         "outBiddingZone_Domain": cfg["zones"][zone]["entsoe"],
         "periodStart": f"{idx[0]:%Y%m%d%H%M}", "periodEnd": f"{idx[-1] + pd.Timedelta(FREQ):%Y%m%d%H%M}"}
    for attempt in range(4):
        r = requests.get("https://web-api.tp.entsoe.eu/api", params=q, timeout=60)
        if r.status_code == 200 or "No matching data" in r.text:
            break
        time.sleep(5 * (attempt + 1))
    if r.status_code != 200:
        log.warning("entsoe load forecast %s %s: HTTP %s", zone, delivery, r.status_code)
        return pd.Series(dtype=float)
    s = _parse_entsoe(r.text).reindex(idx)
    if s.notna().mean() > 0.9:  # cache only complete days
        cache.parent.mkdir(parents=True, exist_ok=True)
        s.rename("value").to_frame().to_parquet(cache)
    return s


# ---------------------------------------------------------------- fuel (optional)
_FUEL_FETCHED: set = set()   # (code, chunk start) already requested by this process


def fuel_quotes(cfg, upto: pd.Timestamp) -> pd.DataFrame:
    """TTF gas (EUR/MWh) and EUA carbon (EUR/t) quotes from oilpriceapi, stamped at or before `upto`.

    Quota-aware like the predico builder: the free plan allows 50 calls a day, so every quote is kept in a
    permanent cache, past 28-day chunks that are already covered are never fetched again, and a chunk is
    requested at most once per process.
    """
    key = os.environ.get("OIL_PRICE_KEY")
    cache = Path(cfg["_root"]) / "data" / "cache" / "fuel_quotes.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame({"time_utc": pd.Series(dtype="datetime64[ns]"), "code": [], "price": []})
    if key:
        lo = (upto - pd.Timedelta(days=cfg["context_days"] + 7)).normalize()
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        new_rows = []
        for code in ("TTF_EUR", "CARBON_EUR"):
            mine = have[have["code"] == code]["time_utc"]
            cur = lo
            while cur < upto:
                chunk_end = min(cur + pd.Timedelta(days=28), upto.normalize() + pd.Timedelta(days=1))
                covered = ((mine >= cur) & (mine < chunk_end)).sum() >= max(1, int((chunk_end - cur).days * 5 / 7 * 0.6))
                if (covered and chunk_end < now - pd.Timedelta(days=2)) or (code, cur) in _FUEL_FETCHED:
                    cur = chunk_end
                    continue
                _FUEL_FETCHED.add((code, cur))
                try:
                    r = requests.get("https://api.oilpriceapi.com/v1/prices/historical", headers={"Authorization": f"Token {key}"},
                                     params={"by_code": code, "start": f"{cur:%Y-%m-%d}", "end": f"{chunk_end:%Y-%m-%d}"}, timeout=90)
                    r.raise_for_status()
                    data = r.json().get("data") or {}
                    for x in (data.get("prices") if isinstance(data, dict) else data) or []:
                        new_rows.append({"time_utc": pd.Timestamp(x["created_at"]).tz_convert("UTC").tz_localize(None),
                                         "code": code, "price": float(x["price"])})
                except Exception as e:  # keep what we have; the next run tops up
                    log.warning("oilpriceapi %s %s: %s", code, f"{cur:%Y-%m-%d}", str(e)[:150])
                    break
                cur = chunk_end
                time.sleep(0.25)
        if new_rows:
            have = pd.concat([have, pd.DataFrame(new_rows)], ignore_index=True)
            have = have.drop_duplicates(["time_utc", "code"], keep="last").sort_values("time_utc")
            cache.parent.mkdir(parents=True, exist_ok=True)
            have.to_parquet(cache, index=False)
    have["time_utc"] = pd.to_datetime(have["time_utc"])
    # Quotes are stamped with their date only, so a quote may be that day's close: count it as known
    # from the end of its date. At the 11:30 cut-off on D-1 the latest usable quote is dated D-2.
    have["available_utc"] = have["time_utc"].dt.normalize() + pd.Timedelta(days=1)
    return have[have["available_utc"] <= upto]


# ---------------------------------------------------------------- assemble
def build_covariates(issue_date, cfg, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Covariates for one issue day: history window + delivery day, each day from its own pre-cut-off vintage."""
    issue_date = as_date(issue_date)
    out_dir = Path(cfg["_root"]) / "data" / "covariates"
    f, fm = out_dir / f"{issue_date}.parquet", out_dir / f"{issue_date}.json"
    if f.exists() and fm.exists() and not force:
        return pd.read_parquet(f), json.loads(fm.read_text())
    # every issue day must see the same kinds of inputs; never cache a frame with a source silently missing
    if not os.environ.get("ENTSOE_TOKEN"):
        raise RuntimeError("ENTSOE_TOKEN is not set: the load forecast would be missing")
    c = centroids(cfg)
    models = list(cfg["covariates"]["nwp"]["models"])
    days = [issue_date - dt.timedelta(days=k) for k in range(cfg["context_days"], -1, -1)]  # issue days of each slice
    slices, runs_used = [], []
    for d in days:
        idx = delivery_index(d, cfg)   # delivery day d+1 as seen from issue day d
        wide = []
        for model in models:
            run = select_run(model, d, cfg)
            try:
                raw = fetch_run(model, run, cfg, c)
            except Exception as e:
                log.error("NWP %s run %s unavailable: %s", model, run, e)
                runs_used.append({"issue_date": str(d), "model": model, "run_utc": str(run), "status": f"missing: {e}"[:200]})
                continue
            agg = _aggregate(raw, c, model).reindex(idx)
            wide.append(agg)
            meta_f = _nwp_cache(cfg, c, model, run).with_suffix(".json")
            fetched = json.loads(meta_f.read_text())["fetched_utc"] if meta_f.exists() else None
            runs_used.append({"issue_date": str(d), "model": model, "run_utc": str(run), "fetched_utc": fetched, "status": "ok"})
        w = pd.concat(wide, axis=1) if wide else pd.DataFrame(index=idx)
        w = pd.concat([w, _ensemble_stats(w, models)], axis=1)
        for zone in cfg["zones"]:
            w[f"{zone}.load_fc.entsoe.load_mw"] = load_forecast(zone, as_date(d) + dt.timedelta(days=1), cfg).reindex(idx).values
        slices.append(w)
    cov = pd.concat(slices).sort_index()
    cov = cov[~cov.index.duplicated(keep="last")]
    cov.index.name = "time_utc"
    fuel = fuel_quotes(cfg, issue_timestamp(issue_date, cfg))
    if not fuel.empty:
        for code, name in (("TTF_EUR", "fuel.gas.ttf.eur_mwh"), ("CARBON_EUR", "fuel.carbon.eua.eur_t")):
            q = fuel[fuel["code"] == code].set_index("available_utc")["price"].sort_index()
            # each quarter-hour gets the last quote known at the cut-off of its own issue day (D-1, Brussels)
            delivery = cov.index.tz_localize("UTC").tz_convert(cfg["timezone"]).date
            known = {}
            for d in set(delivery):
                upto_d = issue_timestamp(d - dt.timedelta(days=1), cfg)
                k = q[q.index <= upto_d]
                known[d] = k.iloc[-1] if len(k) else np.nan
            cov[name] = [known[d] for d in delivery]
        cov["fuel.ccgt.srmc.eur_mwh"] = cov["fuel.gas.ttf.eur_mwh"] / 0.52 + cov["fuel.carbon.eua.eur_t"] * 0.37
    meta = {
        "issue_date": str(issue_date),
        "cutoff_utc": str(issue_timestamp(issue_date, cfg)),
        "built_utc": str(pd.Timestamp.now(tz="UTC")),
        "nwp_models": {m: cfg["covariates"]["nwp"]["models"][m] for m in models},
        "nwp_runs": runs_used,
        "centroids_hash": _centroid_hash(c),
        "load_forecast": "ENTSO-E A65 day-ahead total load forecast (published by 10:00 Brussels on D-1 by regulation)"
                         if os.environ.get("ENTSOE_TOKEN") else "not configured",
        "fuel": "oilpriceapi TTF_EUR + CARBON_EUR daily quotes; at each cut-off the last quote dated two days before delivery or earlier" if not fuel.empty else "not configured (OIL_PRICE_KEY missing)",
        "columns": len(cov.columns),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(f)
    fm.write_text(json.dumps(meta, indent=1))
    return cov, meta
