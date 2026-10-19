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
and TTF gas and EUA carbon from Yahoo Finance daily closes (last close known before the cut-off).

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


def _zone_parts(c: pd.DataFrame) -> dict[str, np.ndarray]:
    """Positions of each zone's centroids in c."""
    return {z: np.flatnonzero(c["zone"].to_numpy() == z) for z in dict.fromkeys(c["zone"])}


def cached_per_zone(c: pd.DataFrame, path_of, fetch) -> pd.DataFrame:
    """Long-format weather (column `centroid` = position in c) cached per zone: path_of(c_zone) is a zone's cache
    file, fetch(c_subset) requests the zones still missing, in one go. Adding a zone then fetches only that zone,
    and the other zones' caches stay valid (they are keyed by their own centroids)."""
    parts, missing = {}, []
    for z, pos in _zone_parts(c).items():
        f = path_of(c.iloc[pos].reset_index(drop=True))
        if f.exists():
            parts[z] = pd.read_parquet(f)
        else:
            missing.append(z)
    if missing:
        cm = c[c["zone"].isin(missing)].reset_index(drop=True)
        raw = fetch(cm)
        for z, pos in _zone_parts(cm).items():
            sub = raw[raw["centroid"].isin(pos)].copy()
            sub["centroid"] = sub["centroid"].map({p: i for i, p in enumerate(pos)})
            parts[z] = sub
            f = path_of(cm.iloc[pos].reset_index(drop=True))
            if f is not None and getattr(raw, "attrs", {}).get("cache", True):
                f.parent.mkdir(parents=True, exist_ok=True)
                sub.to_parquet(f, index=False)
                if raw.attrs.get("meta"):
                    f.with_suffix(".json").write_text(json.dumps(raw.attrs["meta"]))
    out = []
    for z, pos in _zone_parts(c).items():
        sub = parts[z].copy()
        sub["centroid"] = pos[sub["centroid"].to_numpy()]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def run_meta(cfg, c: pd.DataFrame, model: str, run: pd.Timestamp) -> dict:
    """When a run was fetched (the first zone's cache record)."""
    pos = next(iter(_zone_parts(c).values()))
    f = _nwp_cache(cfg, c.iloc[pos].reset_index(drop=True), model, run).with_suffix(".json")
    return json.loads(f.read_text()) if f.exists() else {}


def fetch_run(model: str, run: pd.Timestamp, cfg, c: pd.DataFrame) -> pd.DataFrame:
    """All centroids of c for one model run, long format. Cached per zone, forever: a run never changes."""
    return cached_per_zone(c, lambda cz: _nwp_cache(cfg, cz, model, run), lambda cm: _request_run(model, run, cfg, cm))


def _missing_values(got: list, variables: list, run: pd.Timestamp, min_share: float = 0.99) -> str:
    """Why a reply is not usable although every location has a time axis: a variable mostly null over lead times
    12-48 h, which hold delivery day D for every run the cut-off admits. Open-Meteo publishes a run before it has
    ingested all of it: ICON-EU 06 UTC at the 11:30 cut-off came with timestamps but null cloud, temperature,
    irradiance, 10 m wind and gusts on 3, 4, 5, 10, 13 and 17 Oct 2026, and was complete an hour later."""
    t = pd.to_datetime(got[0]["minutely_15"]["time"])
    win = np.flatnonzero((t >= run + pd.Timedelta(hours=12)) & (t < run + pd.Timedelta(hours=48)))
    if not len(win):
        return ""
    short = {}
    for v in variables:
        vals = [loc["minutely_15"].get(v) or [] for loc in got]
        ok = sum(sum(1 for i in win if i < len(x) and x[i] is not None) for x in vals)
        share = ok / (len(win) * len(got))
        if share < min_share:
            short[v] = round(share, 2)
    return f"incomplete values over lead 12-48 h (non-null share {short})" if short else ""


def _request_run(model: str, run: pd.Timestamp, cfg, c: pd.DataFrame) -> pd.DataFrame:
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
    js, why = None, ""
    for attempt in range(5):
        # a 200 is not enough: on 2026-10-06 Open-Meteo answered 200 with a body that was not JSON
        try:
            r = requests.get(host, params=params, timeout=120)
            if r.status_code == 200:
                got = r.json()
                got = got if isinstance(got, list) else [got]
                if len(got) == len(c) and all((loc.get("minutely_15") or {}).get("time") for loc in got):
                    why = _missing_values(got, variables, run)
                    if not why:
                        js = got
                        break
                else:
                    why = f"incomplete reply ({len(got)} of {len(c)} locations with data)"
            else:
                why = f"HTTP {r.status_code} {r.text[:200]}"
                if r.status_code == 400 and "not available" in r.text:   # the archive lacks this run: retrying won't help
                    break
        except (requests.RequestException, ValueError) as e:   # ValueError: the body is not JSON
            why = f"{type(e).__name__}: {e}"
        log.warning("open-meteo %s %s (try %d): %s", model, run, attempt + 1, why)
        if attempt < 4:
            time.sleep(10 * (attempt + 1))
    if js is None:
        raise RuntimeError(f"no valid reply after 5 tries, last: {why}"[:300])
    frames = []
    for i, loc in enumerate(js):
        m = loc["minutely_15"]
        d = pd.DataFrame({v: pd.to_numeric(pd.Series(m.get(v)), errors="coerce") for v in variables})
        d["time_utc"] = pd.to_datetime(m["time"])
        d["centroid"] = i
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.attrs = {"meta": {"model": model, "run_utc": str(run), "fetched_utc": str(pd.Timestamp.now(tz="UTC"))}}
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
YAHOO_FUEL = {"TTF_EUR": "TTF=F",      # ICE Dutch TTF natural gas, front-month future, EUR/MWh
              "CARBON_EUR": "CO2.L"}   # SparkChange physical carbon EUA ETC, EUR (tracks the EUA price)


def _yahoo_daily(symbol: str, rng: str) -> list[dict]:
    """Daily closes from Yahoo Finance's chart API, stamped with the exchange date."""
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", params={"range": rng, "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    days = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    return [{"time_utc": d, "price": float(c)} for d, c in zip(days, res["indicators"]["quote"][0]["close"]) if c is not None]


def fuel_quotes(cfg, upto: pd.Timestamp) -> pd.DataFrame:
    """TTF gas (EUR/MWh) and EUA carbon (EUR/t) daily closes from Yahoo Finance, known at or before `upto`.

    Kept in a permanent cache, topped up at most twice a day (one request per series, the last month; five years on
    the first fetch). No key, no quota.
    """
    cache = Path(cfg["_root"]) / "data" / "cache" / "fuel_quotes.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame({"time_utc": pd.Series(dtype="datetime64[ns]"), "code": [], "price": [], "source": []})
    if "source" not in have or (have["source"] != "yahoo").any():       # quotes from the previous provider: start over
        have = have.iloc[0:0].assign(source=pd.Series(dtype=str))
    fresh = cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600 and len(have)
    if not fresh:
        rows = []
        for code, symbol in YAHOO_FUEL.items():
            try:
                rng = "1mo" if (have["code"] == code).sum() > 500 else "5y"
                rows += [{**x, "code": code, "source": "yahoo"} for x in _yahoo_daily(symbol, rng)]
            except Exception as e:  # keep what we have; the next run tops up
                log.warning("yahoo %s: %s", symbol, str(e)[:150])
        if rows:
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
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
        meta = json.loads(fm.read_text())
        if meta.get("zones", ["BE", "NL", "FR"]) == list(cfg["zones"]):   # built for another set of zones: rebuild
            return pd.read_parquet(f), meta
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
            cycle = pd.Timedelta(hours=int(cfg["covariates"]["nwp"]["models"][model]["cycle_hours"]))
            raw, used, errors = None, run, []
            # the latest admissible run; if it cannot be fetched, up to two earlier cycles (published even earlier,
            # so just as admissible) rather than an ensemble with a member missing
            for back in range(3):
                cand = run - back * cycle
                try:
                    raw, used = fetch_run(model, cand, cfg, c), cand
                    break
                except Exception as e:  # noqa: BLE001 - try the previous cycle
                    errors.append(f"{cand:%H} UTC run: {e}"[:200])
            if raw is None:
                log.error("NWP %s run %s unavailable: %s", model, run, "; ".join(errors))
                runs_used.append({"issue_date": str(d), "model": model, "run_utc": str(run), "status": f"missing: {'; '.join(errors)}"[:300]})
                continue
            if used != run:
                log.warning("NWP %s: run %s unavailable, using %s (%s)", model, run, used, errors[0])
            agg = _aggregate(raw, c, model).reindex(idx)
            wide.append(agg)
            fetched = run_meta(cfg, c, model, used).get("fetched_utc")
            status = "ok" if used == run else f"fallback: the {run:%H} UTC run was unavailable ({errors[0][:120]})"
            runs_used.append({"issue_date": str(d), "model": model, "run_utc": str(used), "fetched_utc": fetched, "status": status})
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
            # each quarter-hour gets the last quote known at the cut-off of its own issue day (D-1, CET/CEST)
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
        "zones": list(cfg["zones"]),
        "load_forecast": "ENTSO-E A65 day-ahead total load forecast (published by 10:00 CET/CEST on D-1 by regulation)"
                         if os.environ.get("ENTSOE_TOKEN") else "not configured",
        "fuel": "Yahoo Finance daily closes, TTF=F (ICE Dutch TTF front month) and CO2.L (EUA ETC); at each cut-off the last quote dated two days before delivery or earlier" if not fuel.empty else "unavailable",
        "columns": len(cov.columns),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(f)
    fm.write_text(json.dumps(meta, indent=1))
    return cov, meta


def split_combined_cache(cfg, c: pd.DataFrame) -> int:
    """One-off: split cache files written for the whole centroid set c (before per-zone caching) into per-zone
    files, for single runs (data/cache/nwp) and previous runs (data/cache/nwp_prev). Returns files written."""
    n, h = 0, _centroid_hash(c)
    for kind in ("nwp", "nwp_prev"):
        for f in sorted((Path(cfg["_root"]) / "data" / "cache" / kind / h).glob("*/*.parquet")):
            raw = pd.read_parquet(f)
            meta = f.with_suffix(".json")
            for z, pos in _zone_parts(c).items():
                out = Path(cfg["_root"]) / "data" / "cache" / kind / _centroid_hash(c.iloc[pos].reset_index(drop=True)) / f.parent.name / f.name
                if out.exists():
                    continue
                sub = raw[raw["centroid"].isin(pos)].copy()
                sub["centroid"] = sub["centroid"].map({p: i for i, p in enumerate(pos)})
                out.parent.mkdir(parents=True, exist_ok=True)
                sub.to_parquet(out, index=False)
                if meta.exists():
                    out.with_suffix(".json").write_text(meta.read_text())
                n += 1
    return n
