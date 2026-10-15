"""Imbalance prices per zone, for valuing forecasts (the Forecast Value page).

ENTSO-E Transparency Platform, Imbalance Prices (17.1.G, document A85), 15-minute settlement periods. Every zone
publishes two series: the price for a balance-responsible party that is long (surplus, category A04) and short
(deficit, A05). Under single pricing (BE, DE, FR, DK1, DK2) they are equal; NL can differ when both directions are
activated in a period. The reply is a zip of XML documents. Positions left out of a curve repeat the previous one.
The German imbalance price (reBAP) is national, published for the country area 10Y1001A1001A83F.

Stored per delivery day: data/market/imbalance/<D>/<zone>.parquet, columns long, short (EUR/MWh, UTC-naive index).
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .timeutil import delivery_index

log = logging.getLogger(__name__)
API = "https://web-api.tp.entsoe.eu/api"


def _area(cfg, zone: str) -> str:
    z = cfg["zones"][zone]
    return z.get("imbalance_area") or z["entsoe"]


def _path(cfg, zone: str, delivery: dt.date) -> Path:
    return Path(cfg["_root"]) / "data" / "market" / "imbalance" / str(delivery) / f"{zone}.parquet"


def _request(params: dict) -> bytes:
    for attempt in range(5):
        try:
            r = requests.get(API, params=params, timeout=90)
        except requests.RequestException as e:
            log.warning("entsoe imbalance request failed (%s), retrying", e)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code in (400, 429, 503, 599) and attempt < 4 and b"No matching data" not in r.content:
            time.sleep(15 * (attempt + 1))
            continue
        return r.content   # "No matching data" and other refusals: no prices
    return b""


def fetch_imbalance(cfg, zone: str, delivery: dt.date) -> pd.DataFrame:
    """Long and short imbalance prices for the quarter-hours of delivery day D."""
    token = os.environ.get("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN is not set")
    idx = delivery_index(delivery - dt.timedelta(days=1), cfg)
    raw = _request({"securityToken": token, "documentType": "A85", "controlArea_Domain": _area(cfg, zone),
                    "periodStart": f"{idx[0]:%Y%m%d%H%M}", "periodEnd": f"{idx[-1] + pd.Timedelta('15min'):%Y%m%d%H%M}"})
    docs = []
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            docs = [z.read(n).decode("utf-8", "replace") for n in z.namelist()]
    elif raw.lstrip().startswith(b"<"):
        docs = [raw.decode("utf-8", "replace")]
    out = {"long": [], "short": []}
    for xml in docs:
        root = ET.fromstring(re.sub(r' xmlns(:\w+)?="[^"]+"', "", xml))
        for period in root.iter("Period"):
            start = pd.Timestamp(period.findtext("timeInterval/start")).tz_convert("UTC").tz_localize(None)
            end = pd.Timestamp(period.findtext("timeInterval/end")).tz_convert("UTC").tz_localize(None)
            res = pd.Timedelta(period.findtext("resolution"))
            n = int((end - start) / res)
            sides = {"A04": np.full(n, np.nan), "A05": np.full(n, np.nan)}
            for pt in period.iter("Point"):
                cat, amount = pt.findtext("imbalance_Price.category"), pt.findtext("imbalance_Price.amount")
                if cat in sides and amount is not None:
                    sides[cat][int(pt.findtext("position")) - 1] = float(amount)
            grid = pd.date_range(start, periods=n, freq=res)
            for cat, name in (("A04", "long"), ("A05", "short")):
                v = pd.Series(sides[cat], index=grid)
                if v.notna().any():
                    out[name].append(v.ffill())          # a left-out position repeats the previous one (A03 curve)
    out = {k: pd.concat(v).sort_index().groupby(level=0).last().resample("15min").ffill() for k, v in out.items() if v}
    df = pd.DataFrame(out).reindex(idx)
    return df


def update_imbalance(cfg, first: dt.date, last: dt.date) -> list[str]:
    """Fetch and store imbalance prices for delivery days first..last that are missing or incomplete."""
    written = []
    d = first
    while d <= last:
        for zone in cfg["zones"]:
            f = _path(cfg, zone, d)
            if f.exists() and pd.read_parquet(f).notna().all(axis=1).mean() >= 0.99:
                continue
            try:
                df = fetch_imbalance(cfg, zone, d)
            except Exception as e:  # noqa: BLE001 - value is extra; never fail a scoring run on it
                log.warning("imbalance %s %s: %s", zone, d, str(e)[:150])
                continue
            if df.notna().all(axis=1).mean() >= 0.95:
                f.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(f)
                written.append(f"{d} {zone}")
        d += dt.timedelta(days=1)
    return written


def load_imbalance(cfg, zone: str, delivery: dt.date) -> pd.DataFrame | None:
    f = _path(cfg, zone, delivery)
    return pd.read_parquet(f) if f.exists() else None


# ---------- forecast value: a producer sells its forecast on the day-ahead market ----------
# For each quarter-hour the producer commits c MWh day-ahead (its forecast, or a percentile of its forecast
# distribution) and is paid c x day-ahead price. The deviation g - c is settled at the imbalance price: a surplus
# (g > c, long) is paid the long price, a deficit (g < c, short) costs the short price. With perfect foresight
# c = g and the revenue is g x day-ahead price. Price taker: the bid does not move the price.
PERCENTILES = list(range(10, 95, 5))
QCOLS = [f"q0.{k}" for k in range(1, 10)]
VALUE_TARGETS = ("wind", "solar")


def _committed(fc: pd.DataFrame) -> dict[str, np.ndarray]:
    """Quantity committed per strategy: 'point', and 'p10'..'p90' interpolated between the published deciles."""
    out = {"point": fc["point"].to_numpy(float)}
    if all(c in fc for c in QCOLS) and fc[QCOLS].notna().all(axis=None):
        q = np.sort(fc[QCOLS].to_numpy(float), axis=1)          # a crossed quantile pair is read in order
        levels = np.arange(10, 100, 10)
        for p in PERCENTILES:
            j = min(np.searchsorted(levels, p, side="right") - 1, len(levels) - 2)
            w = (p - levels[j]) / 10
            out[f"p{p}"] = q[:, j] * (1 - w) + q[:, j + 1] * w
    return {k: np.clip(v, 0, None) for k, v in out.items()}


def _capacity(cfg, key: str, d: dt.date) -> float | None:
    """Installed capacity (MW) of series key on day d: the registered series, else the configured value."""
    f = Path(cfg["_root"]) / "data" / "static" / "capacity" / f"{key}.parquet"
    if f.exists():
        s = pd.read_parquet(f)["mw"]
        s = s[s.index <= pd.Timestamp(d)]
        if len(s):
            return float(s.iloc[-1])
    return ((cfg.get("value") or {}).get("capacity_mw") or {}).get(key)


def _value_day(cfg, issue_date: str, phase: str, folder: Path, scored: dict) -> dict:
    """series -> {perfect, energy, models: {model: {strategy: revenue}}} for one issue day (EUR, MWh), for a virtual
    asset of value.asset_mw: the zone's output and the forecasts scaled by asset_mw / installed capacity."""
    from .config import paths
    from .timeutil import delivery_date

    p = paths(cfg)
    d = delivery_date(issue_date)
    idx = delivery_index(issue_date, cfg)
    h = 0.25                                                     # MWh per MW in a quarter-hour
    out = {}
    fcs = {}
    for zone in cfg["zones"]:
        da_f = p["actuals"] / str(d) / f"{zone}_price.parquet"
        imb = load_imbalance(cfg, zone, d)
        if not da_f.exists() or imb is None:
            continue
        da = pd.read_parquet(da_f)["value"].reindex(idx).ffill().to_numpy(float)
        imb = imb.reindex(idx).ffill()
        p_long, p_short = imb["long"].to_numpy(float), imb["short"].to_numpy(float)
        if not (np.isfinite(da).all() and np.isfinite(p_long).all() and np.isfinite(p_short).all()):
            continue
        for target in VALUE_TARGETS:
            g_f = p["actuals"] / str(d) / f"{zone}_{target}.parquet"
            models = scored.get((zone, target), [])
            if not g_f.exists() or not models:
                continue
            g = pd.read_parquet(g_f)["value"].reindex(idx).to_numpy(float)
            if not np.isfinite(g).all():
                continue
            cap = _capacity(cfg, f"{zone}_{target}", d)
            if not cap:
                continue
            k = (cfg.get("value") or {}).get("asset_mw", 100) / cap
            g = np.clip(g, 0, None) * k
            ser = {"perfect": round(float(h * (g * da).sum()), 1), "energy": round(float(h * g.sum()), 1), "models": {}}
            for model in sorted(models):
                f = folder / f"{model}.parquet"
                if not f.exists():
                    continue
                if model not in fcs:
                    fcs[model] = pd.read_parquet(f)
                fc = fcs[model]
                fc = fc[(fc["zone"] == zone) & (fc["target"] == target)].set_index("delivery_utc").reindex(idx)
                if fc["point"].isna().any():
                    continue
                rev = {}
                for strat, c in _committed(fc).items():
                    c = c * k
                    dev = g - c
                    rev[strat] = round(float(h * (c * da + np.where(dev > 0, dev * p_long, dev * p_short)).sum()), 1)
                ser["models"][model] = rev
            out[f"{zone}_{target}"] = ser
    return out


def value_export(cfg, days: dict, scored: set, hidden: set) -> dict:
    """Per series and issue day: perfect-foresight revenue, energy, and each model's revenue per strategy (EUR).

    days: issue_date -> (phase, folder) as the site uses them; scored: {(issue_date, model, zone, target)} that
    have a score row (so voided, excluded and not-yet-final forecasts are left out, exactly as on the leaderboard).
    One cache file per issue day (data/market/value/), recomputed when its scored set or any input file changes.
    """
    import hashlib
    import json

    from .config import paths
    from .timeutil import delivery_date

    p = paths(cfg)
    cache = Path(cfg["_root"]) / "data" / "market" / "value"
    cache.mkdir(parents=True, exist_ok=True)
    by_day: dict = {}
    for (i, m, z, t) in scored:
        if t in VALUE_TARGETS and m not in hidden:
            by_day.setdefault(i, {}).setdefault((z, t), []).append(m)
    out = {"percentiles": PERCENTILES, "asset_mw": (cfg.get("value") or {}).get("asset_mw", 100), "series": {}}
    for issue_date, (phase, folder) in sorted(days.items()):
        sc = by_day.get(issue_date)
        if not sc:
            continue
        d = str(delivery_date(issue_date))
        inputs = [*folder.glob("*.parquet"), *(p["actuals"] / d).glob("*.parquet"),
                  *(Path(cfg["_root"]) / "data" / "market" / "imbalance" / d).glob("*.parquet")]
        sig = hashlib.sha1(json.dumps([sorted((z, t, sorted(ms)) for (z, t), ms in sc.items()), phase, cfg.get("value"),
                                       [str(f.stat().st_mtime) for f in (Path(cfg["_root"]) / "data" / "static" / "capacity").glob("*.parquet")],
                                       sorted((str(f), f.stat().st_mtime) for f in inputs)]).encode()).hexdigest()
        cf = cache / f"{issue_date}.json"
        day = None
        if cf.exists():
            c = json.loads(cf.read_text())
            day = c["series"] if c.get("sig") == sig else None
        if day is None:
            day = _value_day(cfg, issue_date, phase, folder, sc)
            cf.write_text(json.dumps({"sig": sig, "series": day}))
        for key, v in day.items():
            ser = out["series"].setdefault(key, {"days": [], "phase": [], "perfect": [], "energy": [], "models": {}})
            k = len(ser["days"])
            ser["days"].append(issue_date)
            ser["phase"].append(phase)
            ser["perfect"].append(v["perfect"])
            ser["energy"].append(v["energy"])
            for model, rev in v["models"].items():
                m = ser["models"].setdefault(model, {})
                for strat, r in rev.items():
                    m.setdefault(strat, {})[k] = r
    # sparse {day position: revenue} -> dense lists with null where the model did not deliver
    for ser in out["series"].values():
        n = len(ser["days"])
        for model, strats in ser["models"].items():
            ser["models"][model] = {s: [v.get(i) for i in range(n)] for s, v in strats.items()}
    return out
