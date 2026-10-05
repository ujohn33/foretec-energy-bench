"""Energy-Charts API (Fraunhofer ISE). Public, no key, CC BY 4.0.

Endpoints used:
  /price?bzn=BE&start=...&end=...            -> {"unix_seconds": [...], "price": [...]}
  /public_power?country=be&start=...&end=... -> {"unix_seconds": [...], "production_types": [{"name": ..., "data": [...]}]}

Run `foretec-live probe` on the server first: it prints what the API returns, so field names
and resolution can be checked before the first live run.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .base import Source, to_15min

log = logging.getLogger(__name__)
BASE = "https://api.energy-charts.info"
CHUNK_DAYS = 7
ANCHOR = dt.date(2020, 1, 6)  # chunks start on fixed Mondays so repeated requests hit the cache

WIND_NAMES = {"wind onshore", "wind offshore"}
SOLAR_NAMES = {"solar"}
LOAD_NAMES = {"load", "load (incl. self-consumption)"}


MIN_INTERVAL = 4.0  # seconds between requests; the public API answers bursts with 429
SETTLED_DAYS = 3  # chunks that ended this long ago no longer change and are cached on disk
_last_request = 0.0


def _get(path: str, params: dict, retries: int = 7) -> dict:
    global _last_request
    for attempt in range(retries):
        wait = _last_request + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            r = requests.get(BASE + path, params=params, timeout=60)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else min(15 * 2**attempt, 300)
                if attempt == retries - 1:
                    r.raise_for_status()
                log.warning("energy-charts %s rate limited, waiting %.0fs", path, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:  # pragma: no cover - network
            if attempt == retries - 1:
                raise
            log.warning("energy-charts %s failed (%s), retrying", path, e)
            time.sleep(5 * (attempt + 1))
    return {}


def _series(unix_seconds, values) -> pd.Series:
    idx = pd.to_datetime(pd.Series(unix_seconds, dtype="int64"), unit="s")
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.Series(vals.values, index=pd.DatetimeIndex(idx.values))


def _pick(types: list[dict], names: set[str]) -> list[dict]:
    exact = [t for t in types if t.get("name", "").strip().lower() in names]
    if exact:
        return exact
    key = next(iter(names)).split()[0]
    return [t for t in types if key in t.get("name", "").lower() and "share" not in t.get("name", "").lower()]


class EnergyChartsSource(Source):
    name = "energycharts"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._cache: dict = {}  # per process: one forecast run or one backfill
        self._disk = Path(cfg["_root"]) / "data" / "cache" / "energycharts" if "_root" in cfg else None

    def _get(self, path: str, params: dict) -> dict:
        key = (path, tuple(sorted(params.items())))
        if key in self._cache:
            return self._cache[key]
        settled = dt.date.fromisoformat(params["end"]) <= dt.date.today() - dt.timedelta(days=SETTLED_DAYS)
        f = None
        if self._disk is not None and settled:
            f = self._disk / (path.strip("/") + "_" + "_".join(f"{k}-{v}" for k, v in key[1]) + ".json")
            if f.exists():
                self._cache[key] = json.loads(f.read_text())
                return self._cache[key]
        # as a fallback it must fail fast: a slow fallback could push the live run past the gate
        js = _get(path, params, retries=2 if self.cfg.get("_fallback") else 7)
        if f is not None and js:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(js))
        self._cache[key] = js
        return js

    def _chunks(self, start: pd.Timestamp, end: pd.Timestamp):
        first = (start - pd.Timedelta(days=1)).date()
        last = (end + pd.Timedelta(days=1)).date()
        d0 = ANCHOR + dt.timedelta(days=((first - ANCHOR).days // CHUNK_DAYS) * CHUNK_DAYS)
        while d0 < last:
            d_next = d0 + dt.timedelta(days=CHUNK_DAYS)
            yield d0.isoformat(), d_next.isoformat()
            d0 = d_next

    def fetch(self, zone, target, start, end):
        z = self.cfg["zones"][zone]
        parts = []
        for a, b in self._chunks(start, end):
            if target == "price":
                js = self._get("/price", {"bzn": z["energycharts_bzn"], "start": a, "end": b})
                parts.append(_series(js.get("unix_seconds", []), js.get("price", [])))
            else:
                js = self._get("/public_power", {"country": z["energycharts_country"], "start": a, "end": b})
                types = js.get("production_types", [])
                if target == "load":   # exact names only: the fuzzy match would also take "Residual load"
                    chosen = [t for t in types if t.get("name", "").strip().lower() in LOAD_NAMES][:1]
                else:
                    chosen = _pick(types, WIND_NAMES if target == "wind" else SOLAR_NAMES)
                if not chosen:
                    log.error("no %s series for %s; available: %s", target, zone, [t.get("name") for t in types])
                    continue
                total = None
                for t in chosen:
                    s = _series(js["unix_seconds"], t["data"])
                    total = s if total is None else total.add(s, fill_value=0)
                parts.append(total)
        if not parts:
            return pd.Series(dtype=float)
        s = to_15min(pd.concat(parts))
        return s[(s.index >= start) & (s.index < end)]
