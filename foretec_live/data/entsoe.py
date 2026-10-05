"""ENTSO-E Transparency Platform, parsed directly from the XML (ENTSOE_TOKEN).

  price  A44 day-ahead prices, in_Domain = out_Domain = bidding zone. When a period is published at
         several resolutions (15-min SDAC prices next to an hourly series) the finest one is kept.
  wind   A75 actual generation per type, process A16, psrType B19 (onshore) + B18 (offshore).
  solar  A75, psrType B16.
  load   A65 actual total load, process A16, outBiddingZone_Domain = bidding zone; the day-ahead total load
         forecast (process A01, by regulation published two hours before the gate) is `tso_forecast(zone, "load")`.

A75 documents can carry generation (inBiddingZone_Domain) and consumption (outBiddingZone_Domain)
series for the same type; only generation is used. A03 curves leave out positions whose value
repeats the previous one, so missing positions are filled forward within their period.

We parse the XML ourselves instead of using entsoe-py, which flattens multi-period documents.
Chunks align to fixed Mondays like the Energy-Charts source, and finished chunks are cached on disk.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import Source, to_15min

log = logging.getLogger(__name__)
API = "https://web-api.tp.entsoe.eu/api"
CHUNK_DAYS = 7
ANCHOR = dt.date(2020, 1, 6)
SETTLED_DAYS = 5   # actuals can still be revised for a few days; only older chunks are cached
PSR = {"wind": ["B19", "B18"], "solar": ["B16"]}


def parse_timeseries(xml: str) -> list[tuple[dict, pd.Series]]:
    """Every TimeSeries in a document as (metadata, UTC-naive series at its own resolution)."""
    xml = re.sub(r' xmlns(:\w+)?="[^"]+"', "", xml)
    root = ET.fromstring(xml)
    out = []
    for ts in root.iter("TimeSeries"):
        meta = {
            "in_domain": (ts.findtext("inBiddingZone_Domain.mRID") or ts.findtext("in_Domain.mRID") or "").strip(),
            "out_domain": (ts.findtext("outBiddingZone_Domain.mRID") or ts.findtext("out_Domain.mRID") or "").strip(),
            "psr": (ts.findtext("MktPSRType/psrType") or "").strip(),
            "curve": (ts.findtext("curveType") or "A01").strip(),
        }
        for period in ts.iter("Period"):
            start = pd.Timestamp(period.findtext("timeInterval/start")).tz_convert("UTC").tz_localize(None)
            end = pd.Timestamp(period.findtext("timeInterval/end")).tz_convert("UTC").tz_localize(None)
            res = pd.Timedelta(period.findtext("resolution"))
            n = int((end - start) / res)
            vals = np.full(n, np.nan)
            for pt in period.iter("Point"):
                q = pt.findtext("quantity") or pt.findtext("price.amount")
                vals[int(pt.findtext("position")) - 1] = float(q)
            s = pd.Series(vals, index=pd.date_range(start, periods=n, freq=res))
            if meta["curve"] == "A03":
                s = s.ffill()
            out.append((dict(meta, resolution=res), s))
    return out


def _finest(parts: list[tuple[dict, pd.Series]]) -> pd.Series:
    """Combine periods; where resolutions overlap, the finest one wins."""
    if not parts:
        return pd.Series(dtype=float)
    parts = sorted(parts, key=lambda p: p[0]["resolution"], reverse=True)   # coarse first, fine overwrites
    out = pd.Series(dtype=float)
    for _, s in parts:
        s15 = to_15min(s)
        out = s15.combine_first(out) if not out.empty else s15
    return out.sort_index()


class EntsoeSource(Source):
    name = "entsoe"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.token = os.environ.get("ENTSOE_TOKEN") or os.environ.get("ENTSOE_API_KEY")
        if not self.token:
            raise RuntimeError("ENTSOE_TOKEN is not set")
        self._mem: dict = {}
        self._disk = Path(cfg["_root"]) / "data" / "cache" / "entsoe" if "_root" in cfg else None

    def _get(self, params: dict) -> str:
        q = {"securityToken": self.token, **params}
        for attempt in range(5):
            try:
                r = requests.get(API, params=q, timeout=90)
            except requests.RequestException as e:
                log.warning("entsoe request failed (%s), retrying", e)
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200 or "No matching data" in r.text:
                return r.text
            if r.status_code in (429, 503):
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"entsoe HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("entsoe: no answer after retries")

    def _chunk(self, kind: str, zone: str, psr: str, a: dt.date, b: dt.date) -> pd.Series:
        key = (kind, zone, psr, a)
        if key in self._mem:
            return self._mem[key]
        settled = b <= dt.date.today() - dt.timedelta(days=SETTLED_DAYS)   # TSO forecasts too: re-fetched until settled
        f = self._disk / f"{kind}_{zone}_{psr or 'all'}_{a}.parquet" if self._disk is not None else None
        if f is not None and settled and f.exists():
            s = pd.read_parquet(f)["value"]
            self._mem[key] = s
            return s
        eic = self.cfg["zones"][zone]["entsoe"]
        period = {"periodStart": f"{a:%Y%m%d}0000", "periodEnd": f"{b:%Y%m%d}0000"}
        if kind == "price":
            xml = self._get({"documentType": "A44", "in_Domain": eic, "out_Domain": eic,
                             "contract_MarketAgreement.type": "A01", **period})
            parts = [] if "No matching data" in xml else parse_timeseries(xml)
        elif kind == "tso":
            xml = self._get({"documentType": "A69", "processType": "A01", "in_Domain": eic, "psrType": psr, **period})
            parts = [] if "No matching data" in xml else parse_timeseries(xml)
        elif kind in ("load", "loadfc"):   # A65 total load: actual (A16) or day-ahead forecast (A01)
            xml = self._get({"documentType": "A65", "processType": "A16" if kind == "load" else "A01",
                             "outBiddingZone_Domain": eic, **period})
            parts = [] if "No matching data" in xml else parse_timeseries(xml)
        else:
            xml = self._get({"documentType": "A75", "processType": "A16", "in_Domain": eic, "psrType": psr, **period})
            parts = [] if "No matching data" in xml else parse_timeseries(xml)
            parts = [p for p in parts if p[0]["in_domain"] and not p[0]["out_domain"]]   # generation only
        s = _finest(parts)
        if f is not None and settled and not s.empty:
            f.parent.mkdir(parents=True, exist_ok=True)
            s.rename("value").to_frame().to_parquet(f)
        self._mem[key] = s
        return s

    def _chunks(self, start: pd.Timestamp, end: pd.Timestamp):
        first = (start - pd.Timedelta(days=1)).date()
        last = (end + pd.Timedelta(days=1)).date()
        d0 = ANCHOR + dt.timedelta(days=((first - ANCHOR).days // CHUNK_DAYS) * CHUNK_DAYS)
        while d0 < last:
            yield d0, d0 + dt.timedelta(days=CHUNK_DAYS)
            d0 += dt.timedelta(days=CHUNK_DAYS)

    def tso_forecast(self, zone: str, target: str, start, end) -> pd.Series:
        """TSO day-ahead forecast as published at the time of the call: A69 for wind or solar, A65 for load."""
        if target == "load":
            parts = [self._chunk("loadfc", zone, "", a, b) for a, b in self._chunks(start, end)]
            parts = [p for p in parts if not p.empty]
            if not parts:
                return pd.Series(dtype=float)
            s = pd.concat(parts).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            return s[(s.index >= start) & (s.index < end)]
        comps = []
        for psr in PSR[target]:
            parts = [self._chunk("tso", zone, psr, a, b) for a, b in self._chunks(start, end)]
            parts = [p for p in parts if not p.empty]
            if parts:
                c = pd.concat(parts).sort_index()
                comps.append(c[~c.index.duplicated(keep="last")])
        if not comps:
            return pd.Series(dtype=float)
        s = pd.concat(comps, axis=1).sum(axis=1, min_count=len(comps))
        return s[(s.index >= start) & (s.index < end)]

    def fetch(self, zone, target, start, end):
        parts = []
        for a, b in self._chunks(start, end):
            if target == "price":
                parts.append(self._chunk("price", zone, "", a, b))
            elif target == "load":
                parts.append(self._chunk("load", zone, "", a, b))
            else:
                comps = [self._chunk("gen", zone, psr, a, b) for psr in PSR[target]]
                comps = [c for c in comps if not c.empty]    # e.g. no offshore wind in a zone
                if comps:
                    # a quarter-hour counts only when every published component has it (offshore often
                    # lags onshore); incomplete ones stay missing for the fallback to fill
                    parts.append(pd.concat(comps, axis=1).sum(axis=1, min_count=len(comps)))
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.Series(dtype=float)
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s[(s.index >= start) & (s.index < end)]
