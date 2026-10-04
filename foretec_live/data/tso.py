"""The TSOs' own open-data services, used to fill quarter-hours ENTSO-E does not deliver. No API key needed.

Elia (BE), opendata.elia.be (Opendatasoft):
  wind   ods031 (historical, `measured`) + ods086 (near real-time, `realtime`): the sum of the disjoint parts
         Federal offshore + Flanders/Wallonia onshore, each on the Elia grid and on distribution grids.
  solar  ods032 + ods087, region "Belgium".
  Elia also publishes its own day-ahead forecast as it stood at 11:00 on D-1 (`dayahead11hforecast`), before
  the 12:00 gate; the version ENTSO-E receives is the 18:00 one. `dayahead_11h` returns it.
RTE (FR), éCO2mix real-time on odre.opendatasoft.com:
  wind   `eolien` (onshore + offshore), solar `solaire`, 15-minute national values.

Each source answers only for its own zone and returns nothing elsewhere, so they can sit in one fallback chain.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from .base import Source, to_15min

log = logging.getLogger(__name__)
ELIA = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
ODRE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"


def _export(base: str, dataset: str, params: dict, retries: int) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(f"{base}/{dataset}/exports/json", params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            log.warning("%s %s failed (%s), retrying", base.split("/")[2], dataset, e)
            time.sleep(5 * (attempt + 1))
    return []


def _series(rows: list[dict], tcol: str, vcol: str = "value") -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None)
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").to_numpy(), index=pd.DatetimeIndex(idx)).dropna()
    return to_15min(s[~s.index.duplicated(keep="last")])


def _when(col: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{col} >= date'{start:%Y-%m-%dT%H:%M:%S}Z' and {col} < date'{end:%Y-%m-%dT%H:%M:%S}Z'"


class EliaSource(Source):
    name = "elia"
    WIND = (("ods031", "measured"), ("ods086", "realtime"))   # historical first, near real-time fills the recent end
    SOLAR = (("ods032", "measured"), ("ods087", "realtime"))

    def fetch(self, zone, target, start, end):
        if zone != "BE" or target not in ("wind", "solar"):
            return pd.Series(dtype=float)
        return self._collect(target, self.WIND if target == "wind" else self.SOLAR, start, end)

    def dayahead_11h(self, target, start, end) -> pd.Series:
        """Elia's day-ahead forecast for Belgium as published at 11:00 on D-1, MW (a fixed snapshot, so it
        can be fetched later without look-ahead)."""
        parts = self.WIND if target == "wind" else self.SOLAR
        return self._collect(target, [(ds, "dayahead11hforecast") for ds, _ in parts], start, end)

    def _collect(self, target, datasets, start, end) -> pd.Series:
        retries = 2 if self.cfg.get("_fallback") else 4
        out = pd.Series(dtype=float)
        for dataset, col in datasets:
            if target == "wind":
                params = {"select": f"datetime, sum({col}) as value, count({col}) as n", "group_by": "datetime",
                          "where": _when("datetime", start, end) + f" and {col} is not null"}
                rows = _export(ELIA, dataset, params, retries)
                parts = max((r["n"] for r in rows), default=0)
                rows = [r for r in rows if r["n"] == parts]          # a quarter-hour counts only with every part reported
            else:
                params = {"select": f"datetime, {col} as value",
                          "where": _when("datetime", start, end) + f" and region = 'Belgium' and {col} is not null"}
                rows = _export(ELIA, dataset, params, retries)
            s = _series(rows, "datetime")
            out = out.combine_first(s) if not out.empty else s
        return out[(out.index >= start) & (out.index < end)] if not out.empty else out


class RteSource(Source):
    name = "rte"
    FIELD = {"wind": "eolien", "solar": "solaire"}

    def fetch(self, zone, target, start, end):
        if zone != "FR" or target not in self.FIELD:
            return pd.Series(dtype=float)
        col = self.FIELD[target]
        params = {"select": f"date_heure, {col} as value", "where": _when("date_heure", start, end) + f" and {col} is not null"}
        s = _series(_export(ODRE, "eco2mix-national-tr", params, 2 if self.cfg.get("_fallback") else 4), "date_heure")
        return s[(s.index >= start) & (s.index < end)] if not s.empty else s
