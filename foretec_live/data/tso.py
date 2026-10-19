"""The TSOs' own open-data services, used to fill quarter-hours ENTSO-E does not deliver. No API key needed.

Elia (BE), opendata.elia.be (Opendatasoft):
  wind   ods031 (historical, `measured`) + ods086 (near real-time, `realtime`): the sum of the disjoint parts
         Federal offshore + Flanders/Wallonia onshore, each on the Elia grid and on distribution grids.
  solar  ods032 + ods087, region "Belgium".
  Elia also publishes its own day-ahead forecast as it stood at 11:00 on D-1 (`dayahead11hforecast`), before
  the 12:00 gate (`dayahead_11h`), and the 18:00 version (`dayaheadforecast`, `dayahead_18h`). The 18:00 one is
  what Elia sends to ENTSO-E, but ENTSO-E gets hourly values; Elia Open Data has it at 15 minutes (hourly means
  identical to the MW, checked 14 Sep - 6 Oct 2026).
RTE (FR), éCO2mix real-time on odre.opendatasoft.com:
  wind   `eolien` (onshore + offshore), solar `solaire`, 15-minute national values.

NED (NL), Nationaal Energie Dashboard (api.ned.nl, key NED_NL_KEY), backed by TenneT and the grid operators:
  national 15-minute onshore wind, offshore wind and solar, as current estimates (NedSource: the NL actuals,
  rooftop PV and DSO-connected turbines included) and as forecasts, updated about every hour and overwritten as
  delivery approaches, so only a forecast fetched live before the gate is a gate-time forecast.

Energinet (DK1, DK2), Energi Data Service (api.energidataservice.dk, no key), ElectricityProdex5MinRealtime:
  onshore + offshore wind, 5-minute upscaled SCADA values published within ~5 minutes and not corrected afterwards,
  averaged to quarter-hours. Used for DK wind because ENTSO-E's DK1 offshore (A75 B18) is a different quantity: it
  ran 1.2x the metered output from April 2025 to May 2026, and since July 2026 it covers only the parks registered
  at ENTSO-E (Anholt, Horns Rev A-C, Vesterhav Nord/Syd), missing 0.3-0.5 GW of new offshore output. Prodex agrees
  with Energinet's settlement within 1-4% (hourly, 2024 - Oct 2026). DK solar stays on ENTSO-E: Prodex solar leaves
  out self-consumption (DK2 19% below settlement in 2026), A75 matches settlement including it.
  DK load stays on ENTSO-E A65, the quantity Energinet's own day-ahead load forecast (the grid-operator benchmark)
  predicts: in 2026 that forecast is 1.00x A65 with an hourly MAE of 40-89 MW in DK1, against 226-287 MW to Prodex's
  production plus net imports. So this source answers wind only: as a fallback it must not fill A65's load gaps.

RTE (FR), Generation Forecast API v3 (key RTE_DATA_KEY = base64 client_id:client_secret): RTE's day-ahead
  ("D-1") wind (onshore + offshore) and solar forecast, published at 16:15 on D-1, after the gate. RTE has no
  earlier vintage for wind or solar, so it is a reference only.

Each source answers only for its own zone and returns nothing elsewhere, so they can sit in one fallback chain.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
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

    def dayahead_18h(self, target, start, end) -> pd.Series:
        """Elia's day-ahead forecast for Belgium as published at 18:00 on D-1 (after the gate), MW, 15 minutes."""
        parts = self.WIND if target == "wind" else self.SOLAR
        return self._collect(target, [(ds, "dayaheadforecast") for ds, _ in parts], start, end)

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
    FIELD = {"wind": "eolien", "solar": "solaire", "load": "consommation"}

    def fetch(self, zone, target, start, end):
        if zone != "FR" or target not in self.FIELD:
            return pd.Series(dtype=float)
        col = self.FIELD[target]
        params = {"select": f"date_heure, {col} as value", "where": _when("date_heure", start, end) + f" and {col} is not null"}
        s = _series(_export(ODRE, "eco2mix-national-tr", params, 2 if self.cfg.get("_fallback") else 4), "date_heure")
        return s[(s.index >= start) & (s.index < end)] if not s.empty else s


class NedForecast:
    """NED.nl national wind/solar/load series for whole local days (MW): classification 1 = forecast (live use only,
    NED overwrites it), 2 = current estimate (the national actual: monitored systems upscaled, rooftop included).
    Load is ElectricityLoad (type 59, activity consuming): gross national consumption, including what rooftop and
    distribution-connected generation supply locally (unlike the net grid load TenneT reports to ENTSO-E)."""
    URL = "https://api.ned.nl/v1/utilizations"
    # wind: onshore (1) + offshore 51, the metered offshore output (identical to ENTSO-E B18). Type 17 is NED's per-park
    # model: capped at a fixed capacity and 7-27% high (checked 2023-2026); used for NL wind actuals until 12 Oct 2026.
    TYPES = {"wind": (1, 51), "solar": (2,), "load": (59,)}   # onshore + offshore wind; solar; electricity load
    ACTIVITY = {"load": 2}                                     # 1 providing (generation), 2 consuming

    def __init__(self, key: str | None = None, retries: int = 3):
        self.key = key or os.environ.get("NED_NL_KEY")
        if not self.key:
            raise RuntimeError("NED_NL_KEY is not set")
        self.retries = retries

    _last = 0.0   # time of the last request, shared by every client in the process (NED rate-limits per key)

    def _get(self, q: dict) -> list:
        """One page. NED rate-limits per key (429): requests are paced, a 429 waits for Retry-After (or backs off)
        and is retried up to 8 times, other failures `retries` times."""
        fails = rate_limited = 0
        while True:
            wait = NedForecast._last + 0.6 - time.time()   # at most ~100 requests a minute
            if wait > 0:
                time.sleep(wait)
            NedForecast._last = time.time()
            try:
                r = requests.get(self.URL, params=q, timeout=60,
                                 headers={"X-AUTH-TOKEN": self.key, "Accept": "application/json", "User-Agent": "foretec-energy-bench/1.0"})
                if r.status_code == 429 and rate_limited < 8:
                    rate_limited += 1
                    delay = float(r.headers.get("Retry-After") or min(60, 5 * 2 ** rate_limited))
                    log.warning("NED rate limit, waiting %.0f s (%d/8)", delay, rate_limited)
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                fails += 1
                if fails >= self.retries:
                    raise
                log.warning("NED %s failed (%s), retrying", q.get("type"), e)
                time.sleep(5 * fails)

    def forecast(self, target: str, first_day: dt.date, last_day: dt.date, types: tuple | None = None,
                 classification: int = 1) -> pd.Series:
        total = None
        for typ in types or self.TYPES[target]:
            rows = []
            a = first_day
            while a <= last_day:   # a month per query: year-long ranges time out at NED's gateway (504)
                b = min(last_day, a + dt.timedelta(days=30))
                page = 1
                while True:
                    # the API accepts plain dates (local, Europe/Amsterdam with granularitytimezone=1), not timestamps
                    got = self._get({"point": 0, "type": typ, "granularity": 4, "granularitytimezone": 1,
                                     "classification": classification, "activity": self.ACTIVITY.get(target, 1),
                                     "validfrom[after]": a.isoformat(),
                                     "validfrom[strictly_before]": (b + dt.timedelta(days=1)).isoformat(),
                                     "itemsPerPage": 200, "page": page})
                    rows += got
                    if len(got) < 200:
                        break
                    page += 1
                a = b + dt.timedelta(days=1)
            s = _series([{"t": x["validfrom"], "value": x["capacity"] / 1000.0} for x in rows], "t")   # kW -> MW
            total = s if total is None else total.add(s)       # both parts needed for a quarter-hour
        return total if total is not None else pd.Series(dtype=float)


class NedSource(Source):
    """NL wind (onshore + offshore), solar and load as NED's national current estimate. ENTSO-E's NL wind is offshore
    plus TSO-metered onshore, its NL solar only TSO-metered plants (about 1% of the fleet) and its NL load net of
    embedded solar; NED covers all of it (load gross, as Elia and RTE report it)."""
    name = "ned"

    def fetch(self, zone, target, start, end):
        if zone != "NL" or target not in NedForecast.TYPES:
            return pd.Series(dtype=float)
        ned = NedForecast(retries=2 if self.cfg.get("_fallback") else 4)
        tz = "Europe/Amsterdam"
        first = pd.Timestamp(start).tz_localize("UTC").tz_convert(tz).date()
        last = (pd.Timestamp(end) - pd.Timedelta("15min")).tz_localize("UTC").tz_convert(tz).date()
        s = ned.forecast(target, first, last, classification=2)
        return s[(s.index >= start) & (s.index < end)] if not s.empty else s


class EnerginetSource(Source):
    """DK1/DK2 wind (onshore + offshore) from Energinet's real-time production, quarter-hour means of the 5-minute
    values (a quarter-hour needs at least 2 of its 3)."""
    name = "energinet"
    URL = "https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime"
    WIND = ("OnshoreWindPower", "OffshoreWindPower")

    def _get(self, params: dict) -> list[dict]:
        """One request; Energi Data Service rate-limits bursts (429): wait for Retry-After (or back off), up to 6 times."""
        retries, fails, limited = (2 if self.cfg.get("_fallback") else 4), 0, 0
        while True:
            try:
                r = requests.get(self.URL, params=params, timeout=120)
                if r.status_code == 429 and limited < 6:
                    limited += 1
                    time.sleep(float(r.headers.get("Retry-After") or 10 * limited))
                    continue
                r.raise_for_status()
                return r.json()["records"]
            except (requests.RequestException, ValueError, KeyError) as e:
                fails += 1
                if fails >= retries:
                    raise
                log.warning("energidataservice failed (%s), retrying", e)
                time.sleep(5 * fails)

    def fetch(self, zone, target, start, end):
        if zone not in ("DK1", "DK2") or target != "wind":
            return pd.Series(dtype=float)
        cols = self.WIND
        rows, a = [], pd.Timestamp(start)
        while a < end:   # a quarter of a year per request
            b = min(pd.Timestamp(end), a + pd.Timedelta(days=92))
            rows += self._get({"start": f"{a:%Y-%m-%dT%H:%M}", "end": f"{b:%Y-%m-%dT%H:%M}", "timezone": "UTC", "limit": 0,
                               "filter": f'{{"PriceArea":["{zone}"]}}', "columns": "Minutes5UTC," + ",".join(cols)})
            a = b
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows)
        five = pd.Series(df[list(cols)].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(cols)).to_numpy(),
                         index=pd.DatetimeIndex(pd.to_datetime(df["Minutes5UTC"]))).dropna().sort_index()
        five = five[~five.index.duplicated(keep="last")]
        q = five.resample("15min")
        s = q.mean()[q.count() >= 2]
        s.index.name = "time_utc"
        return s[(s.index >= start) & (s.index < end)]


class RteForecast:
    """RTE's day-ahead (D-1) wind and solar forecast for France, MW, 15-minute values."""
    TOKEN = "https://digital.iservices.rte-france.com/token/oauth/"
    URL = "https://digital.iservices.rte-france.com/open_api/generation_forecast/v3/forecasts"
    TYPES = {"wind": ("WIND_ONSHORE", "WIND_OFFSHORE"), "solar": ("SOLAR",)}

    def __init__(self, key: str | None = None):
        key = key or os.environ.get("RTE_DATA_KEY")
        if not key:
            raise RuntimeError("RTE_DATA_KEY is not set")
        r = requests.post(self.TOKEN, headers={"Authorization": f"Basic {key}", "Content-Type": "application/x-www-form-urlencoded"}, timeout=60)
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def forecast(self, target: str, first_day: dt.date, last_day: dt.date, tz: str = "Europe/Paris") -> pd.Series:
        """Sum of the production types of `target`, local days first_day..last_day (at most 21 per call)."""
        a = pd.Timestamp(first_day.isoformat()).tz_localize(tz).isoformat()
        b = pd.Timestamp((last_day + dt.timedelta(days=1)).isoformat()).tz_localize(tz).isoformat()
        total = None
        for pt in self.TYPES[target]:
            r = requests.get(self.URL, params={"production_type": pt, "type": "D-1", "start_date": a, "end_date": b},
                             headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"}, timeout=60)
            r.raise_for_status()
            rows = [{"t": v["start_date"], "value": v["value"]} for f in r.json().get("forecasts", []) for v in f.get("values", [])]
            s = _series(rows, "t")
            total = s if total is None else total.add(s)
        return total if total is not None else pd.Series(dtype=float)
