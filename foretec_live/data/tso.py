"""The TSOs' own open-data services, used to fill quarter-hours ENTSO-E does not deliver. No API key needed.

Elia (BE), opendata.elia.be (Opendatasoft):
  wind   ods031 (historical, `measured`) + ods086 (near real-time, `realtime`): the sum of the disjoint parts
         Federal offshore + Flanders/Wallonia onshore, each on the Elia grid and on distribution grids.
  solar  region "Belgium": ods032 (historical, to 25 Oct 2026), then ods209 (historical) + ods210 (near real-time) from
         26 Oct 2026, which split each region by grid connection (DSO, TSO, Total): the Total rows. ods087 (the old
         near-real-time set) is discontinued and answers 400. ods209 has no `measured`: its column is
         `measured_upscaled`, published as TEXT with a comma decimal ("5252,070"); ods210 keeps only today onward.
         A dataset that fails is skipped (logged) so the others still deliver; without this, ods209's 400 made the
         whole source fail and the hourly ENTSO-E fallback drew BE solar as a staircase (24-29 Oct 2026).
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
  with Energinet's settlement within 1-4% (hourly, 2024 - Oct 2026).
  DK solar from Energinet's hourly settlement (ProductionConsumptionSettlement, its four solar columns including
  self-consumption) since ENTSO-E stopped publishing it (A75 B16 empty from 4 Nov 2026 08:00 UTC). Settlement matched
  A75 at 0.98x (DK1) and 0.94x (DK2), hourly MAE 15 and 11 MW, 15 Sep - 3 Nov 2026 (without its self-consumption
  column 0.84x and 0.66x). Settlement is published about ten days after delivery, so a day is frozen and scored then.
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
import json
import logging
import os
import time
from pathlib import Path

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
    vals = df[vcol]
    if not pd.api.types.is_numeric_dtype(vals):   # Elia ods209 publishes measured_upscaled as text with a comma
        vals = vals.astype(str).str.replace(",", ".", regex=False)   # decimal; pandas 3 types it `str`, not object
    s = pd.Series(pd.to_numeric(vals, errors="coerce").to_numpy(), index=pd.DatetimeIndex(idx)).dropna()
    return to_15min(s[~s.index.duplicated(keep="last")])


def _when(col: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{col} >= date'{start:%Y-%m-%dT%H:%M:%S}Z' and {col} < date'{end:%Y-%m-%dT%H:%M:%S}Z'"


class EliaSource(Source):
    name = "elia"
    # (dataset, actual column, extra filter): historical first, near real-time fills the recent end
    WIND = (("ods031", "measured", ""), ("ods086", "realtime", ""))
    SOLAR = (("ods032", "measured", ""), ("ods209", "measured_upscaled", "gridconnectiontype = 'Total'"),
             ("ods210", "measured", "gridconnectiontype = 'Total'"))

    def fetch(self, zone, target, start, end):
        if zone != "BE" or target not in ("wind", "solar"):
            return pd.Series(dtype=float)
        return self._collect(target, self.WIND if target == "wind" else self.SOLAR, start, end)

    def dayahead_11h(self, target, start, end) -> pd.Series:
        """Elia's day-ahead forecast for Belgium as published at 11:00 on D-1, MW (a fixed snapshot, so it
        can be fetched later without look-ahead)."""
        parts = self.WIND if target == "wind" else self.SOLAR
        return self._collect(target, [(ds, "dayahead11hforecast", extra) for ds, _, extra in parts], start, end)

    def most_recent(self, target, start, end) -> pd.Series:
        """Elia's latest forecast for Belgium as it stands now, MW, 15 minutes. Elia overwrites it, so it is a forecast
        made before the gate only if this is called before the gate."""
        parts = self.WIND if target == "wind" else self.SOLAR
        return self._collect(target, [(ds, "mostrecentforecast", extra) for ds, _, extra in parts], start, end)

    def dayahead_18h(self, target, start, end) -> pd.Series:
        """Elia's day-ahead forecast for Belgium as published at 18:00 on D-1 (after the gate), MW, 15 minutes."""
        parts = self.WIND if target == "wind" else self.SOLAR
        return self._collect(target, [(ds, "dayaheadforecast", extra) for ds, _, extra in parts], start, end)

    def _collect(self, target, datasets, start, end) -> pd.Series:
        retries = 2 if self.cfg.get("_fallback") else 4
        out = pd.Series(dtype=float)
        failed = []
        for dataset, col, extra in datasets:
            extra = f" and {extra}" if extra else ""
            try:
                if target == "wind":
                    params = {"select": f"datetime, sum({col}) as value, count({col}) as n", "group_by": "datetime",
                              "where": _when("datetime", start, end) + f" and {col} is not null{extra}"}
                    rows = _export(ELIA, dataset, params, retries)
                    parts = max((r["n"] for r in rows), default=0)
                    rows = [r for r in rows if r["n"] == parts]      # a quarter-hour counts only with every part reported
                else:
                    params = {"select": f"datetime, {col} as value",
                              "where": _when("datetime", start, end) + f" and region = 'Belgium' and {col} is not null{extra}"}
                    rows = _export(ELIA, dataset, params, retries)
            except requests.RequestException as e:   # one dataset down (or renamed) must not take the others with it
                log.warning("elia %s %s skipped: %s", dataset, target, e)
                failed.append(e)
                continue
            s = _series(rows, "datetime")
            out = out.combine_first(s) if not out.empty else s
        if out.empty and failed:
            raise failed[-1]
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


class RateLimited(requests.HTTPError):
    """Energi Data Service answered 429 and the caller does not wait for it."""


class EnerginetSource(Source):
    """DK1/DK2 wind (onshore + offshore) from Energinet's real-time production, quarter-hour means of the 5-minute
    values (a quarter-hour needs at least 2 of its 3). DK1/DK2 solar from the hourly settlement: see _solar.

    Prodex is keyed on Danish local time with 24 hours a day: on the autumn clock change it has no records for the
    first (summer-time) 02:00-03:00 or for 23:05-23:55 (26 Oct 2025 and 25 Oct 2026, 23 five-minute records each; both
    spring days are complete). That empties 8 of the 100 quarter-hours, below the 95% needed to freeze the day. So in
    a window of at most FILL_WINDOW (a delivery day being frozen, revised or drawn, never model history), quarter-hours
    Prodex leaves empty that ended SETTLED_AFTER ago are filled from Energinet's hourly settlement
    (ProductionConsumptionSettlement, keyed on UTC hours, complete across the clock change): the hour's wind MWh, i.e.
    its mean MW, held across its quarter-hours. Settlement agrees with Prodex within 1-4% hourly. Until it is
    published the quarter-hours stay empty and the day is frozen on a later run.
    """
    name = "energinet"
    URL = "https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime"
    SETTLEMENT_URL = "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement"
    WIND = ("OnshoreWindPower", "OffshoreWindPower")
    SETTLED_WIND = ("OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh", "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh")
    SETTLED_SOLAR = ("SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh", "SolarPowerGe40kW_MWh", "SolarPowerSelfConMWh")
    SETTLED_AFTER = pd.Timedelta("6h")   # Prodex arrives within ~5 min; a younger gap may still fill itself
    SCALE_DAYS = pd.Timedelta("14D")     # solar history: the days settlement and Prodex share that set Prodex's scale
    # the settlement dataset is rate-limited per IP (429 with Retry-After of minutes, Nov 2026): settled solar is kept
    # in a disk cache, asked for new hours at most every SOLAR_REFRESH, and a 429 is not waited out on that path
    SOLAR_REFRESH = pd.Timedelta("6h")
    PATIENCE = 30.0                      # seconds of Retry-After an impatient request still waits
    _settlement_down = False             # set on a 429 for settled solar: no more requests from this source object
    FILL_WINDOW = pd.Timedelta("3D")

    def _get(self, params: dict, url: str | None = None, patient: bool = True) -> list[dict]:
        """One request; Energi Data Service rate-limits bursts (429): wait for Retry-After (or back off), up to 6 times.
        An impatient request waits only up to PATIENCE seconds and otherwise fails at once."""
        retries, fails, limited = (2 if self.cfg.get("_fallback") else 4), 0, 0
        while True:
            try:
                r = requests.get(url or self.URL, params=params, timeout=120)
                wait = float(r.headers.get("Retry-After") or 10 * (limited + 1)) if r.status_code == 429 else 0.0
                if r.status_code == 429 and limited < 6 and (patient or wait <= self.PATIENCE):
                    limited += 1
                    time.sleep(wait)
                    continue
                if r.status_code == 429:   # impatient: give up at once, below
                    raise RateLimited(f"429: rate-limited for {wait:.0f} s")
                r.raise_for_status()
                return r.json()["records"]
            except (requests.RequestException, ValueError, KeyError) as e:
                fails += 1
                if fails >= retries or isinstance(e, RateLimited):
                    raise
                log.warning("energidataservice failed (%s), retrying", e)
                time.sleep(5 * fails)

    def _records(self, zone, start, end, cols, time_col, url=None, patient=True) -> pd.Series:
        """Sum of `cols` per record of [start, end) for one area, a quarter of a year per request, sorted, NaN dropped."""
        rows, a = [], pd.Timestamp(start)
        while a < end:
            b = min(pd.Timestamp(end), a + pd.Timedelta(days=92))
            rows += self._get({"start": f"{a:%Y-%m-%dT%H:%M}", "end": f"{b:%Y-%m-%dT%H:%M}", "timezone": "UTC", "limit": 0,
                               "filter": f'{{"PriceArea":["{zone}"]}}', "columns": f"{time_col}," + ",".join(cols)},
                              url=url, patient=patient)
            a = b
        df = pd.DataFrame(rows)
        if df.empty or time_col not in df or not set(cols) <= set(df.columns):
            return pd.Series(dtype=float)
        out = pd.Series(df[list(cols)].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(cols)).to_numpy(),
                        index=pd.DatetimeIndex(pd.to_datetime(df[time_col]))).dropna().sort_index()
        return out[~out.index.duplicated(keep="last")]

    @staticmethod
    def _quarters(five: pd.Series) -> pd.Series:
        q = five.resample("15min")
        return q.mean()[q.count() >= 2]

    def fetch(self, zone, target, start, end):
        if zone not in ("DK1", "DK2") or target not in ("wind", "solar"):
            return pd.Series(dtype=float)
        if target == "solar":
            return self._solar(zone, start, end)
        five = self._records(zone, start, end, self.WIND, "Minutes5UTC")
        if five.empty:
            return five
        s = self._quarters(five)
        if pd.Timestamp(end) - pd.Timestamp(start) <= self.FILL_WINDOW:
            s = self._fill_from_settlement(zone, s, start, end)
        s.index.name = "time_utc"
        return s[(s.index >= start) & (s.index < end)]

    def _solar(self, zone: str, start, end) -> pd.Series:
        """Settled solar of each UTC hour (its MWh, i.e. its mean MW) held across the hour's quarter-hours.

        A delivery day (a window of at most FILL_WINDOW: frozen, revised or drawn) is settlement only, so it stays
        empty until Energinet publishes it, about ten days after delivery. Model history (a longer window) would
        otherwise end ten days before the cut-off; after the last settled hour it continues with Prodex's real-time
        solar times the settlement / Prodex ratio of the SCALE_DAYS they share (1.03 in DK1 and 0.96 in DK2 for
        27 Oct - 9 Nov 2026), so every model sees the same estimate of the recent days.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        hourly = self._settled_solar(zone, start.floor("h"), end)
        s = pd.Series(dtype=float)
        if not hourly.empty:
            idx = pd.date_range(hourly.index[0], hourly.index[-1] + pd.Timedelta("1h"), freq="15min", inclusive="left")
            s = pd.Series(hourly.reindex(idx.floor("h")).to_numpy(), index=idx).dropna()
        if end - start > self.FILL_WINDOW:
            done = hourly.index[-1] + pd.Timedelta("1h") if not hourly.empty else start
            if done < end:
                five = self._records(zone, max(start, done - self.SCALE_DAYS), end, ("SolarPower",), "Minutes5UTC")
                live = self._quarters(five) if not five.empty else five
                both = pd.concat([live.resample("1h").mean().rename("live"), hourly.rename("settled")], axis=1).dropna() \
                    if not live.empty and not hourly.empty else pd.DataFrame(columns=["live", "settled"])
                if len(both) >= 24 and both["live"].sum() > 0:
                    ratio = float(both["settled"].sum() / both["live"].sum())
                    tail = live[live.index >= done] * ratio
                    if not tail.empty:
                        log.info("energinet %s solar: history after %s from real-time solar x %.3f", zone, done, ratio)
                        s = tail if s.empty else pd.concat([s, tail]).sort_index()
        if s.empty:
            return pd.Series(dtype=float)
        s.index.name = "time_utc"
        return s[(s.index >= start) & (s.index < end)]

    def _settled_solar(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Hourly settled solar of [start, end) from data/cache/energinet/settled_solar_<zone>.parquet, which grows as
        Energinet publishes: a request only for hours before the cache, or, at most every SOLAR_REFRESH, for the
        hours after it. A failed or rate-limited request leaves the cache as it is (those hours stay unsettled)."""
        f = Path(self.cfg["_root"]) / "data" / "cache" / "energinet" / f"settled_solar_{zone}.parquet" if "_root" in self.cfg else None
        stamp = f.with_suffix(".json") if f is not None else None
        have = pd.read_parquet(f)["value"] if f is not None and f.exists() else pd.Series(dtype=float)
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        checked = pd.Timestamp(json.loads(stamp.read_text())["checked_utc"]) if stamp is not None and stamp.exists() else None
        need = []
        if have.empty:   # up to now: one request also covers the later days a scoring run is about to ask for
            need.append((start, max(end, now.floor("h")), True))
        else:
            if start < have.index[0]:
                need.append((start, have.index[0], False))
            after = have.index[-1] + pd.Timedelta("1h")
            if end > after and (checked is None or now - checked >= self.SOLAR_REFRESH):
                need.append((after, max(end, now.floor("h")), True))
        for a, b, tail in need:
            if self._settlement_down:   # one 429 per run: asking again for every pending day keeps the limit hot
                break
            try:
                got = self._records(zone, a, b, self.SETTLED_SOLAR, "HourUTC", url=self.SETTLEMENT_URL, patient=False)
            except (requests.RequestException, ValueError, KeyError) as e:
                self._settlement_down = isinstance(e, RateLimited)
                log.warning("energinet settled solar %s %s .. %s not fetched (%s); using what is cached", zone, a, b, e)
                continue
            if not got.empty:
                have = got if have.empty else pd.concat([have, got]).sort_index()
                have = have[~have.index.duplicated(keep="last")]
            if tail and stamp is not None:
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.write_text(json.dumps({"checked_utc": str(now)}))
        if f is not None and need and not have.empty:
            f.parent.mkdir(parents=True, exist_ok=True)
            have.rename("value").to_frame().to_parquet(f)
        return have[(have.index >= start) & (have.index < end)] if not have.empty else have

    def _fill_from_settlement(self, zone: str, s: pd.Series, start, end) -> pd.Series:
        """Quarter-hours of [start, end) Prodex left empty and that ended SETTLED_AFTER ago, from the hourly settlement.
        No request when there is no such gap (every ordinary day); a failed request keeps Prodex alone."""
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="15min", inclusive="left")
        settled = pd.Timestamp.now(tz="UTC").tz_localize(None) - self.SETTLED_AFTER
        gaps = idx[~idx.isin(s.index) & (idx + pd.Timedelta("15min") <= settled)]
        if gaps.empty:
            return s
        a, b = gaps[0].floor("h"), gaps[-1].floor("h") + pd.Timedelta("1h")
        cols = list(self.SETTLED_WIND)
        try:
            rows = self._get({"start": f"{a:%Y-%m-%dT%H:%M}", "end": f"{b:%Y-%m-%dT%H:%M}", "timezone": "UTC", "limit": 0,
                              "filter": f'{{"PriceArea":["{zone}"]}}', "columns": "HourUTC," + ",".join(cols)},
                             url=self.SETTLEMENT_URL)
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("energinet settlement %s failed (%s); %d quarter-hours stay empty", zone, e, len(gaps))
            return s
        df = pd.DataFrame(rows)
        if df.empty or "HourUTC" not in df or not set(cols) <= set(df.columns):
            return s
        hourly = pd.Series(df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(cols)).to_numpy(),
                           index=pd.DatetimeIndex(pd.to_datetime(df["HourUTC"]))).dropna()
        hourly = hourly[~hourly.index.duplicated(keep="last")]
        fill = pd.Series(hourly.reindex(gaps.floor("h")).to_numpy(), index=gaps).dropna()
        if fill.empty:
            return s
        log.info("energinet %s: %d quarter-hours Prodex lacks filled from settlement (%s .. %s)",
                 zone, len(fill), fill.index[0], fill.index[-1])
        return pd.concat([s, fill]).sort_index()


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
