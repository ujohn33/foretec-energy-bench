"""The TSO's own day-ahead wind/solar forecast as it stood before the gate. Live only (model.yaml): what was
published before the gate cannot be verified after the fact, so this entry is never backtested.

BE: Elia's day-ahead forecast published at 11:00 on D-1 (Elia Open Data, `dayahead11hforecast`).
NL: the NED.nl offshore wind forecast (Nationaal Energie Dashboard, backed by TenneT), fetched at the run.
    Offshore only: the NL wind actual on ENTSO-E is offshore plus TSO-metered onshore (~28 MW of ~305 MW on
    2-3 Oct 2026), while NED's onshore forecast is national (~6x that); NED offshore matched ENTSO-E offshore
    (287 vs 277 MW). NED overwrites its forecast as delivery approaches, so it is used only in a live run
    before the gate. NL solar is excluded for the same reason.
Fallback, live runs only: ENTSO-E A69 as published when this runs (often not there yet; TSOs have until 18:00
on D-1, and after the gate it would be that later version).
Point forecasts only: quantiles of parts do not add up to quantiles of the total.
"""
import datetime as dt

import numpy as np
import pandas as pd
from fl_io import read_request, write_output
from foretec_live.config import load_config
from foretec_live.data.entsoe import EntsoeSource
from foretec_live.data.tso import EliaSource, NedForecast
from foretec_live.timeutil import gate_timestamp

req, series = read_request()
cfg = load_config()
elia = EliaSource({**cfg, "_fallback": True})
inp = pd.read_parquet(req["input"])
now = pd.Timestamp.now(tz="UTC").tz_localize(None)
results, errors = {}, {}
for key in series:
    zone, target = key.split("_")
    last = inp.loc[inp["series"] == key, "time_utc"].max()
    idx = pd.DatetimeIndex(last + pd.Timedelta("15min") * np.arange(1, req["horizon"][key] + 1))
    end = idx[-1] + pd.Timedelta("15min")
    # the horizon ends at local midnight after delivery day D; the issue day is D-1
    delivery = (end.tz_localize("UTC").tz_convert(cfg["timezone"]) - pd.Timedelta("1h")).date()
    issue = delivery - dt.timedelta(days=1)
    live = now <= gate_timestamp(issue, cfg)

    def ned():
        if not live:
            raise RuntimeError("live only (NED overwrites its forecast)")
        return NedForecast().forecast(target, issue, delivery, types=(17,))   # offshore: see the docstring

    sources = {"BE": [("Elia 11:00", lambda: elia.dayahead_11h(target, idx[0], end))],
               "NL": [("NED.nl", ned)]}.get(zone, [])
    if live:   # after the gate ENTSO-E holds the 18:00 version, which would be look-ahead
        sources.append(("ENTSO-E A69", lambda: EntsoeSource(cfg).tso_forecast(zone, target, idx[0], end)))
    tried = []
    for name, get in sources:
        try:
            s = get().reindex(idx)
        except Exception as e:  # noqa: BLE001 - try the next source
            tried.append(f"{name}: {str(e)[:80] or type(e).__name__}")
            continue
        day = s.iloc[-96:]                               # the delivery day is the end of the horizon
        if day.notna().mean() >= 0.95:
            results[key] = (s.ffill().bfill().to_numpy(), None)
            print(f"{key}: {name}, {int(day.notna().sum())}/{len(day)} quarter-hours of D")
            break
        tried.append(f"{name}: {int(day.notna().sum())} of {len(day)} quarter-hours of D")
    else:
        errors[key] = "TSO forecast not available before the gate: " + "; ".join(tried)
write_output(req, results, errors)
