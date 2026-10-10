"""The TSO's own day-ahead wind/solar forecast as it stood before the gate. Live only (model.yaml): what was
published before the gate cannot be verified after the fact, so this entry is never backtested.

BE: Elia's day-ahead forecast published at 11:00 on D-1 (Elia Open Data, `dayahead11hforecast`).
NL: the NED.nl national wind (onshore + offshore) and solar forecast (Nationaal Energie Dashboard, backed by TenneT),
    fetched at the run, matching the NL actuals (NED's national estimate). NED overwrites its forecast as delivery
    approaches, so it is used only in a live run before the gate.
FR: none. RTE's next-day wind/solar forecast is published at 16:15 on D-1 (after the gate); its API has
    no D-2/D-3 vintage for wind or solar (checked 5 Oct 2026).
Load: ENTSO-E A65 day-ahead total load forecast, every zone (published by 10:00 by regulation).
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
        return NedForecast().forecast(target, issue, delivery)   # national: onshore + offshore wind, solar

    if target == "load":   # day-ahead total load forecast, due two hours before the gate
        sources = [("ENTSO-E A65", lambda: EntsoeSource(cfg).tso_forecast(zone, "load", idx[0], end))]
    else:
        sources = {"BE": [("Elia 11:00", lambda: elia.dayahead_11h(target, idx[0], end))],
                   "NL": [("NED.nl", ned)]}.get(zone, [])
    if live and target != "load":   # after the gate ENTSO-E holds the 18:00 version, which would be look-ahead
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
