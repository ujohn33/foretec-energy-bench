"""Reference, published after the gate: the TSOs' final day-ahead wind/solar forecast.

FR: RTE's day-ahead forecast published at 16:15 on D-1 (RTE Generation Forecast API v3, onshore + offshore wind,
    solar), so it is there from the 20:45 scoring run. Fallback: ENTSO-E A69.
BE: Elia's day-ahead forecast published at 18:00 on D-1 (Elia Open Data, 15 minutes; ENTSO-E gets the same
    forecast in hourly values). Fallback: ENTSO-E A69.
NL: ENTSO-E A69 (process A01), as published when this runs (by 18:00 on D-1 by regulation).
"""
import numpy as np
import pandas as pd
from fl_io import read_request, write_output
from foretec_live.config import load_config
from foretec_live.data.entsoe import EntsoeSource
from foretec_live.data.tso import EliaSource, RteForecast

cfg = load_config()
req, series = read_request()
src = EntsoeSource(cfg)
elia = EliaSource({**cfg, "_fallback": True})
inp = pd.read_parquet(req["input"])
_rte = []


def rte(target, idx):
    if not _rte:
        _rte.append(RteForecast())
    local = pd.DatetimeIndex([idx[0], idx[-1]]).tz_localize("UTC").tz_convert("Europe/Paris")
    return _rte[0].forecast(target, local[0].date(), local[1].date())


results, errors = {}, {}
for key in series:
    zone, target = key.split("_")
    last = inp.loc[inp["series"] == key, "time_utc"].max()
    idx = pd.DatetimeIndex(last + pd.Timedelta("15min") * np.arange(1, req["horizon"][key] + 1))
    end = idx[-1] + pd.Timedelta("15min")
    # Elia's 18:00 field may hold a preliminary value earlier in the afternoon: use it only from 18:00 on D-1
    delivery = (end.tz_localize("UTC").tz_convert(cfg["timezone"]) - pd.Timedelta("1h")).date()
    after_18 = pd.Timestamp.now(tz=cfg["timezone"]) >= pd.Timestamp(f"{delivery - pd.Timedelta(days=1)} 18:00", tz=cfg["timezone"])
    sources = ([("RTE D-1", lambda: rte(target, idx))] if zone == "FR" else []) + \
              ([("Elia 18:00", lambda: elia.dayahead_18h(target, idx[0], end))] if zone == "BE" and after_18 else []) + \
              [("ENTSO-E A69", lambda: src.tso_forecast(zone, target, idx[0], end))]
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
        errors[key] = "TSO forecast not published (yet): " + "; ".join(tried)
write_output(req, results, errors)
