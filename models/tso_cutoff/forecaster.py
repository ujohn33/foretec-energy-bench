"""The TSO's own day-ahead wind/solar forecast as it stood before the gate.

BE: Elia's day-ahead forecast published at 11:00 on D-1 (Elia Open Data, `dayahead11hforecast`), before the
11:30 cut-off. It is a fixed snapshot, so backtests see exactly what was public at the time. Point forecast
only: Elia publishes P10/P90 per wind part, and quantiles of parts do not add up to quantiles of the total.
Fallback: ENTSO-E A69 as published when this runs (often not there yet at 11:30; TSOs have until 18:00).
"""
import numpy as np
import pandas as pd
from fl_io import read_request, write_output
from foretec_live.config import load_config
from foretec_live.data.entsoe import EntsoeSource
from foretec_live.data.tso import EliaSource

req, series = read_request()
cfg = load_config()
elia = EliaSource({**cfg, "_fallback": True})
inp = pd.read_parquet(req["input"])
results, errors = {}, {}
for key in series:
    zone, target = key.split("_")
    last = inp.loc[inp["series"] == key, "time_utc"].max()
    idx = pd.DatetimeIndex(last + pd.Timedelta("15min") * np.arange(1, req["horizon"][key] + 1))
    end = idx[-1] + pd.Timedelta("15min")
    tried = []
    for name, get in (("Elia 11:00", lambda: elia.dayahead_11h(target, idx[0], end) if zone == "BE" else pd.Series(dtype=float)),
                      ("ENTSO-E A69", lambda: EntsoeSource(cfg).tso_forecast(zone, target, idx[0], end))):
        try:
            s = get().reindex(idx)
        except Exception as e:  # noqa: BLE001 - try the next source
            tried.append(f"{name}: {type(e).__name__}")
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
