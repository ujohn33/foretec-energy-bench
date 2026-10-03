"""TSO day-ahead wind/solar forecast from ENTSO-E (A69, process A01), as published when this runs."""
import numpy as np
import pandas as pd
from fl_io import read_request, write_output
from foretec_live.config import load_config
from foretec_live.data.entsoe import EntsoeSource

req, series = read_request()
src = EntsoeSource(load_config())
inp = pd.read_parquet(req["input"])
results, errors = {}, {}
for key in series:
    zone, target = key.split("_")
    last = inp.loc[inp["series"] == key, "time_utc"].max()
    idx = pd.DatetimeIndex(last + pd.Timedelta("15min") * np.arange(1, req["horizon"][key] + 1))
    s = src.tso_forecast(zone, target, idx[0], idx[-1] + pd.Timedelta("15min")).reindex(idx)
    day = s.iloc[-96:]                                   # the delivery day is the end of the horizon
    if day.notna().mean() < 0.95:
        errors[key] = f"TSO forecast not published (yet): {int(day.notna().sum())} of {len(day)} quarter-hours of D"
        continue
    results[key] = (s.ffill().bfill().to_numpy(), None)
write_output(req, results, errors)
