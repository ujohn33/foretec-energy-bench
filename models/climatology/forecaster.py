"""Climatology: same Brussels time of day, +/-14 calendar days, every year in data/static/climatology_history.parquet."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fl_io import read_request, write_output

WINDOW_DAYS = 14
TZ = "Europe/Brussels"

req, series = read_request()
hist = pd.read_parquet(Path(os.environ["FORETEC_HOME"]) / "data" / "static" / "climatology_history.parquet")
loc = hist["time_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ)
hist = hist.assign(qh=loc.dt.hour * 4 + loc.dt.minute // 15, doy=loc.dt.dayofyear)
inp = pd.read_parquet(req["input"])
last = inp.groupby("series")["time_utc"].max()

results = {}
for key in series:
    zone, target = key.split("_")
    h = hist[(hist["zone"] == zone) & (hist["target"] == target)].dropna(subset=["value"])
    steps = req["horizon"][key]
    t = pd.DatetimeIndex(last[key] + pd.Timedelta("15min") * np.arange(1, steps + 1)).tz_localize("UTC").tz_convert(TZ)
    point, quant = np.empty(steps), np.empty((steps, len(req["quantiles"])))
    for i, ts in enumerate(t):
        dd = np.abs(h["doy"].to_numpy() - ts.dayofyear)
        dd = np.minimum(dd, 365 - dd)                                    # circular day-of-year distance
        v = h["value"].to_numpy()[(h["qh"].to_numpy() == ts.hour * 4 + ts.minute // 15) & (dd <= WINDOW_DAYS)]
        point[i] = v.mean()
        quant[i] = np.quantile(v, req["quantiles"])
    results[key] = (point, quant)
write_output(req, results)
