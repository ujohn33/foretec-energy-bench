"""Chronos-2 with known-future covariates, one predict call per series."""
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from fl_cov import compact_features, future_index, load_covariates
from fl_io import read_request, write_output

req, series = read_request()
cov = load_covariates(req)
inp = pd.read_parquet(req["input"])
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)

results = {}
for key in series:   # one call per series: Chronos-2 cannot batch series with different covariate sets
    zone, target = key.split("_")
    h = req["horizon"][key]
    t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
    t_fut = future_index(t_hist[-1], h)
    cols = compact_features(cov.columns, zone, target)
    c = cov.reindex(t_hist.append(t_fut))[cols].interpolate(limit_direction="both").fillna(0.0)
    inputs = [{
        "target": series[key],
        "past_covariates": {n: c.loc[t_hist, n].to_numpy(np.float32) for n in cols},
        "future_covariates": {n: c.loc[t_fut, n].to_numpy(np.float32) for n in cols},
    }]
    q, _ = pipe.predict_quantiles(inputs, prediction_length=h, quantile_levels=req["quantiles"])
    qk = q[0].squeeze(0).float().numpy()
    point = qk[:, req["quantiles"].index(0.5)]
    if target != "price":
        point, qk = np.clip(point, 0, None), np.clip(qk, 0, None)
    results[key] = (point, np.sort(qk, axis=1))
write_output(req, results)
