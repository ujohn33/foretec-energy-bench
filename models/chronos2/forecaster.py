"""Chronos-2, zero-shot. Env: chronos (chronos-forecasting>=2)."""
import numpy as np
import torch
from chronos import Chronos2Pipeline
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
keys = list(series)
q, _ = pipe.predict_quantiles([series[k] for k in keys], prediction_length=max_horizon(req), quantile_levels=req["quantiles"])
results = {}
for k, qk in zip(keys, q):
    qk = qk.squeeze(0).float().numpy()                     # (H, Q)
    results[k] = (qk[:, req["quantiles"].index(0.5)], np.sort(qk, axis=1))
write_output(req, results)
