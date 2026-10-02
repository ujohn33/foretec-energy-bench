"""Chronos-Bolt base, zero-shot. Env: chronos. Context limit 2048 steps."""
import numpy as np
import torch
from chronos import ChronosBoltPipeline
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
keys = list(series)
ctx = [torch.tensor(series[k][-2048:]) for k in keys]
q, _ = pipe.predict_quantiles(ctx, prediction_length=max_horizon(req), quantile_levels=req["quantiles"])
q = q.float().numpy()                                       # (B, H, Q)
write_output(req, {k: (q[i, :, req["quantiles"].index(0.5)], np.sort(q[i], axis=1)) for i, k in enumerate(keys)})
