"""TimesFM 3.0, zero-shot. Env: timesfm."""
import numpy as np
import timesfm
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
H = max_horizon(req)
m = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
keys = list(series)
outs = list(m.predict_batch(contexts=[series[k].astype(np.float32) for k in keys], horizon=H, return_quantiles=True))
results = {}
for k, o in zip(keys, outs):
    q = np.asarray(o.quantiles, dtype=float).reshape(-1, np.asarray(o.quantiles).shape[-1])
    if q.shape[1] == 10:   # leading mean column, as in TimesFM 2.5
        q = q[:, 1:]
    results[k] = (np.asarray(o.forecast, dtype=float).reshape(-1), np.sort(q, axis=1) if q.shape[1] == 9 else None)
write_output(req, results)
