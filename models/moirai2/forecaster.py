"""Moirai 2.0 small, zero-shot. Env: moirai (uni2ts from github)."""
import numpy as np
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
keys = list(series)
T = len(next(iter(series.values())))
model = Moirai2Forecast(module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                        prediction_length=max_horizon(req), context_length=T,
                        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
levels = list(np.round(np.asarray(model.module.quantile_levels, dtype=float), 3))
with torch.no_grad():
    out = np.asarray(model.predict([series[k][:, None] for k in keys]))   # (B, Q, H[, 1])
out = out.reshape(out.shape[0], out.shape[1], -1)
cols = [levels.index(round(a, 3)) for a in req["quantiles"]]
write_output(req, {k: (out[i, levels.index(0.5)], np.sort(out[i, cols].T, axis=1)) for i, k in enumerate(keys)})
