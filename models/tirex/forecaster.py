"""TiRex 35M, zero-shot. Env: tirex (tirex-ts), CPU torch backend."""
import numpy as np
import torch
from tirex import load_model
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
model = load_model("NX-AI/TiRex", device="cpu", backend="torch")
keys = list(series)
ctx = [torch.tensor(series[k]) for k in keys]                 # histories differ in length (price vs wind/solar cut-off)
q, mean = model.forecast(context=ctx, prediction_length=max_horizon(req), output_type="numpy")
write_output(req, {k: (q[i, :, 4], np.sort(q[i], axis=1)) for i, k in enumerate(keys)})
