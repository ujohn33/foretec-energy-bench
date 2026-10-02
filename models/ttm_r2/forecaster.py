"""IBM Granite TinyTimeMixer r2, zero-shot point forecast. Env: ttm (granite-tsfm)."""
import numpy as np
import torch
from tsfm_public.toolkit.get_model import get_model
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
H = max_horizon(req)
CTX = 1536
model = get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=CTX, prediction_length=H)
model.eval()
keys = list(series)
x = np.stack([series[k][-CTX:] for k in keys]).astype(np.float32)
mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
with torch.no_grad():
    out = model(past_values=torch.tensor((x - mu) / sd)[:, :, None]).prediction_outputs.numpy()[:, :, 0]
out = out * sd + mu
write_output(req, {k: (out[i], None) for i, k in enumerate(keys)})
