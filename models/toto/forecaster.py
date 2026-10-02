"""Toto Open Base 1.0, zero-shot, sample paths. Env: toto (toto-ts)."""
import numpy as np
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to("cpu")
toto.eval()
fc = TotoForecaster(toto.model)
results = {}
for k, y in series.items():
    x = torch.tensor(y[-4096:], dtype=torch.float32)[None, :]                 # (variates=1, T)
    inp = MaskedTimeseries(series=x, padding_mask=torch.ones_like(x, dtype=torch.bool),
                           id_mask=torch.zeros_like(x, dtype=torch.int), timestamp_seconds=torch.zeros_like(x, dtype=torch.int),
                           time_interval_seconds=torch.full((1,), 900, dtype=torch.int))
    with torch.no_grad():
        f = fc.forecast(inp, prediction_length=req["horizon"][k], num_samples=128, samples_per_batch=128)
    s = f.samples.squeeze().cpu().numpy()                                     # (H, samples)
    results[k] = (np.median(s, axis=-1), np.quantile(s, req["quantiles"], axis=-1).T)
write_output(req, results)
