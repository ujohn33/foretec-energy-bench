"""Toto Open Base 1.0, covariates as trailing variates; known future values injected during decoding."""
import numpy as np
import pandas as pd
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

req, series = read_request()
cov, inp = load_covariates(req), pd.read_parquet(req["input"])
toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to("cpu")
toto.eval()
fc = TotoForecaster(toto.model)
results = {}
for key, y in series.items():
    t_hist, t_fut, cp, cf, _ = series_frames(req, cov, inp, key)
    x = torch.tensor(np.vstack([y[None, :], cp.T]), dtype=torch.float32)[:, -2048:]   # target first, exogenous last; 2048 steps keeps memory in bounds
    k = cp.shape[1]
    inp_ts = MaskedTimeseries(series=x, padding_mask=torch.ones_like(x, dtype=torch.bool), id_mask=torch.zeros_like(x, dtype=torch.int),
                              timestamp_seconds=torch.zeros_like(x, dtype=torch.int),
                              time_interval_seconds=torch.full((x.shape[0],), 900, dtype=torch.int), num_exogenous_variables=k)
    with torch.no_grad():
        f = fc.forecast(inp_ts, prediction_length=len(t_fut), num_samples=64, samples_per_batch=16,
                        future_exogenous_variables=torch.tensor(cf.T[None, :, :], dtype=torch.float32))
    s = f.samples.squeeze(0)[0].cpu().numpy()                          # target variate: (H, samples)
    results[key] = nonneg(key, np.median(s, axis=-1), np.quantile(s, req["quantiles"], axis=-1).T)
write_output(req, results)
