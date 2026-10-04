"""The Forecasting Company's t0 (tfc-t0, Apache-2.0), with or without known-future covariates. Env: tfc."""
import numpy as np
import pandas as pd
import torch
from t0 import T0Forecaster

from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output


def run(repo: str, covariates: bool) -> None:
    req, series = read_request()
    model = T0Forecaster.from_pretrained(repo).eval()
    inp = pd.read_parquet(req["input"])
    cov = load_covariates(req) if covariates else None
    qs = req["quantiles"]
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        kwargs = {}
        if covariates:
            _, _, cp, cf, _ = series_frames(req, cov, inp, key)
            c = np.concatenate([cp, cf]).astype(np.float32)               # (T + H, F)
            c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-6)              # comparable scales across covariates
            kwargs["future_covariates"] = torch.tensor(c.T[None])          # (1, F, T + H): known over context + horizon
        with torch.no_grad():
            out = model.predict(torch.tensor(y[None], dtype=torch.float32), horizon=h, quantile_levels=qs, **kwargs)
        q = np.asarray(out.quantiles, dtype=float)[0]                      # (H, Q)
        results[key] = nonneg(key, q[:, qs.index(0.5)], np.sort(q, axis=1))
    write_output(req, results)
