"""Timestral (Mistral AI, October 2026), zero-shot. Env: timestral.

Timestral-209M, the largest of the four sizes, through Mistral's standalone inference package (pinned in
scripts/install_model_envs.sh). It returns 21 quantiles (0.01 ... 0.99); the benchmark's levels are taken from the
nearest ones. The package defaults are kept: symmetric averaging (the context and its negation are forecast and
averaged) and clipping at zero for series whose context is never negative. With covariates=True the compact covariate
set (NWP ensemble means/spreads, ENTSO-E load forecast, fuel; fl_cov.compact_features) goes in as known-future
covariates over the history and the delivery day, which Timestral reads natively.

Weights: Modified MIT licence (no use by companies with more than $20M monthly revenue); inference code Apache-2.0.
"""
from __future__ import annotations

import numpy as np
import torch

from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

REPO, REVISION = "mistralai/Timestral-209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"


def run(covariates: bool = False) -> None:
    from timestral import TimestralForecaster
    req, series = read_request()
    fc = TimestralForecaster.from_pretrained(REPO, revision=REVISION)
    if covariates:
        import pandas as pd
        cov, inp = load_covariates(req), pd.read_parquet(req["input"])
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        y = np.asarray(y, dtype=np.float32)
        kw = {}
        if covariates:
            _, _, cp, cf, _ = series_frames(req, cov, inp, key)
            kw["known_covariates"] = np.concatenate([cp, cf]).T[None].astype(np.float32)   # (1, C, T + H)
        with torch.no_grad():
            f = fc.forecast(y[None, None], prediction_length=h, **kw)
        levels = np.asarray(f.quantile_levels)
        q = np.asarray(f.quantiles, dtype=float)[0, 0]                                     # (H, 21)
        cols = [int(np.argmin(np.abs(levels - p))) for p in req["quantiles"]]
        results[key] = nonneg(key, q[:, int(np.argmin(np.abs(levels - 0.5)))], np.sort(q[:, cols], axis=1))
    write_output(req, results)
