"""Simons (Abel AI Lab, November 2026), zero-shot. Env: simons.

Simons-297M, the largest of the four released sizes, through Abel's package (pinned in scripts/install_model_envs.sh;
code Apache-2.0, weights OpenMDW-1.1). It returns nine quantiles, 0.1 ... 0.9, the benchmark's levels; the point
forecast is the median. With covariates=True the compact covariate set (NWP ensemble means/spreads, ENTSO-E load
forecast where the target gets it, fuel; fl_cov.compact_features) goes in as known-future covariates over the history
and the delivery day. Simons reads at most 16 covariates, taken in fl_cov's order (own zone first).
"""
from __future__ import annotations

import numpy as np
import torch

from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

REPO, REVISION = "abel-lab/Simons-297M", "335dc22e4ce26ca0a8bc567de9e93dbb1bd2ce3a"


def run(covariates: bool = False) -> None:
    from simons import Simons
    req, series = read_request()
    model = Simons.from_pretrained(REPO, device="cpu", revision=REVISION)
    if covariates:
        import pandas as pd
        cov, inp = load_covariates(req), pd.read_parquet(req["input"])
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        kw = {}
        if covariates:
            _, _, cp, cf, _ = series_frames(req, cov, inp, key)
            kw["covariates"] = torch.tensor(np.concatenate([cp, cf]).T[None], dtype=torch.float32)    # [1, C, T + H]
        with torch.no_grad():
            q = model.forecast(torch.tensor(np.asarray(y, dtype=np.float32))[None], horizon=h, **kw)
        q = np.asarray(q, dtype=float)[0]                                                          # [H, 9]
        results[key] = nonneg(key, q[:, 4], np.sort(q, axis=1))
    write_output(req, results)
