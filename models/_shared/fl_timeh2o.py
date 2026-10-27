"""TimeH2O-v1 (H2O.ai, October 2026, 108M parameters), zero-shot. Env: timeh2o.

Through H2O.ai's standalone inference package h2ofm (Apache-2.0, pinned in scripts/install_model_envs.sh). The weights
are gated on Hugging Face under H2O.ai's non-commercial model licence: the server reads them with the HF_TOKEN of an
account that accepted it. It returns the benchmark's nine quantile levels directly; the point forecast is the median.
Package defaults are kept (flip-equivariant averaging, patch ensemble). With covariates=True the compact covariate set
(NWP ensemble means/spreads, ENTSO-E load forecast where a target gets it, fuel; fl_cov.compact_features) goes in as
future covariates over the history and the delivery day, which TimeH2O reads natively.
"""
from __future__ import annotations

import numpy as np

from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

REPO, REVISION = "h2oai/timeh2o-v1", "1310be782553943a4c520d2810a98ce8f56b2709"


def run(covariates: bool = False) -> None:
    from h2ofm import TimeH2O
    req, series = read_request()
    model = TimeH2O.from_pretrained(REPO, revision=REVISION, device="cpu")
    if covariates:
        import pandas as pd
        cov, inp = load_covariates(req), pd.read_parquet(req["input"])
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        kw = {}
        if covariates:
            _, _, cp, cf, _ = series_frames(req, cov, inp, key)
            kw["future_covariates"] = np.concatenate([cp, cf]).T.astype(np.float64)    # (C, T + H)
        f = model.forecast(np.asarray(y, dtype=np.float64), h, quantile_levels=req["quantiles"], **kw)
        q = np.asarray(f.quantiles, dtype=float)                                        # (H, 9)
        results[key] = nonneg(key, np.asarray(f.point, dtype=float), np.sort(q, axis=1))
    write_output(req, results)
