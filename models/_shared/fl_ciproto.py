"""ci-proto (Continuous Intelligence, November 2026), zero-shot. Env: ciproto.

Through Continuous Intelligence's package `continuousintelligence` (pinned in scripts/install_model_envs.sh); weights
from the Hub bundle pinned below (model.ckpt + tta.json, loaded with torch weights_only). ci-proto returns the
benchmark's nine levels (0.1 ... 0.9) directly; the point forecast is the median. Timestamps with freq="15min" let the
model build its own calendar features. With covariates=True the compact covariate set (NWP ensemble means/spreads,
ENTSO-E load forecast where the target gets it, fuel; fl_cov.compact_features) goes in as future covariates over the
history and the delivery day.

Run without the bundle's sign-flip test-time augmentation (tta_sign_flip: also forecasting the negated series); every
other setting of the released tta.json is kept. On the server's CPU the flip costs 2.5x (history only) to 6x (16
covariates) the time, which would not fit in the live run.

Licence: ci-proto Code and Model Licence 1.0 (Apache-2.0 based, commercial use allowed); a public product using it must
show "Uses ci-proto, released by Continuous Intelligence. Continuous Intelligence does not endorse or approve this
product or service." and notify Continuous Intelligence once.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

REPO, REVISION = "ContinuousIntelligence/ci-proto", "ee3031351a151e3f6febf8c62b53e6ea1008bef6"


def run(covariates: bool = False) -> None:
    import continuousintelligence as ci
    from huggingface_hub import snapshot_download
    req, series = read_request()
    bundle = snapshot_download(REPO, revision=REVISION, allow_patterns=["model.ckpt", "tta.json", "config.json"])
    tta = {**json.load(open(os.path.join(bundle, "tta.json"))), "tta_sign_flip": False}
    tta_file = os.path.join(tempfile.mkdtemp(), "tta.json")
    with open(tta_file, "w") as f:
        json.dump(tta, f)
    model = ci.load_model(os.path.join(bundle, "model.ckpt"), tta_config=tta_file, device="cpu")
    levels = np.asarray(model.quantiles, dtype=float)
    inp = pd.read_parquet(req["input"])
    cov = load_covariates(req) if covariates else None
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        t_hist = pd.DatetimeIndex(inp.loc[inp["series"] == key, "time_utc"].sort_values())
        stamps = t_hist.append(pd.DatetimeIndex(t_hist[-1] + pd.Timedelta("15min") * np.arange(1, h + 1)))
        kw = {"timestamps": torch.tensor(stamps.as_unit("ns").asi8, dtype=torch.int64), "freq": "15min"}
        if covariates:
            _, _, cp, cf, _ = series_frames(req, cov, inp, key)
            kw["future_covariates"] = torch.tensor(np.concatenate([cp, cf]).T, dtype=torch.float32)   # [C, T + H]
        with torch.no_grad():
            out = model.forecast(torch.tensor(np.asarray(y, dtype=np.float32))[None], horizon=h, **kw)
        q = np.asarray(out, dtype=float)[0].T                                                      # [H, 9]
        cols = [int(np.argmin(np.abs(levels - p))) for p in req["quantiles"]]
        results[key] = nonneg(key, q[:, int(np.argmin(np.abs(levels - 0.5)))], np.sort(q[:, cols], axis=1))
    write_output(req, results)
