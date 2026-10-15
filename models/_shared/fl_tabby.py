"""Tabby and Tabby-Pretrain (Huawei Noah's Ark Paris, CC BY-NC 4.0), zero-shot on the target's own history. Env: tabby.

Tabby is the frozen Tabby-Pretrain backbone plus a prompt post-trained on GIFT-Eval training tasks; Tabby-Pretrain is
the backbone alone, run through the package's predict path with the prompt switched off. Both forecast each series
on its own (no covariates) and return 99 quantiles (0.01 ... 0.99). As in the release, a span of max(H, 128) steps is
forecast and the first H kept. Weights and remote code are pinned to the revisions below.

The _cov variants: Tabby takes no covariates (its context_aware option only summarises the target's own history), so
covariates enter as in TimesFM's "xreg + timesfm" mode: a ridge regression of the history on the known-future
covariates (NWP ensemble means/spreads, fuel; fl_cov.compact_features) explains what they can, Tabby forecasts the
residual, and the ridge forecast over the delivery day is added back to every quantile. Load is forecast without the
ridge: with weather alone the ridge stands in for the daily cycle (radiation ~ daytime) and a sunny or cloudy delivery
day shifts it by up to 2 GW. Backtest 14 Sep - 1 Oct 2026, Rel. MAE vs naive_weekly, Tabby alone against Tabby on the
weather-ridge residual: NL 0.65 vs 1.9 (1.16 with calendar terms in the ridge), BE 0.64 vs 0.65 (with calendar terms).
"""
from __future__ import annotations

import numpy as np
import torch

from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

REVISIONS = {"paris-noah/Tabby": "1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01",
             "paris-noah/Tabby-Pretrain": "b26c277bf96087712ea3ae65740947ffe276ae61"}
MIN_SPAN = 128
LEVELS = np.round(np.arange(1, 100) / 100, 2)
RIDGE = 1.0


def load(repo: str):
    if repo == "paris-noah/Tabby":
        from transformers import AutoModel
        m = AutoModel.from_pretrained(repo, revision=REVISIONS[repo], trust_remote_code=True).eval()
        return lambda x, span: m.predict(x, prediction_length=span)                     # [B, 99, span]
    from huggingface_hub import snapshot_download
    from tabby.posttraining.model import PatchTSTFMPromptCFG, PromptedPatchTSTFM, load_patchtstfm_backbone
    path = snapshot_download(repo, revision=REVISIONS[repo])
    backbone, _, _ = load_patchtstfm_backbone(path, device=torch.device("cpu"))
    cache = {}

    def predict(x, span):
        if span not in cache:
            cfg = PatchTSTFMPromptCFG(prompt_len=0, context_aware=False, prediction_length=span, context_length=8192 - span)
            cache[span] = PromptedPatchTSTFM(backbone, cfg).eval()
        return cache[span](context=x, prediction_length=span)["quantile_preds"]
    return predict


def ridge(cp: np.ndarray, cf: np.ndarray, y: np.ndarray):
    """Fitted values over the history and forecast over the horizon of a ridge on standardised covariates."""
    mu, sd = cp.mean(0), cp.std(0) + 1e-6
    A, F = np.c_[np.ones(len(cp)), (cp - mu) / sd], np.c_[np.ones(len(cf)), (cf - mu) / sd]
    ok = ~np.isnan(y)
    pen = RIDGE * np.eye(A.shape[1])
    pen[0, 0] = 0.0                                                                     # intercept not penalised
    beta = np.linalg.solve(A[ok].T @ A[ok] + pen, A[ok].T @ y[ok])
    return A @ beta, F @ beta


def run(repo: str, covariates: bool = False) -> None:
    req, series = read_request()
    predict = load(repo)
    qs = req["quantiles"]
    cols = [int(np.argmin(np.abs(LEVELS - q))) for q in qs]
    mid = int(np.argmin(np.abs(LEVELS - 0.5)))
    if covariates:
        import pandas as pd
        cov, inp = load_covariates(req), pd.read_parquet(req["input"])
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        span = max(h, MIN_SPAN)
        y = np.asarray(y, dtype=float)
        base = np.zeros(h)
        if covariates and not key.endswith("_load"):
            _, _, cp, cf, _ = series_frames(req, cov, inp, key)
            fit, base = ridge(cp.astype(float), cf.astype(float), y)
            y = y - fit
        x = torch.tensor(y.astype(np.float32)[-(8192 - span):][None])
        with torch.no_grad():
            out = np.asarray(predict(x, span), dtype=float)[0][:, :h] + base             # [99, h]
        results[key] = nonneg(key, out[mid], np.sort(out[cols].T, axis=1))
    write_output(req, results)
