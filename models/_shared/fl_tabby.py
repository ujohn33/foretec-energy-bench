"""Tabby and Tabby-Pretrain (Huawei Noah's Ark Paris, CC BY-NC 4.0), zero-shot on the target's own history. Env: tabby.

Tabby is the frozen Tabby-Pretrain backbone plus a prompt post-trained on GIFT-Eval training tasks; Tabby-Pretrain is
the backbone alone, run through the package's predict path with the prompt switched off. Both forecast each series
on its own (no covariates) and return 99 quantiles (0.01 ... 0.99). As in the release, a span of max(H, 128) steps is
forecast and the first H kept. Weights and remote code are pinned to the revisions below.
"""
from __future__ import annotations

import numpy as np
import torch

from fl_cov import nonneg
from fl_io import read_request, write_output

REVISIONS = {"paris-noah/Tabby": "1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01",
             "paris-noah/Tabby-Pretrain": "b26c277bf96087712ea3ae65740947ffe276ae61"}
MIN_SPAN = 128
LEVELS = np.round(np.arange(1, 100) / 100, 2)


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


def run(repo: str) -> None:
    req, series = read_request()
    predict = load(repo)
    qs = req["quantiles"]
    cols = [int(np.argmin(np.abs(LEVELS - q))) for q in qs]
    mid = int(np.argmin(np.abs(LEVELS - 0.5)))
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        span = max(h, MIN_SPAN)
        x = torch.tensor(np.asarray(y, dtype=np.float32)[-(8192 - span):][None])
        with torch.no_grad():
            out = np.asarray(predict(x, span), dtype=float)[0][:, :h]                    # [99, h]
        results[key] = nonneg(key, out[mid], np.sort(out[cols].T, axis=1))
    write_output(req, results)
