"""Toto 2.0 (Datadog, Apache-2.0), zero-shot on the target's own history. Env: toto2.

Toto 2.0 has no covariate input yet. Its quantile head returns exactly the benchmark's nine levels (0.1 ... 0.9); the
point forecast is the median. One forward pass over the delivery day (no block decoding, as in Datadog's leaderboard
runs); the context is trimmed to a whole number of patches. Weights pinned per size below.
"""
import numpy as np
import torch

from fl_cov import nonneg
from fl_io import read_request, write_output

REVISIONS = {"Datadog/Toto-2.0-313m": "a7bab288f5e95f8606f8306f86659357e1c001ef",
             "Datadog/Toto-2.0-1B": "1604e1a5242884fb9848f88c4ced14f4dc62d9d3"}


def run(repo: str) -> None:
    from toto2 import Toto2Model
    req, series = read_request()
    model = Toto2Model.from_pretrained(repo, revision=REVISIONS[repo]).eval()
    patch = model.config.patch_size
    results = {}
    for key, y in series.items():
        h = req["horizon"][key]
        y = np.asarray(y, dtype=np.float32)
        y = y[len(y) % patch:]
        target = torch.tensor(y)[None, None]
        with torch.no_grad():
            q = model.forecast({"target": target, "target_mask": torch.isfinite(target),
                                "series_ids": torch.zeros(1, 1, dtype=torch.long)},
                               horizon=h, has_missing_values=bool(np.isnan(y).any()))
        q = np.asarray(q, dtype=float)[:, 0, 0, :h].T                    # (h, 9): 0.1 ... 0.9
        results[key] = nonneg(key, q[:, 4], np.sort(q, axis=1))
    write_output(req, results)
