"""NX-AI TiRex-2 (univariate mode), zero-shot quantile forecasts; each series on its own history."""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # one block is @torch.compile'd; eager needs no C++ compiler
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fl_cov import nonneg  # noqa: E402
from fl_io import read_request, write_output  # noqa: E402
from tirex2 import TimeseriesType, load_model  # noqa: E402

req, series = read_request()
model = load_model("NX-AI/TiRex-2", device="cpu")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # the checkpoint's quantile levels
assert req["quantiles"] == LEVELS, req["quantiles"]
results = {}
for key, y in series.items():
    h = req["horizon"][key]
    ts = TimeseriesType(target=torch.tensor(y, dtype=torch.float32).unsqueeze(0), past_covariates=None, future_covariates=None)
    q = np.asarray(model.forecast([ts], prediction_length=h, output_type="numpy")[0])[0].T   # [H, 9]
    results[key] = nonneg(key, q[:, LEVELS.index(0.5)], np.sort(q, axis=1))
write_output(req, results)
