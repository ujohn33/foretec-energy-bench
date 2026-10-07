"""NX-AI TiRex-2 with known-future covariates (NWP ensemble, load forecast, fuel, calendar): the same compact
feature set as chronos2_cov, passed as future-known covariates over history plus horizon."""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # one block is @torch.compile'd; eager needs no C++ compiler
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from fl_cov import load_covariates, nonneg, series_frames  # noqa: E402
from fl_io import read_request, write_output  # noqa: E402
from tirex2 import TimeseriesType, load_model  # noqa: E402

req, series = read_request()
cov = load_covariates(req)
inp = pd.read_parquet(req["input"])
model = load_model("NX-AI/TiRex-2", device="cpu")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
assert req["quantiles"] == LEVELS, req["quantiles"]
results = {}
for key, y in series.items():
    h = req["horizon"][key]
    t_hist, t_fut, past, fut, cols = series_frames(req, cov, inp, key)
    # standardise each covariate on its history so wind speeds, MW and EUR sit on comparable scales
    both = np.vstack([past, fut])
    mu, sd = past.mean(axis=0), past.std(axis=0) + 1e-6
    future = torch.tensor(((both - mu) / sd).T, dtype=torch.float32) if cols else None   # [k, T+H]
    ts = TimeseriesType(target=torch.tensor(y, dtype=torch.float32).unsqueeze(0), past_covariates=None, future_covariates=future)
    q = np.asarray(model.forecast([ts], prediction_length=h, output_type="numpy")[0])[0].T   # [H, 9]
    results[key] = nonneg(key, q[:, LEVELS.index(0.5)], np.sort(q, axis=1))
write_output(req, results)
