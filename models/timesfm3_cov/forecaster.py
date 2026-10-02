"""TimesFM 3.0 with known-future covariates (past_future_covariates: n_cov x (T + H))."""
import numpy as np
import pandas as pd
import timesfm
from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

req, series = read_request()
cov, inp = load_covariates(req), pd.read_parquet(req["input"])
m = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
results = {}
for key, y in series.items():
    t_hist, t_fut, cp, cf, _ = series_frames(req, cov, inp, key)
    pf = np.concatenate([cp, cf]).T                                    # (n_cov, T + H)
    o = m.predict(context=y.astype(np.float32), horizon=len(t_fut), past_future_covariates=pf, return_quantiles=True)
    q = np.asarray(o.quantiles, dtype=float).reshape(-1, np.asarray(o.quantiles).shape[-1])
    q = q[:, 1:] if q.shape[1] == 10 else q
    results[key] = nonneg(key, np.asarray(o.forecast, dtype=float).reshape(-1), np.sort(q, axis=1))
write_output(req, results)
