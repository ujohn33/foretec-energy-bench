"""TimesFM 2.5 with in-context covariate regression (xreg + timesfm)."""
import numpy as np
import pandas as pd
import timesfm
from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

req, series = read_request()
cov, inp = load_covariates(req), pd.read_parquet(req["input"])
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=3072, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                                 force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True, return_backcast=True))
results = {}
for key, y in series.items():
    t_hist, t_fut, cp, cf, cols = series_frames(req, cov, inp, key)
    dyn = {n: [np.concatenate([cp[:, j], cf[:, j]]).astype(float)] for j, n in enumerate(cols)}
    out, _ = m.forecast_with_covariates(inputs=[y.astype(float)], dynamic_numerical_covariates=dyn, xreg_mode="xreg + timesfm", ridge=1.0)
    point = np.asarray(out[0], dtype=float).reshape(-1)[: len(t_fut)]
    results[key] = nonneg(key, point, None)       # the xreg path returns point forecasts only
write_output(req, results)
