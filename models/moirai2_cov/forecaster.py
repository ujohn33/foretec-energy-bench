"""Moirai 2.0 small with known-future covariates, through the GluonTS predictor.

Moirai2Forecast.predict() slices every input to the context length, which cuts the future part off
feat_dynamic_real; the GluonTS predictor aligns past and future covariates correctly.
"""
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from fl_cov import load_covariates, nonneg, series_frames
from fl_io import read_request, write_output

req, series = read_request()
cov, inp = load_covariates(req), pd.read_parquet(req["input"])
module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
results = {}
for key, y in series.items():
    t_hist, t_fut, cp, cf, _ = series_frames(req, cov, inp, key)
    model = Moirai2Forecast(module=module, prediction_length=len(t_fut), context_length=len(y), target_dim=1,
                            feat_dynamic_real_dim=cp.shape[1], past_feat_dynamic_real_dim=0)
    ds = ListDataset([{"start": pd.Period(t_hist[0], freq="15min"), "target": y.astype(np.float32),
                       "feat_dynamic_real": np.concatenate([cp, cf]).T.astype(np.float32)}], freq="15min")
    f = next(iter(model.create_predictor(batch_size=1).predict(ds)))
    q = np.column_stack([f.quantile(a) for a in req["quantiles"]])
    results[key] = nonneg(key, f.quantile(0.5), np.sort(q, axis=1))
write_output(req, results)
