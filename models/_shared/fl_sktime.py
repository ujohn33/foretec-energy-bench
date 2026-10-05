"""Run an sktime forecaster named in model.yaml inside a subprocess env (same spec as in-process sktime models).

model.yaml:
    estimator: StatsForecastAutoETS                 # any sktime forecaster, or an sktime spec string, e.g.
    params: {season_length: 96}                     #   "TransformedTargetForecaster([Deseasonalizer(sp=96), ...])"
    n_jobs: 4                                       # series fitted in parallel (default 4)
    nonnegative: true                               # clip wind, solar and load at zero (default true)

Used for estimators whose dependencies do not fit the main venv (StatsForecast needs pandas 2). Every series is
fitted on its own history with fh = 1..horizon; quantiles come from predict_quantiles when the estimator
supports them, otherwise the series gets a point forecast only.
"""
import inspect
import warnings

import numpy as np
import pandas as pd
import yaml
from fl_io import read_request, write_output
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
spec = yaml.safe_load(open("model.yaml"))


def make():
    from sktime.registry import all_estimators, craft
    est, params = spec["estimator"], spec.get("params") or {}
    if "(" in est:                       # a full sktime spec string
        return craft(est)
    cls = dict(all_estimators("forecaster"))[est]
    return cls(**params)


def one(key, values, h, quantiles, target):
    warnings.filterwarnings("ignore")
    y = pd.Series(values.astype(float), index=pd.RangeIndex(len(values)))
    f = make()
    if "remember_data" in inspect.signature(f.set_config).parameters or True:
        try:
            f.set_config(remember_data=False)
        except Exception:  # noqa: BLE001 - older sktime: config not known
            pass
    fh = np.arange(1, h + 1)
    f.fit(y, fh=fh)
    point = np.asarray(f.predict(), dtype=float).reshape(-1)
    q = None
    if f.get_tag("capability:pred_int", False):
        try:
            q = np.asarray(f.predict_quantiles(alpha=quantiles), dtype=float)
            q = np.sort(q, axis=1)
        except Exception:  # noqa: BLE001 - point forecast only
            q = None
    if target != "price" and spec.get("nonnegative", True):
        point = np.clip(point, 0, None)
        q = None if q is None else np.clip(q, 0, None)
    return key, point, q


req, series = read_request()
jobs = [(k, v, req["horizon"][k], req["quantiles"], k.split("_")[1]) for k, v in series.items()]
out = Parallel(n_jobs=int(spec.get("n_jobs", 4)), backend="loky")(delayed(one)(*j) for j in jobs)
write_output(req, {k: (p, q) for k, p, q in out})
