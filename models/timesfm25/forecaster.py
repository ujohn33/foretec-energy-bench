"""TimesFM 2.5 200M, zero-shot. Env: timesfm (github google-research/timesfm)."""
import numpy as np
import timesfm
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
H = max_horizon(req)
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=3072, max_horizon=256, normalize_inputs=True,
                                 use_continuous_quantile_head=True, force_flip_invariance=True,
                                 infer_is_positive=True, fix_quantile_crossing=True))
keys = list(series)
point, quant = m.forecast(horizon=H, inputs=[series[k].astype(np.float64) for k in keys])
# quant[..., 0] is the mean, 1..9 are the 10th..90th percentiles
write_output(req, {k: (point[i], np.sort(quant[i, :, 1:10], axis=1)) for i, k in enumerate(keys)})
