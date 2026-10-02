"""Sundial base 128M, zero-shot generative. Env: sundial (transformers 4.40.1, trust_remote_code)."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from fl_io import max_horizon, read_request, write_output

req, series = read_request()
model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True)
keys = list(series)
n = min(2880, min(len(series[k]) for k in keys))             # equal lengths for one batch
ctx = torch.tensor(np.stack([series[k][-n:] for k in keys]), dtype=torch.float32)
with torch.no_grad():
    s = model.generate(ctx, max_new_tokens=max_horizon(req), num_samples=50).numpy()   # (B, samples, H)
write_output(req, {k: (np.median(s[i], axis=0), np.quantile(s[i], req["quantiles"], axis=0).T) for i, k in enumerate(keys)})
