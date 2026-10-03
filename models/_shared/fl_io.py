"""I/O contract for subprocess models. Needs only numpy, pandas and pyarrow, so it works in any model env.

The runner calls   <env python> forecaster.py request.json   with

request.json  {"input": ..., "output": ..., "quantiles": [0.1, ..., 0.9], "freq": "15min",
               "horizon": {"BE_price": 96, "BE_wind": 156, ...}}
input         parquet: series, time_utc, value      (gap-free 15-min history, oldest first)
output        parquet: series, step, point, q0.1 ... q0.9   (step 1 = first quarter-hour after the history)

A model script only has to turn {series: history array} into {series: (point[h], quantiles[h, Q] or None)}.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd


def read_request(argv=None) -> tuple[dict, dict[str, np.ndarray]]:
    argv = sys.argv if argv is None else argv
    with open(argv[1]) as f:
        req = json.load(f)
    df = pd.read_parquet(req["input"])
    series = {k: g.sort_values("time_utc")["value"].to_numpy(dtype="float32") for k, g in df.groupby("series", sort=False)}
    return req, series


def write_output(req: dict, results: dict[str, tuple], errors: dict[str, str] | None = None) -> None:
    """errors: why a series has no forecast (shown in the run tracker instead of a generic message)."""
    if errors:
        with open(req["output"] + ".errors.json", "w") as f:
            json.dump(errors, f)
    if not results:
        raise SystemExit("no series forecast: " + "; ".join(f"{k}: {v}" for k, v in (errors or {}).items()))
    frames = []
    for key, (point, q) in results.items():
        point = np.asarray(point, dtype=float).reshape(-1)
        h = req["horizon"][key]
        if len(point) < h:
            raise ValueError(f"{key}: model returned {len(point)} steps, need {h}")
        d = {"series": key, "step": np.arange(1, h + 1), "point": point[:h]}
        if q is not None:
            q = np.asarray(q, dtype=float)
            for j, a in enumerate(req["quantiles"]):
                d[f"q{a:.1f}"] = q[:h, j]
        frames.append(pd.DataFrame(d))
    pd.concat(frames, ignore_index=True).to_parquet(req["output"], index=False)


def max_horizon(req: dict) -> int:
    return max(req["horizon"].values())
