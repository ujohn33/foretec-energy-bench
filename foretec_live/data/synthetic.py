"""Deterministic synthetic data for tests and offline development. Not a real market."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .base import Source

_START, _END = pd.Timestamp("2025-01-01"), pd.Timestamp("2028-01-01")
_CACHE: dict[tuple[str, str], pd.Series] = {}


def _make(zone: str, target: str) -> pd.Series:
    idx = pd.date_range(_START, _END, freq="15min", inclusive="left")
    rng = np.random.default_rng(zlib.crc32(f"{zone}-{target}".encode()))
    n = len(idx)
    hour = idx.hour + idx.minute / 60
    doy = idx.dayofyear.values
    ar = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        ar[i] = 0.995 * ar[i - 1] + eps[i]
    if target == "price":
        v = 80 + 25 * np.sin((hour - 7) / 24 * 2 * np.pi) + 10 * (idx.dayofweek < 5) + 1.5 * ar + rng.normal(0, 4, n)
    elif target == "wind":
        v = np.clip(2500 + 600 * ar + 400 * np.cos(doy / 365 * 2 * np.pi), 0, None)
    else:
        sun = np.clip(np.sin((hour - 6) / 14 * np.pi), 0, None)
        season = 0.6 + 0.4 * np.sin((doy - 80) / 365 * 2 * np.pi)
        cloud = 1 / (1 + np.exp(-0.05 * ar))
        v = 9000 * sun * season * cloud
    return pd.Series(np.asarray(v, dtype=float), index=idx)


class SyntheticSource(Source):
    name = "synthetic"

    def fetch(self, zone, target, start, end):
        key = (zone, target)
        if key not in _CACHE:
            _CACHE[key] = _make(zone, target)
        s = _CACHE[key]
        return s[(s.index >= start) & (s.index < end)].rename_axis("time_utc")
