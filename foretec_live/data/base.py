"""Data source interface. A source returns one series per (zone, target) on a UTC-naive 15-minute grid."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..timeutil import FREQ

log = logging.getLogger(__name__)


class Source:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fetch(self, zone: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Return values for [start, end) as a float Series indexed by UTC-naive timestamps."""
        raise NotImplementedError


def to_15min(s: pd.Series) -> pd.Series:
    """Put a series on the 15-minute grid. Hourly values are held flat across their four quarter-hours."""
    s = s.dropna().astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return s
    step = s.index.to_series().diff().median()
    if step is not pd.NaT and step > pd.Timedelta(FREQ):
        # coarser than 15 min (e.g. hourly): extend to the end of the last period, then hold flat
        full = pd.date_range(s.index[0], s.index[-1] + step - pd.Timedelta(FREQ), freq=FREQ)
        s = s.reindex(full).ffill(limit=int(step / pd.Timedelta(FREQ)) - 1)
    else:
        s = s.resample(FREQ).mean()
    s.index.name = "time_utc"
    return s


class FallbackSource(Source):
    """Primary source first; quarter-hours it does not deliver are filled from the fallback.

    `provenance[(zone, target)]` counts, for the last fetch, how many points came from each source.
    """

    def __init__(self, cfg: dict, primary: Source, fallback: Source):
        super().__init__(cfg)
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+{fallback.name}"
        self.provenance: dict = {}
        self.fallback_down = False   # circuit breaker: after one failure, skip the fallback for this run

    def fetch(self, zone, target, start, end):
        idx = pd.date_range(start, end, freq=FREQ, inclusive="left")
        try:
            p = self.primary.fetch(zone, target, start, end).reindex(idx)
        except Exception as e:  # noqa: BLE001 - a dead primary must not stop the run
            log.warning("%s %s %s failed (%s); using %s", self.primary.name, zone, target, e, self.fallback.name)
            p = pd.Series(np.nan, index=idx)
        out, n_fb = p, 0
        if p.isna().any() and not self.fallback_down:
            try:
                f = self.fallback.fetch(zone, target, start, end).reindex(idx)
                out = p.combine_first(f)
                n_fb = int((p.isna() & f.notna()).sum())
            except Exception as e:  # noqa: BLE001
                self.fallback_down = True
                log.warning("fallback %s %s %s failed (%s); not trying it again in this run", self.fallback.name, zone, target, e)
        self.provenance[(zone, target)] = {self.primary.name: int(p.notna().sum()), self.fallback.name: n_fb,
                                           "missing": int(out.isna().sum())}
        if n_fb:
            log.info("%s %s: %d of %d quarter-hours from %s", zone, target, n_fb, len(idx), self.fallback.name)
        return out.dropna()


def record_provenance(source: Source, zone: str, target: str, folder) -> None:
    """Append which source delivered how many points to <folder>/_sources.json (no-op for single sources)."""
    prov = getattr(source, "provenance", {}).get((zone, target))
    if not prov:
        return
    import json
    from pathlib import Path

    f = Path(folder) / "_sources.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[f"{zone}_{target}"] = prov
    f.write_text(json.dumps(data, indent=1, sort_keys=True))


def get_source(cfg: dict) -> Source:
    name = cfg.get("source", "energycharts")
    if cfg.get("fallback_source") and name != "synthetic":
        primary = get_source({**cfg, "source": name, "fallback_source": None})
        fallback = get_source({**cfg, "source": cfg["fallback_source"], "fallback_source": None, "_fallback": True})
        return FallbackSource(cfg, primary, fallback)
    if name == "energycharts":
        from .energycharts import EnergyChartsSource
        return EnergyChartsSource(cfg)
    if name == "entsoe":
        from .entsoe import EntsoeSource
        return EntsoeSource(cfg)
    if name == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(cfg)
    raise ValueError(f"unknown source {name!r}")
