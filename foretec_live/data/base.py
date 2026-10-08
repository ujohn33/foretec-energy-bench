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
    """Primary source first; quarter-hours it does not deliver are filled from the fallbacks, in order.

    `provenance[(zone, target)]` counts, for the last fetch, how many points came from each source.
    Each fallback has its own circuit breaker: after one failure it is skipped for the rest of the run,
    so a dead service can never hold up the live run before the gate.
    """

    def __init__(self, cfg: dict, primary: Source, *fallbacks: Source):
        super().__init__(cfg)
        self.primary, self.fallbacks = primary, list(fallbacks)
        self.name = "+".join([primary.name] + [f.name for f in self.fallbacks])
        self.provenance: dict = {}
        self.down: set = set()

    @property
    def fallback_down(self) -> bool:   # kept for callers that know a single fallback
        return bool(self.down)

    def fetch(self, zone, target, start, end):
        idx = pd.date_range(start, end, freq=FREQ, inclusive="left")
        # source_by_series puts one of the fallbacks first for a series (e.g. Elia's quarter-hours for BE wind)
        first, rest = self.primary, list(self.fallbacks)
        pref = (self.cfg.get("source_by_series") or {}).get(f"{zone}_{target}")
        if pref and pref != self.primary.name:
            chosen = next((s for s in self.fallbacks if s.name == pref), None)
            if chosen is not None:
                first, rest = chosen, [self.primary] + [s for s in self.fallbacks if s is not chosen]
        try:
            out = first.fetch(zone, target, start, end).reindex(idx)
        except Exception as e:  # noqa: BLE001 - a dead primary must not stop the run
            log.warning("%s %s %s failed (%s); using the fallbacks", first.name, zone, target, e)
            out = pd.Series(np.nan, index=idx)
        prov = {first.name: int(out.notna().sum())}
        for fb in rest:
            prov[fb.name] = 0
            if not out.isna().any() or fb.name in self.down:
                continue
            try:
                f = fb.fetch(zone, target, start, end).reindex(idx)
            except Exception as e:  # noqa: BLE001
                self.down.add(fb.name)
                log.warning("fallback %s %s %s failed (%s); not trying it again in this run", fb.name, zone, target, e)
                continue
            filled = out.isna() & f.notna()
            prov[fb.name] = int(filled.sum())
            out = out.combine_first(f)
            if prov[fb.name]:
                log.info("%s %s: %d of %d quarter-hours from %s", zone, target, prov[fb.name], len(idx), fb.name)
        prov["missing"] = int(out.isna().sum())
        self.provenance[(zone, target)] = prov
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
        names = cfg["fallback_source"] if isinstance(cfg["fallback_source"], list) else [cfg["fallback_source"]]
        fallbacks = [get_source({**cfg, "source": n, "fallback_source": None, "_fallback": True}) for n in names]
        return FallbackSource(cfg, primary, *fallbacks)
    if name == "energycharts":
        from .energycharts import EnergyChartsSource
        return EnergyChartsSource(cfg)
    if name == "entsoe":
        from .entsoe import EntsoeSource
        return EntsoeSource(cfg)
    if name == "elia":
        from .tso import EliaSource
        return EliaSource(cfg)
    if name == "rte":
        from .tso import RteSource
        return RteSource(cfg)
    if name == "synthetic":
        from .synthetic import SyntheticSource
        return SyntheticSource(cfg)
    raise ValueError(f"unknown source {name!r}")
