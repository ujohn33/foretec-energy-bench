"""Time conventions.

Everything inside the pipeline is UTC, timezone-naive, on a 15-minute grid.
Local Brussels time is only used to define the issue time and the delivery day,
so clock-change days have 92 or 100 quarter-hours instead of 96.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

FREQ = "15min"


def to_utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None)


def as_date(d) -> dt.date:
    if isinstance(d, dt.date):
        return d
    return pd.Timestamp(d).date()


def issue_timestamp(issue_date, cfg) -> pd.Timestamp:
    """Forecast lock time, D-1 at cfg.issue_time local, as UTC-naive."""
    d = as_date(issue_date)
    local = pd.Timestamp(f"{d.isoformat()} {cfg['issue_time']}").tz_localize(cfg["timezone"])
    return to_utc_naive(local)


def gate_timestamp(issue_date, cfg) -> pd.Timestamp:
    """Submission gate, D-1 at cfg.gate local (SDAC gate closure), as UTC-naive. Defaults to the issue time."""
    d = as_date(issue_date)
    local = pd.Timestamp(f"{d.isoformat()} {cfg.get('gate', cfg['issue_time'])}").tz_localize(cfg["timezone"])
    return to_utc_naive(local)


def delivery_date(issue_date) -> dt.date:
    return as_date(issue_date) + dt.timedelta(days=1)


def delivery_index(issue_date, cfg) -> pd.DatetimeIndex:
    """All quarter-hours of delivery day D (local midnight to midnight), as UTC-naive."""
    d = delivery_date(issue_date)
    start = pd.Timestamp(f"{d.isoformat()} 00:00").tz_localize(cfg["timezone"])
    end = pd.Timestamp(f"{(d + dt.timedelta(days=1)).isoformat()} 00:00").tz_localize(cfg["timezone"])
    idx = pd.date_range(start, end, freq=FREQ, inclusive="left")
    return idx.tz_convert("UTC").tz_localize(None).rename("delivery_utc")


def availability_cutoff(issue_date, target: str, cfg) -> pd.Timestamp:
    """Latest timestamp (exclusive) of target history a model may see at issue time."""
    tcfg = cfg["targets"][target]
    rule = tcfg.get("available_until", "issue_time")
    if rule == "delivery_start":
        return delivery_index(issue_date, cfg)[0]
    lag = pd.Timedelta(tcfg.get("publication_lag", "0h"))
    return (issue_timestamp(issue_date, cfg) - lag).floor(FREQ)


def local_now(cfg) -> pd.Timestamp:
    return pd.Timestamp.now(tz=cfg["timezone"])


def local_today(cfg) -> dt.date:
    return local_now(cfg).date()
