"""Leaderboard: rolling window of relative MAE, written as CSV."""
from __future__ import annotations

import pandas as pd

from .config import paths

TARGET_ORDER = ["price", "wind", "solar"]


def leaderboard(cfg, results_subdir="results") -> pd.DataFrame:
    p = paths(cfg, results_subdir)
    if not p["scores"].exists():
        return pd.DataFrame()
    s = pd.read_parquet(p["scores"])
    if s.empty:
        return pd.DataFrame()
    last = pd.to_datetime(s["issue_date"]).max()
    s = s[pd.to_datetime(s["issue_date"]) > last - pd.Timedelta(days=cfg["rolling_days"])]
    n_days = s["issue_date"].nunique()
    series = s.assign(series=s["zone"] + " " + s["target"])
    wide = series.pivot_table(index="model", columns="series", values="rel_mae", aggfunc="mean")
    order = [f"{z} {t}" for t in TARGET_ORDER for z in cfg["zones"] if f"{z} {t}" in wide.columns]
    wide = wide[order]
    lb = pd.DataFrame({
        "rel_mae": wide.mean(axis=1),
        "series": wide.notna().sum(axis=1),
        "days": series.groupby("model")["issue_date"].nunique(),
    }).join(wide)
    lb["coverage"] = lb["days"] / n_days
    lb = lb.sort_values("rel_mae")
    lb.insert(0, "rank", range(1, len(lb) + 1))
    lb.attrs["window"] = (s["issue_date"].min(), s["issue_date"].max(), n_days)
    return lb


def write_report(cfg, results_subdir="results") -> pd.DataFrame:
    """Write the rolling leaderboard CSV. The website is built separately by site.build_site."""
    p = paths(cfg, results_subdir)
    lb = leaderboard(cfg, results_subdir)
    if not lb.empty:
        p["leaderboard"].parent.mkdir(parents=True, exist_ok=True)
        lb.to_csv(p["leaderboard"])
    return lb
