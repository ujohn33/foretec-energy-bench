"""Score forecasts once actuals are in. Accuracy only in the PoC."""
from __future__ import annotations

import json

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .config import paths
from .data import get_source, record_provenance
from .timeutil import as_date, delivery_date, delivery_index, gate_timestamp, local_now

log = logging.getLogger(__name__)
MIN_COVERAGE = 0.95
KEY = ["issue_date", "model", "zone", "target"]


def scorable(issue_date, target, cfg, now=None) -> bool:
    """Has enough time passed for the actuals of this target to be final?"""
    now = pd.Timestamp(now) if now is not None else local_now(cfg)
    if now.tzinfo is None:
        now = now.tz_localize(cfg["timezone"])
    tcfg = cfg["targets"][target]
    d = as_date(issue_date)
    if "score_after" in tcfg:  # same-day scoring, e.g. day-ahead price
        ready = pd.Timestamp(f"{d.isoformat()} {tcfg['score_after']}").tz_localize(cfg["timezone"])
    else:
        dd = delivery_date(d) + dt.timedelta(days=tcfg.get("score_after_days", 2))
        ready = pd.Timestamp(dd.isoformat()).tz_localize(cfg["timezone"])
    return now >= ready


def actuals(issue_date, zone, target, cfg, source=None) -> pd.Series | None:
    """Actuals for delivery day D, frozen to disk the first time they are complete."""
    p = paths(cfg)
    f = p["actuals"] / str(delivery_date(issue_date)) / f"{zone}_{target}.parquet"
    if f.exists():
        return pd.read_parquet(f)["value"]
    idx = delivery_index(issue_date, cfg)
    source = source or get_source(cfg)
    s = source.fetch(zone, target, idx[0], idx[-1] + pd.Timedelta("15min")).reindex(idx)
    if s.notna().mean() < MIN_COVERAGE:
        log.info("actuals %s %s %s incomplete (%.0f%%)", delivery_date(issue_date), zone, target, 100 * s.notna().mean())
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    s.rename("value").to_frame().to_parquet(f)
    record_provenance(source, zone, target, f.parent)
    return s


def pinball(y: np.ndarray, qf: pd.DataFrame) -> float:
    losses = []
    for col in qf.columns:
        q = float(col[1:])
        e = y - qf[col].values
        losses.append(np.nanmean(np.maximum(q * e, (q - 1) * e)))
    return float(np.mean(losses))


def metrics(fc: pd.DataFrame, y: pd.Series) -> dict:
    m = fc.set_index("delivery_utc").reindex(y.index).dropna(axis=1, how="all")
    mask = y.notna() & m["point"].notna()
    yy, pp = y[mask].values, m.loc[mask, "point"].values
    err = pp - yy
    out = {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "pinball": np.nan,
    }
    qcols = [c for c in m.columns if c.startswith("q")] if "point" in m.columns else []
    if qcols:
        out["pinball"] = pinball(yy, m.loc[mask, qcols])
    return out


def locked_before_gate(day_dir, cfg) -> bool:
    """Live forecasts count only if the run finished before the day-ahead gate. Backtests are exempt."""
    f = day_dir / "_run.json"
    if not f.exists():
        log.warning("%s: no run log, not scored", day_dir.name)
        return False
    finished = pd.Timestamp(json.loads(f.read_text())["run_finished_utc"]).tz_convert("UTC").tz_localize(None)
    if finished > gate_timestamp(day_dir.name, cfg):
        log.warning("%s: run finished %s UTC, after the gate %s UTC; not scored", day_dir.name, finished, gate_timestamp(day_dir.name, cfg))
        return False
    return True


def score_pending(cfg, results_subdir="results", now=None, source=None) -> pd.DataFrame:
    p = paths(cfg, results_subdir)
    scores = pd.read_parquet(p["scores"]) if p["scores"].exists() else pd.DataFrame(columns=KEY)
    done = set(map(tuple, scores[KEY].astype(str).values)) if len(scores) else set()
    source = source or get_source(cfg)
    rows = []
    for day_dir in sorted(p["forecasts"].glob("*")):
        if not day_dir.is_dir():
            continue
        issue_date = day_dir.name
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        if results_subdir == "results" and not locked_before_gate(day_dir, cfg):
            continue
        fc_all = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for (zone, target), grp in fc_all.groupby(["zone", "target"]):
            todo = [m for m in grp["model"].unique() if (issue_date, m, zone, target) not in done]
            if not todo or not scorable(issue_date, target, cfg, now):
                continue
            y = actuals(issue_date, zone, target, cfg, source)
            if y is None:
                continue
            for model in todo:
                r = metrics(grp[grp["model"] == model], y)
                rows.append({"issue_date": issue_date, "model": model, "zone": zone, "target": target, **r})
    if not rows:
        return scores
    new = pd.DataFrame(rows)
    scores = pd.concat([scores, new], ignore_index=True) if len(scores) else new
    # relative MAE against the baseline model on the same day and series (skill)
    base = scores[scores["model"] == cfg["baseline"]][KEY[:1] + KEY[2:] + ["mae"]].rename(columns={"mae": "mae_base"})
    scores = scores.drop(columns=["mae_base", "rel_mae"], errors="ignore").merge(base, on=["issue_date", "zone", "target"], how="left")
    scores["rel_mae"] = scores["mae"] / scores["mae_base"]
    p["scores"].parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(p["scores"], index=False)
    log.info("scored %d new rows", len(new))
    return scores
