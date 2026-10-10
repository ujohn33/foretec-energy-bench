"""Score forecasts once actuals are in. Accuracy only in the PoC."""
from __future__ import annotations

import json

import datetime as dt
import logging
import os
import shutil

import numpy as np
import pandas as pd

from .config import paths
from .data import get_source, record_provenance
from .registry import load_models
from .timeutil import as_date, delivery_date, delivery_index, gate_timestamp, local_now

log = logging.getLogger(__name__)
MIN_COVERAGE = 0.95
KEY = ["issue_date", "model", "zone", "target"]


def _now(cfg, now=None) -> pd.Timestamp:
    now = pd.Timestamp(now) if now is not None else local_now(cfg)
    return now.tz_localize(cfg["timezone"]) if now.tzinfo is None else now.tz_convert(cfg["timezone"])


def scorable(issue_date, target, cfg, now=None) -> bool:
    """Has enough time passed for the actuals of this target to be final?"""
    now = _now(cfg, now)
    tcfg = cfg["targets"][target]
    d = as_date(issue_date)
    if "score_after" in tcfg:  # same-day scoring, e.g. day-ahead price
        ready = pd.Timestamp(f"{d.isoformat()} {tcfg['score_after']}").tz_localize(cfg["timezone"])
    else:
        dd = delivery_date(d) + dt.timedelta(days=tcfg.get("score_after_days", 2))
        ready = pd.Timestamp(dd.isoformat()).tz_localize(cfg["timezone"])
    return now >= ready


def required_coverage(issue_date, target, cfg, now=None) -> float:
    """Share of quarter-hours that must be published before the actuals are frozen.

    Wind and solar become scorable at midnight after delivery. Before `complete_before` on that day only a
    complete day is frozen (the early run must not freeze a day whose last hour is still missing); from then
    on MIN_COVERAGE suffices, as before.
    """
    tcfg = cfg["targets"][target]
    if "score_after_days" not in tcfg or not tcfg.get("complete_before"):
        return MIN_COVERAGE
    day = delivery_date(issue_date) + dt.timedelta(days=tcfg["score_after_days"])
    until = pd.Timestamp(f"{day.isoformat()} {tcfg['complete_before']}").tz_localize(cfg["timezone"])
    return 1.0 if _now(cfg, now) < until else MIN_COVERAGE


def actuals(issue_date, zone, target, cfg, source=None, min_coverage=MIN_COVERAGE) -> pd.Series | None:
    """Actuals for delivery day D, frozen to disk the first time they are complete."""
    p = paths(cfg)
    f = p["actuals"] / str(delivery_date(issue_date)) / f"{zone}_{target}.parquet"
    if f.exists():
        return pd.read_parquet(f)["value"]
    idx = delivery_index(issue_date, cfg)
    source = source or get_source(cfg)
    s = source.fetch(zone, target, idx[0], idx[-1] + pd.Timedelta("15min")).reindex(idx)
    if s.notna().mean() < min_coverage:
        log.info("actuals %s %s %s incomplete (%.0f%%, need %.0f%%)", delivery_date(issue_date), zone, target,
                 100 * s.notna().mean(), 100 * min_coverage)
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


def check_revisions(cfg, source=None, now=None) -> list[dict]:
    """Re-fetch wind/solar actuals frozen in the last few days and re-score a series that moved.

    Every check is appended to data/actuals/<D>/_revisions.json (also the unchanged ones, so revision sizes
    are on record). A series whose absolute change exceeds `threshold` of its daily energy is re-frozen and
    its score rows (live and backtest) are dropped, so the next score_pending computes them again.
    Prices are final at the auction and are never re-checked.
    """
    rc = cfg.get("revision_check") or {}
    days, threshold = int(rc.get("days", 3)), float(rc.get("threshold", 0.005))
    p = paths(cfg)
    if now is None:
        today = local_now(cfg).date()
    else:
        now = pd.Timestamp(now)
        today = (now.tz_convert(cfg["timezone"]) if now.tzinfo else now).date()
    source = source or get_source(cfg)
    revised = []
    for ddir in sorted(p["actuals"].glob("*")):
        try:
            delivery = dt.date.fromisoformat(ddir.name)
        except ValueError:
            continue
        if not 1 <= (today - delivery).days <= days:
            continue
        log_f = ddir / "_revisions.json"
        checks = json.loads(log_f.read_text()) if log_f.exists() else []
        for f in sorted(ddir.glob("*_*.parquet")):
            zone, target = f.stem.split("_")
            if "score_after_days" not in cfg["targets"].get(target, {}):
                continue
            old = pd.read_parquet(f)["value"]
            try:
                new = source.fetch(zone, target, old.index[0], old.index[-1] + pd.Timedelta("15min")).reindex(old.index)
            except Exception as e:  # noqa: BLE001 - a failed check must not stop scoring
                log.warning("revision check %s %s failed: %s", ddir.name, f.stem, e)
                continue
            both = old.notna() & new.notna()
            if both.mean() < MIN_COVERAGE:
                continue
            diff = (new - old)[both]
            rel = float(diff.abs().sum() / max(float(old[both].abs().sum()), 1e-9))
            entry = {"checked_utc": str(pd.Timestamp.now(tz="UTC")), "series": f.stem, "rel_change": round(rel, 6),
                     "max_abs_mw": round(float(diff.abs().max()), 2), "revised": rel > threshold}
            checks.append(entry)
            if rel > threshold:
                new.rename("value").to_frame().to_parquet(f)
                record_provenance(source, zone, target, ddir)
                issue = str(delivery - dt.timedelta(days=1))
                for sub in ("results", "results/backtest"):
                    sf = paths(cfg, sub)["scores"]
                    if sf.exists():
                        s = pd.read_parquet(sf)
                        keep = ~((s["issue_date"].astype(str) == issue) & (s["zone"] == zone) & (s["target"] == target))
                        if (~keep).any():
                            s[keep].to_parquet(sf, index=False)
                revised.append({"delivery": ddir.name, **entry})
                log.warning("actuals %s %s revised by %.2f%%: re-frozen, will be re-scored", ddir.name, f.stem, 100 * rel)
        log_f.write_text(json.dumps(checks, indent=1))
    return revised


def score_pending(cfg, results_subdir="results", now=None, source=None) -> pd.DataFrame:
    p = paths(cfg, results_subdir)
    scores = pd.read_parquet(p["scores"]) if p["scores"].exists() else pd.DataFrame(columns=KEY)
    done = set(map(tuple, scores[KEY].astype(str).values)) if len(scores) else set()
    source = source or get_source(cfg)
    # a forecast written before a series was excluded from its model (exclude: in model.yaml) is not scored
    specs = {s.name: s for s in load_models(p["models"])} if p["models"].exists() else {}
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
            todo = [m for m in grp["model"].unique() if (issue_date, m, zone, target) not in done
                    and (m not in specs or specs[m].applies(zone, target))]
            if not todo or not scorable(issue_date, target, cfg, now):
                continue
            y = actuals(issue_date, zone, target, cfg, source, required_coverage(issue_date, target, cfg, now))
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


def provisional_actuals(cfg, source=None, now=None) -> list[str]:
    """Metered-so-far wind and solar for delivery days under way or not frozen yet (today and yesterday).

    Written to data/actuals_provisional/<D>/<zone>_<target>.parquet with a _fetched.json time stamp. The site
    draws them as provisional actuals; they are never scored. A series is dropped once its frozen actuals
    exist, and folders older than yesterday are removed.
    """
    p = paths(cfg)
    root = p["actuals_provisional"]
    today = _now(cfg, now).date()
    targets = [t for t, v in cfg["targets"].items() if "score_after_days" in v]
    source = source or get_source(cfg)
    written = []
    for delivery in (today - dt.timedelta(days=1), today):
        idx = delivery_index(delivery - dt.timedelta(days=1), cfg)
        ddir = root / str(delivery)
        for zone in cfg["zones"]:
            for target in targets:
                key = f"{zone}_{target}"
                if (p["actuals"] / str(delivery) / f"{key}.parquet").exists():
                    (ddir / f"{key}.parquet").unlink(missing_ok=True)
                    continue
                try:
                    s = source.fetch(zone, target, idx[0], idx[-1] + pd.Timedelta("15min")).reindex(idx)
                except Exception as e:  # noqa: BLE001 - provisional values are a convenience, never block on them
                    log.warning("provisional %s %s failed: %s", delivery, key, e)
                    continue
                if not s.notna().any():
                    continue
                ddir.mkdir(parents=True, exist_ok=True)
                tmp = ddir / f".{key}.parquet.tmp"
                s.rename("value").to_frame().to_parquet(tmp)
                os.replace(tmp, ddir / f"{key}.parquet")
                record_provenance(source, zone, target, ddir)
                written.append(f"{delivery} {key} {int(s.notna().sum())}/{len(idx)}")
        if ddir.exists():
            (ddir / "_fetched.json").write_text(json.dumps({"fetched_utc": str(pd.Timestamp.now(tz="UTC").floor("s"))}))
    for d in root.glob("*"):
        try:
            if dt.date.fromisoformat(d.name) < today - dt.timedelta(days=1):
                shutil.rmtree(d)
        except ValueError:
            continue
    return written
