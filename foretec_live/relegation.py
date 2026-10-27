"""Relegation: keep the live run inside its half hour by stopping models that cost much and are clearly beaten.

A live model is relegated when another live model that costs at most half as much (`rival_cost_ratio`) and takes the
same kind of input (history only, or covariates) beats it clearly over the last `window_days` of live races: pooled
Rel. MAE better by at least `min_gap` and a lower MAE in at least `min_win` of the races both ran. Cost is the model's
time in the live run (median of the last 7 runs); models under `min_cost_s` cost the window next to nothing and are never
relegated. Both need `min_live_days` of scored live days. Never relegated, and never counted as the rival: baselines,
the grid operators' forecasts, ensembles; ensemble members and `protected` models are never relegated either.

A relegated model stops running live (data/relegated.json, the date and the reason); its live record and backtest stay
on the site. Putting it back is deleting its entry. The check also warns when the median run takes longer than
`budget_minutes`, which is left to a person to act on.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import paths
from .registry import load_models
from .timeutil import local_today

log = logging.getLogger(__name__)
DEFAULTS = {"window_days": 28, "min_live_days": 10, "min_cost_s": 30, "rival_cost_ratio": 0.5, "min_gap": 0.10,
            "min_win": 0.60, "protected": [], "budget_minutes": 22}
COVARIATES = {"nwp", "load_forecast", "fuel"}


def state_file(cfg) -> Path:
    return Path(cfg["_root"]) / "data" / "relegated.json"


def relegated(cfg) -> dict:
    f = state_file(cfg)
    return json.loads(f.read_text()) if f.exists() else {}


def run_costs(cfg, runs: int = 7) -> tuple[pd.Series, float | None]:
    """Seconds per model per live run (median over the last `runs` runs) and the median run length in minutes."""
    rows, minutes = [], []
    for f in sorted(paths(cfg)["forecasts"].glob("*/_run.json"))[-runs:]:
        m = json.loads(f.read_text())
        minutes.append((pd.Timestamp(m["run_finished_utc"]) - pd.Timestamp(m["run_started_utc"])).total_seconds() / 60)
        rows += [{"model": r["model"], "day": f.parent.name, "s": float(r.get("seconds") or 0)} for r in m.get("runs", [])]
    if not rows:
        return pd.Series(dtype=float), None
    sec = pd.DataFrame(rows).groupby(["model", "day"])["s"].sum().groupby("model").median()
    return sec, float(pd.Series(minutes).median())


def check(cfg, today=None) -> tuple[list[dict], dict]:
    """(models to relegate with their reason, facts: median run minutes and budget)."""
    rc = {**DEFAULTS, **(cfg.get("relegation") or {})}
    specs = {s.name: s.raw for s in load_models(paths(cfg)["models"])}
    members = {m for r in specs.values() if r.get("runner") == "ensemble" for m in r.get("members", [])}
    done = relegated(cfg)
    sec, minutes = run_costs(cfg)
    p = paths(cfg)
    s = pd.read_parquet(p["scores"]) if p["scores"].exists() else pd.DataFrame()
    facts = {"median_run_minutes": minutes, "budget_minutes": rc["budget_minutes"]}
    if s.empty:
        return [], facts
    s = s.assign(issue_date=s["issue_date"].astype(str))
    start = str((pd.Timestamp(today or local_today(cfg)) - pd.Timedelta(days=rc["window_days"])).date())
    s = s[s["issue_date"] >= start]
    days = s.groupby("model")["issue_date"].nunique()
    g = s.assign(a=s["n"] * s["mae"], b=s["n"] * s["mae_base"]).groupby(["model", "zone", "target"])[["a", "b"]].sum()
    rel = (g["a"] / g["b"]).groupby("model").mean()
    mae = s.set_index(["issue_date", "zone", "target", "model"])["mae"]

    def ok(m):   # live now, scored often enough, cost measured
        return m in specs and m not in done and m in rel.index and m in sec.index and days.get(m, 0) >= rc["min_live_days"] \
            and specs[m].get("live") is not False and not specs[m].get("reference") and not specs[m].get("live_only")
    uses_cov = lambda m: bool(set(specs[m].get("inputs", ["history"])) & COVARIATES)
    bench = lambda m: specs[m].get("family") in ("baseline", "tso", "ensemble")
    out = []
    for m in sorted(sec.index, key=lambda x: -sec[x]):
        if not ok(m) or bench(m) or m in members or m in rc["protected"] or sec[m] < rc["min_cost_s"]:
            continue
        for o in sorted(rel.index, key=lambda x: rel[x]):   # the strongest qualifying rival is named
            if o == m or not ok(o) or bench(o) or uses_cov(o) != uses_cov(m) or sec[o] > sec[m] * rc["rival_cost_ratio"]:
                continue
            if rel[m] < rel[o] * (1 + rc["min_gap"]):
                continue
            a, b = mae.xs(m, level="model"), mae.xs(o, level="model")
            both = a.index.intersection(b.index)
            win = float((b.loc[both] < a.loc[both]).mean()) if len(both) else 0.0
            if win >= rc["min_win"]:
                out.append({"model": m, "seconds": round(float(sec[m])), "rel_mae": round(float(rel[m]), 3),
                            "rival": o, "rival_seconds": round(float(sec[o])), "rival_rel_mae": round(float(rel[o]), 3),
                            "rival_wins": round(win, 2), "races": int(len(both)),
                            "reason": f"beaten by {o} at {sec[o]:.0f} s against {sec[m]:.0f} s per run: Rel. MAE "
                                      f"{rel[o]:.3f} against {rel[m]:.3f}, lower MAE in {win:.0%} of {len(both)} live races "
                                      f"(last {rc['window_days']} days)"})
                break
    return out, facts


def apply(cfg, verdicts: list[dict], today=None) -> dict:
    state = relegated(cfg)
    day = str(today or local_today(cfg))
    for v in verdicts:
        state[v["model"]] = {"date": day, **v}
        log.warning("relegated %s: %s", v["model"], v["reason"])
    f = state_file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=1))
    return state
