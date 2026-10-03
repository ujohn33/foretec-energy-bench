"""Command line: foretec-live <command>.

  probe     check the data source returns sensible series (run this first on a new server)
  forecast  snapshot + run all models for an issue date (default: today, Brussels)
  score     score every forecast whose actuals are final
  report    rebuild leaderboard CSV and static page
  backfill  forecast + score a range of past issue dates into results/backtest
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import warnings

import pandas as pd

from .config import load_config, paths
from .data import get_source
from .report import write_report
from .site import build_site
from .registry import load_models
from .run import run_forecasts, take_snapshot
from .score import check_revisions, locked_before_gate, score_pending
from .timeutil import availability_cutoff, delivery_index, gate_timestamp, issue_timestamp, local_today


def _date(s):
    return dt.date.fromisoformat(s)


def cmd_probe(cfg, args):
    src = get_source(cfg)
    d = args.date or (local_today(cfg) - dt.timedelta(days=2))
    idx = delivery_index(d, cfg)
    print(f"source={src.name}  delivery day {idx[0]} .. {idx[-1]} UTC ({len(idx)} quarter-hours)")
    bad = 0
    for zone in cfg["zones"]:
        for target in cfg["targets"]:
            try:
                s = src.fetch(zone, target, idx[0], idx[-1] + pd.Timedelta("15min"))
                step = s.index.to_series().diff().median() if len(s) > 1 else None
                cov = s.reindex(idx).notna().mean()
                print(f"  {zone} {target:5s}: {len(s):4d} pts  coverage {cov:5.0%}  step {step}  "
                      f"min {s.min():9.1f}  mean {s.mean():9.1f}  max {s.max():9.1f}")
                bad += cov < 0.95
            except Exception as e:
                bad += 1
                print(f"  {zone} {target:5s}: ERROR {type(e).__name__}: {e}")
    print("cutoffs for today's issue:", {t: str(availability_cutoff(local_today(cfg), t, cfg)) for t in cfg["targets"]})
    return 1 if bad else 0


def _site(cfg):
    """Rebuild the website; a failure here must not fail the forecast or scoring run."""
    try:
        build_site(cfg)
    except Exception:  # pragma: no cover
        logging.exception("website build failed")


def cmd_site(cfg, args):
    print(build_site(cfg))
    return 0


def cmd_forecast(cfg, args):
    d = args.date or local_today(cfg)
    # Live forecasts are only valid before the day-ahead gate, and a locked day is never replaced.
    # (A persistent timer catching up after a reboot or a schedule change must not overwrite results.)
    day_dir = paths(cfg)["forecasts"] / str(d)
    if not args.force:
        if pd.Timestamp.now(tz="UTC").tz_localize(None) > gate_timestamp(d, cfg):
            logging.error("issue %s: past the %s gate (%s UTC); not running. Use --force for a run that will not be scored.",
                          d, cfg.get("gate"), gate_timestamp(d, cfg))
            return 1
        if (day_dir / "_run.json").exists() and locked_before_gate(day_dir, cfg):
            logging.error("issue %s: already locked before the gate; not overwriting (use --force)", d)
            return 1
    logging.info("issue date %s, lock %s UTC", d, issue_timestamp(d, cfg))
    meta = run_forecasts(d, cfg, only_models=args.models)
    ok = sum(r["status"] == "ok" for r in meta["runs"])
    print(f"{d}: {ok} model-series ok, {len(meta['runs']) - ok} not")
    _site(cfg)
    return 0


def cmd_score(cfg, args):
    src = get_source(cfg)
    revised = check_revisions(cfg, source=src)
    s = score_pending(cfg, source=src)
    if revised:   # the backtest window shares the actuals, so re-score it too
        score_pending(cfg, results_subdir="results/backtest", source=src)
        write_report(cfg, results_subdir="results/backtest")
        print(f"{len(revised)} series revised and re-scored: " + ", ".join(f"{r['delivery']} {r['series']}" for r in revised))
    print(f"{len(s)} scored rows in total")
    return 0


def cmd_report(cfg, args):
    lb = write_report(cfg)
    print(lb.head(20).to_string() if not lb.empty else "no scores yet")
    _site(cfg)
    return 0


def cmd_catchup(cfg, args):
    """Backtest every model that is missing from the backtest window, so adding a model folder is enough
    to deploy it: backtested overnight, live from the next issue time, scored daily."""
    bt = cfg.get("backtest")
    if not bt:
        print("no backtest window in config.yaml")
        return 0
    start, end = dt.date.fromisoformat(str(bt["start"])), dt.date.fromisoformat(str(bt["end"]))
    p = paths(cfg, "results/backtest")
    names = [s.name for s in load_models(p["models"]) if not s.raw.get("live_only")]
    src = get_source(cfg)
    ran = []
    d = start
    while d <= end:
        missing = [m for m in names if not (p["forecasts"] / str(d) / f"{m}.parquet").exists()]
        if missing:
            snaps = take_snapshot(d, cfg, source=src, save=False)
            run_forecasts(d, cfg, results_subdir="results/backtest", only_models=missing, snapshots=snaps, source=src)
            ran.append(f"{d}: {', '.join(missing)}")
        d += dt.timedelta(days=1)
    if ran:
        score_pending(cfg, results_subdir="results/backtest", source=src)
        write_report(cfg, results_subdir="results/backtest")
        _site(cfg)
    print("\n".join(ran) if ran else "backtest complete for every model")
    return 0


def cmd_reference(cfg, args):
    """Reference forecasts published after the gate (e.g. the TSO's final day-ahead forecast) for today's issue.
    Runs before each scoring; never touches the live run log, so the gate check is unaffected."""
    d = args.date or local_today(cfg)
    if not (paths(cfg)["forecasts"] / str(d)).exists():
        print(f"no live run for {d}; nothing to add references to")
        return 0
    # the frozen 11:30 snapshot stays as it is: take a fresh one without saving it
    meta = run_forecasts(d, cfg, reference_run=True, snapshots=take_snapshot(d, cfg, save=False))
    ok = sum(r["status"] == "ok" for r in meta["runs"])
    print(f"{d}: {ok} reference series ok, {len(meta['runs']) - ok} not")
    return 0


def cmd_backfill(cfg, args):
    src = get_source(cfg)
    d = args.start
    while d <= args.end:
        snaps = take_snapshot(d, cfg, source=src, save=False)
        run_forecasts(d, cfg, results_subdir="results/backtest", only_models=args.models, snapshots=snaps, source=src)
        logging.info("backfilled %s", d)
        d += dt.timedelta(days=1)
    score_pending(cfg, results_subdir="results/backtest", source=src)
    lb = write_report(cfg, results_subdir="results/backtest")
    print(lb.to_string() if not lb.empty else "no scores")
    _site(cfg)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="foretec-live")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="override config source (energycharts|entsoe|synthetic)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--date", type=_date)
    p = sub.add_parser("forecast"); p.add_argument("--date", type=_date); p.add_argument("--models", nargs="*")
    p.add_argument("--force", action="store_true", help="run even after the gate or over a locked day")
    sub.add_parser("score")
    sub.add_parser("report")
    sub.add_parser("site")
    sub.add_parser("catchup")
    p = sub.add_parser("reference"); p.add_argument("--date", type=_date)
    p = sub.add_parser("backfill"); p.add_argument("--start", type=_date, required=True)
    p.add_argument("--end", type=_date, required=True); p.add_argument("--models", nargs="*")
    args = ap.parse_args(argv)
    # sktime 1.2 announces a 1.3 default change on every estimator; also reaches worker processes via env
    warnings.filterwarnings("ignore", message=".*remember_data.*", category=FutureWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.source:
        cfg["source"] = args.source
    return {"probe": cmd_probe, "forecast": cmd_forecast, "score": cmd_score,
            "report": cmd_report, "backfill": cmd_backfill, "site": cmd_site, "catchup": cmd_catchup, "reference": cmd_reference}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
