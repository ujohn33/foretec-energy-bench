"""Command line: foretec-live <command>.

  probe     check the data source returns sensible series (run this first on a new server)
  forecast  snapshot + run all models for an issue date (default: today, CET/CEST)
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
from pathlib import Path

import pandas as pd

from .config import load_config, paths
from .data import get_source
from .report import write_report
from .site import build_site
from .registry import load_models
from .run import add_ensembles, pretrain, run_forecasts, take_snapshot
from .score import check_revisions, locked_before_gate, provisional_actuals, score_pending
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
    if os.environ.get("FORETEC_SKIP_SITE"):   # e.g. a backfill of zones not yet public
        return
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
    try:      # imbalance prices for the Forecast Value page: the last week of delivery days, so late publications fill in
        from .market import update_imbalance
        today = local_today(cfg)
        got = update_imbalance(cfg, today - dt.timedelta(days=7), today - dt.timedelta(days=1))
        if got:
            print(f"imbalance prices: {len(got)} zone-days stored")
    except Exception:  # noqa: BLE001 - the value page is extra; never fail scoring on it
        logging.exception("imbalance price update failed")
    if revised:   # the backtest window shares the actuals, so re-score it too
        score_pending(cfg, results_subdir="results/backtest", source=src)
        write_report(cfg, results_subdir="results/backtest")
        print(f"{len(revised)} series revised and re-scored: " + ", ".join(f"{r['delivery']} {r['series']}" for r in revised))
    print(f"{len(s)} scored rows in total")
    return 0


def cmd_provisional(cfg, args):
    """Fetch metered-so-far wind/solar for days not frozen yet (shown as provisional, never scored)."""
    for line in provisional_actuals(cfg):
        print(line)
    return 0


def cmd_report(cfg, args):
    lb = write_report(cfg)
    print(lb.head(20).to_string() if not lb.empty else "no scores yet")
    _site(cfg)
    return 0


def _warm_long_history_models(cfg, models_dir):
    """Models trained on long history (train_days in model.yaml) retrain monthly: train this month's artifact
    overnight, so the first run of the month does not spend its time before the gate training."""
    import subprocess

    from .run import external_python

    for s in load_models(models_dir):
        if not s.raw.get("train_days"):
            continue
        env = {**os.environ, "PYTHONPATH": str(Path(cfg["_root"]) / "models" / "_shared"), "FORETEC_HOME": str(cfg["_root"])}
        script = s.folder / s.raw.get("script", "forecaster.py")
        r = subprocess.run([str(external_python(s, cfg)), str(script), "warm", str(s.folder), str(local_today(cfg))],
                           cwd=s.folder, env=env, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            logging.error("warming %s failed: %s", s.name, r.stderr.strip()[-400:])


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
    _warm_long_history_models(cfg, p["models"])
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
    # a series runs only once its reference is published (model.yaml `published`: local time on the issue day, per zone
    # or default), so an earlier scoring run neither marks it failed nor replaces the published version with a preliminary
    now = pd.Timestamp.now(tz=cfg["timezone"])
    due, waiting = set(), set()
    for spec in load_models(paths(cfg)["models"]):
        if not spec.raw.get("reference"):
            continue
        times = spec.raw.get("published") or {}
        for z in cfg["zones"]:
            for t in cfg["targets"]:
                if spec.applies(z, t):
                    at = pd.Timestamp(f"{d} {times.get(z, times.get('default', '00:00'))}", tz=cfg["timezone"])
                    (due if now >= at else waiting).add(f"{z}_{t}")
    if not due:
        print(f"{d}: no reference published yet")
        return 0
    # the frozen 11:30 snapshot stays as it is: take a fresh one without saving it
    meta = run_forecasts(d, cfg, reference_run=True, snapshots=take_snapshot(d, cfg, save=False),
                         series=sorted(due) if waiting else None)
    ok = sum(r["status"] == "ok" for r in meta["runs"])
    print(f"{d}: {ok} reference series ok, {len(meta['runs']) - ok} not")
    return 0


def cmd_intervals(cfg, args):
    """One-off: interval scores (WIS, Winkler 80%, coverage) for score rows made before they were computed."""
    from .score import backfill_interval_scores
    for sub in ("results", "results/backtest"):
        print(sub, backfill_interval_scores(cfg, sub), "rows")
    _site(cfg)
    return 0


def cmd_covhist(cfg, args):
    """Training covariates for long-history covariate models (see covariates_history.py); fetched once, kept."""
    from .covariates_history import build_history, refresh_fuel

    written = build_history(cfg, args.start, args.end)
    print(f"covariate history: {len(written)} month files written, fuel refreshed in {refresh_fuel(cfg)}")
    return 0


def cmd_ensembles(cfg, args):
    """Add the ensembles to past issue days from the members' stored forecasts (run times kept), then score."""
    sub = "results/backtest" if args.backtest else "results"
    d = args.start
    while d <= args.end:
        rows = add_ensembles(d, cfg, sub, args.models)
        print(d, {s: sum(r["status"] == s for r in rows) for s in ("ok", "error")} if rows else "no run log")
        d += dt.timedelta(days=1)
    score_pending(cfg, results_subdir=sub)
    _site(cfg)
    return 0


def cmd_relegate(cfg, args):
    """Weekly relegation check (relegation.py): print the verdicts; --apply stops those models running live."""
    from .relegation import apply, check
    verdicts, facts = check(cfg)
    m, b = facts["median_run_minutes"], facts["budget_minutes"]
    if m is not None and m > b:
        logging.warning("live run takes %.1f min (median of 7), over the %s min budget: consider relegating by hand", m, b)
    print(f"median live run {m if m is None else round(m, 1)} min (budget {b})")
    for v in verdicts:
        print(f"{'RELEGATE' if args.apply else 'would relegate'} {v['model']}: {v['reason']}")
    if not verdicts:
        print("no model to relegate")
    if args.apply and verdicts:
        apply(cfg, verdicts)
        _site(cfg)
    return 0


def cmd_pretrain(cfg, args):
    """Morning training of the models too slow to train inside the live run (pretrain: true), for today's issue."""
    d = args.date or local_today(cfg)
    done = pretrain(d, cfg, args.models)
    print(f"{d}: trained " + (", ".join(done) if done else "nothing"))
    return 0


def cmd_backfill(cfg, args):
    src = get_source(cfg)
    # ensembles alone need no snapshot: they combine the members' forecasts already on disk
    from .registry import load_models
    ens_only = bool(args.models) and all(s.raw.get("runner") == "ensemble" for s in load_models(paths(cfg)["models"], args.models))
    d = args.start
    while d <= args.end:
        targets = args.targets or (sorted({s.split("_")[1] for s in args.series}) if args.series else None)
        snaps = None if ens_only else take_snapshot(d, cfg, source=src, save=False, targets=targets)
        run_forecasts(d, cfg, results_subdir="results/backtest", only_models=args.models, snapshots=snaps, source=src,
                      targets=args.targets, series=args.series)
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
    sub.add_parser("provisional")
    sub.add_parser("report")
    sub.add_parser("site")
    sub.add_parser("catchup")
    p = sub.add_parser("reference"); p.add_argument("--date", type=_date)
    sub.add_parser("intervals")
    p = sub.add_parser("ensembles"); p.add_argument("--start", type=_date, required=True); p.add_argument("--end", type=_date, required=True)
    p.add_argument("--models", nargs="*"); p.add_argument("--backtest", action="store_true")
    p = sub.add_parser("relegate"); p.add_argument("--apply", action="store_true")
    p = sub.add_parser("pretrain"); p.add_argument("--date", type=_date); p.add_argument("--models", nargs="*")
    p = sub.add_parser("covhist"); p.add_argument("--start", type=_date, required=True); p.add_argument("--end", type=_date, required=True)
    p = sub.add_parser("backfill"); p.add_argument("--start", type=_date, required=True)
    p.add_argument("--targets", nargs="*", help="only these targets, merged into the existing backtest days (a new target)")
    p.add_argument("--series", nargs="*", help="only these series (ZONE_target), merged into the existing backtest days")
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
    return {"probe": cmd_probe, "forecast": cmd_forecast, "score": cmd_score, "provisional": cmd_provisional,
            "report": cmd_report, "backfill": cmd_backfill, "site": cmd_site, "catchup": cmd_catchup, "reference": cmd_reference, "covhist": cmd_covhist, "pretrain": cmd_pretrain, "intervals": cmd_intervals, "ensembles": cmd_ensembles, "relegate": cmd_relegate}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
