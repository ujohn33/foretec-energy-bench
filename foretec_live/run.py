"""Daily forecast run: take the as-of snapshot, run every model, write forecasts."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import traceback
import warnings

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from .config import paths
from .data import get_source, record_provenance
from .registry import load_models
from .timeutil import FREQ, availability_cutoff, delivery_index, issue_timestamp

log = logging.getLogger(__name__)
MIN_HISTORY_DAYS = 7


def take_snapshot(issue_date, cfg, source=None, save=True, targets=None) -> dict[tuple[str, str], pd.Series]:
    """Collect the history every model is allowed to see, and store it so the run can be repeated exactly."""
    source = source or get_source(cfg)
    p = paths(cfg)
    out = {}
    for zone in cfg["zones"]:
        for target in targets or cfg["targets"]:
            end = availability_cutoff(issue_date, target, cfg)
            start = end - pd.Timedelta(days=cfg["context_days"])
            try:
                s = source.fetch(zone, target, start, end)
            except Exception as e:
                log.error("snapshot %s %s failed: %s", zone, target, e)
                continue
            s = s[s.index < end]
            s = s.reindex(pd.date_range(start, end, freq=FREQ, inclusive="left")).ffill(limit=8)
            s = s[s.first_valid_index():] if s.first_valid_index() is not None else s.iloc[0:0]
            if s.isna().any():
                s = s.interpolate(limit_direction="both")
            if len(s) < MIN_HISTORY_DAYS * 96:
                log.error("snapshot %s %s: only %d points, skipping", zone, target, len(s))
                continue
            s.index.name = "time_utc"
            s.name = "value"
            out[(zone, target)] = s
            if save:
                d = p["snapshots"] / str(issue_date)
                d.mkdir(parents=True, exist_ok=True)
                s.to_frame().to_parquet(d / f"{zone}_{target}.parquet")
                record_provenance(source, zone, target, d)
    return out


def _forecast_one(spec, y: pd.Series, fh_index: pd.DatetimeIndex, target: str, zone: str, quantiles):
    y = y.asfreq(FREQ)
    fh = ForecastingHorizon(fh_index, is_relative=False, freq=FREQ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = spec.build(target=target, zone=zone)
        est.fit(y, fh=fh)
        point = est.predict(fh)
        q = None
        if est.get_tag("capability:pred_int", False):
            try:
                q = est.predict_quantiles(fh, alpha=quantiles)
            except Exception as e:
                log.info("%s: no quantiles (%s)", spec.name, e)
    df = pd.DataFrame({"delivery_utc": fh_index, "point": pd.Series(point).values})
    if q is not None:
        for (_, a), col in zip(q.columns, q.columns):
            df[f"q{a:.1f}"] = q[col].values
    return df


def _gap_free(y: pd.Series) -> pd.Series:
    """Foundation models take dense arrays: fill short gaps, then the edges."""
    return y.asfreq(FREQ).interpolate(limit=8, limit_direction="both").ffill().bfill()


def external_python(spec, cfg) -> Path:
    if spec.raw["env"] == "_self":  # light models that only need the benchmark's own environment
        return Path(sys.executable)
    envs = Path(cfg["_root"]) / cfg.get("envs_dir", "../envs")
    # not resolve(): the venv python is a symlink to the base interpreter
    return Path(os.path.abspath(envs / spec.raw["env"] / "bin" / "python"))


def _forecast_external(spec, items, fh_index: pd.DatetimeIndex, cfg, extra: dict | None = None) -> dict:
    """Run a subprocess model once for all its series. Returns {(zone, target): DataFrame | Exception}."""
    step = pd.Timedelta(FREQ)
    out = {}
    with tempfile.TemporaryDirectory(prefix=f"fl-{spec.name}-") as tmp:
        tmp = Path(tmp)
        rows, horizon, last = [], {}, {}
        for (zone, target), y in items:
            key = f"{zone}_{target}"
            y = _gap_free(y)
            last[key] = y.index[-1]
            horizon[key] = int((fh_index[-1] - y.index[-1]) / step)
            rows.append(pd.DataFrame({"series": key, "time_utc": y.index, "value": y.values.astype("float32")}))
        pd.concat(rows, ignore_index=True).to_parquet(tmp / "input.parquet", index=False)
        req = {"input": str(tmp / "input.parquet"), "output": str(tmp / "output.parquet"),
               "quantiles": cfg["quantiles"], "freq": FREQ, "horizon": horizon, **(extra or {})}
        (tmp / "request.json").write_text(json.dumps(req))
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(Path(cfg["_root"]).parent / ".hf-cache"))
        env["PYTHONPATH"] = str(Path(cfg["_root"]) / "models" / "_shared")
        env["FORETEC_HOME"] = str(cfg["_root"])
        env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
        cmd = [str(external_python(spec, cfg)), str(spec.folder / spec.raw.get("script", "forecaster.py")), str(tmp / "request.json")]
        try:
            r = subprocess.run(cmd, cwd=spec.folder, env=env, capture_output=True, text=True,
                               timeout=int(spec.raw.get("timeout", 1800)))
            if r.returncode != 0:
                raise RuntimeError(f"exit {r.returncode}: {r.stderr.strip()[-600:]}")
            res = pd.read_parquet(tmp / "output.parquet")
            ef = tmp / "output.parquet.errors.json"
            reasons = json.loads(ef.read_text()) if ef.exists() else {}
        except Exception as e:  # whole model failed: every series gets the error
            return {k: e for k, _ in items}
        for (zone, target), _ in items:
            key = f"{zone}_{target}"
            g = res[res["series"] == key]
            if g.empty:
                out[(zone, target)] = RuntimeError(reasons.get(key, "no output for this series"))
                continue
            g = g.assign(delivery_utc=last[key] + g["step"] * step).set_index("delivery_utc").reindex(fh_index)
            cols = ["point"] + [c for c in g.columns if c.startswith("q")]
            df = g[cols].reset_index()
            if df["point"].isna().any():
                out[(zone, target)] = RuntimeError(f"{int(df['point'].isna().sum())} missing quarter-hours")
            else:
                out[(zone, target)] = df
    return out


def run_forecasts(issue_date, cfg, results_subdir="results", only_models=None, snapshots=None, source=None, reference_run=False,
                  targets=None, series=None):
    """reference_run: run only `reference: true` models (after the gate) and log to _reference.json, so the
    gate check on _run.json is untouched. In the live run reference models are left out; in backtests they run.
    targets: run only these targets and merge into the day's existing forecasts and log (a target joining the backtest).
    series: the same for single series, as "ZONE_target" (e.g. a series whose source changed)."""
    p = paths(cfg, results_subdir)
    started = str(pd.Timestamp.now(tz="UTC"))
    snaps = snapshots if snapshots is not None else take_snapshot(issue_date, cfg, source=source, targets=targets)
    if targets:
        snaps = {k: v for k, v in snaps.items() if k[1] in targets}
    if series:
        snaps = {k: v for k, v in snaps.items() if f"{k[0]}_{k[1]}" in series}
    merge = bool(targets or series)
    in_subset = (lambda z, t: (not targets or t in targets) and (not series or f"{z}_{t}" in series))
    fh_index = delivery_index(issue_date, cfg)
    outdir = p["forecasts"] / str(issue_date)
    outdir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    covariates = None   # built once per issue day, only if some model asks for more than its own history
    live = results_subdir == "results"
    for spec in load_models(p["models"], only_models):
        is_ref = bool(spec.raw.get("reference"))
        if live and is_ref != reference_run:
            continue
        if not live and spec.raw.get("live_only"):   # e.g. "as published at the cut-off": unverifiable after the fact
            continue
        external = spec.raw.get("runner") == "subprocess"
        if external:
            py = external_python(spec, cfg)
            ok, why = py.exists(), f"model env not installed ({py})"
        else:
            ok, why = spec.dependencies_ok()
        if not ok:
            log.warning("skip %s: %s", spec.name, why)
            log_rows.append({"model": spec.name, "status": "skipped", "detail": why})
            continue
        frames = []
        items = [((zone, target), y) for (zone, target), y in snaps.items() if spec.applies(zone, target)
                 # live_only_zones: zones whose source only exists live (e.g. a forecast the publisher overwrites)
                 and (live or zone not in (spec.raw.get("live_only_zones") or []))]
        if not items:   # e.g. a wind/solar-only model in a run restricted to load
            continue
        extra = {}
        if external and set(spec.raw.get("inputs", ["history"])) - {"history", "calendar"}:
            if covariates is None:
                try:
                    from .covariates import build_covariates
                    build_covariates(issue_date, cfg)
                    covariates = str(Path(cfg["_root"]) / "data" / "covariates" / f"{issue_date}.parquet")
                except Exception as e:
                    log.error("covariates for %s unavailable: %s", issue_date, e)
                    covariates = ""
            if not covariates:
                log_rows.append({"model": spec.name, "status": "skipped", "detail": "covariates unavailable"})
                continue
            extra = {"covariates": covariates}
        if external:
            t0 = time.time()
            results = _forecast_external(spec, items, fh_index, cfg, extra)
            per_series = round((time.time() - t0) / max(len(items), 1), 2)
        for (zone, target), y in items:
            t0 = time.time()
            try:
                if external:
                    df = results[(zone, target)]
                    if isinstance(df, Exception):
                        raise df
                else:
                    df = _forecast_one(spec, y, fh_index, target, zone, cfg["quantiles"])
                df.insert(0, "target", target)
                df.insert(0, "zone", zone)
                frames.append(df)
                status, detail = "ok", ""
            except Exception as e:
                status, detail = "error", f"{type(e).__name__}: {e}"
                log.error("%s %s %s failed: %s", spec.name, zone, target, detail)
                log.debug(traceback.format_exc())
            log_rows.append({"model": spec.name, "zone": zone, "target": target, "status": status,
                             "seconds": per_series if external else round(time.time() - t0, 2), "detail": detail})
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.insert(0, "model", spec.name)
            out.insert(0, "issue_date", str(issue_date))
            target_f = outdir / f"{spec.name}.parquet"
            if merge and target_f.exists():   # keep the other series' forecasts as they are
                old = pd.read_parquet(target_f)
                mine = [in_subset(z, t) for z, t in zip(old["zone"], old["target"])]
                out = pd.concat([old[[not m for m in mine]], out], ignore_index=True)
            out.to_parquet(target_f, index=False)
    meta = {
        "issue_date": str(issue_date),
        "run_started_utc": started,
        "issue_time_utc": str(issue_timestamp(issue_date, cfg)),
        "run_finished_utc": str(pd.Timestamp.now(tz="UTC")),
        "series": [f"{z}_{t}" for z, t in snaps],
        "runs": log_rows,
    }
    f = outdir / ("_reference.json" if reference_run else "_run.json")
    if merge and f.exists():   # series added or re-run on a day: keep every other entry and the original run times
        old = json.loads(f.read_text())
        rerun = {r["model"] for r in meta["runs"]}
        keep = [r for r in old.get("runs", []) if not (r["model"] in rerun and ("target" not in r or in_subset(r.get("zone"), r.get("target"))))]
        stamp = {x: meta["run_finished_utc"] for x in (targets or series)}
        meta = {**old, "series": old.get("series", []) + [x for x in meta["series"] if x not in old.get("series", [])],
                "runs": keep + meta["runs"],
                "series_rerun_utc" if series else "targets_added_utc": {**old.get("series_rerun_utc" if series else "targets_added_utc", {}), **stamp}}
    elif (only_models or reference_run) and f.exists():  # partial re-run: keep the log entries of the other models
        old = json.loads(f.read_text())
        rerun = {r["model"] for r in meta["runs"]}
        meta["runs"] = [r for r in old.get("runs", []) if r["model"] not in rerun] + meta["runs"]
    f.write_text(json.dumps(meta, indent=1))
    return meta
