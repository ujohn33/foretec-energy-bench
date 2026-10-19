"""Daily forecast run: take the as-of snapshot, run every model, write forecasts."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import datetime as dt
import time
from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from .config import paths
from .data import get_source, record_provenance
from .registry import load_models
from .timeutil import FREQ, as_date, availability_cutoff, delivery_index, issue_timestamp

log = logging.getLogger(__name__)
MIN_HISTORY_DAYS = 7


def take_snapshot(issue_date, cfg, source=None, save=True, targets=None) -> dict[tuple[str, str], pd.Series]:
    """Collect the history every model is allowed to see, and store it so the run can be repeated exactly."""
    from concurrent.futures import ThreadPoolExecutor

    source = source or get_source(cfg)
    p = paths(cfg)
    out = {}
    keys = [(zone, target) for zone in cfg["zones"] for target in targets or cfg["targets"]]

    def window(target):
        end = availability_cutoff(issue_date, target, cfg)
        return end - pd.Timedelta(days=cfg["context_days"]), end

    def get(key):   # the downloads run in parallel (they dominate the run's time before the gate); the rest in order
        try:
            return source.fetch(key[0], key[1], *window(key[1])), None
        except Exception as e:  # noqa: BLE001 - reported below, the series is skipped
            return None, e

    with ThreadPoolExecutor(int(cfg.get("snapshot_workers", 4))) as ex:
        fetched = dict(zip(keys, ex.map(get, keys)))
    for (zone, target), (s, err) in fetched.items():
        start, end = window(target)
        if err is not None:
            log.error("snapshot %s %s failed: %s", zone, target, err)
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
        threads = str(cfg.get("_threads_per_model") or os.cpu_count() or 1)   # shared out when models run in parallel
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = threads
        cmd = [str(external_python(spec, cfg)), str(spec.folder / spec.raw.get("script", "forecaster.py")), str(tmp / "request.json")]
        try:
            r = subprocess.run(cmd, cwd=spec.folder, env=env, capture_output=True, text=True,
                               timeout=int(spec.raw.get("timeout", 1800)))
            if r.returncode != 0:
                raise RuntimeError(f"exit {r.returncode}: {r.stderr.strip()[-600:]}")
            if (extra or {}).get("mode") == "train":   # a training run saves a model and forecasts nothing
                return {}
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
    live = results_subdir == "results"
    specs = []
    for spec in load_models(p["models"], only_models):
        is_ref = bool(spec.raw.get("reference"))
        if live and is_ref != reference_run:
            continue
        if not live and spec.raw.get("live_only"):   # e.g. "as published at the cut-off": unverifiable after the fact
            continue
        if live and spec.raw.get("live") is False:   # backtested, not yet run live (e.g. too slow for the gate)
            continue
        specs.append(spec)
    # ensembles combine the members' stored forecasts of the day, so they run after every other model
    ensembles = [sp for sp in specs if sp.raw.get("runner") == "ensemble"]
    specs = [sp for sp in specs if sp.raw.get("runner") != "ensemble"]
    if snapshots is not None:
        snaps = snapshots
    elif specs:
        snaps = take_snapshot(issue_date, cfg, source=source, targets=targets)
    else:   # ensembles only (e.g. one added later): nothing to fetch, the members' forecasts are on disk
        snaps = {}
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
    # covariates: built once per issue day, before the models run, if any model asks for more than its own history
    covariates = None
    needs_cov = lambda sp: sp.raw.get("runner") == "subprocess" and set(sp.raw.get("inputs", ["history"])) - {"history", "calendar"}
    if any(needs_cov(sp) for sp in specs):
        try:
            from .covariates import build_covariates
            build_covariates(issue_date, cfg)
            covariates = str(Path(cfg["_root"]) / "data" / "covariates" / f"{issue_date}.parquet")
        except Exception as e:
            log.error("covariates for %s unavailable: %s", issue_date, e)
    # models run model_workers at a time (config), each subprocess with its share of the CPU threads
    workers = max(1, int(cfg.get("model_workers", 1)))
    run_cfg = {**cfg, "_threads_per_model": max(1, (os.cpu_count() or 1) // workers)}

    def run_one(spec) -> list[dict]:
        rows = []
        external = spec.raw.get("runner") == "subprocess"
        if external:
            py = external_python(spec, cfg)
            ok, why = py.exists(), f"model env not installed ({py})"
        else:
            ok, why = spec.dependencies_ok()
        if not ok:
            log.warning("skip %s: %s", spec.name, why)
            return [{"model": spec.name, "status": "skipped", "detail": why}]
        frames = []
        items = [((zone, target), y) for (zone, target), y in snaps.items() if spec.applies(zone, target)
                 # live_only_zones: zones whose source only exists live (e.g. a forecast the publisher overwrites)
                 and (live or zone not in (spec.raw.get("live_only_zones") or []))]
        if not items:   # e.g. a wind/solar-only model in a run restricted to load
            return []
        extra = {"issue_date": str(issue_date), "live": live,    # live runs never train (models are trained ahead)
                 "artifact_dir": str(Path(cfg["_root"]) / "data" / "models" / spec.name)}
        if needs_cov(spec):
            if not covariates:
                return [{"model": spec.name, "status": "skipped", "detail": "covariates unavailable"}]
            extra["covariates"] = covariates
        if external:
            t0 = time.time()
            results = _forecast_external(spec, items, fh_index, run_cfg, extra)
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
            rows.append({"model": spec.name, "zone": zone, "target": target, "status": status,
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
        return rows

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            for rows in ex.map(run_one, specs):   # results in model order, whatever finishes first
                log_rows += rows
    else:
        for spec in specs:
            log_rows += run_one(spec)
    for spec in ensembles:
        rows, out = run_ensemble(spec, outdir, fh_index, cfg["quantiles"], in_subset if merge else None)
        log_rows += rows
        if out is not None:
            target_f = outdir / f"{spec.name}.parquet"
            if merge and target_f.exists():
                old = pd.read_parquet(target_f)
                mine = [in_subset(z, t) for z, t in zip(old["zone"], old["target"])]
                out = pd.concat([old[[not m for m in mine]], out], ignore_index=True)
            out.insert(0, "model", spec.name)
            out.insert(0, "issue_date", str(issue_date))
            out.to_parquet(target_f, index=False)
    meta = {
        "issue_date": str(issue_date),
        "run_started_utc": started,
        "issue_time_utc": str(issue_timestamp(issue_date, cfg)),
        "run_finished_utc": str(pd.Timestamp.now(tz="UTC")),
        "series": [f"{z}_{t}" for z, t in snaps] or sorted({f"{r['zone']}_{r['target']}" for r in log_rows if "zone" in r}),
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


def run_ensemble(spec, outdir: Path, fh_index: pd.DatetimeIndex, quantiles, in_subset=None):
    """An ensemble: per series and delivery quarter-hour, the median of its members' forecasts of the same issue day
    (point and each quantile separately, quantiles then sorted). Members are fixed in model.yaml (`members`); a series
    is combined when at least `min_members` of them (default all but one) delivered it in full, else it is not
    forecast. Returns (log rows, forecast frame or None)."""
    members = list(spec.raw["members"])
    need = int(spec.raw.get("min_members", len(members) - 1))
    qcols = [f"q{a:.1f}" for a in quantiles]
    frames = {}
    for m in members:
        f = outdir / f"{m}.parquet"
        if f.exists():
            frames[m] = pd.read_parquet(f)
    keys = sorted({(z, t) for fr in frames.values() for z, t in zip(fr["zone"], fr["target"])
                   if spec.applies(z, t) and (in_subset is None or in_subset(z, t))})
    rows, out = [], []
    for zone, target in keys:
        t0 = time.time()
        got = {}
        for m, fr in frames.items():
            sub = fr[(fr["zone"] == zone) & (fr["target"] == target)].set_index("delivery_utc").reindex(fh_index)
            if len(sub) and sub["point"].notna().all():
                got[m] = sub
        missing = [m for m in members if m not in got]
        if len(got) < need:
            rows.append({"model": spec.name, "zone": zone, "target": target, "status": "error", "seconds": 0.0,
                         "detail": f"{len(got)} of {len(members)} members delivered (need {need}); missing {', '.join(missing)}"})
            continue
        df = pd.DataFrame({"delivery_utc": fh_index,
                           "point": pd.concat([g["point"] for g in got.values()], axis=1).median(axis=1).to_numpy()})
        qs = {c: pd.concat([g[c] for g in got.values() if c in g and g[c].notna().all()], axis=1) for c in qcols}
        if all(len(v.columns) for v in qs.values()):
            qv = np.sort(np.column_stack([qs[c].median(axis=1).to_numpy() for c in qcols]), axis=1)
            for j, c in enumerate(qcols):
                df[c] = qv[:, j]
        df.insert(0, "target", target)
        df.insert(0, "zone", zone)
        out.append(df)
        rows.append({"model": spec.name, "zone": zone, "target": target, "status": "ok", "seconds": round(time.time() - t0, 2),
                     "detail": f"missing {', '.join(missing)}" if missing else ""})
    return rows, (pd.concat(out, ignore_index=True) if out else None)


def pretrain(issue_date, cfg, only_models=None) -> list[str]:
    """Morning training, outside the gate window, for models with `pretrain: true` in model.yaml (too slow to train
    inside the live run): each trains on the window of delivery days up to yesterday, from yesterday's frozen
    covariates and the history public now, and saves its model for today's issue day; the live run only predicts."""
    p = paths(cfg)
    specs = [s for s in load_models(p["models"], only_models) if s.raw.get("pretrain") and s.raw.get("live") is not False]
    if not specs:
        return []
    frame = Path(cfg["_root"]) / "data" / "covariates" / f"{as_date(issue_date) - dt.timedelta(days=1)}.parquet"
    if not frame.exists():
        raise RuntimeError(f"no covariates of yesterday's run ({frame.name}): nothing to train on")
    snaps = take_snapshot(issue_date, cfg, save=False)
    fh_index = delivery_index(issue_date, cfg)
    done = []
    for spec in specs:
        items = [((z, t), y) for (z, t), y in snaps.items() if spec.applies(z, t)]
        t0 = time.time()
        res = _forecast_external(spec, items, fh_index, cfg, {
            "mode": "train", "issue_date": str(issue_date), "covariates": str(frame),
            "artifact_dir": str(Path(cfg["_root"]) / "data" / "models" / spec.name)})
        failed = [k for k, v in res.items() if isinstance(v, Exception)]
        if failed:
            log.error("pretrain %s failed: %s", spec.name, res[failed[0]])
        else:
            done.append(f"{spec.name} ({time.time() - t0:.0f}s)")
    return done
