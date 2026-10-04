"""Static website: copy the page template and export the data it reads.

site/
  index.html, app.js, style.css      copied from foretec_live/web/
  data/summary.json                  every score row (live and backtest), models, config
  data/days/<issue_date>.json        actuals + every model's forecast for one issue day
Live results win over backtest results for the same issue day.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from types import SimpleNamespace
import pandas as pd

from .config import paths
from .timeutil import delivery_date, delivery_index, local_now

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"
PHASES = {"backtest": "results/backtest", "live": "results"}
BAND = ("q0.1", "q0.9")


def _atomic_write(path: Path, text: str) -> None:
    """Write next to the target, then rename: a visitor never gets a half-written file during a rebuild."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _round(a, nd=1):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def _scores(cfg) -> pd.DataFrame:
    frames = []
    for phase, sub in PHASES.items():
        f = paths(cfg, sub)["scores"]
        if f.exists():
            frames.append(pd.read_parquet(f).assign(phase=phase))
    if not frames:
        return pd.DataFrame()
    s = pd.concat(frames, ignore_index=True)
    s["issue_date"] = s["issue_date"].astype(str)
    live_days = set(s.loc[s["phase"] == "live", "issue_date"])
    return s[(s["phase"] == "live") | ~s["issue_date"].isin(live_days)]


def _forecast_days(cfg) -> dict[str, tuple[str, Path]]:
    """issue_date -> (phase, folder); live wins."""
    days = {}
    for phase, sub in PHASES.items():
        root = paths(cfg, sub)["forecasts"]
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir():
                    days[d.name] = (phase, d)
    return days


def _day_json(issue_date: str, folder: Path, cfg) -> dict:
    p = paths(cfg)
    idx = delivery_index(issue_date, cfg)
    out = {"issue_date": issue_date, "delivery_date": str(delivery_date(issue_date)),
           "t0": idx[0].isoformat() + "Z", "n": len(idx), "series": {}}
    fcs = [pd.read_parquet(f) for f in sorted(folder.glob("*.parquet"))]
    fc = pd.concat(fcs, ignore_index=True) if fcs else pd.DataFrame()
    for zone in cfg["zones"]:
        for target in cfg["targets"]:
            key = f"{zone}_{target}"
            a = p["actuals"] / out["delivery_date"] / f"{key}.parquet"
            prov = p["actuals_provisional"] / out["delivery_date"] / f"{key}.parquet"
            provisional = not a.exists() and prov.exists()     # metered so far, not frozen and never scored
            src = a if a.exists() else prov if provisional else None
            actual = pd.read_parquet(src)["value"].reindex(idx) if src else pd.Series(np.nan, index=idx)
            entry = {"actual": _round(actual) if actual.notna().any() else None, "models": {}}
            if provisional and entry["actual"]:
                entry["provisional"] = True
            if not fc.empty:
                sub = fc[(fc["zone"] == zone) & (fc["target"] == target)]
                for model, g in sub.groupby("model"):
                    g = g.set_index("delivery_utc").reindex(idx)
                    m = {"p": _round(g["point"])}
                    if all(c in g.columns for c in BAND) and g[BAND[0]].notna().any():
                        m["lo"], m["hi"] = _round(g[BAND[0]]), _round(g[BAND[1]])
                    entry["models"][model] = m
            out["series"][key] = entry
    out["inputs"] = _day_inputs(issue_date, idx, cfg)
    srcs = {}
    for folder in (p["actuals_provisional"], p["actuals"]):      # frozen provenance wins over provisional
        srcf = folder / out["delivery_date"] / "_sources.json"
        if srcf.exists():
            srcs.update(json.loads(srcf.read_text()))
    out["actual_sources"] = srcs or None
    fetched = p["actuals_provisional"] / out["delivery_date"] / "_fetched.json"
    if fetched.exists() and any(v.get("provisional") for v in out["series"].values()):
        out["provisional_fetched_utc"] = json.loads(fetched.read_text()).get("fetched_utc")
    revf = p["actuals"] / out["delivery_date"] / "_revisions.json"
    out["revisions"] = [c for c in json.loads(revf.read_text()) if c.get("revised")] if revf.exists() else []
    hist = p["snapshots"] / issue_date / "_sources.json"   # live days only; backtests do not store snapshots
    out["history_sources"] = json.loads(hist.read_text()) if hist.exists() else None
    return out


# key covariates shown on the site: (label, column template, unit)
SHOWN_COVARIATES = [
    ("wind_on", "Wind onshore, 100 m wind speed", "{z}.wind_onshore.{m}.wind_speed_100m", "m/s"),
    ("wind_off", "Wind offshore, 100 m wind speed", "{z}.wind_offshore.{m}.wind_speed_100m", "m/s"),
    ("gti", "Solar, tilted irradiance", "{z}.solar.{m}.global_tilted_irradiance", "W/m²"),
    ("cloud", "Solar, cloud cover", "{z}.solar.{m}.cloud_cover", "%"),
    ("temp", "Load centres, temperature", "{z}.load.{m}.temperature_2m", "°C"),
]


def _day_inputs(issue_date: str, idx, cfg) -> dict | None:
    """What covariate models received for this issue day, as frozen at the cut-off."""
    d = Path(cfg["_root"]) / "data" / "covariates"
    f, fm = d / f"{issue_date}.parquet", d / f"{issue_date}.json"
    if not (f.exists() and fm.exists()):
        return None
    meta = json.loads(fm.read_text())
    cov = pd.read_parquet(f).reindex(idx)
    models = list(meta.get("nwp_models", {}))
    shown = {}
    for zone in cfg["zones"]:
        z = {}
        for key, label, tmpl, unit in SHOWN_COVARIATES:
            members = {m: _round(cov[tmpl.format(z=zone, m=m)], 2) for m in models if tmpl.format(z=zone, m=m) in cov}
            if members:
                z[key] = {"label": label, "unit": unit, "members": members}
        col = f"{zone}.load_fc.entsoe.load_mw"
        if col in cov and cov[col].notna().any():
            z["load_fc"] = {"label": "ENTSO-E day-ahead load forecast", "unit": "MW", "members": {"entsoe": _round(cov[col], 0)}}
        if "fuel.ccgt.srmc.eur_mwh" in cov and cov["fuel.ccgt.srmc.eur_mwh"].notna().any():
            z["fuel"] = {"label": "CCGT fuel cost (TTF/0.52 + 0.37 x EUA)", "unit": "EUR/MWh", "members": {"fuel": _round(cov["fuel.ccgt.srmc.eur_mwh"], 2)}}
        shown[zone] = z
    # published download: the delivery-day slice every covariate model saw for D (earlier days are in earlier files)
    dl = Path(cfg["_root"]) / "site" / "data" / "inputs"
    dl.mkdir(parents=True, exist_ok=True)
    cov.to_csv(dl / f"{issue_date}_covariates.csv.gz", float_format="%.3f", compression="gzip")
    (dl / f"{issue_date}_inputs.json").write_text(json.dumps(meta, indent=1))
    return {
        "cutoff_utc": meta["cutoff_utc"],
        "runs": [r for r in meta["nwp_runs"] if r["issue_date"] == issue_date],
        "nwp_models": meta["nwp_models"],
        "load_forecast": meta["load_forecast"],
        "fuel": meta["fuel"],
        "n_columns": meta["columns"],
        "shown": shown,
        "download": {"covariates": f"data/inputs/{issue_date}_covariates.csv.gz", "meta": f"data/inputs/{issue_date}_inputs.json"},
    }


def _timeline(cfg) -> dict:
    """What the pipeline diagram draws, straight from the config (Brussels times, days relative to D)."""
    t = cfg["targets"]
    nwp = (cfg.get("covariates") or {}).get("nwp", {}).get("models", {})
    return {
        "issue_time": cfg["issue_time"], "gate": cfg.get("gate", cfg["issue_time"]), "schedule": cfg.get("schedule", {}),
        "price_score_after": t["price"].get("score_after"),
        "lag_hours": {k: float(str(v.get("publication_lag", "0h")).rstrip("h") or 0) for k, v in t.items()},
        "score_after_days": {k: v.get("score_after_days") for k, v in t.items()},
        "nwp": {m: {"label": v["label"], "lag_hours": v["lag_hours"], "cycle_hours": v["cycle_hours"]} for m, v in nwp.items()},
    }


def _tracker(cfg) -> dict:
    """Per issue day and model: forecasts generated, on time, scored. Built from the run logs and scores."""
    from .score import scorable
    from .timeutil import gate_timestamp

    s = _scores(cfg)
    scored = set() if s.empty else set(zip(s["issue_date"], s["model"], s["zone"], s["target"]))
    days, cells = [], {}
    for day, (phase, folder) in sorted(_forecast_days(cfg).items()):
        f = folder / "_run.json"
        meta = json.loads(f.read_text()) if f.exists() else {}
        fr = folder / "_reference.json"   # references published after the gate, logged separately
        ref_runs = json.loads(fr.read_text()).get("runs", []) if fr.exists() else []
        finished = meta.get("run_finished_utc")
        gate = gate_timestamp(day, cfg)
        on_time = None
        if phase == "live" and finished:
            on_time = bool(pd.Timestamp(finished).tz_convert("UTC").tz_localize(None) <= gate)
        due = {(z, t) for z in cfg["zones"] for t in cfg["targets"] if scorable(day, t, cfg)}
        per = {}
        for r in meta.get("runs", []) + ref_runs:
            c = per.setdefault(r["model"], {"ok": 0, "err": 0, "skip": 0, "due": 0, "scored": 0, "sec": 0.0, "msgs": []})
            if r["status"] == "ok":
                c["ok"] += 1
                if (r.get("zone"), r.get("target")) in due:
                    c["due"] += 1
                    c["scored"] += (day, r["model"], r["zone"], r["target"]) in scored
            elif r["status"] == "skipped":
                c["skip"] += 1
                c["msgs"].append(f"skipped: {r.get('detail', '')}"[:240])
            else:
                c["err"] += 1
                c["msgs"].append(f"{r.get('zone', '')} {r.get('target', '')}: {r.get('detail', '')}"[:240])
            c["sec"] += float(r.get("seconds") or 0)
        for c in per.values():
            c["sec"] = round(c["sec"], 1)
            c["msgs"] = c["msgs"][:6]
        cells[day] = per
        revf = paths(cfg)["actuals"] / str(delivery_date(day)) / "_revisions.json"
        revised = sorted({c["series"] for c in json.loads(revf.read_text()) if c.get("revised")}) if revf.exists() else []
        days.append({"d": day, "phase": phase, "revised": revised, "started": meta.get("run_started_utc"), "finished": finished,
                     "gate_utc": str(gate), "on_time": on_time, "n_series": len(meta.get("series", [])), "n_due": len(due)})
    return {"days": days, "cells": cells}


def _covariate_summary(cfg) -> dict | None:
    ccfg = cfg.get("covariates")
    if not ccfg:
        return None
    out = {"nwp_models": ccfg["nwp"]["models"], "centroids": [], "sources_doc": None}
    f = Path(cfg["_root"]) / ccfg["centroids"]
    if f.exists():
        c = pd.read_csv(f)
        out["centroids"] = [{"zone": r.zone, "bucket": r.bucket, "lat": round(r.latitude, 4), "lon": round(r.longitude, 4),
                             "weight": round(float(r.weight), 1), "n": int(r.unit_count), "source": getattr(r, "source", "")}
                            for r in c.itertuples()]
        readme = f.parent / "README.md"
        if readme.exists():   # published verbatim so every centroid source is inspectable
            (Path(cfg["_root"]) / "site" / "data").mkdir(parents=True, exist_ok=True)
            shutil.copy2(readme, Path(cfg["_root"]) / "site" / "data" / "centroids_README.md")
            shutil.copy2(f, Path(cfg["_root"]) / "site" / "data" / "centroids.csv")
            out["sources_doc"] = "data/centroids_README.md"
    return out


def build_site(cfg) -> Path:
    p = paths(cfg)
    site = p["site"]
    (site / "data" / "days").mkdir(parents=True, exist_ok=True)
    for f in WEB.iterdir():
        if f.is_file():
            _atomic_copy(f, site / f.name)
        elif f.is_dir():          # vendor/: self-hosted Chart.js and Leaflet, no third-party CDN at page load
            (site / f.name).mkdir(exist_ok=True)
            for g in f.iterdir():
                _atomic_copy(g, site / f.name / g.name)
    # cache-busting: browsers must pick up a new app.js/style.css as soon as it is deployed
    digests = {name: hashlib.sha1((site / name).read_bytes()).hexdigest()[:10] for name in ("app.js", "style.css")}
    for page in WEB.glob("*.html"):
        html = (site / page.name).read_text()
        for name, digest in digests.items():
            html = html.replace(f'"{name}"', f'"{name}?v={digest}"')
        _atomic_write(site / page.name, html)
    # old single-table page: point it at the new site
    (site / "backtest.html").write_text('<!doctype html><meta http-equiv="refresh" content="0;url=./#standings">')

    s = _scores(cfg)
    cols = ["issue_date", "phase", "model", "zone", "target", "mae", "rmse", "bias", "pinball", "rel_mae"]
    rows = [] if s.empty else [
        [r.issue_date, r.phase, r.model, r.zone, r.target] + _round([r.mae, r.rmse, r.bias, r.pinball], 2) + _round([r.rel_mae], 4)
        for r in s[cols].itertuples(index=False)
    ]
    days = _forecast_days(cfg)
    settled = str(local_now(cfg).date() - dt.timedelta(days=4))
    for issue_date, (phase, folder) in days.items():
        f = site / "data" / "days" / f"{issue_date}.json"
        # older days rarely change: rewrite only recent ones, or when forecasts/inputs are newer than the page data
        if f.exists() and issue_date < settled:
            sources = list(folder.glob("*.parquet")) + list((Path(cfg["_root"]) / "data" / "covariates").glob(f"{issue_date}.*"))
            if max((x.stat().st_mtime for x in sources), default=0) <= f.stat().st_mtime:
                continue
        js = _day_json(issue_date, folder, cfg)
        js["phase"] = phase
        _atomic_write(f, json.dumps(js, separators=(",", ":"), allow_nan=False))

    models = []
    # every model folder, including retired ones: their backtest stays public, with the reason
    for yml in sorted(p["models"].glob("*/model.yaml")):
        r = yaml.safe_load(yml.read_text()) or {}
        spec = SimpleNamespace(name=r.get("name", yml.parent.name))
        models.append({"name": spec.name, "author": r.get("author", ""), "description": r.get("description", ""),
                       "family": r.get("family", spec.name), "inputs": r.get("inputs", ["history"]),
                       "enabled": r.get("enabled", True), "retired": r.get("retired", ""),
                       "reference": bool(r.get("reference")), "live_only": bool(r.get("live_only")),
                       "kind": f"subprocess · env {r['env']}" if r.get("runner") == "subprocess" else (r.get("estimator") or r.get("module", ""))})
    summary = {
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "config": {
            "zones": list(cfg["zones"]),
            "targets": {t: {"unit": v["unit"]} for t, v in cfg["targets"].items()},
            "issue_time": cfg["issue_time"], "gate": cfg.get("gate", cfg["issue_time"]), "timezone": cfg["timezone"], "baseline": cfg["baseline"],
            "context_days": cfg["context_days"], "source": cfg["source"],
        },
        "models": models,
        "covariates": _covariate_summary(cfg),
        "timeline": _timeline(cfg),
        "tracker": _tracker(cfg),
        "score_columns": cols,
        "scores": rows,
        "days": {d: ph for d, (ph, _) in days.items()},
    }
    _atomic_write(site / "data" / "summary.json", json.dumps(summary, separators=(",", ":"), allow_nan=False))
    log.info("site: %d score rows, %d forecast days", len(rows), len(days))
    return site
