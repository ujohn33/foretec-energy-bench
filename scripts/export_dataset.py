"""Export the benchmark as a dataset (Hugging Face / Zenodo): every forecast, actual and score, as parquet.

  scores/{live,backtest}.parquet     one row per issue day, model and series: MAE, RMSE, bias, pinball, WIS, Winkler,
                                     coverage, the baseline's MAE and Rel. MAE, quarter-hours scored (n)
  forecasts/{live,backtest}.parquet  every forecast: issue day, model, zone, target, delivery quarter-hour (UTC), point
                                     and the 10th..90th percentiles
  actuals/actuals.parquet            the actuals the forecasts were scored against (frozen), per delivery quarter-hour
  models.csv                         every model shown on the site: author, family, inputs, description
  README.md                          the dataset card (Hugging Face YAML header)
Models hidden on the site are left out. Live results win over backtest results for the same issue day, as on the site.

Usage (on the server): FORETEC_HOME=/srv/foretec/foretec-live python scripts/export_dataset.py /srv/foretec/dataset
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foretec_live.config import load_config, paths  # noqa: E402
from foretec_live.site import PHASES, _hidden, _scores  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
cfg = load_config()
hidden = _hidden(cfg)
for d in ("scores", "forecasts", "actuals"):
    (out / d).mkdir(parents=True, exist_ok=True)

s = _scores(cfg)
live_days = set(s.loc[s["phase"] == "live", "issue_date"])
for phase in PHASES:
    s[s["phase"] == phase].drop(columns="phase").sort_values(["issue_date", "zone", "target", "model"]) \
        .to_parquet(out / "scores" / f"{phase}.parquet", index=False, compression="zstd")

n_fc = {}
for phase, sub in PHASES.items():
    frames = []
    for day in sorted(p for p in paths(cfg, sub)["forecasts"].iterdir() if p.is_dir()):
        if phase == "backtest" and day.name in live_days:
            continue
        for f in day.glob("*.parquet"):
            if f.stem not in hidden:
                frames.append(pd.read_parquet(f))
    fc = pd.concat(frames, ignore_index=True)
    fc = fc[~fc["model"].isin(hidden)]
    fc["issue_date"] = fc["issue_date"].astype(str)
    fc.sort_values(["issue_date", "zone", "target", "model", "delivery_utc"]).to_parquet(
        out / "forecasts" / f"{phase}.parquet", index=False, compression="zstd")
    n_fc[phase] = len(fc)

frames = []
for day in sorted(p for p in paths(cfg)["actuals"].iterdir() if p.is_dir()):
    for f in day.glob("*_*.parquet"):
        zone, target = f.stem.rsplit("_", 1)
        a = pd.read_parquet(f).reset_index()
        frames.append(a.assign(zone=zone, target=target, delivery_date=day.name))
act = pd.concat(frames, ignore_index=True)[["delivery_date", "zone", "target", "delivery_utc", "value"]]
act.sort_values(["delivery_date", "zone", "target", "delivery_utc"]).to_parquet(out / "actuals" / "actuals.parquet", index=False, compression="zstd")

rows = []
for yml in sorted(paths(cfg)["models"].glob("*/model.yaml")):
    r = yaml.safe_load(yml.read_text()) or {}
    name = r.get("name", yml.parent.name)
    if name in hidden:
        continue
    rows.append({"model": name, "author": r.get("author", ""), "family": r.get("family", name),
                 "inputs": ",".join(r.get("inputs", ["history"])), "enabled": r.get("enabled", True),
                 "reference": bool(r.get("reference")), "description": r.get("description", "")})
pd.DataFrame(rows).to_csv(out / "models.csv", index=False)

zones = {z: v.get("label", z) for z, v in cfg["zones"].items()}
first, last = s["issue_date"].min(), s["issue_date"].max()
card = Path(__file__).with_name("dataset_card.md").read_text().format(
    zones=", ".join(zones.values()), first=first, last=last, n_live=len(live_days),
    n_models=len(rows), n_scores=len(s), n_fc_live=n_fc["live"], n_fc_bt=n_fc["backtest"], baseline=cfg["baseline"])
(out / "README.md").write_text(card)
print(f"{out}: {len(s)} score rows, {n_fc} forecast rows, {len(act)} actual rows, {len(rows)} models")
for f in sorted(out.rglob("*.*")):
    print(f"  {f.relative_to(out)}  {f.stat().st_size / 1e6:.1f} MB")
