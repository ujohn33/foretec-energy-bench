"""Benchmark configuration and repository paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Repository root: $FORETEC_HOME if set, else the current working directory."""
    return Path(os.environ.get("FORETEC_HOME", Path.cwd())).resolve()


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.parent.resolve())
    return cfg


def paths(cfg: dict, results_subdir: str = "results") -> dict[str, Path]:
    root = Path(cfg["_root"])
    res = root / results_subdir
    p = {
        "root": root,
        "models": root / "models",
        "snapshots": root / "data" / "snapshots",
        "actuals": root / "data" / "actuals",
        "actuals_provisional": root / "data" / "actuals_provisional",
        "results": res,
        "forecasts": res / "forecasts",
        "scores": res / "scores.parquet",
        "leaderboard": res / "leaderboard.csv",
        "site": root / "site",
    }
    return p
