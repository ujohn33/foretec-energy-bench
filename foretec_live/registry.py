"""Model registry. Every folder in models/ with a model.yaml is one entry.

model.yaml, either a plain sktime estimator:

    name: naive_weekly
    author: Foretec
    description: Same quarter-hour one week earlier
    estimator: NaiveForecaster          # sktime class name, or a dotted import path
    params: {strategy: last, sp: 672}
    targets: [price, wind, solar]        # optional, default all
    zones: [BE, NL, FR]                  # optional, default all

or custom code in the same folder:

    name: lgbm_lags
    module: forecaster.py                # must define build(target, zone) -> sktime forecaster
"""
from __future__ import annotations

import warnings

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def _resolve_class(name: str):
    if "." in name:
        mod, cls = name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    from sktime.registry import all_estimators

    found = dict(all_estimators("forecaster"))
    if name not in found:
        raise KeyError(f"sktime has no forecaster called {name!r}")
    return found[name]


def _build_value(v):
    """Allow nested estimators in params: {class: lightgbm.LGBMRegressor, params: {...}}."""
    if isinstance(v, dict) and "class" in v:
        return _resolve_class(v["class"])(**{k: _build_value(x) for k, x in (v.get("params") or {}).items()})
    return v


@dataclass
class ModelSpec:
    name: str
    folder: Path
    raw: dict = field(repr=False)
    targets: list[str] | None = None
    zones: list[str] | None = None
    enabled: bool = True

    def applies(self, zone: str, target: str) -> bool:
        return (self.targets is None or target in self.targets) and (self.zones is None or zone in self.zones)

    def build(self, target: str, zone: str):
        if "module" in self.raw:
            path = self.folder / self.raw["module"]
            spec = importlib.util.spec_from_file_location(f"foretec_models.{self.name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.build(target=target, zone=zone)
        cls = _resolve_class(self.raw["estimator"])
        params = {k: _build_value(v) for k, v in (self.raw.get("params") or {}).items()}
        return cls(**params)

    def dependencies_ok(self) -> tuple[bool, str]:
        """Check soft dependencies before spending time on a run."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = self.build(target="price", zone="BE")
        except Exception as e:  # missing package, bad yaml, ...
            return False, f"{type(e).__name__}: {e}"
        try:
            from sktime.utils.dependencies import _check_estimator_deps

            if not _check_estimator_deps(est, severity="none"):
                return False, f"missing python dependencies: {est.get_tag('python_dependencies', None)}"
        except ImportError:
            pass
        return True, ""


def load_models(models_dir: Path, only: list[str] | None = None) -> list[ModelSpec]:
    specs = []
    for yml in sorted(Path(models_dir).glob("*/model.yaml")):
        raw = yaml.safe_load(yml.read_text()) or {}
        name = raw.get("name", yml.parent.name)
        if only and name not in only:
            continue
        spec = ModelSpec(
            name=name,
            folder=yml.parent,
            raw=raw,
            targets=raw.get("targets"),
            zones=raw.get("zones"),
            enabled=raw.get("enabled", True),
        )
        if spec.enabled:
            specs.append(spec)
    return specs
