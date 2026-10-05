"""End-to-end check on the synthetic source: forecast -> score -> report, plus DST and leakage cutoffs."""
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from foretec_live.config import load_config, paths
from foretec_live.report import write_report
from foretec_live.run import run_forecasts, take_snapshot
from foretec_live.score import score_pending
from foretec_live.site import build_site
from foretec_live.timeutil import availability_cutoff, delivery_index, issue_timestamp

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")
    shutil.copytree(REPO / "models", tmp_path / "models")
    monkeypatch.setenv("FORETEC_HOME", str(tmp_path))
    c = load_config(tmp_path / "config.yaml")
    c["source"] = "synthetic"
    return c


@pytest.mark.parametrize("delivery,n", [("2026-03-29", 92), ("2026-10-25", 100), ("2026-10-02", 96)])
def test_dst_lengths(cfg, delivery, n):
    issue = dt.date.fromisoformat(delivery) - dt.timedelta(days=1)
    assert len(delivery_index(issue, cfg)) == n


def test_no_leakage(cfg):
    d = dt.date(2026, 10, 1)
    idx = delivery_index(d, cfg)
    assert availability_cutoff(d, "price", cfg) == idx[0]
    # generation is only known up to issue time minus publication lag
    assert availability_cutoff(d, "wind", cfg) <= issue_timestamp(d, cfg) - pd.Timedelta("1h")
    snaps = take_snapshot(d, cfg, save=False)
    for (zone, target), y in snaps.items():
        assert y.index.max() < availability_cutoff(d, target, cfg), (zone, target)


def test_forecast_score_report(cfg):
    d = dt.date(2026, 9, 20)
    models = ["naive_weekly", "naive_daily"]
    meta = run_forecasts(d, cfg, results_subdir="results/backtest", only_models=models)
    assert all(r["status"] == "ok" for r in meta["runs"])
    p = paths(cfg, "results/backtest")
    fc = pd.read_parquet(p["forecasts"] / d.isoformat() / "naive_daily.parquet")
    assert len(fc) == 96 * len(cfg["zones"]) * len(cfg["targets"])  # quarter-hours x zones x targets

    scores = score_pending(cfg, results_subdir="results/backtest", now=pd.Timestamp("2026-10-01"))
    assert set(scores["model"]) == set(models)
    assert (scores.loc[scores.model == "naive_weekly", "rel_mae"].round(9) == 1).all()

    lb = write_report(cfg, results_subdir="results/backtest")
    assert set(lb.index) == set(models) and p["leaderboard"].exists()

    site = build_site(cfg)
    summary = json.loads((site / "data" / "summary.json").read_text())
    assert {r[1] for r in summary["scores"]} == {"backtest"} and summary["days"] == {d.isoformat(): "backtest"}
    day = json.loads((site / "data" / "days" / f"{d.isoformat()}.json").read_text())
    assert len(day["series"]["BE_price"]["models"]["naive_daily"]["p"]) == 96
    assert all((site / f).exists() for f in ["index.html", "explorer.html", "methodology.html", "app.js", "style.css"])


def test_subprocess_model(cfg, tmp_path):
    """A subprocess model gets gap-free history and its steps land on the delivery grid."""
    import os
    import sys
    from foretec_live.registry import load_models
    from foretec_live.run import _forecast_external

    root = Path(cfg["_root"])
    cfg["envs_dir"] = str(tmp_path / "envs")
    env = tmp_path / "envs" / "dummy" / "bin"
    env.mkdir(parents=True)
    (env / "python").write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")   # a bare symlink would drop the venv
    (env / "python").chmod(0o755)
    m = root / "models" / "dummy_ext"
    m.mkdir()
    (m / "model.yaml").write_text("name: dummy_ext\nrunner: subprocess\nenv: dummy\n")
    (m / "forecaster.py").write_text(
        "import numpy as np\nfrom fl_io import read_request, write_output\n"
        "req, s = read_request()\n"
        "write_output(req, {k: (np.full(req['horizon'][k], y[-1]), np.tile(y[-1] + np.arange(9.0), (req['horizon'][k], 1))) for k, y in s.items()})\n")
    d = dt.date(2026, 9, 20)
    snaps = take_snapshot(d, cfg, save=False)
    y = snaps[("BE", "wind")].copy()
    y.iloc[-20:-15] = float("nan")                      # a short gap must be filled
    spec = load_models(root / "models", ["dummy_ext"])[0]
    idx = delivery_index(d, cfg)
    res = _forecast_external(spec, [(("BE", "wind"), y.dropna())], idx, cfg)[("BE", "wind")]
    assert not isinstance(res, Exception), res
    assert list(res["delivery_utc"]) == list(idx) and res["point"].notna().all()
    assert (res["q0.9"] - res["q0.1"]).eq(8).all()


def test_catchup_backtests_new_models(cfg, capsys):
    """A model folder without backtest results gets backtested; a second pass has nothing to do."""
    from foretec_live.cli import cmd_catchup

    cfg["backtest"] = {"start": "2026-09-20", "end": "2026-09-21"}
    root = Path(cfg["_root"])
    for m in root.glob("models/*/model.yaml"):
        if m.parent.name not in ("naive_daily", "naive_weekly"):
            shutil.rmtree(m.parent)
    cmd_catchup(cfg, None)
    p = paths(cfg, "results/backtest")
    assert all((p["forecasts"] / d / f"{m}.parquet").exists() for d in ["2026-09-20", "2026-09-21"] for m in ["naive_daily", "naive_weekly"])
    capsys.readouterr()
    cmd_catchup(cfg, None)
    assert "backtest complete" in capsys.readouterr().out


def test_live_run_after_gate_is_not_scored(cfg):
    """A live run that finishes after the day-ahead gate never gets scored; an on-time one does."""
    d = dt.date(2026, 9, 20)
    run_forecasts(d, cfg, only_models=["naive_weekly"])
    p = paths(cfg)
    log = p["forecasts"] / d.isoformat() / "_run.json"
    meta = json.loads(log.read_text())
    meta["run_finished_utc"] = "2026-09-20 10:30:00+00:00"   # 12:30 Brussels, after the 12:00 gate
    log.write_text(json.dumps(meta))
    assert score_pending(cfg, now=pd.Timestamp("2026-10-01")).empty
    meta["run_finished_utc"] = "2026-09-20 09:35:00+00:00"   # 11:35 Brussels
    log.write_text(json.dumps(meta))
    assert len(score_pending(cfg, now=pd.Timestamp("2026-10-01"))) == len(cfg["zones"]) * len(cfg["targets"])


def test_forecast_refuses_after_gate_and_over_locked_day(cfg):
    from types import SimpleNamespace
    from foretec_live.cli import cmd_forecast

    late = SimpleNamespace(date=dt.date(2026, 9, 20), models=["naive_weekly"], force=False)
    assert cmd_forecast(cfg, late) == 1                                   # 2026-09-20 12:00 is long past
    assert not (paths(cfg)["forecasts"] / "2026-09-20").exists()


def test_covariate_building_blocks(cfg):
    """Run selection respects publication lags (DST-safe), A03 curves repeat, ensemble stats are predico-style."""
    from foretec_live.covariates import _ensemble_stats, _parse_entsoe, select_run

    nwp = {"icon_eu": {"cycle_hours": 3, "lag_hours": 3.5}, "ecmwf_ifs": {"cycle_hours": 6, "lag_hours": 7}}
    cfg["covariates"] = {"nwp": {"models": nwp}}
    # cut-off 11:30 Brussels = 09:30 UTC in summer, 10:30 UTC in winter
    assert select_run("icon_eu", "2026-09-20", cfg) == pd.Timestamp("2026-09-20 06:00")
    assert select_run("ecmwf_ifs", "2026-09-20", cfg) == pd.Timestamp("2026-09-20 00:00")
    assert select_run("icon_eu", "2026-12-01", cfg) == pd.Timestamp("2026-12-01 06:00")

    xml = ('<GL_MarketDocument xmlns="urn:x"><TimeSeries><Period><timeInterval><start>2026-09-20T22:00Z</start>'
           '<end>2026-09-20T23:00Z</end></timeInterval><resolution>PT15M</resolution>'
           '<Point><position>1</position><quantity>10</quantity></Point>'
           '<Point><position>3</position><quantity>30</quantity></Point></Period></TimeSeries></GL_MarketDocument>')
    s = _parse_entsoe(xml)
    assert list(s.values) == [10, 10, 30, 30]          # A03: missing positions repeat the previous value

    idx = pd.date_range("2026-09-21", periods=2, freq="15min")
    wide = pd.DataFrame({"BE.solar.a.shortwave_radiation": [100.0, 0.0], "BE.solar.b.shortwave_radiation": [200.0, 0.0]}, index=idx)
    st = _ensemble_stats(wide, ["a", "b"])
    assert st["BE.solar.ens_mean.shortwave_radiation"].tolist() == [150.0, 0.0]
    assert st["BE.solar.ens_range.shortwave_radiation"].tolist() == [100.0, 0.0]
    assert st["BE.solar.diff_a_b.shortwave_radiation"].tolist() == [-100.0, 0.0]
    assert st["BE.solar.ens_cv.shortwave_radiation"].iloc[1] == 0.0   # no division by zero at night


def test_fuel_quote_counts_only_after_its_date(cfg, monkeypatch):
    """A quote dated X may be X's close: it must not be usable at the 11:30 cut-off on day X."""
    from foretec_live.covariates import fuel_quotes

    monkeypatch.delenv("OIL_PRICE_KEY", raising=False)
    cache = Path(cfg["_root"]) / "data" / "cache" / "fuel_quotes.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_utc": pd.to_datetime(["2026-09-18", "2026-09-19"]), "code": "TTF_EUR", "price": [70.0, 99.0]}).to_parquet(cache)
    q = fuel_quotes(cfg, pd.Timestamp("2026-09-19 09:30"))      # cut-off of issue day 2026-09-19
    assert q["price"].tolist() == [70.0]


def test_entsoe_parser_and_fallback(cfg):
    """A75: generation kept, consumption dropped, A03 gaps repeat; the fallback fills only what the primary lacks."""
    from foretec_live.data.base import FallbackSource, Source
    from foretec_live.data.entsoe import _finest, parse_timeseries

    xml = ('<GL_MarketDocument xmlns="urn:x">'
           '<TimeSeries><curveType>A03</curveType><inBiddingZone_Domain.mRID>10YBE----------2</inBiddingZone_Domain.mRID>'
           '<MktPSRType><psrType>B16</psrType></MktPSRType><Period><timeInterval><start>2026-09-20T22:00Z</start>'
           '<end>2026-09-20T23:00Z</end></timeInterval><resolution>PT15M</resolution>'
           '<Point><position>1</position><quantity>5</quantity></Point><Point><position>4</position><quantity>8</quantity></Point>'
           '</Period></TimeSeries>'
           '<TimeSeries><curveType>A01</curveType><outBiddingZone_Domain.mRID>10YBE----------2</outBiddingZone_Domain.mRID>'
           '<MktPSRType><psrType>B16</psrType></MktPSRType><Period><timeInterval><start>2026-09-20T22:00Z</start>'
           '<end>2026-09-20T22:15Z</end></timeInterval><resolution>PT15M</resolution>'
           '<Point><position>1</position><quantity>999</quantity></Point></Period></TimeSeries></GL_MarketDocument>')
    parts = parse_timeseries(xml)
    gen = [p for p in parts if p[0]["in_domain"] and not p[0]["out_domain"]]
    assert len(parts) == 2 and len(gen) == 1
    assert _finest(gen).tolist() == [5, 5, 5, 8]

    idx = pd.date_range("2026-09-20 22:00", periods=4, freq="15min")

    class Fake(Source):
        def __init__(self, name, values):
            super().__init__(cfg); self.name = name; self.values = values

        def fetch(self, zone, target, start, end):
            return pd.Series(self.values, index=idx).dropna()

    src = FallbackSource(cfg, Fake("entsoe", [1.0, None, 3.0, None]), Fake("energycharts", [9.0, 2.0, 9.0, 4.0]))
    out = src.fetch("BE", "wind", idx[0], idx[-1] + pd.Timedelta("15min"))
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]                 # primary wins wherever it has a value
    assert src.provenance[("BE", "wind")] == {"entsoe": 2, "energycharts": 2, "missing": 0}


def test_schedule_matches_systemd_timers():
    """The website draws the schedule from config.yaml; the timers must say the same."""
    import re

    import yaml
    sched = yaml.safe_load((REPO / "config.yaml").read_text())["schedule"]
    def times(unit):
        return sorted(re.findall(r"OnCalendar=\*-\*-\* (\d\d:\d\d):00 Europe/Brussels", (REPO / "scripts/systemd" / unit).read_text()))
    assert times("foretec-forecast.timer") == [sched["forecast"]]
    assert times("foretec-score.timer") == sorted(sched["score"])
    assert times("foretec-catchup.timer") == [sched["catchup"]]


def test_reference_and_live_only_models(cfg):
    """Reference models stay out of the live run and log separately; live-only models never backtest."""
    root = Path(cfg["_root"])
    for name, flags in (("ref_m", "reference: true\n"), ("live_m", "live_only: true\n")):
        d = root / "models" / name
        d.mkdir()
        (d / "model.yaml").write_text(f"name: {name}\nestimator: NaiveForecaster\nparams: {{strategy: last, sp: 96}}\n{flags}")
    d = dt.date(2026, 9, 20)
    live = run_forecasts(d, cfg, only_models=["naive_daily", "ref_m", "live_m"])
    assert {r["model"] for r in live["runs"]} == {"naive_daily", "live_m"}
    ref = run_forecasts(d, cfg, only_models=["naive_daily", "ref_m", "live_m"], reference_run=True)
    assert {r["model"] for r in ref["runs"]} == {"ref_m"}
    p = paths(cfg)
    assert json.loads((p["forecasts"] / d.isoformat() / "_run.json").read_text())["run_finished_utc"] == live["run_finished_utc"]
    bt = run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["naive_daily", "ref_m", "live_m"])
    assert {r["model"] for r in bt["runs"]} == {"naive_daily", "ref_m"}


def test_revision_check_refreezes_and_rescores(cfg):
    """Wind/solar scored on D+1; a small revision is only logged, a large one re-freezes and re-scores."""
    from foretec_live.score import check_revisions

    d = dt.date(2026, 9, 20)                                   # delivery 2026-09-21
    run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["naive_weekly", "naive_daily"])
    now = pd.Timestamp("2026-09-22 15:00")                     # D+1 afternoon
    s = score_pending(cfg, results_subdir="results/backtest", now=now)
    be_wind = (s["zone"] == "BE") & (s["target"] == "wind") & (s["issue_date"].astype(str) == "2026-09-20")
    assert be_wind.sum() == 2                                  # scored on D+1, not D+2

    p = paths(cfg, "results/backtest")
    f = paths(cfg)["actuals"] / "2026-09-21" / "BE_wind.parquet"
    original = pd.read_parquet(f)
    original.assign(value=original["value"] * 1.002).to_parquet(f)   # 0.2%: under the 0.5% threshold
    assert check_revisions(cfg, now=now) == []
    original.assign(value=original["value"] * 1.05).to_parquet(f)    # 5%: re-frozen from the source
    rev = check_revisions(cfg, now=now)
    assert [r["series"] for r in rev] == ["BE_wind"]
    assert pd.read_parquet(f)["value"].round(6).equals(original["value"].round(6))
    s = pd.read_parquet(p["scores"])
    assert not ((s["zone"] == "BE") & (s["target"] == "wind") & (s["issue_date"].astype(str) == "2026-09-20")).any()
    s = score_pending(cfg, results_subdir="results/backtest", now=now)
    assert ((s["zone"] == "BE") & (s["target"] == "wind") & (s["issue_date"].astype(str) == "2026-09-20")).sum() == 2
    checks = json.loads((f.parent / "_revisions.json").read_text())
    assert [c["revised"] for c in checks if c["series"] == "BE_wind"] == [False, True]


def test_fallback_circuit_breaker(cfg):
    """A failing fallback is tried once per run, then skipped: it must never stall the live run."""
    from foretec_live.data.base import FallbackSource, Source

    idx = pd.date_range("2026-09-20 22:00", periods=4, freq="15min")
    calls = []

    class Gappy(Source):
        name = "entsoe"
        def fetch(self, zone, target, start, end):
            return pd.Series([1.0, None, 3.0, 4.0], index=idx).dropna()

    class Down(Source):
        name = "energycharts"
        def fetch(self, zone, target, start, end):
            calls.append(zone)
            raise RuntimeError("503")

    src = FallbackSource(cfg, Gappy(cfg), Down(cfg))
    for z in ("BE", "NL", "FR"):
        out = src.fetch(z, "wind", idx[0], idx[-1] + pd.Timedelta("15min"))
        assert out.tolist() == [1.0, 3.0, 4.0]
    assert calls == ["BE"]


def test_fallback_chain_order(cfg):
    """Fallbacks fill in order: the first that has a quarter-hour wins, later ones only fill what is left."""
    from foretec_live.data.base import FallbackSource, Source

    idx = pd.date_range("2026-09-20 22:00", periods=4, freq="15min")

    def src(name, values):
        class S(Source):
            def fetch(self, zone, target, start, end):
                return pd.Series(values, index=idx).dropna()
        s = S(cfg); s.name = name
        return s

    chain = FallbackSource(cfg, src("entsoe", [1.0, None, None, None]), src("elia", [9.0, 2.0, None, None]),
                           src("rte", [None, None, None, None]), src("energycharts", [9.0, 9.0, 3.0, None]))
    out = chain.fetch("BE", "wind", idx[0], idx[-1] + pd.Timedelta("15min"))
    assert out.tolist() == [1.0, 2.0, 3.0]
    assert chain.provenance[("BE", "wind")] == {"entsoe": 1, "elia": 1, "rte": 0, "energycharts": 1, "missing": 1}


def test_early_scoring_needs_complete_day_and_provisional_actuals(cfg):
    """07:00 on D+1 freezes wind/solar only if D is complete; until then the site shows provisional values, never scored."""
    from foretec_live.score import provisional_actuals, required_coverage

    d = dt.date(2026, 9, 20)                                   # delivery 2026-09-21
    assert required_coverage(d, "wind", cfg, now=pd.Timestamp("2026-09-22 07:00")) == 1.0
    assert required_coverage(d, "solar", cfg, now=pd.Timestamp("2026-09-22 14:45")) == 0.95
    assert required_coverage(d, "price", cfg, now=pd.Timestamp("2026-09-22 07:00")) == 0.95

    run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["naive_daily"])
    during = pd.Timestamp("2026-09-21 15:00")                  # delivery day under way
    written = provisional_actuals(cfg, now=during)
    assert any(w.startswith("2026-09-21 BE_wind") for w in written)
    prov = paths(cfg)["actuals_provisional"] / "2026-09-21" / "BE_wind.parquet"
    assert prov.exists() and (prov.parent / "_fetched.json").exists()
    s = score_pending(cfg, results_subdir="results/backtest", now=during)
    assert not (s["target"].isin(["wind", "solar"])).any()     # provisional values are never scored
    assert not (paths(cfg)["actuals"] / "2026-09-21" / "BE_wind.parquet").exists()

    s = score_pending(cfg, results_subdir="results/backtest", now=pd.Timestamp("2026-09-22 07:00"))
    assert ((s["target"] == "wind") & (s["issue_date"].astype(str) == "2026-09-20")).sum() == 3   # complete day, scored at 07:00
    provisional_actuals(cfg, now=pd.Timestamp("2026-09-22 07:10"))
    assert not prov.exists()                                   # frozen actuals replace the provisional ones


def test_new_target_merges_into_backtest_day(cfg):
    """backfill --targets load adds the new target to a backtest day without touching the other targets."""
    d = dt.date(2026, 9, 20)
    others = [t for t in cfg["targets"] if t != "load"]
    run_forecasts(d, {**cfg, "targets": {t: cfg["targets"][t] for t in others}}, results_subdir="results/backtest",
                  only_models=["naive_daily"])
    f = paths(cfg, "results/backtest")["forecasts"] / d.isoformat() / "naive_daily.parquet"
    before = pd.read_parquet(f)
    assert set(before["target"]) == set(others)
    run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["naive_daily"], targets=["load"])
    # a model without load (wind/solar only) is skipped, not called with no series
    root = Path(cfg["_root"]) / "models" / "ws_only"
    root.mkdir()
    (root / "model.yaml").write_text("name: ws_only\nestimator: NaiveForecaster\nparams: {strategy: last}\ntargets: [wind, solar]\n")
    meta = run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["ws_only"], targets=["load"])
    assert not [r for r in meta["runs"] if r["model"] == "ws_only"]
    after = pd.read_parquet(f)
    assert set(after["target"]) == set(cfg["targets"])
    pd.testing.assert_frame_equal(after[after["target"] != "load"].reset_index(drop=True), before.reset_index(drop=True))
    log = json.loads((f.parent / "_run.json").read_text())
    assert {r["target"] for r in log["runs"]} == set(cfg["targets"])
