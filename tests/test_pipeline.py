"""End-to-end check on the synthetic source: forecast -> score -> report, plus DST and leakage cutoffs."""
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
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
    c["void_live"] = []   # the production voids are dated; tests set their own
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
    meta["run_finished_utc"] = "2026-09-20 10:30:00+00:00"   # 12:30 CET/CEST, after the 12:00 gate
    log.write_text(json.dumps(meta))
    assert score_pending(cfg, now=pd.Timestamp("2026-10-01")).empty
    meta["run_finished_utc"] = "2026-09-20 09:35:00+00:00"   # 11:35 CET/CEST
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
    # cut-off 11:30 CET/CEST = 09:30 UTC in summer, 10:30 UTC in winter
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


def test_load_forecast_is_a_price_input_only():
    """Load models never get the TSO's load forecast (the benchmark they are meant to beat); price models do."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "_shared"))
    from fl_cov import compact_features, full_features

    cols = [f"{z}.{b}.ens_mean.{v}" for z in ("BE", "NL") for b, v in
            (("wind_onshore", "wind_speed_100m"), ("wind_offshore", "wind_speed_100m"), ("solar", "global_tilted_irradiance"),
             ("load", "temperature_2m"), ("load", "shortwave_radiation"), ("load", "cloud_cover"))]
    cols += ["BE.load_fc.entsoe.load_mw", "NL.load_fc.entsoe.load_mw", "fuel.ccgt.srmc.eur_mwh"]
    for zone in ("BE", "NL"):
        for pick in (compact_features, full_features):
            load = pick(cols, zone, "load")
            assert load and not [c for c in load if ".load_fc." in c], (pick.__name__, zone, load)
            assert f"{zone}.load_fc.entsoe.load_mw" in pick(cols, zone, "price"), (pick.__name__, zone)


def test_half_ingested_nwp_run_is_rejected(cfg, monkeypatch):
    """A reply with a time axis but null values over the delivery day is retried, then refused (so the builder falls
    back to the previous cycle and nothing is cached), as ICON-EU 06 UTC came at the cut-off on 17 Oct 2026."""
    from foretec_live import covariates as cv

    run = pd.Timestamp("2026-10-17 06:00")
    times = [str(t) for t in pd.date_range(run, periods=60 * 4, freq="15min")]
    variables = sorted({v for vs in cv.BUCKET_VARS.values() for v in vs})

    def reply(null_var):
        m = {"time": times, **{v: [None if v == null_var and i >= 48 else 1.0 for i in range(len(times))] for v in variables}}
        return [{"minutely_15": m}, {"minutely_15": m}]

    class R:
        status_code = 200
        def __init__(self, js): self._js = js
        def json(self): return self._js

    cfg["covariates"] = {"nwp": {"host": "https://example.invalid", "forecast_hours": 60}}
    c = pd.DataFrame({"latitude": [50.0, 51.0], "longitude": [4.0, 5.0]})
    monkeypatch.setattr(cv.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(cv.requests, "get", lambda *a, **k: calls.append(1) or R(reply("cloud_cover")))
    with pytest.raises(RuntimeError, match="incomplete values"):
        cv._request_run("icon_eu", run, cfg, c)
    assert len(calls) == 5
    monkeypatch.setattr(cv.requests, "get", lambda *a, **k: R(reply(None)))
    out = cv._request_run("icon_eu", run, cfg, c)
    assert out["cloud_cover"].notna().all() and set(out["centroid"]) == {0, 1}


def test_capacity_scaling_per_zone(cfg, tmp_path):
    """scale: {solar: capacity} scales every zone (proxy without a series); {wind: {capacity: [FR]}} only FR, given a series."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "_shared"))
    from fl_covmodel import scale_kind

    d = Path(cfg["_root"]) / "data" / "static" / "capacity"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mw": [1.0]}, index=pd.DatetimeIndex(["2026-01-01"], name="date")).to_parquet(d / "FR_wind.parquet")
    pd.DataFrame({"mw": [1.0]}, index=pd.DatetimeIndex(["2026-01-01"], name="date")).to_parquet(d / "BE_solar.parquet")
    spec = {"scale": {"solar": "capacity", "wind": {"capacity": ["FR", "DK1"]}}}
    assert scale_kind(cfg, spec, "BE_solar") == "capacity"
    assert scale_kind(cfg, spec, "NL_solar") == "proxy"
    assert scale_kind(cfg, spec, "FR_wind") == "capacity"
    assert scale_kind(cfg, spec, "DK1_wind") is None          # listed, but no capacity series here
    assert scale_kind(cfg, spec, "DE_wind") is None
    assert scale_kind(cfg, spec, "FR_price") is None


def test_fuel_quote_counts_only_after_its_date(cfg, monkeypatch):
    """A quote dated X may be X's close: it must not be usable at the 11:30 cut-off on day X."""
    from foretec_live.covariates import fuel_quotes

    cache = Path(cfg["_root"]) / "data" / "cache" / "fuel_quotes.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_utc": pd.to_datetime(["2026-09-18", "2026-09-19"]), "code": "TTF_EUR", "price": [70.0, 99.0],
                  "source": "yahoo"}).to_parquet(cache)                # fresh cache: no request is made
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

    # hourly load sent as separate periods around gaps (Energinet DK, Oct 2026): a one-point hour fills its 4 quarters
    hourly = ('<GL_MarketDocument xmlns="urn:x">'
              '<TimeSeries><curveType>A01</curveType><outBiddingZone_Domain.mRID>10YDK-1--------W</outBiddingZone_Domain.mRID>'
              '<Period><timeInterval><start>2026-10-16T22:00Z</start><end>2026-10-16T23:00Z</end></timeInterval>'
              '<resolution>PT60M</resolution><Point><position>1</position><quantity>2000</quantity></Point></Period>'
              '<Period><timeInterval><start>2026-10-17T01:00Z</start><end>2026-10-17T03:00Z</end></timeInterval>'
              '<resolution>PT60M</resolution><Point><position>1</position><quantity>1800</quantity></Point>'
              '<Point><position>2</position><quantity>1700</quantity></Point></Period></TimeSeries></GL_MarketDocument>')
    load = _finest(parse_timeseries(hourly))
    assert len(load) == 12 and (load.loc["2026-10-16 22:00":"2026-10-16 22:45"] == 2000).all()
    assert load.index[4] == pd.Timestamp("2026-10-17 01:00") and load.iloc[-1] == 1700   # 23:00-01:00 stays a gap

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
    assert times("foretec-pretrain.timer") == [sched["pretrain"]]


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

    # actuals_locked_through: frozen under an earlier series definition, never re-checked, scores kept
    cfg["actuals_locked_through"] = {"BE_wind": "2026-09-21"}
    original.assign(value=original["value"] * 1.05).to_parquet(f)
    assert check_revisions(cfg, now=now) == []
    assert pd.read_parquet(f)["value"].round(6).equals((original["value"] * 1.05).round(6))


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
    out = chain.fetch("NL", "wind", idx[0], idx[-1] + pd.Timedelta("15min"))   # NL: no source_by_series entry
    assert out.tolist() == [1.0, 2.0, 3.0]
    assert chain.provenance[("NL", "wind")] == {"entsoe": 1, "elia": 1, "rte": 0, "energycharts": 1, "missing": 1}


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
    assert ((s["target"] == "wind") & (s["issue_date"].astype(str) == "2026-09-20")).sum() == len(cfg["zones"])   # complete day, scored at 07:00
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


def test_nwp_fetch_retries_a_200_that_is_not_json(cfg, monkeypatch):
    """Open-Meteo once answered 200 with a non-JSON body: that must be retried, not taken as the run."""
    from foretec_live import covariates as cv

    c = pd.DataFrame({"zone": ["BE", "BE"], "bucket": ["solar", "load"], "cluster_id": [0, 0],
                      "latitude": [50.85, 51.05], "longitude": [4.35, 4.45], "weight": [1.0, 1.0]})
    good = [{"minutely_15": {"time": ["2026-10-06T00:00", "2026-10-06T00:15"], **{v: [1.0, 2.0] for vs in cv.BUCKET_VARS.values() for v in vs}}}
            for _ in range(len(c))]

    class Reply:
        def __init__(self, body):
            self.status_code, self.body, self.text = 200, body, "" if body is None else "json"

        def json(self):
            if self.body is None:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return self.body

    replies = iter([Reply(None), Reply(None), Reply(good)])
    monkeypatch.setattr(cv.requests, "get", lambda *a, **k: next(replies))
    monkeypatch.setattr(cv.time, "sleep", lambda s: None)
    raw = cv.fetch_run("icon_eu", pd.Timestamp("2026-10-06 06:00"), cfg, c)
    assert len(raw) == 2 * len(c) and set(raw["centroid"]) == set(range(len(c)))


def test_source_by_series_puts_a_fallback_first(cfg):
    """source_by_series: for BE wind the preferred source is asked first, the primary fills its gaps."""
    from foretec_live.data.base import FallbackSource, Source

    class Fake(Source):
        def __init__(self, name, value, gap=False):
            super().__init__(cfg); self.name, self.value, self.gap = name, value, gap

        def fetch(self, zone, target, start, end):
            idx = pd.date_range(start, end, freq="15min", inclusive="left")
            s = pd.Series(self.value, index=idx, dtype=float)
            return s.iloc[1:] if self.gap else s

    a, b = pd.Timestamp("2026-09-20"), pd.Timestamp("2026-09-20 02:00")
    chain = FallbackSource({**cfg, "source_by_series": {"BE_wind": "elia"}}, Fake("entsoe", 1.0), Fake("elia", 2.0, gap=True))
    be = chain.fetch("BE", "wind", a, b)
    assert be.iloc[0] == 1.0 and (be.iloc[1:] == 2.0).all()          # Elia first, ENTSO-E fills its gap
    assert chain.provenance[("BE", "wind")] == {"elia": 7, "entsoe": 1, "missing": 0}
    assert (chain.fetch("NL", "wind", a, b) == 1.0).all()             # other series: the primary as before
    # exclusive: a different quantity elsewhere, so the preferred source's gap stays a gap
    excl = FallbackSource({**cfg, "source_by_series": {"NL_solar": "ned"}, "exclusive_series": ["NL_solar"]},
                          Fake("entsoe", 1.0), Fake("ned", 3.0, gap=True))
    nl = excl.fetch("NL", "solar", a, b)
    assert len(nl) == 7 and (nl == 3.0).all() and excl.provenance[("NL", "solar")]["missing"] == 1


def test_series_rerun_merges_into_backtest_day(cfg):
    """backfill --series BE_wind re-runs one series and leaves every other series of the day untouched."""
    d = dt.date(2026, 9, 20)
    run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["naive_daily"])
    f = paths(cfg, "results/backtest")["forecasts"] / d.isoformat() / "naive_daily.parquet"
    before = pd.read_parquet(f)
    run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["naive_daily"], series=["BE_wind"])
    after = pd.read_parquet(f)
    key = after["zone"] + "_" + after["target"]
    bkey = before["zone"] + "_" + before["target"]
    pd.testing.assert_frame_equal(after[key != "BE_wind"].reset_index(drop=True), before[bkey != "BE_wind"].reset_index(drop=True))
    assert (key == "BE_wind").sum() == (bkey == "BE_wind").sum()
    log = json.loads((f.parent / "_run.json").read_text())
    assert sum(1 for r in log["runs"] if r["model"] == "naive_daily") == len(cfg["zones"]) * len(cfg["targets"])


def test_voided_live_series_are_never_scored(cfg):
    """void_live: live forecasts made from a different quantity stay on disk and in the tracker, never scored."""
    d = dt.date(2026, 9, 20)
    cfg["void_live"] = [{"series": ["NL_wind", "NL_load"], "through": "2026-09-20", "reason": "test"}]
    run_forecasts(d, cfg, only_models=["naive_weekly"])
    log = paths(cfg)["forecasts"] / d.isoformat() / "_run.json"
    log.write_text(json.dumps({**json.loads(log.read_text()), "run_finished_utc": "2026-09-20 09:35:00+00:00"}))   # on time
    s = score_pending(cfg, now=pd.Timestamp("2026-10-01"))
    keys = set(s["zone"] + "_" + s["target"])
    assert not keys & {"NL_wind", "NL_load"} and "NL_solar" in keys
    assert len(score_pending(cfg, now=pd.Timestamp("2026-10-02"))) == len(s)   # a later pass does not re-score them
    from foretec_live.site import _tracker
    c = _tracker(cfg)["cells"][d.isoformat()]["naive_weekly"]
    assert c["s"]["NL_wind"] == "V" and c["why"]["NL_wind"] == "test" and c["void"] == 2


def test_void_live_window(cfg):
    """void_live with `from`: only issue days inside [from, through] are voided."""
    from foretec_live.score import voided
    cfg["void_live"] = [{"series": ["DK1_wind"], "from": "2026-10-17", "through": "2026-10-18", "reason": "w"}]
    assert [bool(voided(d, "DK1", "wind", cfg)) for d in ("2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19")] == [False, True, True, False]
    assert not voided("2026-10-17", "DK1", "solar", cfg)


def test_energinet_wind_quarter_hours(cfg, monkeypatch):
    """Energinet DK wind: onshore + offshore, 5-minute values averaged to quarter-hours (2 of 3 needed), one area only."""
    from foretec_live.data import tso

    t = pd.date_range("2026-10-01 00:00", periods=9, freq="5min")
    rows = [{"Minutes5UTC": f"{x:%Y-%m-%dT%H:%M:%S}", "OnshoreWindPower": 100.0 + i, "OffshoreWindPower": 50.0} for i, x in enumerate(t)]
    del rows[4], rows[3]                                     # 00:15 quarter has only 00:25 left
    rows[-1]["OffshoreWindPower"] = None                     # 00:40 lacks offshore: not counted
    seen = []

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"records": rows}

    monkeypatch.setattr(tso.requests, "get", lambda url, params, timeout: seen.append(params) or R())
    src = tso.EnerginetSource(cfg)
    s = src.fetch("DK1", "wind", pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-01 01:00"))
    assert list(s.index) == [pd.Timestamp("2026-10-01 00:00"), pd.Timestamp("2026-10-01 00:30")]
    assert s.iloc[0] == 151.0 and s.iloc[1] == 156.5        # mean of 150,151,152; of 156,157 (00:40 dropped)
    assert seen[0]["filter"] == '{"PriceArea":["DK1"]}' and seen[0]["timezone"] == "UTC"
    assert src.fetch("DK1", "solar", pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")).empty   # solar stays on ENTSO-E
    assert src.fetch("NL", "wind", pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")).empty
    # DK load stays on ENTSO-E: as a fallback this source must not fill A65's gaps with a different quantity
    n = len(seen)
    assert src.fetch("DK1", "load", pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")).empty and len(seen) == n


def test_long_history_lag_models(cfg):
    """lgbm_lags / lin_lags: trained through the end of the previous month (never the issue month), with intervals."""
    import pickle

    import yaml

    from foretec_live.data import get_source

    root = Path(cfg["_root"])
    for m in ("lgbm_lags", "lin_lags"):           # a short window keeps the test fast
        f = root / "models" / m / "model.yaml"
        f.write_text(yaml.safe_dump({**yaml.safe_load(f.read_text()), "train_days": 21, "holdout_days": 5}))
    src, ext = get_source(cfg), root / "data" / "static" / "history_ext"
    ext.mkdir(parents=True)
    for z in cfg["zones"]:
        for t in cfg["targets"]:
            s = src.fetch(z, t, pd.Timestamp("2026-07-20"), pd.Timestamp("2026-09-02"))
            s.rename("value").to_frame().to_parquet(ext / f"{z}_{t}.parquet")
    d = dt.date(2026, 9, 20)
    meta = run_forecasts(d, cfg, results_subdir="results/backtest", only_models=["lgbm_lags", "lin_lags"])
    assert meta["runs"] and all(r["status"] == "ok" for r in meta["runs"]), [r for r in meta["runs"] if r["status"] != "ok"][:2]
    fc = pd.read_parquet(paths(cfg, "results/backtest")["forecasts"] / d.isoformat() / "lin_lags.parquet")
    assert fc["q0.9"].ge(fc["q0.1"]).all() and fc["point"].notna().all()
    art = pickle.loads((root / "data" / "models" / "lgbm_lags" / "2026-08-31.pkl").read_bytes())
    assert art["train_end"] == "2026-08-31" and len(art["series"]) == len(cfg["zones"]) * len(cfg["targets"])


def test_previous_runs_lead_matching(cfg):
    """Training weather from the previous-runs archive: per quarter-hour the offset nearest the live run's lead."""
    from foretec_live.covariates import select_run
    from foretec_live.covariates_history import VARIABLES, previous_runs_day

    d = dt.date(2026, 9, 21)                                   # delivery day; issue 20 Sep, 11:30 CET/CEST
    idx = delivery_index(d - dt.timedelta(days=1), cfg)
    prev = pd.DataFrame({"time_utc": idx.repeat(2), "centroid": [0, 1] * len(idx)})
    for v in VARIABLES:
        prev[f"{v}@1"], prev[f"{v}@2"] = 1.0, 2.0
    for model, run_h in (("icon_eu", 6), ("gfs_seamless", 0)):
        assert select_run(model, d - dt.timedelta(days=1), cfg).hour == run_h
        got = previous_runs_day(model, d, cfg, prev)
        lead = (got["time_utc"] - select_run(model, d - dt.timedelta(days=1), cfg)).dt.total_seconds() / 3600
        assert (got.loc[lead < 36, "wind_speed_100m"] == 1).all() and (got.loc[lead > 36, "wind_speed_100m"] == 2).all()
    assert set(got.columns) == {"time_utc", "centroid", *VARIABLES}


def test_weather_cache_per_zone(cfg, tmp_path):
    """A run is cached per zone: adding a zone fetches only that zone; the result equals one combined request."""
    from foretec_live.covariates import cached_per_zone

    c = pd.DataFrame({"zone": ["BE", "BE", "NL", "FR", "FR"], "bucket": "solar", "cluster_id": [0, 1, 0, 0, 1],
                      "latitude": [50.0, 51.0, 52.0, 45.0, 46.0], "longitude": [4.0, 5.0, 5.0, 2.0, 3.0]})
    calls = []

    def fetch(cm):
        calls.append(sorted(set(cm["zone"])))
        return pd.DataFrame({"centroid": range(len(cm)), "v": cm["latitude"].to_numpy()})

    path = lambda cz: tmp_path / f"{cz['zone'].iloc[0]}.parquet"
    first = cached_per_zone(c[c.zone != "FR"].reset_index(drop=True), path, fetch)
    both = cached_per_zone(c, path, fetch)
    assert calls == [["BE", "NL"], ["FR"]]                         # the second call fetched FR only
    assert list(both.sort_values("centroid")["v"]) == list(c["latitude"])
    assert list(first.sort_values("centroid")["v"]) == [50.0, 51.0, 52.0]


def test_inactive_zones_and_coarse_scoring(tmp_path, monkeypatch):
    """Zones with active: false are skipped unless FORETEC_ALL_ZONES is set; a series scored at 1 h compares
    hourly means, so 15-minute detail its hourly actuals do not have is not penalised."""
    from foretec_live.score import metrics

    text = (REPO / "config.yaml").read_text().replace(
        "zones:\n", 'zones:\n  XX: {entsoe: "10Y-TEST", active: false}\n', 1)
    (tmp_path / "config.yaml").write_text(text)
    monkeypatch.delenv("FORETEC_ALL_ZONES", raising=False)
    zones = load_config(tmp_path / "config.yaml")["zones"]
    assert "XX" not in zones and {"DE", "DK1", "DK2"} <= set(zones)
    monkeypatch.setenv("FORETEC_ALL_ZONES", "1")
    assert "XX" in load_config(tmp_path / "config.yaml")["zones"]
    idx = pd.date_range("2026-09-20 22:00", periods=96, freq="15min")
    shape = np.tile([-1.0, 1.0, -1.0, 1.0], 24) * 50                 # intra-hour wiggle around the hourly level
    y = pd.Series(np.repeat(np.arange(24.0) * 10, 4), index=idx)      # hourly actuals, stepwise
    fc = pd.DataFrame({"delivery_utc": idx, "point": y.to_numpy() + shape})
    assert metrics(fc, y)["mae"] == 50 and metrics(fc, y, "1h")["mae"] == 0


def test_solar_night_zeros_filled_daytime_gaps_kept():
    """ENTSO-E A75 without solar's night zeros: nights become 0, a daytime outage stays missing."""
    from foretec_live.data.entsoe import fill_night_zeros

    idx = pd.date_range("2026-03-01", periods=96 * 2, freq="15min")
    hour = idx.hour + idx.minute / 60
    solar = pd.Series(np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) ** 4 * 1000, index=idx)   # slow ramp at dawn
    published = solar[solar > 0]                                           # night zeros left out
    published = published.drop(published.index[(published.index.day == 2) & (published.index.hour == 12)])  # outage
    out = fill_night_zeros(published, idx[0], idx[-1] + pd.Timedelta("15min"))
    night = (hour < 6) | (hour > 18)
    assert (out.reindex(idx)[night] == 0).all()
    assert out.reindex(idx)[(idx.day == 2) & (idx.hour == 12)].isna().all()


def test_live_runs_never_train(cfg):
    """Models are trained ahead (nightly catch-up); a live run without one reports it and trains nothing."""
    d = dt.date(2026, 9, 20)
    meta = run_forecasts(d, cfg, only_models=["lin_lags"])              # live: nothing trained ahead
    runs = [r for r in meta["runs"] if r["model"] == "lin_lags"]
    assert runs and all(r["status"] != "ok" for r in runs)
    assert "no trained model" in runs[0].get("detail", "")
    assert not (Path(cfg["_root"]) / "data" / "models" / "lin_lags").exists()


def test_ensemble_median_of_members(tmp_path):
    """Ensembles: median per quarter-hour of point and each quantile, sorted; one member short is tolerated, two not."""
    from foretec_live.registry import ModelSpec
    from foretec_live.run import run_ensemble
    idx = pd.date_range("2026-10-20", periods=4, freq="15min")
    qs = [0.1, 0.5, 0.9]

    def member(name, level, series=(("BE", "wind"), ("NL", "solar"))):
        rows = [pd.DataFrame({"issue_date": "2026-10-19", "model": name, "zone": z, "target": t, "delivery_utc": idx,
                              "point": level, "q0.1": level - 1, "q0.5": level, "q0.9": level + 1}) for z, t in series]
        pd.concat(rows).to_parquet(tmp_path / f"{name}.parquet", index=False)

    member("a", 1.0); member("b", 2.0); member("c", 10.0); member("d", 3.0, series=(("BE", "wind"),))
    spec = ModelSpec(name="ens", folder=tmp_path, raw={"runner": "ensemble", "members": ["a", "b", "c", "d"]})
    rows, out = run_ensemble(spec, tmp_path, idx, qs)
    st = {(r["zone"], r["target"]): r for r in rows}
    be = out[(out.zone == "BE") & (out.target == "wind")]
    assert np.allclose(be["point"], 2.5) and np.allclose(be["q0.9"], 3.5)        # median of 1, 2, 3, 10
    nl = out[(out.zone == "NL") & (out.target == "solar")]
    assert np.allclose(nl["point"], 2.0) and st[("NL", "solar")]["detail"] == "missing d"   # 3 of 4: still combined
    member("c", 10.0, series=(("BE", "wind"),))                                    # NL solar now has 2 of 4
    rows, out = run_ensemble(spec, tmp_path, idx, qs)
    assert {(r["zone"], r["target"]): r["status"] for r in rows}[("NL", "solar")] == "error"
    assert set(zip(out.zone, out.target)) == {("BE", "wind")}
