# Foretec Energy Bench

The live benchmark for energy forecasts: an open, daily benchmark of forecasting models on European power markets, built on [sktime](https://www.sktime.net) 1.2. Live at **https://energybench.foretec.co**. (The Python package and command keep their working names, `foretec_live` and `foretec-live`.)

Every day at 11:30 Brussels time, before the 12:00 SDAC day-ahead gate closure, each model in `models/` forecasts the next delivery day for:

| | Belgium | Netherlands | France |
|---|---|---|---|
| Day-ahead price (EUR/MWh) | BE | NL | FR |
| Wind generation (MW, on + offshore) | BE | NL | FR |
| Solar generation (MW) | BE | NL | FR |

Forecasts are on the 15-minute day-ahead grid: 96 quarter-hours, or 92/100 on clock-change days. Once the actuals are final they are scored, and the leaderboard is rebuilt.

This proof of concept scores accuracy only. Trading and money metrics come later.

## How a day works

| When (Brussels) | What |
|---|---|
| D-1 11:30 | **forecast**: snapshot the data that was public at the 11:30 cut-off, run every model, write `results/forecasts/D-1/*.parquet` |
| D-1 12:00 | **gate**: SDAC day-ahead gate closure. A live run that finishes later is marked late and never scored |
| D-1 14:45, 20:45 | **score + report**: the day-ahead auction for D has cleared (results around 13:00), so prices are scored |
| D+1 14:45 | wind and solar for D are scored (metered day complete on ENTSO-E); until D+3 every scoring run re-fetches them, logs the change in `data/actuals/D/_revisions.json` and re-scores a series that moved by more than 0.5% of its daily energy |

**No look-ahead.** The data available to a model is cut off as follows:

- **Prices:** everything up to the start of the delivery day. Prices for D-1 were published on D-2.
- **Wind and solar:** actuals up to issue time minus one hour of publication lag.
- **TSO day-ahead renewable forecasts:** not used. They are only published by 18:00 on D-1, after issue time.

## Scoring

These metrics are computed per model, day, zone and target:

- MAE, RMSE and bias.
- Pinball loss over the quantiles 0.1–0.9, only for models that produce intervals.
- **rel_mae:** MAE divided by the MAE of the `naive_weekly` baseline on the same day and series. A score below 1 beats "same quarter-hour last week".

The leaderboard is the mean `rel_mae` over all 9 series across the last 28 scored days, alongside coverage (the share of days the model actually ran).

Data comes from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) (`ENTSOE_TOKEN`; day-ahead prices A44, actual generation per type A75), parsed from the XML. Quarter-hours ENTSO-E does not deliver are filled from the TSOs' own open data, [Elia Open Data](https://opendata.elia.be) for BE and [RTE éCO2mix](https://odre.opendatasoft.com) for FR (no keys), then from [Energy-Charts](https://energy-charts.info) by Fraunhofer ISE (CC BY 4.0) as a last resort. Energy-Charts republishes ENTSO-E: over 7 Sep to 2 Oct 2026 the two agree to rounding wherever both have a value, while ENTSO-E missed 12% of BE wind and 2.6% of FR wind. Which source delivered each point is logged per series in `_sources.json` next to every snapshot and every set of actuals.

## Adding a model

Open a pull request that adds a folder `models/<your_model>/` containing a `model.yaml`. There are two ways to define the model.

**1. Any sktime estimator, by name**

```yaml
name: my_ets
author: Your Name
description: One line on what it does.
estimator: AutoETS            # any name in the sktime registry, or a dotted import path
params: {sp: 96, auto: true}
targets: [price]              # optional, default all
zones: [BE, NL]               # optional, default all
```

**2. Your own code**

```yaml
name: my_model
author: Your Name
description: One line.
module: forecaster.py
```

Then `forecaster.py` exposes a function that returns an sktime forecaster:

```python
def build(target: str, zone: str):
    ...
    return forecaster   # fit(y, fh) then predict(fh); predict_quantiles is used if supported
```

Each run calls `fit` on 28 days of 15-minute history (`y`, UTC timestamps) and then `predict` for the next delivery day.

**3. A separate environment (foundation models and anything with heavy dependencies)**

```yaml
name: my_fm
author: Your Name
description: One line.
family: my_fm          # models in one family share a colour on the leaderboard
runner: subprocess
env: my_env            # runs with /srv/foretec/envs/my_env/bin/python
script: forecaster.py
timeout: 1800          # seconds, optional
```

The runner calls the script once per issue day with all series, so weights load once. Use `models/_shared/fl_io.py` (on the path automatically):

```python
from fl_io import max_horizon, read_request, write_output

req, series = read_request()          # {"BE_price": np.ndarray, ...}: gap-free 15-min history, oldest first
results = {}
for key, y in series.items():
    h = req["horizon"][key]           # steps from the end of this history to the end of the delivery day
    results[key] = (point_h, quantiles_h_by_9)   # quantiles for req["quantiles"], or None
write_output(req, results)
```

Add the env to `scripts/install_model_envs.sh`. The foundation models in this repo (Chronos-2, Chronos-Bolt, TimesFM 2.5 and 3, Moirai 2, TiRex, Toto, Sundial, TTM r2) all use this route.

- **Dependencies:** a model whose soft dependencies or env are not installed is skipped and logged, not failed.
- **Quantiles:** models that return quantiles (`predict_quantiles`, or the subprocess `quantiles` array) are also scored on pinball loss.
- **Covariates:** add `inputs: [history, nwp, load_forecast, fuel, calendar]` to `model.yaml` and the request gets a `covariates` parquet frozen at the cut-off (see below and `models/_shared/fl_cov.py`).

## Covariates

`foretec_live/covariates.py` builds, per issue day, `data/covariates/<issue_date>.parquet` plus a JSON log of exactly what was used:

- **NWP ensemble:** ICON-EU (DWD), GFS (NOAA) and IFS HRES (ECMWF) from the Open-Meteo single-runs archive. A run counts only if `init <= cut-off - lag_hours` (config), i.e. it was published before the cut-off: at 11:30 that is ICON-EU 06 UTC and GFS/IFS 00 UTC.
- **Where:** capacity-weighted k-means centroids per zone and bucket (`geo/centroids.csv`, built by `geo/build_centroids.py`, sources in `geo/README.md`). Zone values are capacity-weighted means; across the models we add the ensemble mean, std, min, max, range, coefficient of variation and pairwise differences (the predico recipe).
- **Same vintage for training:** the values for each past delivery day come from the runs that were available at that day's own cut-off, so models train on forecasts with the lead time they forecast with.
- **ENTSO-E day-ahead load forecast** (A65; TSOs publish it 2 h before the gate). Backtests use the currently published version.
- **Fuel cost:** TTF and EUA from oilpriceapi, a quote counting only from the end of its date, plus a CCGT short-run cost.

API keys come from the environment (`OPENMETEOKEY`, `ENTSOE_TOKEN`, `OIL_PRICE_KEY`); the systemd units read them from `/root/.env`.

## Running it

```bash
pip install -e .                         # foundation models: scripts/install_model_envs.sh
foretec-live probe                       # check the data source: points, coverage, step, ranges
foretec-live forecast                    # today's issue (or --date YYYY-MM-DD, --models a b)
foretec-live score                       # score everything that has final actuals
foretec-live report                      # results/leaderboard.csv
foretec-live site                        # rebuild the website in site/
foretec-live backfill --start 2026-09-14 --end 2026-10-01   # into results/backtest
foretec-live catchup                     # backtest every model missing from the config backtest window
foretec-live --source synthetic forecast # offline, deterministic fake data
pytest                                   # end-to-end test on synthetic data
```

Set `FORETEC_HOME` to run from outside the repo folder.

## Server

On Ubuntu (tested on 26.04, Python 3.14) a script installs everything under `/srv/foretec` and adds systemd timers: forecast 11:30, score 14:45 and 20:45, catch-up 03:30 (Brussels). See the header of `scripts/setup_server.sh`.

```bash
sudo bash scripts/setup_server.sh [--foundation] [--web]
```

To deploy changes, pull on the server and re-run the script; it keeps `data/` and `results/`:

```bash
cd /root/foretec-live && git pull && sudo bash scripts/setup_server.sh --web   # add --foundation when model envs change
```

The first time, pass the domain for HTTPS and any extra names that should redirect to it (`--domain=energybench.foretec.co --alias=transparency.foretec.co`); both are remembered, so later `--web` runs keep HTTPS and the plain-IP address redirects too. A name is only configured once its DNS points at this server, so add the DNS record first and re-run `--web` afterwards; until the main domain resolves, the site is served on the aliases that do.

```bash
sudo bash scripts/setup_server.sh --domain=energybench.foretec.co --alias=transparency.foretec.co
```

API keys (`ENTSOE_TOKEN`, `OPENMETEOKEY`, `OIL_PRICE_KEY`) live in `/root/.env` on the server, never in the repo.

## Layout

```
config.yaml            zones, targets, issue time, quantiles, window
foretec_live/          data sources, snapshot + run, covariates, scoring, report, website, CLI
foretec_live/web/      the static website (index.html, app.js, style.css)
models/<name>/         one folder per entry; models/_shared/ holds the subprocess I/O helpers
geo/                   weather centroids and how they were built (raw downloads not in git)
scripts/               server setup, model environments, systemd units
data/                  snapshots and frozen actuals (not in git)
results/               forecasts, scores.parquet, leaderboard.csv (not in git)
site/                  generated website (not in git)
```

Apache-2.0.
