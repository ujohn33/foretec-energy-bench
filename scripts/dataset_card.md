---
license: cc-by-4.0
pretty_name: Foretec Energy Bench
language:
- en
task_categories:
- time-series-forecasting
tags:
- energy
- electricity
- power-markets
- day-ahead
- wind
- solar
- load-forecasting
- price-forecasting
- benchmark
- foundation-models
- weather
size_categories:
- 1M<n<10M
configs:
- config_name: scores
  data_files:
  - split: live
    path: scores/live.parquet
  - split: backtest
    path: scores/backtest.parquet
- config_name: forecasts
  data_files:
  - split: live
    path: forecasts/live.parquet
  - split: backtest
    path: forecasts/backtest.parquet
- config_name: actuals
  data_files:
  - split: all
    path: actuals/actuals.parquet
---

# Foretec Energy Bench

An open, live benchmark for day-ahead electricity forecasting. Every day at 11:30 CET/CEST, before the 12:00
SDAC day-ahead auction gate, every model forecasts each quarter-hour of the next delivery day for **price, wind,
solar and load** in **{zones}**, and is scored against the grid operators' published actuals.

- **Live leaderboard:** https://energybench.foretec.co
- **Methodology:** https://energybench.foretec.co/methodology.html
- **Code:** https://github.com/ujohn33/foretec-energy-bench

This dataset holds every forecast, the actuals they were scored against, and every score: issue days {first} to
{last} ({n_live} live issue days; the earlier days are a backtest run the same way), {n_models} models,
{n_scores} score rows, {n_fc_live} live and {n_fc_bt} backtest forecast rows (one per delivery quarter-hour).

## How the forecasts were made

- **Gate-aligned by construction.** Inputs are fetched at the 11:30 cut-off and frozen: the history of each
  series, the numerical weather prediction runs actually available at that time (ICON-EU, GFS, ECMWF IFS, at
  capacity-weighted points per zone), the ENTSO-E day-ahead load forecast and fuel prices.
- **One information set for every model.** The benchmark runs every model itself, on the same frozen inputs:
  time-series foundation models (Chronos-2, TimesFM, TiRex, Moirai, Toto, Sundial, Tabby, t0 and others, with
  and without the weather covariates), LightGBM and ridge models, naive baselines, and the grid operators' own
  forecasts (`tso_cutoff`, published before the gate; `tso_final`, published after it, a reference only).
- **Live versus backtest.** Live forecasts were made before the gate on the issue day. Backtest forecasts were
  made later for past issue days with the same code and the archived inputs; they are kept apart.

## Files

| Config | File | Contents |
|---|---|---|
| `scores` | `scores/live.parquet`, `scores/backtest.parquet` | per issue day, model, zone and target: `mae`, `rmse`, `bias`, `pinball`, `wis`, `winkler`, `cover`, the baseline's MAE (`mae_base`), `rel_mae`, quarter-hours scored (`n`) |
| `forecasts` | `forecasts/live.parquet`, `forecasts/backtest.parquet` | `issue_date`, `model`, `zone`, `target`, `delivery_utc`, `point`, quantiles `q0.1` … `q0.9` |
| `actuals` | `actuals/actuals.parquet` | `delivery_date`, `zone`, `target`, `delivery_utc`, `value` |
| | `models.csv` | every model: author, family, inputs, description |

Units: price in EUR/MWh; wind, solar and load in MW. Times are UTC; the delivery day is the local (CET/CEST) day
after the issue day. `zone` uses `DE` for the DE-LU bidding zone.

## Metrics

- **Rel. MAE**: a model's MAE divided by that of `{baseline}` (the same quarter-hour one week earlier) on the same
  day and series. On the leaderboard, errors are first pooled per series over the period (sum of absolute errors
  over sum of the baseline's), then combined across series by the mean, median or geometric mean.
- **Elo**: one race per issue day and series; every pair of models in a race is a game won by the lower error;
  a Bradley–Terry fit over all games gives the rating on the Elo scale, with a bootstrap over issue days for a
  95% interval.

## Sources and licence

Actuals and the load forecast come from the ENTSO-E Transparency Platform, with gaps filled from Elia Open Data,
RTE éCO2mix, NED.nl and Energy-Charts; weather forecasts from Open-Meteo. The dataset is released under
**CC BY 4.0**; please cite it as below and credit the original data sources. Some models' weights carry their own
licences (for example Tabby, CC BY-NC 4.0); check them before reusing those models.

## Citation

```bibtex
@misc{{genov2026energybench,
  author       = {{Genov, Evgenii}},
  title        = {{Foretec Energy Bench: a live, gate-aligned benchmark of day-ahead forecasts for European power markets}},
  year         = {{2026}},
  howpublished = {{\url{{https://energybench.foretec.co}}}}
}}
```
