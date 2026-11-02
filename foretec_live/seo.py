"""What search engines and AI agents read: the site rendered without JavaScript, written at every site build.

The pages draw their tables and charts in the browser from data/summary.json, so a crawler that does not run
JavaScript (most AI agents do not) would see an empty benchmark. This module adds, from the same scores:

  index.html etc.   a plain-HTML summary of the live standings (in <details>), the Open Graph preview image and
                    schema.org JSON-LD (the benchmark as a Dataset, for Google Dataset Search)
  leaderboard.csv   the standings per phase as a flat table
  llms.txt          a Markdown overview for AI agents (llmstxt.org): rules, metrics, current standings, data links
  models/           one static page per model and an index of them, so each model has its own URL to link and to find
  robots.txt        every crawler allowed, AI crawlers named explicitly; sitemap.xml with each page's last update

The standings are computed as the Leaderboard computes them by default (all zones and targets, every live issue
day, Rel. MAE races, Bradley-Terry Elo, mean across series), so the static summary and the page agree.
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SITE_URL = "https://energybench.foretec.co"
LICENSE = ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
REPO = "https://github.com/ujohn33/foretec-energy-bench"
HF_DATASET = "https://huggingface.co/datasets/ujohn33/foretec-energy-bench"
DATASET_DOI = "10.57967/hf/11052"
MIN_RACES = 24                       # as app.js: fewer races (or under half the most) is provisional
PAGES = {
    "index.html": ("", "Leaderboard"),
    "explorer.html": ("explorer.html", "Forecast Explorer"),
    "value.html": ("value.html", "Forecast Value"),
    "methodology.html": ("methodology.html", "Methodology"),
    "compete.html": ("compete.html", "Bidding Competition"),
}
TARGET_LABEL = {"price": "Day-ahead price", "wind": "Wind", "solar": "Solar", "load": "Load"}
INPUT_LABEL = {"history": "the series' own history", "nwp": "weather forecasts from three numerical weather prediction models",
               "load_forecast": "the ENTSO-E load forecast", "fuel": "fuel prices", "calendar": "calendar features"}
# the names people search for; a *_cov variant is the same model with covariates
MODEL_LABEL = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt", "timesfm3": "TimesFM 3", "timesfm25": "TimesFM 2.5", "timestral": "Timestral",
               "t0_alpha": "t0-alpha", "t0_beta": "t0-beta", "t0_gamma": "t0-gamma", "tirex": "TiRex", "tirex2": "TiRex-2", "moirai2": "Moirai 2",
               "tabby": "Tabby", "tabby_pretrain": "Tabby (pretrained)", "toto": "Toto", "toto2": "Toto 2", "toto2_1b": "Toto 2 (1B)", "sundial": "Sundial",
               "timeh2o": "TimeH2O", "ttm_r2": "TinyTimeMixer", "autoarima": "AutoARIMA", "autoets": "AutoETS", "autotheta": "AutoTheta", "autoces": "AutoCES",
               "tbats": "TBATS", "lgbm_prod": "LightGBM (tuned)", "lgbm_cov": "LightGBM with weather", "lgbm_lags": "LightGBM on lags",
               "naive_daily": "Naive daily", "naive_weekly": "Naive weekly", "climatology": "Climatology", "ens_fm": "Foundation model ensemble",
               "ens_fm_cov": "Foundation model ensemble with weather", "ens_hybrid": "Hybrid ensemble", "tso_cutoff": "Grid operators' forecast at the cut-off",
               "tso_final": "Grid operators' final forecast"}
DISPLAY = {"tso_cutoff": "TSO_cutoff", "tso_final": "TSO_final"}   # as app.js
AI_CRAWLERS = ["GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "Claude-SearchBot", "Claude-User", "anthropic-ai",
               "PerplexityBot", "Perplexity-User", "Google-Extended", "Applebot-Extended", "CCBot", "Bingbot"]


def _bt(p_wins: np.ndarray, games: np.ndarray) -> np.ndarray:
    """Bradley-Terry strengths by the MM algorithm, as app.js btFit: one virtual game (half a win) against an
    average opponent keeps every strength finite; geometric mean 1, so the average model rates 1000."""
    k = len(p_wins)
    wins = 0.5 + p_wins.sum(axis=1)
    p = np.ones(k)
    for _ in range(300):
        den = 1 / (p + 1) + (games / (p[:, None] + p[None, :])).sum(axis=1)
        q = wins / den
        q /= math.exp(np.log(q).mean())
        delta = np.abs(np.log(q / p)).max()
        p = q
        if delta < 1e-7:
            break
    return p


def standings(s: pd.DataFrame, cfg, refs: set[str], phase: str = "live") -> pd.DataFrame:
    """One row per model: Elo, races, provisional, Rel. MAE / Rel. RMSE pooled per series against the baseline on
    the same days and combined across series by mean, median and geometric mean."""
    s = s[(s["phase"] == phase) & s["rel_mae"].notna()]
    if s.empty:
        return pd.DataFrame()
    race = s[~s["model"].isin(refs)]
    models = sorted(race["model"].unique())
    ix = {m: i for i, m in enumerate(models)}
    k = len(models)
    W, N = np.zeros((k, k)), np.zeros((k, k))
    races = dict.fromkeys(models, 0)
    for _, g in race.groupby(["issue_date", "zone", "target"]):
        if len(g) < 2:
            continue
        v = dict(zip(g["model"], g["rel_mae"]))
        for m in v:
            races[m] += 1
        ms = list(v)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                i, j = ix[ms[a]], ix[ms[b]]
                w = 0.5 if v[ms[a]] == v[ms[b]] else float(v[ms[a]] < v[ms[b]])
                W[i, j] += w; W[j, i] += 1 - w; N[i, j] += 1; N[j, i] += 1
    elo = 1000 + 400 * np.log10(_bt(W, N)) if k >= 2 else np.full(k, 1000.0)
    need = min(MIN_RACES, math.ceil(0.5 * max(races.values(), default=0)))

    base = s[s["model"] == cfg["baseline"]].set_index(["issue_date", "zone", "target"])
    rows = []
    for m, g in s.groupby("model"):
        g = g.join(base[["mae", "rmse"]], on=["issue_date", "zone", "target"], rsuffix="_b").dropna(subset=["mae_b"])
        n = g["n"].fillna(96) if "n" in g else pd.Series(96, index=g.index)
        g = g.assign(w=n, ae=n * g["mae"], se=n * g["rmse"] ** 2, bae=n * g["mae_b"], bse=n * g["rmse_b"] ** 2)
        per = g.groupby(["zone", "target"])[["ae", "se", "bae", "bse"]].sum()
        rel = (per["ae"] / per["bae"]).replace([np.inf], np.nan).dropna()
        relr = np.sqrt(per["se"] / per["bse"]).replace([np.inf], np.nan).dropna()
        gm = lambda x: float(np.exp(np.log(x[x > 0]).mean())) if (x > 0).any() else None   # noqa: E731
        rows.append({"model": m, "reference": m in refs, "elo": None if m not in ix else round(float(elo[ix[m]]), 1),
                     "races": races.get(m, 0), "provisional": m in ix and races[m] < need,
                     "rel_mae": float(rel.mean()) if len(rel) else None, "rel_mae_median": float(rel.median()) if len(rel) else None,
                     "rel_mae_gmean": gm(rel), "rel_rmse": float(relr.mean()) if len(relr) else None, "series": int(len(per)),
                     "first_day": g["issue_date"].min(), "last_day": g["issue_date"].max()})
    out = pd.DataFrame(rows)
    out["order"] = list(zip(out["reference"], out["provisional"], -pd.to_numeric(out["elo"]).fillna(-1e9)))
    out = out.sort_values("order").drop(columns="order").reset_index(drop=True)
    ranked = ~out["reference"] & ~out["provisional"]
    out["rank"] = None
    out.loc[ranked, "rank"] = range(1, int(ranked.sum()) + 1)
    return out


def _best_by_target(s: pd.DataFrame, cfg, refs: set[str], phase: str = "live", top: int = 3) -> dict:
    """Per target: the models with the lowest Rel. MAE (mean over the zones) among those in every zone of it."""
    out = {}
    for t in cfg["targets"]:
        st = standings(s[s["target"] == t], cfg, refs, phase)
        if st.empty:
            continue
        full = st[(st["series"] == st["series"].max()) & ~st["reference"] & st["rel_mae"].notna() & (st["model"] != cfg["baseline"])]
        out[t] = full.nsmallest(top, "rel_mae")[["model", "rel_mae"]].to_dict("records")
    return out


def _f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def _snapshot(st: pd.DataFrame, best: dict, s: pd.DataFrame, cfg, generated: str) -> str:
    live = s[s["phase"] == "live"]
    d0, d1, nd = live["issue_date"].min(), live["issue_date"].max(), live["issue_date"].nunique()
    zones = ", ".join(v.get("label", z) for z, v in cfg["zones"].items())
    ranked = st[st["rank"].notna()].head(15)
    rows = "".join(
        f"<tr><td>{int(r['rank'])}</td><td><a href=\"models/{html.escape(r['model'])}/\">{html.escape(r['model'])}</a></td><td>{r['elo']:.0f}</td><td>{_f(r['rel_mae'])}</td>"
        f"<td>{r['series']}</td><td>{r['races']}</td></tr>" for _, r in ranked.iterrows())
    per_target = "".join(
        f"<li>{TARGET_LABEL.get(t, t)}: " + ", ".join(f"{html.escape(x['model'])} ({x['rel_mae']:.3f})" for x in lst) + "</li>"
        for t, lst in best.items())
    return f"""
<section id="snapshot" class="wrap snapshot">
<details>
<summary>Plain-text summary of the live standings (updated {html.escape(generated)})</summary>
<p>EnergyBench by Foretec is an open, live benchmark for day-ahead electricity forecasting. Every day at
{html.escape(cfg['issue_time'])} {html.escape(cfg['timezone'])} time, before the {html.escape(cfg.get('gate', '12:00'))} day-ahead
auction gate, every model forecasts each quarter-hour of the next delivery day for price, wind, solar and load in
{html.escape(zones)} ({len(cfg['zones']) * len(cfg['targets'])} series), from one frozen set of public inputs
(the series' history, three numerical weather prediction models, the ENTSO-E load forecast, fuel prices).
Forecasts are scored against the grid operators' published actuals.</p>
<p>Live standings over {nd} issue days ({d0} to {d1}), all zones and targets. Elo: a Bradley–Terry rating on
the Elo scale from head-to-head games, one race per issue day and series, won by the lower error.
Rel. MAE: the model's absolute errors pooled per series and divided by those of {html.escape(cfg['baseline'])}
(the same quarter-hour one week earlier) on the same days, then averaged across series; below 1 beats it.</p>
<table><thead><tr><th>#</th><th>Model</th><th>Elo</th><th>Rel. MAE</th><th>Series</th><th>Races</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>Lowest Rel. MAE per target (live, models forecasting every zone):</p><ul>{per_target}</ul>
<p>Data: <a href="data/summary.json">every score (JSON)</a>, <a href="leaderboard.csv">standings (CSV)</a>,
<a href="llms.txt">overview for AI agents</a>, <a href="methodology.html">methodology</a>.
Licence: <a href="{LICENSE[1]}" rel="license">{LICENSE[0]}</a>.</p>
</details>
</section>
"""


def _head(page: str, title: str, cfg, jsonld: list[dict] | None) -> str:
    path, _ = PAGES[page]
    tags = [
        f'<meta property="og:url" content="{SITE_URL}/{path}">',
        f'<meta property="og:image" content="{SITE_URL}/og-card.png?v=2">',
        '<meta property="og:image:width" content="2400">', '<meta property="og:image:height" content="1260">',
        '<meta property="og:image:alt" content="EnergyBench by Foretec: the live day-ahead forecasting benchmark">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<link rel="license" href="{LICENSE[1]}">',
        '<link rel="alternate" type="text/markdown" href="llms.txt" title="Overview for AI agents">',
    ]
    for obj in jsonld or []:
        tags.append('<script type="application/ld+json">' + json.dumps(obj, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    return "\n".join(tags) + "\n"


def _dataset(cfg, s: pd.DataFrame, generated: str) -> list[dict]:
    zones = [v.get("label", z) for z, v in cfg["zones"].items()]
    org = {"@type": "Organization", "name": "Foretec", "url": "https://foretec.be/"}
    return [
        {"@context": "https://schema.org", "@type": "WebSite", "name": "EnergyBench", "alternateName": ["Energy Bench", "Foretec Energy Bench"], "url": SITE_URL + "/", "publisher": org},
        {"@context": "https://schema.org", "@type": "Dataset",
         "name": "EnergyBench: live day-ahead electricity forecasting benchmark", "alternateName": ["Energy Bench", "Foretec Energy Bench"],
         "description": ("Daily day-ahead forecasts of electricity price, wind, solar and load for " + ", ".join(zones) +
                         ", made by foundation models, gradient boosting and baselines before the 12:00 day-ahead auction gate "
                         "from one frozen set of public inputs (history, NWP weather forecasts, load forecast, fuel prices), "
                         "with their scores against the grid operators' actuals: MAE, RMSE, bias, pinball loss, Rel. MAE "
                         "against a seasonal naive baseline, and a Bradley-Terry Elo rating."),
         "url": SITE_URL + "/", "sameAs": [REPO, HF_DATASET], "identifier": f"https://doi.org/{DATASET_DOI}", "license": LICENSE[1], "isAccessibleForFree": True,
         "creator": org, "publisher": org, "dateModified": generated,
         "temporalCoverage": f"{s['issue_date'].min()}/..",
         "spatialCoverage": [{"@type": "Place", "name": f"{z} bidding zone"} for z in zones],
         "keywords": ["energy forecasting", "electricity price forecasting", "wind power forecasting", "solar power forecasting",
                      "load forecasting", "time series foundation models", "benchmark", "day-ahead market", "ENTSO-E"],
         "variableMeasured": ["day-ahead electricity price (EUR/MWh)", "wind generation (MW)", "solar generation (MW)", "total load (MW)"],
         "measurementTechnique": "Forecasts locked before the day-ahead gate, scored against published actuals",
         "distribution": [
             {"@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": SITE_URL + "/data/summary.json"},
             {"@type": "DataDownload", "encodingFormat": "text/csv", "contentUrl": SITE_URL + "/leaderboard.csv"}]},
    ]


def _llms(st: pd.DataFrame, bt: pd.DataFrame, best: dict, s: pd.DataFrame, cfg, generated: str) -> str:
    live = s[s["phase"] == "live"]
    zones = ", ".join(v.get("label", z) for z, v in cfg["zones"].items())
    lines = ["# EnergyBench by Foretec", "",
             "> An open, live benchmark for day-ahead electricity forecasting. Every day at "
             f"{cfg['issue_time']} ({cfg['timezone']}), before the {cfg.get('gate', '12:00')} day-ahead auction gate, every model "
             "forecasts each quarter-hour of the next delivery day for price, wind, solar and load in "
             f"{zones}, from one frozen set of public inputs, and is scored against the grid operators' actuals.", "",
             f"Updated {generated}. Data licence: {LICENSE[0]} ({LICENSE[1]}). Code: {REPO}", "",
             "## Rules", "",
             f"- Gate-aligned by construction: inputs are fetched at the {cfg['issue_time']} cut-off and frozen; the weather is the "
             "numerical weather prediction run actually available then (ICON-EU, GFS, ECMWF IFS), plus the ENTSO-E load forecast and fuel prices.",
             "- The benchmark runs every model itself, so every model sees exactly the same inputs; entries are code, not forecasts.",
             "- The grid operators' own forecasts are included: tso_cutoff (published before the gate, ranked) and tso_final "
             "(published after the gate, a reference that is not ranked).",
             "- Backtest days are scored the same way but kept apart from live days.", "",
             "## Metrics", "",
             f"- Rel. MAE: a model's absolute errors pooled per series (zone and target) and divided by those of {cfg['baseline']} "
             "(the same quarter-hour one week earlier) on the same days. Below 1 beats it; 0.5 halves its error. Rel. RMSE likewise with squared errors.",
             "- Across series, Rel. MAE is combined by the mean (default), median or geometric mean. The mean is moved by one very good "
             "or very bad series, the median rewards winning most series, the geometric mean treats halving and doubling the error alike.",
             "- Elo: one race per issue day and series; every pair of models in a race is a game won by the lower error. A Bradley-Terry "
             "fit on all games gives the rating on the Elo scale (1000 = the field's average, +400 = 10:1 odds); a bootstrap over issue days gives a 95% interval.",
             "- Also reported: MAE, RMSE, bias, pinball loss, weighted interval score, interval coverage, and the revenue value of forecasts.", "",
             f"## Live standings ({live['issue_date'].nunique()} issue days, {live['issue_date'].min()} to {live['issue_date'].max()}, all series)", "",
             "| # | Model | Elo | Rel. MAE (mean) | Rel. MAE (median) | Series | Races |", "|---|---|---|---|---|---|---|"]
    for _, r in st[st["rank"].notna()].head(20).iterrows():
        lines.append(f"| {int(r['rank'])} | {r['model']} | {r['elo']:.0f} | {_f(r['rel_mae'])} | {_f(r['rel_mae_median'])} | {r['series']} | {r['races']} |")
    lines += ["", "Lowest live Rel. MAE per target (models forecasting every zone):", ""]
    for t, lst in best.items():
        lines.append(f"- {TARGET_LABEL.get(t, t)}: " + ", ".join(f"{x['model']} ({x['rel_mae']:.3f})" for x in lst))
    if not bt.empty:
        lines += ["", "Backtest leaders (Elo): " + ", ".join(f"{r['model']} ({r['elo']:.0f})" for _, r in bt[bt["rank"].notna()].head(5).iterrows())]
    lines += ["", "## Pages", "",
              f"- [Leaderboard]({SITE_URL}/): standings by Elo, Rel. MAE and other metrics, filterable by zone, target and period",
              f"- [Forecast Explorer]({SITE_URL}/explorer.html): every model's forecast against the actual, per issue day and series, with the frozen inputs",
              f"- [Forecast Value]({SITE_URL}/value.html): revenue of a virtual 100 MW wind or solar asset selling each forecast day-ahead",
              f"- [Models]({SITE_URL}/models/): one page per model with its ranks, its score per zone and target, and its inputs",
              f"- [Methodology]({SITE_URL}/methodology.html): task, data sources, scoring and ranking in full", "",
              "## Data", "",
              f"- [Standings (CSV)]({SITE_URL}/leaderboard.csv): live and backtest, one row per model",
              f"- [All scores (JSON)]({SITE_URL}/data/summary.json): every score row (issue day, model, zone, target, metrics), models, config",
              f"- Per issue day: {SITE_URL}/data/days/YYYY-MM-DD.json (actuals and every forecast)",
              f"- [Full dataset on Hugging Face]({HF_DATASET}): every forecast, actual and score as parquet; DOI {DATASET_DOI}", ""]
    return "\n".join(lines)


def _label(m: str) -> str:
    base = m[:-4] if m.endswith("_cov") and m not in MODEL_LABEL else m
    label = MODEL_LABEL.get(base, DISPLAY.get(m, m))
    return f"{label} with weather inputs" if base != m and base in MODEL_LABEL else label


def _series_scores(s: pd.DataFrame, cfg, ranked: set[str]) -> pd.DataFrame:
    """Live Rel. MAE per model and series, pooled as in standings(), with the place among the ranked models."""
    s = s[(s["phase"] == "live") & s["rel_mae"].notna()]
    keys = ["issue_date", "zone", "target"]
    g = s.join(s[s["model"] == cfg["baseline"]].set_index(keys)["mae"].rename("mae_b"), on=keys).dropna(subset=["mae_b"])
    n = g["n"].fillna(96)
    per = g.assign(ae=n * g["mae"], bae=n * g["mae_b"]).groupby(["model", "zone", "target"])[["ae", "bae"]].sum()
    per = per[per["bae"] > 0].assign(rel=lambda x: x["ae"] / x["bae"]).reset_index()
    per["place"] = per[per["model"].isin(ranked)].groupby(["zone", "target"])["rel"].rank(method="min")
    return per


def _up(fragment: str, pre: str) -> str:
    """Header and footer of the root pages, with their relative links rewritten for a page deeper in the site."""
    return re.sub(r'href="(?![a-z]+:|#)(?:\./)?([^"]*)"', lambda x: f'href="{pre}{x[1]}"', fragment)


def _page(pre: str, path: str, title: str, desc: str, body: str, crumbs: list[tuple[str, str]], chrome: tuple[str, str]) -> str:
    e = html.escape
    url = f"{SITE_URL}/{path}"
    trail = {"@context": "https://schema.org", "@type": "BreadcrumbList",
             "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n, "item": f"{SITE_URL}/{u}"} for i, (n, u) in enumerate(crumbs)]}
    fonts = ("https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400..700;1,400&family=Source+Code+Pro:wght@400"
             "&family=Inter:ital,wght@0,600..800;1,500..900&display=swap")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(desc)}">
<meta name="theme-color" content="#E6F0F2" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0B1F3A" media="(prefers-color-scheme: dark)">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="EnergyBench by Foretec">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{SITE_URL}/og-card.png?v=2">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{pre}favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{fonts}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@125,800&text=LIVE&display=swap">
<link rel="stylesheet" href="{pre}style.css">
<script type="application/ld+json">{json.dumps(trail, ensure_ascii=False).replace("</", "<\\/")}</script>
</head>
<body>
{chrome[0]}
<main class="wrap mp">
<nav class="mp-crumbs" aria-label="Breadcrumb">{" / ".join(f'<a href="{pre}{e(u)}">{e(n)}</a>' for n, u in crumbs[:-1])} / <span>{e(crumbs[-1][0])}</span></nav>
{body}
</main>
{chrome[1]}
</body>
</html>
"""


def _link(m: str, pre: str = "../") -> str:
    return f'<a href="{pre}{html.escape(m)}/">{html.escape(DISPLAY.get(m, m))}</a>'


def _model_body(info: dict, r, st: pd.DataFrame, by_target: dict, per: pd.DataFrame, bt: pd.DataFrame, cfg, field: int) -> tuple[str, str]:
    """The page body and its meta description, in words drawn from this model's own scores."""
    e, m = html.escape, info["name"]
    label, rel = _label(m), r["rel_mae"]
    days = f"{r['first_day']} to {r['last_day']}"
    if pd.notna(r["rank"]):
        head = f"{DISPLAY.get(m, m)} ranks #{int(r['rank'])} of {field} on the EnergyBench live leaderboard"
    elif info["ensemble"]:
        head = f"{m} is an ensemble of models in the field, so it stays out of the default ranking"
    elif info["reference"]:
        head = f"{DISPLAY.get(m, m)} is a reference published after the gate, so it is not ranked"
    else:
        head = f"{m} has {r['races']} live races so far, too few for a rank"
    gain = " It is the baseline every other model is measured against." if m == cfg["baseline"] else \
        f" Its error is {rel:.0%} of a forecast that repeats the same quarter-hour one week earlier." if rel < 1 else \
        " It does not beat a forecast that repeats the same quarter-hour one week earlier."
    lede = f"{head}, with a Rel. MAE of {rel:.3f} over {r['series']} series ({days}).{gain}"
    rows = [("Rank", f"#{int(r['rank'])} of {field}" if pd.notna(r["rank"]) else "Not ranked"),
            ("Elo", f"{r['elo']:.0f}" if pd.notna(r["elo"]) else "–"), ("Rel. MAE", f"{rel:.3f}"), ("Races", str(r["races"]))]
    stats = "".join(f"<div><b>{e(v)}</b><span>{e(k)}</span></div>" for k, v in rows)

    mine = per[per["model"] == m]
    tgts = "".join(f"<tr><th>{TARGET_LABEL.get(t, t)}</th><td>{'#%d of %d' % (x['rank'], x['of']) if x['rank'] else 'Not ranked'}</td><td>{x['rel_mae']:.3f}</td><td>{x['series']}</td></tr>"
                   for t, x in by_target.items())
    zones = [z for z in cfg["zones"] if (mine["zone"] == z).any()]
    tg = [t for t in cfg["targets"] if (mine["target"] == t).any()]
    cell = {(x.zone, x.target): (x.rel, x.place) for x in mine.itertuples()}

    def td(z, t):
        if (z, t) not in cell:
            return "<td>–</td>"
        v, place = cell[(z, t)]
        return f"<td>{v:.3f}" + (f" <small>#{int(place)}</small>" if pd.notna(place) else "") + "</td>"
    grid = "".join(f"<tr><th>{e(cfg['zones'][z].get('label', z))}</th>" + "".join(td(z, t) for t in tg) + "</tr>" for z in zones)
    best = mine.nsmallest(1, "rel").iloc[0] if len(mine) else None
    worst = mine.nlargest(1, "rel").iloc[0] if len(mine) else None
    note = "" if best is None or len(mine) < 2 else (
        f"<p>Closest to the actuals on {e(TARGET_LABEL.get(best['target'], best['target']).lower())} in {e(cfg['zones'][best['zone']].get('label', best['zone']))} "
        f"({best['rel']:.3f}); furthest on {e(TARGET_LABEL.get(worst['target'], worst['target']).lower())} in {e(cfg['zones'][worst['zone']].get('label', worst['zone']))} ({worst['rel']:.3f}).</p>")

    facts = [("Family", info["family"]), ("Inputs", "; ".join(INPUT_LABEL.get(i, i) for i in info["inputs"]))]
    if info["members"]:
        facts.append(("Members", None))
    if info["retired"]:
        facts.append(("Status", f"Retired: {info['retired']}"))
    elif info["relegated"]:
        facts.append(("Status", f"Out of the live run since {info['relegated']}: {info['relegated_reason']}"))
    if info["caveat"]:
        facts.append(("Caveat", info["caveat"]))
    about = "".join(f"<tr><th>{e(k)}</th><td>{e(v) if v is not None else ', '.join(_link(x) for x in info['members'])}</td></tr>" for k, v in facts)

    order = st[st["rank"].notna()].reset_index(drop=True)
    near = order.index[order["model"] == m]
    around = order.iloc[max(near[0] - 2, 0):near[0] + 3] if len(near) else order.head(0)
    around = "".join(f"<tr><td>{int(x['rank'])}</td><td>{_link(x['model'])}</td><td>{x['elo']:.0f}</td><td>{x['rel_mae']:.3f}</td></tr>" for _, x in around.iterrows() if x["model"] != m)
    kin = [x for x in info["kin"] if x != m]
    related = ('<section class="mp-sec"><h2>Near it on the leaderboard</h2><div class="panel tablewrap"><table><thead><tr><th>#</th><th>Model</th><th>Elo</th><th>Rel. MAE</th></tr></thead>'
               f"<tbody>{around}</tbody></table></div></section>") if around else ""
    if kin:
        related += f'<section class="mp-sec"><h2>Same family</h2><p>{", ".join(_link(x) for x in kin)}</p></section>'
    back = bt[bt["model"] == m]
    backtest = "" if back.empty or pd.isna(back.iloc[0]["rel_mae"]) else (
        f"<p>On the backtest days, kept apart from the live record, its Rel. MAE is {back.iloc[0]['rel_mae']:.3f}"
        + (f" and it ranks #{int(back.iloc[0]['rank'])}" if pd.notna(back.iloc[0]["rank"]) else "") + ".</p>")
    run = ("The grid operators publish this forecast themselves; the benchmark collects it and scores it like any other." if info["family"] == "tso"
           else "This model no longer runs live; its record stays public." if info["retired"] or info["relegated"]
           else f"The model runs every day at {cfg['issue_time']}, before the {cfg.get('gate', '12:00')} day-ahead auction gate, on the same frozen inputs as the rest of the field.")
    body = f"""<h1>{e(label)}</h1>
<p class="lede">{e(lede)}</p>
<div class="mp-stats">{stats}</div>
<section class="mp-sec"><h2>About this model</h2>
<p>{e(info['description'])}</p>
<div class="panel tablewrap"><table class="mini"><tbody>{about}</tbody></table></div></section>
<section class="mp-sec"><h2>Accuracy by target</h2>
<div class="panel tablewrap"><table><thead><tr><th>Target</th><th>Rank</th><th>Rel. MAE</th><th>Series</th></tr></thead><tbody>{tgts}</tbody></table></div></section>
<section class="mp-sec"><h2>Accuracy by zone and target</h2>
<p>Live Rel. MAE, with the place among the ranked models in that series. Below 1 beats the same quarter-hour one week earlier.</p>
<div class="panel tablewrap"><table><thead><tr><th>Zone</th>{"".join(f"<th>{TARGET_LABEL.get(t, t)}</th>" for t in tg)}</tr></thead><tbody>{grid}</tbody></table></div>
{note}{backtest}</section>
{related}
<section class="mp-sec"><h2>Check it yourself</h2>
<p>{e(run)}
<a href="../../?m={e(m)}">See it on the leaderboard</a>, <a href="../../explorer.html">compare its forecasts with the actuals</a>,
read <a href="../../methodology.html">how scoring works</a>, or open <a href="{REPO}/tree/main/models/{e(m)}" rel="noopener">its configuration</a>.</p></section>"""
    bits = [head + (f": Rel. MAE {rel:.3f} over {r['series']} series" if pd.notna(rel) else ""), "Day-ahead price, wind, solar and load forecasts scored live"]
    return body, ". ".join(bits) + "."


def _models_index(st: pd.DataFrame, cfg, info: dict, field: int) -> str:
    def row(x, cells):
        return f"<tr>{cells}<td>{_link(x['model'], '')}</td><td>{html.escape(_label(x['model']))}</td><td>{_f(x['rel_mae'])}</td><td>{x['series']}</td></tr>"
    ranked = "".join(row(x, f"<td>{int(x['rank'])}</td><td>{x['elo']:.0f}</td>") for _, x in st[st["rank"].notna()].iterrows())
    rest = st[st["rank"].isna() & st["model"].isin(info)]
    others = "".join(row(x, "") for _, x in rest.iterrows())
    return f"""<h1>Models in the benchmark</h1>
<p class="lede">{field} ranked forecasting models, from time-series foundation models to gradient boosting, statistical baselines and the grid operators' own forecasts,
all scored live on day-ahead electricity price, wind, solar and load in {len(cfg['zones'])} European bidding zones. Each has its own page with its scores per zone and target.</p>
<section class="mp-sec"><h2>Ranked on live days</h2>
<div class="panel tablewrap"><table><thead><tr><th>#</th><th>Elo</th><th>Model</th><th>Name</th><th>Rel. MAE</th><th>Series</th></tr></thead><tbody>{ranked}</tbody></table></div></section>
<section class="mp-sec"><h2>Outside the ranking</h2>
<p>Ensembles, references published after the gate, and models with too few live races so far.</p>
<div class="panel tablewrap"><table><thead><tr><th>Model</th><th>Name</th><th>Rel. MAE</th><th>Series</th></tr></thead><tbody>{others}</tbody></table></div></section>"""


def write_models(site: Path, s: pd.DataFrame, cfg, models: list[dict], unranked: set[str], st: pd.DataFrame, bt: pd.DataFrame, atomic_write) -> list[str]:
    """models/ and models/<name>/: the page text comes from the same standings as the leaderboard. Returns the paths, for the sitemap."""
    root = (site / "index.html").read_text() if (site / "index.html").exists() else ""
    header = re.search(r'<header class="top">.*?</header>', root, re.S)
    footer = re.search(r"<footer>.*?</footer>", root, re.S)
    chrome = tuple(_up(x[0], "../../") if x else "" for x in (header, footer))
    chrome_i = tuple(_up(x[0], "../") if x else "" for x in (header, footer))
    field = int(st["rank"].notna().sum())
    ranked = set(st.loc[st["rank"].notna(), "model"])
    per = _series_scores(s, cfg, ranked)
    tables = {t: standings(s[s["target"] == t], cfg, unranked, "live") for t in cfg["targets"]}
    live = set(st.loc[st["rel_mae"].notna(), "model"])
    info = {m["name"]: m for m in models if m["name"] in live}
    family = {}
    for m in info.values():
        family.setdefault(m["family"], []).append(m["name"])
    out = site / "models"
    out.mkdir(exist_ok=True)
    for old in out.iterdir():      # a model hidden or removed since the last build loses its page
        if old.is_dir() and old.name not in info:
            shutil.rmtree(old)
    paths = ["models/"]
    for m, i in info.items():
        r = st[st["model"] == m].iloc[0]
        by_target = {}
        for t, x in tables.items():
            row = x[x["model"] == m] if not x.empty else x
            if len(row) and pd.notna(row.iloc[0]["rel_mae"]):
                y = row.iloc[0]
                by_target[t] = {"rank": y["rank"] if pd.notna(y["rank"]) else None, "of": int(x["rank"].notna().sum()), "rel_mae": y["rel_mae"], "series": y["series"]}
        body, desc = _model_body({**i, "kin": family[i["family"]] if i["family"] != "baseline" else []}, r, st, by_target, per, bt, cfg, field)
        title = f"{_label(m)}: day-ahead energy forecast accuracy · EnergyBench"
        (out / m).mkdir(exist_ok=True)
        atomic_write(out / m / "index.html", _page("../../", f"models/{m}/", title, desc, body, [("Leaderboard", ""), ("Models", "models/"), (DISPLAY.get(m, m), f"models/{m}/")], chrome))
        paths.append(f"models/{m}/")
    desc = "Every forecasting model in EnergyBench with its live rank, Elo and Rel. MAE on day-ahead electricity price, wind, solar and load."
    atomic_write(out / "index.html", _page("../", "models/", "Energy forecasting models compared · EnergyBench", desc, _models_index(st, cfg, info, field),
                                           [("Leaderboard", ""), ("Models", "models/")], chrome_i))
    return paths


def write_seo(site: Path, s: pd.DataFrame, cfg, refs: set[str], generated: str, atomic_write, models: list[dict] | None = None) -> None:
    refs = refs | {m["name"] for m in models or [] if m["ensemble"]}   # as the Leaderboard's default: the ensembles leave the races
    st = standings(s, cfg, refs, "live")
    bt = standings(s, cfg, refs, "backtest")
    best = _best_by_target(s, cfg, refs, "live")
    cols = ["phase", "rank", "model", "elo", "provisional", "reference", "rel_mae", "rel_mae_median", "rel_mae_gmean",
            "rel_rmse", "series", "races", "first_day", "last_day"]
    csv = pd.concat([x.assign(phase=ph) for ph, x in (("live", st), ("backtest", bt)) if not x.empty], ignore_index=True)
    atomic_write(site / "leaderboard.csv", csv[cols].round(4).to_csv(index=False))
    atomic_write(site / "llms.txt", _llms(st, bt, best, s, cfg, generated))
    # /api/ is the waiting-list form's endpoint (POST only), not a page: no crawler needs it
    atomic_write(site / "robots.txt", "User-agent: *\nAllow: /\nDisallow: /api/\n\n" + "".join(f"User-agent: {b}\nAllow: /\nDisallow: /api/\n\n" for b in AI_CRAWLERS)
                 + f"Sitemap: {SITE_URL}/sitemap.xml\n")
    day = generated[:10]
    extra = []
    if models and not st.empty:
        try:      # one bad model must not cost the rest of the export
            extra = write_models(site, s, cfg, models, refs, st, bt, atomic_write)
        except Exception:  # noqa: BLE001
            log.exception("model pages failed")
    urls = [f"{SITE_URL}/{p}" for p, _ in PAGES.values()] + [f"{SITE_URL}/{p}" for p in extra] + [f"{SITE_URL}/llms.txt"]
    atomic_write(site / "sitemap.xml", '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                 + "".join(f"  <url><loc>{u}</loc><lastmod>{day}</lastmod><changefreq>daily</changefreq></url>\n" for u in urls) + "</urlset>\n")
    snap = _snapshot(st, best, s, cfg, generated) if not st.empty else ""
    for page, (_, title) in PAGES.items():
        f = site / page
        if not f.exists():
            continue
        text = f.read_text()
        text = text.replace("</head>", _head(page, title, cfg, _dataset(cfg, s, generated) if page == "index.html" else None) + "</head>", 1)
        if page == "index.html" and snap:
            text = text.replace("</main>", snap + "</main>", 1)
        atomic_write(f, text)
