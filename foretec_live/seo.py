"""What search engines and AI agents read: the site rendered without JavaScript, written at every site build.

The pages draw their tables and charts in the browser from data/summary.json, so a crawler that does not run
JavaScript (most AI agents do not) would see an empty benchmark. This module adds, from the same scores:

  index.html etc.   the Open Graph preview image and schema.org JSON-LD (the benchmark as a Dataset, for Google Dataset Search)
  leaderboard.csv   the standings per phase as a flat table
  llms.txt          a Markdown overview for AI agents (llmstxt.org): rules, metrics, current standings, data links
  models/           one static page per model and an index of them, so each model has its own URL to link and to find
  robots.txt        every crawler allowed, AI crawlers named explicitly; sitemap.xml with each page's last update

The standings are computed as the Leaderboard computes them by default (all zones and targets, every live issue
day, Rel. MAE races, Bradley-Terry Elo, mean across series), so llms.txt, the model pages and the page agree.
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
_PLUG = '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0V8z"/><path d="M12 18v4"/>'
TARGET_ICON = {"price": '<path d="M18 7a7 7 0 1 0 0 10"/><path d="M4 10h9M4 14h9"/>',
               "wind": '<path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 12h16a3 3 0 1 1-3 3"/><path d="M3 16h7a2.5 2.5 0 1 1-2.5 2.5"/>',
               "solar": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19"/>',
               "load": _PLUG}
INPUTS = {"history": ('<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>', "History"),
          "nwp": ('<circle cx="8" cy="8" r="3"/><path d="M8 2v1.5M2 8h1.5M3.8 3.8l1 1M12.2 3.8l-1 1"/><path d="M8 20h9a4 4 0 0 0 .6-7.95A6 6 0 0 0 6.2 13.2 3.5 3.5 0 0 0 8 20z"/>', "Weather", "ICON-EU · GFS · IFS"),
          "load_forecast": (_PLUG, "Load forecast"),
          "fuel": ('<path d="M12 22c4 0 7-3 7-7 0-3-2-5-3-7-1 2-2 3-3 3 1-3 0-6-3-9 0 4-5 7-5 13 0 4 3 7 7 7z"/>', "Fuel prices"),
          "calendar": ('<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>', "Calendar")}
# maker, released, parameters, licence, source: only what the model's card, paper or announcement states; a *_cov variant shares its base model's
MODEL_FACTS = {"chronos2": ("Amazon", "Oct 2025", "120M", "Apache-2.0", "https://arxiv.org/abs/2510.15821"),
               "chronos_bolt": ("Amazon", None, "205M", None, None),
               "timesfm3": ("Google", "Aug 2026", None, "Non-commercial weights", "https://huggingface.co/google/timesfm-3.0-pytorch"),
               "timesfm25": ("Google", "Sep 2025", "200M", None, "https://huggingface.co/google/timesfm-2.5-200m-pytorch"),
               "t0_alpha": ("The Forecasting Company", "Jun 2026", "102M", "Apache-2.0", "https://arxiv.org/abs/2609.24559"),
               "t0_beta": ("The Forecasting Company", "2026", "256M", None, "https://arxiv.org/abs/2609.24559"),
               "t0_gamma": ("The Forecasting Company", "Oct 2026", "256M", None, "https://pypi.org/project/tfc-t0/"),
               "timestral": ("Mistral AI", "Oct 2026", "209M", "Modified MIT", "https://huggingface.co/mistralai/Timestral-209M"),
               "tirex": ("NX-AI", "May 2025", "35M", None, "https://arxiv.org/abs/2505.23719"),
               "tirex2": ("NX-AI", "2026", "38M", "Apache-2.0", None),
               "moirai2": ("Salesforce", "Nov 2025", None, "CC BY-NC", "https://arxiv.org/abs/2511.11698"),
               "toto": ("Datadog", "May 2025", "151M", "Apache-2.0", None),
               "toto2": ("Datadog", "2026", "313M", "Apache-2.0", "https://huggingface.co/Datadog/Toto-2.0-313m"),
               "toto2_1b": ("Datadog", "2026", "1B", "Apache-2.0", "https://huggingface.co/Datadog/Toto-2.0-1B"),
               "sundial": ("Tsinghua University", None, "128M", "Apache-2.0", None),
               "tabby": ("Huawei Noah's Ark Lab", "Sep 2026", "147M", "CC BY-NC 4.0", "https://huggingface.co/paris-noah/Tabby"),
               "tabby_pretrain": ("Huawei Noah's Ark Lab", "Sep 2026", "146M", "CC BY-NC 4.0", "https://huggingface.co/paris-noah/Tabby-Pretrain"),
               "timeh2o": ("H2O.ai", "Oct 2026", "108M", "Non-commercial", "https://huggingface.co/h2oai/timeh2o-v1"),
               "ttm_r2": ("IBM", "2024", None, "Apache-2.0", "https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2")}
# the names people search for; a *_cov variant is the same model with covariates
MODEL_LABEL = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt", "timesfm3": "TimesFM 3", "timesfm25": "TimesFM 2.5", "timestral": "Timestral",
               "t0_alpha": "t0-alpha", "t0_beta": "t0-beta", "t0_gamma": "t0-gamma", "tirex": "TiRex", "tirex2": "TiRex-2", "moirai2": "Moirai 2",
               "tabby": "Tabby", "tabby_pretrain": "Tabby (pretrained)", "toto": "Toto", "toto2": "Toto 2", "toto2_1b": "Toto 2 (1B)", "sundial": "Sundial",
               "timeh2o": "TimeH2O", "ttm_r2": "TinyTimeMixer", "autoarima": "AutoARIMA", "autoets": "AutoETS", "autotheta": "AutoTheta", "autoces": "AutoCES",
               "tbats": "TBATS", "lgbm_prod": "LightGBM (tuned)", "lgbm_cov": "LightGBM with weather", "lgbm_lags": "LightGBM on lags",
               "lgbm_cov_2y": "LightGBM with weather (2 years)", "lin_cov": "Linear regression with weather", "lin_cov_2y": "Linear regression with weather (2 years)",
               "lin_lags": "Linear regression on lags", "theta_daily": "Theta", "naive_daily": "Naive daily", "naive_weekly": "Naive weekly", "climatology": "Climatology", "ens_fm": "Foundation model ensemble",
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


def _when(d: str) -> str:
    x = pd.Timestamp(d)
    return f"{x.day} {x.strftime('%b %Y')}"


def _icon(paths: str) -> str:
    return f'<svg class="mp-ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false">{paths}</svg>'


def _kind(i: dict) -> str:
    m, fam = i["name"], i["family"]
    if fam in ("ensemble", "tso"):
        return {"ensemble": "Ensemble", "tso": "Grid operator forecast"}[fam]
    if m.startswith("lgbm"):
        return "Gradient boosting"
    if m.startswith("lin_"):
        return "Linear regression"
    if fam == "baseline":
        return "Statistical model" if m.startswith(("auto", "tbats", "theta")) else "Baseline"
    return "Foundation model"


def _facts(m: str) -> dict:
    """What is known about the model itself. Unknown fields stay out: never a guess in the infobox."""
    base = m[:-4] if m.endswith("_cov") else m
    return dict(zip(("maker", "released", "size", "licence", "source"), MODEL_FACTS.get(base, ())))


def _trend(days: list[str], me: pd.Series, field: pd.Series) -> str:
    """7-day running Rel. MAE of the model against the field's median, with the baseline at 1."""
    W, H, L, R, T, B = 720, 190, 34, 92, 10, 24
    top = max(1.05, float(np.nanmax(me.values)) * 1.05, float(np.nanmax(field.values)) * 1.05)
    x = lambda i: L + (W - L - R) * i / max(len(days) - 1, 1)   # noqa: E731
    y = lambda v: T + (H - T - B) * (1 - v / top)               # noqa: E731
    line = lambda s: " ".join(f"{x(i):.1f},{y(s.iloc[i]):.1f}" for i in range(len(days)) if pd.notna(s.iloc[i]))   # noqa: E731
    grid = "".join(f'<line class="g" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text class="t" x="{L - 7}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
                   for v in (0, 0.5, 1) if v <= top)
    last = [i for i in range(len(days)) if pd.notna(me.iloc[i])][-1]
    lastf = [i for i in range(len(days)) if pd.notna(field.iloc[i])][-1]
    my, fy = y(me.iloc[last]), y(field.iloc[lastf])
    if abs(my - fy) < 15:      # the end labels never overlap
        my, fy = (my - 8, fy + 8) if my <= fy else (my + 8, fy - 8)
    hit = "".join(f'<circle class="h" cx="{x(i):.1f}" cy="{y(me.iloc[i]):.1f}" r="9"><title>{_when(days[i])}: {me.iloc[i]:.3f}</title></circle>'
                  for i in range(len(days)) if pd.notna(me.iloc[i]))
    return (f'<svg class="mp-trend" viewBox="0 0 {W} {H}" role="img" aria-label="Running error over time against the field">{grid}'
            f'<polyline class="f" points="{line(field)}"/><polyline class="m" points="{line(me)}"/>'
            f'<circle class="d" cx="{x(last):.1f}" cy="{y(me.iloc[last]):.1f}" r="4.5"/>'
            f'<text class="l" x="{x(last) + 10:.1f}" y="{my + 4:.1f}">This model</text><text class="t" x="{x(lastf) + 10:.1f}" y="{fy + 4:.1f}">Field median</text>'
            f'<text class="t" x="{L}" y="{H - 5}">{_when(days[0])}</text><text class="t" x="{W - R}" y="{H - 5}" text-anchor="end">{_when(days[-1])}</text>{hit}</svg>')


def _strip(values: list[float], mine: float, name: str) -> str:
    """Where the model sits in the field for one target: every ranked model a dot, this one marked."""
    W, H, CAP = 420, 34, 1.25
    x = lambda v: 8 + (W - 16) * min(v, CAP) / CAP   # noqa: E731
    dots = "".join(f'<circle class="o" cx="{x(v):.1f}" cy="{H / 2}" r="3.5"/>' for v in values)
    return (f'<svg class="mp-strip" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(name)}: Rel. MAE {mine:.3f} among the field">'
            f'<line class="g" x1="{x(1):.1f}" x2="{x(1):.1f}" y1="3" y2="{H - 3}"/>{dots}'
            f'<circle class="m" cx="{x(mine):.1f}" cy="{H / 2}" r="7.5"><title>{html.escape(name)}: {mine:.3f}</title></circle></svg>')


def _model_body(i: dict, r, st: pd.DataFrame, targets: dict, per: pd.DataFrame, trend: str, cfg, field: int, days_live: int) -> tuple[str, str]:
    """The page: big numbers, charts and icons; the words are in the infobox and the meta description."""
    e, m, rel = html.escape, i["name"], r["rel_mae"]
    facts, kind = _facts(m), _kind(i)
    ranked = pd.notna(r["rank"])
    who = facts.get("maker") or ("Grid operators" if i["family"] == "tso" else "Foretec")
    tag = " · ".join(x for x in (kind, who, facts.get("released")) if x)
    numbers = [(str(days_live), "days run live")]
    if ranked:
        numbers = [(f"#{int(r['rank'])}", f"of {field} ranked models"), (f"{r['elo']:.0f}", "Elo rating"), *numbers]
    big = "".join(f"<div><b>{e(a)}</b><span>{e(b)}</span></div>" for a, b in numbers)

    strips = "".join(
        f'<div class="mp-row">{_icon(TARGET_ICON[t])}<span class="mp-rl">{e(TARGET_LABEL.get(t, t))}</span>{_strip(x["field"], x["rel_mae"], TARGET_LABEL.get(t, t))}'
        f'<b class="mp-rk">{"#%d" % x["rank"] if x["rank"] else format(x["rel_mae"], ".2f")}</b></div>' for t, x in targets.items())

    mine = per[per["model"] == m]
    tg = [t for t in cfg["targets"] if (mine["target"] == t).any()]
    cell = {(x.zone, x.target): x for x in mine.itertuples()}
    heads = "".join(f'<div class="mp-hc">{_icon(TARGET_ICON[t])}<span>{e(TARGET_LABEL.get(t, t).replace("Day-ahead ", "").capitalize())}</span></div>' for t in tg)
    rows = ""
    for z in cfg["zones"]:
        if not (mine["zone"] == z).any():
            continue
        rows += f'<div class="mp-zone">{e(cfg["zones"][z].get("label", z))}</div>'
        for t in tg:
            c = cell.get((z, t))
            if c is None:
                rows += '<div class="mp-cell na">–</div>'
                continue
            n = per[(per["zone"] == z) & (per["target"] == t)]["place"].max()
            v = 0.0 if pd.isna(c.place) or not n or n < 2 else 1 - (c.place - 1) / (n - 1)
            top = f"#{int(c.place)}" if pd.notna(c.place) else "–"
            rows += f'<div class="mp-cell" title="{e(top)}, Rel. MAE {c.rel:.3f}"><span><i style="--s:{8 + 30 * v:.0f}px"></i></span><b>{e(top)}</b></div>'
    heat = f'<div class="mp-heat" style="--c:{len(tg)}"><div></div>{heads}{rows}</div>'

    zone_rank = "" if mine["place"].isna().all() else (f'<section><h2>Rank by zone</h2>{heat}'
                                                       '<p class="mp-key">Bigger is a better place among the ranked models in that series.</p></section>')
    used = set(i["inputs"])
    inputs = "" if i["family"] == "tso" else '<section><h2>What it sees</h2><div class="mp-inputs">' + "".join(
        f'<div class="{"on" if k in used else "off"}" title="{"Uses" if k in used else "Does not use"} {e(v[1].lower())}">{_icon(v[0])}<span>{e(v[1])}</span>'
        + (f"<small>{e(v[2])}</small>" if len(v) > 2 else "") + "</div>" for k, v in INPUTS.items()) + "</div></section>"

    box = [("Type", e(kind)), ("Developer", e(who))]
    box += [(k, e(facts[f])) for k, f in (("Released", "released"), ("Parameters", "size"), ("Licence", "licence")) if facts.get(f)]
    if facts.get("source"):
        host = re.sub(r"^https?://(www\.)?", "", facts["source"]).split("/")[0]
        box.append(("Source", f'<a href="{e(facts["source"])}" rel="noopener">{e(host)}</a>'))
    box += [("Live since", e(_when(r["first_day"]))), ("Series", str(r["series"]))]
    if i["members"]:
        box.append(("Members", ", ".join(_link(x) for x in i["members"])))
    kin = [x for x in i["kin"] if x != m][:6]
    if kin:
        box.append(("Family", ", ".join(_link(x) for x in kin)))
    if i["retired"] or i["relegated"]:
        box.append(("Status", "Retired" if i["retired"] else f"Out of the live run since {e(_when(i['relegated']))}"))
    infobox = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in box)

    order = st[st["rank"].notna()].reset_index(drop=True)
    at = order.index[order["model"] == m]
    near = order.iloc[max(at[0] - 1, 0):at[0] + 2] if len(at) else order.head(0)
    near = "".join(f'<a href="../{e(x["model"])}/"><b>#{int(x["rank"])}</b> {e(DISPLAY.get(x["model"], x["model"]))}</a>' for _, x in near.iterrows() if x["model"] != m)
    body = f"""<div class="mp-top">
<div class="mp-main">
<p class="mp-tag">{e(tag)}</p>
<h1>{e(_label(m))}</h1>
<p class="mp-id">{e(DISPLAY.get(m, m))}</p>
<div class="mp-big">{big}</div>
<section><h2>Error over time</h2>{trend}<p class="mp-key">Seven-day running error against repeating last week, where 1 is no better than that.</p></section>
<section><h2>In the field</h2><div class="mp-strips">{strips}</div>
<p class="mp-key">Each dot is a ranked model; further left is better. The line marks repeating last week.</p></section>
{zone_rank}
{inputs}
</div>
<aside class="mp-box"><dl>{infobox}</dl>
<p class="mp-links"><a href="../../?z=all&amp;t=all&amp;r=all&amp;c=all&amp;e=0&amp;a=mean&amp;m={e(m)}">On the leaderboard</a><a href="../../explorer.html">Forecast Explorer</a><a href="{REPO}/tree/main/models/{e(m)}" rel="noopener">Configuration</a></p></aside>
</div>
{f'<nav class="mp-near" aria-label="Nearby models">{near}</nav>' if near else ''}"""
    first = re.split(r"(?<=[.])\s", i["description"], maxsplit=1)[0][:150]
    lead = f"#{int(r['rank'])} of {field} on EnergyBench" if ranked else "On EnergyBench"
    return body, f"{_label(m)}: {lead}, Rel. MAE {rel:.3f} on live day-ahead energy forecasts. {first}"


def _models_index(st: pd.DataFrame, cfg, info: dict, field: int) -> str:
    def row(x):
        i, rel = info[x["model"]], x["rel_mae"]
        return (f'<a class="mp-li" href="{html.escape(x["model"])}/"><b>{int(x["rank"]) if pd.notna(x["rank"]) else "–"}</b><span class="n">{html.escape(_label(x["model"]))}</span>'
                f'<span class="k">{html.escape(_kind(i))}</span><span class="b"><i style="width:{min(rel, 1.25) / 1.25:.1%}"></i></span><span class="v">{_f(rel, 2)}</span></a>')
    rows = st[st["model"].isin(info)]
    return f"""<h1>Models</h1>
<div class="mp-big"><div><b>{field}</b><span>ranked models</span></div><div><b>{len(cfg['targets'])}</b><span>targets</span></div><div><b>{len(cfg['zones'])}</b><span>bidding zones</span></div></div>
<p class="mp-key">Bars show error against repeating last week: shorter is better.</p>
<div class="mp-list">{"".join(row(x) for _, x in rows[rows["rank"].notna()].iterrows())}</div>
<section><h2>Outside the ranking</h2><div class="mp-list">{"".join(row(x) for _, x in rows[rows["rank"].isna()].iterrows())}</div></section>"""


def write_models(site: Path, s: pd.DataFrame, cfg, models: list[dict], unranked: set[str], st: pd.DataFrame, atomic_write) -> list[str]:
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
    live = s[(s["phase"] == "live") & s["rel_mae"].notna()]
    daily = live.pivot_table(index="issue_date", columns="model", values="rel_mae", aggfunc="mean").sort_index()
    running = daily.rolling(7, min_periods=3).mean()
    median = running[[c for c in running if c in ranked]].median(axis=1)
    day_list = [str(d) for d in running.index]
    info = {m["name"]: m for m in models if m["name"] in set(st.loc[st["rel_mae"].notna(), "model"])}
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
        targets = {}
        for t, x in tables.items():
            row = x[x["model"] == m] if not x.empty else x
            if len(row) and pd.notna(row.iloc[0]["rel_mae"]):
                y = row.iloc[0]
                targets[t] = {"rank": y["rank"] if pd.notna(y["rank"]) else None, "rel_mae": y["rel_mae"],
                              "field": x.loc[x["rank"].notna(), "rel_mae"].dropna().tolist()}
        trend = _trend(day_list, running[m], median) if running[m].notna().sum() >= 2 else ""
        body, desc = _model_body({**i, "kin": family[i["family"]] if i["family"] not in ("baseline", "nwp") else []}, r, st, targets, per, trend or "", cfg, field,
                                 int(daily[m].notna().sum()))
        title = f"{_label(m)}: day-ahead energy forecast accuracy · EnergyBench"
        (out / m).mkdir(exist_ok=True)
        atomic_write(out / m / "index.html", _page("../../", f"models/{m}/", title, desc, body, [("Leaderboard", ""), ("Models", "models/"), (DISPLAY.get(m, m), f"models/{m}/")], chrome))
        paths.append(f"models/{m}/")
    desc = "Every forecasting model in EnergyBench with its live rank and error on day-ahead electricity price, wind, solar and load."
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
            extra = write_models(site, s, cfg, models, refs, st, atomic_write)
        except Exception:  # noqa: BLE001
            log.exception("model pages failed")
    urls = [f"{SITE_URL}/{p}" for p, _ in PAGES.values()] + [f"{SITE_URL}/{p}" for p in extra] + [f"{SITE_URL}/llms.txt"]
    atomic_write(site / "sitemap.xml", '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                 + "".join(f"  <url><loc>{u}</loc><lastmod>{day}</lastmod><changefreq>daily</changefreq></url>\n" for u in urls) + "</urlset>\n")
    for page, (_, title) in PAGES.items():
        f = site / page
        if not f.exists():
            continue
        text = f.read_text()
        text = text.replace("</head>", _head(page, title, cfg, _dataset(cfg, s, generated) if page == "index.html" else None) + "</head>", 1)
        atomic_write(f, text)
