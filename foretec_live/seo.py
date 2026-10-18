"""What search engines and AI agents read: the site rendered without JavaScript, written at every site build.

The pages draw their tables and charts in the browser from data/summary.json, so a crawler that does not run
JavaScript (most AI agents do not) would see an empty benchmark. This module adds, from the same scores:

  index.html etc.   a plain-HTML summary of the live standings (in <details>), the Open Graph preview image and
                    schema.org JSON-LD (the benchmark as a Dataset, for Google Dataset Search)
  leaderboard.csv   the standings per phase as a flat table
  llms.txt          a Markdown overview for AI agents (llmstxt.org): rules, metrics, current standings, data links
  robots.txt        every crawler allowed, AI crawlers named explicitly; sitemap.xml with each page's last update

The standings are computed as the Leaderboard computes them by default (all zones and targets, every live issue
day, Rel. MAE races, Bradley-Terry Elo, mean across series), so the static summary and the page agree.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

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
        f"<tr><td>{int(r['rank'])}</td><td>{html.escape(r['model'])}</td><td>{r['elo']:.0f}</td><td>{_f(r['rel_mae'])}</td>"
        f"<td>{r['series']}</td><td>{r['races']}</td></tr>" for _, r in ranked.iterrows())
    per_target = "".join(
        f"<li>{TARGET_LABEL.get(t, t)}: " + ", ".join(f"{html.escape(x['model'])} ({x['rel_mae']:.3f})" for x in lst) + "</li>"
        for t, lst in best.items())
    return f"""
<section id="snapshot" class="wrap snapshot">
<details>
<summary>Plain-text summary of the live standings (updated {html.escape(generated)})</summary>
<p>Foretec Energy Bench is an open, live benchmark for day-ahead electricity forecasting. Every day at
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
        f'<meta property="og:image" content="{SITE_URL}/og-card.png">',
        '<meta property="og:image:width" content="1200">', '<meta property="og:image:height" content="630">',
        '<meta property="og:image:alt" content="Foretec Energy Bench: the live day-ahead forecasting benchmark">',
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
        {"@context": "https://schema.org", "@type": "WebSite", "name": "Foretec Energy Bench", "url": SITE_URL + "/", "publisher": org},
        {"@context": "https://schema.org", "@type": "Dataset",
         "name": "Foretec Energy Bench: live day-ahead electricity forecasting benchmark",
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
    lines = ["# Foretec Energy Bench", "",
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
              f"- [Methodology]({SITE_URL}/methodology.html): task, data sources, scoring and ranking in full", "",
              "## Data", "",
              f"- [Standings (CSV)]({SITE_URL}/leaderboard.csv): live and backtest, one row per model",
              f"- [All scores (JSON)]({SITE_URL}/data/summary.json): every score row (issue day, model, zone, target, metrics), models, config",
              f"- Per issue day: {SITE_URL}/data/days/YYYY-MM-DD.json (actuals and every forecast)",
              f"- [Full dataset on Hugging Face]({HF_DATASET}): every forecast, actual and score as parquet; DOI {DATASET_DOI}", ""]
    return "\n".join(lines)


def write_seo(site: Path, s: pd.DataFrame, cfg, refs: set[str], generated: str, atomic_write) -> None:
    st = standings(s, cfg, refs, "live")
    bt = standings(s, cfg, refs, "backtest")
    best = _best_by_target(s, cfg, refs, "live")
    cols = ["phase", "rank", "model", "elo", "provisional", "reference", "rel_mae", "rel_mae_median", "rel_mae_gmean",
            "rel_rmse", "series", "races", "first_day", "last_day"]
    csv = pd.concat([x.assign(phase=ph) for ph, x in (("live", st), ("backtest", bt)) if not x.empty], ignore_index=True)
    atomic_write(site / "leaderboard.csv", csv[cols].round(4).to_csv(index=False))
    atomic_write(site / "llms.txt", _llms(st, bt, best, s, cfg, generated))
    atomic_write(site / "robots.txt", "User-agent: *\nAllow: /\n\n" + "".join(f"User-agent: {b}\nAllow: /\n\n" for b in AI_CRAWLERS)
                 + f"Sitemap: {SITE_URL}/sitemap.xml\n")
    day = generated[:10]
    urls = [f"{SITE_URL}/{p}" for p, _ in PAGES.values()] + [f"{SITE_URL}/llms.txt"]
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
