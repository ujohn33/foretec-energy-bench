"""Foundation models against the pre-gate TSO forecast, in order of model release, every model on the same races
(usage, from the benchmark root: SCORES=OUT/scores_basis.parquet python preprint_tso_gap.py OUT). Writes release.dat, tso_gap.json."""
import json, os, sys
import numpy as np, pandas as pd

out = sys.argv[1]
s = pd.read_parquet(os.environ.get("SCORES", "results/scores.parquet")); s["issue_date"] = s.issue_date.astype(str)
s = s[s.mae.notna()]; s["key"] = s.zone + "_" + s.target
K = ["issue_date", "key"]
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))
def ratio(a):
    return a.groupby("key").apply(lambda q: (q.n * q.m).sum() / (q.n * q.t).sum(), include_groups=False)
def boot(a, N=1000, seed=0):
    days = a.issue_date.unique(); rng = np.random.default_rng(seed); ix = {d: g.index.values for d, g in a.groupby("issue_date")}
    nk = a.key.nunique(); b = []
    for _ in range(N):
        y = a.loc[np.concatenate([ix[d] for d in rng.choice(days, len(days))])]
        if y.key.nunique() == nk: b.append(gm(ratio(y)))
    return [round(100 * (gm(ratio(a)) - 1), 2), round(100 * (float(np.percentile(b, 2.5)) - 1), 2), round(100 * (float(np.percentile(b, 97.5)) - 1), 2)]
def pair(m, ref, races):
    x = s[s.model == m][K + ["mae", "n"]].rename(columns={"mae": "m"}).merge(s[s.model == ref][K + ["mae"]].rename(columns={"mae": "t"}), on=K)
    return x.merge(races, on=K).reset_index(drop=True)

# in release order; months from MODEL_FACTS in foretec_live/seo.py (t0-beta and TiRex-2 have none and stay out of the order)
RELEASE = [("timesfm25_cov", "TimesFM 2.5", "Sep 2025"), ("chronos2_cov", "Chronos-2", "Oct 2025"), ("t0_alpha_cov", "t0-alpha", "Jun 2026"),
           ("timesfm3_cov", "TimesFM 3.0", "Aug 2026"), ("tabby_cov", "Tabby", "Sep 2026"), ("t0_gamma_cov", "t0-gamma", "Oct 2026"), ("timestral_cov", "Timestral", "Oct 2026")]
COMMON = [m for m, _, _ in RELEASE] + ["t0_beta_cov", "tirex2_cov", "tabby_pretrain_cov", "lgbm_prod", "chronos2", "timesfm3", "timesfm25", "t0_alpha", "t0_gamma", "timestral"]
races = s[s.model == "tso_cutoff"][K]
for m in COMMON:
    races = races.merge(s[s.model == m][K], on=K)
res = {"races": len(races), "days": races.issue_date.nunique(), "first": races.issue_date.min(), "last": races.issue_date.max(), "per_series": races.groupby("key").size().to_dict(), "models": {}}
ws = races[races.key.str.endswith(("wind", "solar"))]; ld = races[races.key.str.endswith("load")]
for m in [r[0] for r in RELEASE] + ["t0_beta_cov", "tirex2_cov", "lgbm_prod"]:
    res["models"][m] = {"all": boot(pair(m, "tso_cutoff", races)), "wind_solar": boot(pair(m, "tso_cutoff", ws)), "load": boot(pair(m, "tso_cutoff", ld)),
                        "series": (100 * (ratio(pair(m, "tso_cutoff", races)) - 1)).round(1).to_dict()}
res["same_family"] = {"timesfm3_vs_25": boot(pair("timesfm3_cov", "timesfm25_cov", races)), "t0_gamma_vs_alpha": boot(pair("t0_gamma_cov", "t0_alpha_cov", races))}
# the ensembles ran on fewer days: their own races with the TSO forecast
ens = s[s.model == "tso_cutoff"][K].merge(s[s.model == "ens_hybrid"][K], on=K)
e_ws = ens[ens.key.str.endswith(("wind", "solar"))]; e_ld = ens[ens.key.str.endswith("load")]
res["ensembles"] = {"races": len(ens), "days": ens.issue_date.nunique()}
for m in ("ens_hybrid", "ens_fm_cov"):
    res["ensembles"][m] = {"all": boot(pair(m, "tso_cutoff", ens)), "wind_solar": boot(pair(m, "tso_cutoff", e_ws)), "load": boot(pair(m, "tso_cutoff", e_ld))}
# what the weather does on these races: history, covariates, TSO, relative to the baseline
base = s[s.model == "naive_weekly"].set_index(K).mae.rename("b")
res["weather"] = {}
for tg in ("wind", "solar", "load"):
    cm = races[races.key.str.endswith(tg)]; v = {}
    for lab, m in (("history", "timesfm3"), ("covariates", "timesfm3_cov"), ("tso", "tso_cutoff")):
        q = s[s.model == m][K + ["mae", "n"]].merge(cm, on=K).join(base, on=K)
        v[lab] = round(gm(q.groupby("key").apply(lambda x: (x.n * x.mae).sum() / (x.n * x.b).sum(), include_groups=False)), 3)
    res["weather"][tg] = v
json.dump(res, open(f"{out}/tso_gap.json", "w"), indent=1)
# release frontier figure: x in months since Sep 2025 (same-month releases set apart by 0.4), the best released so far,
# ridge-covariate models flagged; models without a known release month go in a separate file, drawn apart
MON = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
months = lambda r: (int(r.split()[1]) - 2025) * 12 + MON[r.split()[0]] - 9
RIDGE = {"timesfm25_cov", "tabby_cov", "tabby_pretrain_cov"}
best, seen = None, {}
with open(f"{out}/release.dat", "w") as o:
    o.write("x gap lo hi best ridge\n")
    for m, _, r in RELEASE:
        x = months(r); k = seen.get(x, 0); seen[x] = k + 1
        g, lo, hi = res["models"][m]["all"]; best = g if best is None else min(best, g)
        o.write(f"{x + (0.4 * k - 0.2 if sum(1 for _, _, q in RELEASE if months(q) == x) > 1 else 0):.1f} {g:.1f} {g - lo:.1f} {hi - g:.1f} {best:.1f} {int(m in RIDGE)}\n")
with open(f"{out}/release_unknown.dat", "w") as o:
    o.write("x gap lo hi ridge\n")
    for i, m in enumerate(("t0_beta_cov", "tirex2_cov")):
        g, lo, hi = res["models"][m]["all"]; o.write(f"{15.6 + 0.8 * i:.1f} {g:.1f} {g - lo:.1f} {hi - g:.1f} 0\n")
print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=None)); [print(m, v["all"], v["wind_solar"], v["load"]) for m, v in res["models"].items()]
