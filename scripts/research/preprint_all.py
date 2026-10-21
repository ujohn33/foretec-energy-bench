"""Preprint numbers and figure data, live record, all six zones (usage: python preprint_all.py OUTDIR LAST_ISSUE_DATE).
Run from the benchmark root. Writes standings.json, covpairs.csv, tso_pairs.json, perseries.csv, coverage.csv."""
import glob, json, os, sys
import numpy as np, pandas as pd, yaml

out, last = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
hidden = set()
for f in glob.glob("models/*/model.yaml"):
    y = yaml.safe_load(open(f)) or {}
    if y.get("hidden"): hidden.add(y.get("name", f.split("/")[1]))
REF = {"tso_final"}
s = pd.read_parquet("results/scores.parquet")
s["issue_date"] = s.issue_date.astype(str)
s = s[(s.issue_date <= last) & ~s.model.isin(hidden)].copy()
s["n"] = s["n"].fillna(96)
K = ["issue_date", "zone", "target"]
base = s[s.model == "naive_weekly"].set_index(K)[["mae", "rmse"]]
gm = lambda a: float(np.exp(np.mean(np.log(np.asarray(a, float))))) if len(a) else None

def per_series(x):
    j = x.join(base, on=K, rsuffix="_b").dropna(subset=["mae", "mae_b"])
    g = j.groupby(["model", "zone", "target"])
    return pd.DataFrame({
        "rel": g.apply(lambda q: (q.n * q.mae).sum() / (q.n * q.mae_b).sum(), include_groups=False),
        "relr": g.apply(lambda q: np.sqrt((q.n * q.rmse ** 2).sum() / (q.n * q.rmse_b ** 2).sum()), include_groups=False),
        "days": g.size()}).reset_index()

def elo(x, boot=200):
    x = x[~x.model.isin(REF) & x.rel_mae.notna()]
    models = sorted(x.model.unique()); k = len(models); ix = {m: i for i, m in enumerate(models)}
    days = sorted(x.issue_date.unique()); per = []
    for d in days:
        W = np.zeros((k, k)); N = np.zeros((k, k))
        for _, r in x[x.issue_date == d].groupby(["zone", "target"]):
            v = r.set_index("model").rel_mae
            idx = [ix[m] for m in v.index]; vals = v.to_numpy()
            for a in range(len(vals)):
                for b in range(len(vals)):
                    if a == b: continue
                    N[idx[a], idx[b]] += 1
                    W[idx[a], idx[b]] += 1.0 if vals[a] < vals[b] else 0.5 if vals[a] == vals[b] else 0.0
        per.append((W, N))
    def fit(ids, start=None):
        W = sum(per[i][0] for i in ids); N = sum(per[i][1] for i in ids)
        p = np.ones(k) if start is None else start.copy(); wins = 0.5 + W.sum(1)
        for _ in range(300):
            q = wins / (1 / (p + 1) + (N / (p[:, None] + p[None, :])).sum(1)); q /= np.exp(np.mean(np.log(q)))
            d = np.max(np.abs(np.log(q / p))); p = q
            if d < 1e-7: break
        return p
    to = lambda p: 1000 + 400 * np.log10(p)
    allp = fit(range(len(days))); rng = np.random.default_rng(42)
    dr = np.array([to(fit(rng.integers(0, len(days), len(days)), allp)) for _ in range(boot)])
    races = x.groupby("model").size()
    return {m: {"elo": float(to(allp[i])), "lo": float(np.percentile(dr[:, i], 2.5)), "hi": float(np.percentile(dr[:, i], 97.5)), "races": int(races[m])} for m, i in ix.items()}

ps = per_series(s)
ps.to_csv(f"{out}/perseries.csv", index=False)
e = elo(s)
st = {}
for m, g in ps.groupby("model"):
    r = {"rel": gm(g.rel), "relr": gm(g.relr), "nser": len(g)}
    for t, gt in g.groupby("target"): r[t] = gm(gt.rel)
    for z, gz in g.groupby("zone"): r["z_" + z] = gm(gz.rel)
    r.update(e.get(m, {}))
    st[m] = r
json.dump(st, open(f"{out}/standings.json", "w"), indent=1)

# coverage: scored series-days per zone and target (baseline)
cov = s[s.model == "naive_weekly"].groupby(["zone", "target"]).issue_date.agg(["count", "min", "max"]).reset_index()
cov.to_csv(f"{out}/coverage.csv", index=False)

# covariate pairs on common races, per zone x target
rows = []
for m in ["chronos2", "timesfm3", "timesfm25", "t0_alpha", "t0_beta", "t0_gamma", "tirex2", "timestral", "tabby", "tabby_pretrain"]:
    a, b = s[s.model == m], s[s.model == m + "_cov"]
    if a.empty or b.empty: continue
    common = a[K].merge(b[K])
    ra = per_series(a.merge(common)).set_index(["zone", "target"]).rel
    rb = per_series(b.merge(common)).set_index(["zone", "target"]).rel
    ratio = rb / ra
    r = {"model": m, "races": len(common), "days": common.issue_date.nunique(), "rel": gm(ra), "rel_cov": gm(rb), "all": gm(ratio) - 1}
    for t in ["price", "wind", "solar", "load"]:
        q = ratio[ratio.index.get_level_values("target") == t]
        r[t] = gm(q) - 1 if len(q) else None
        r["n_" + t] = len(q)
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{out}/covpairs.csv", index=False)

# TSO references on the races they cover
tp = {}
for ref in ["tso_cutoff", "tso_final"]:
    rr = s[s.model == ref][K]
    tp[ref] = {"series": sorted(set(rr.zone + "_" + rr.target)), "races": len(rr), "models": {}}
    for m in [ref, "chronos2_cov", "timesfm3_cov", "t0_beta_cov", "t0_alpha_cov", "tirex2_cov", "lgbm_cov", "lgbm_prod", "ens_hybrid", "ens_fm_cov"]:
        x = s[s.model == m].merge(rr)
        if x.empty: continue
        p = per_series(x)
        tp[ref]["models"][m] = {"races": len(x), **{t: gm(g.rel) for t, g in p.groupby("target")}}
json.dump(tp, open(f"{out}/tso_pairs.json", "w"), indent=1)
print("done", len(st), "models")
