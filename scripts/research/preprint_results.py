"""Preprint tables: pooled per-series Rel. MAE / Rel. RMSE (as on the site) and the Bradley-Terry Elo with a
day-bootstrap 95% interval, for the backtest and the live record. Run from /srv/foretec/foretec-live."""
import json, sys
import numpy as np, pandas as pd, yaml, glob

hidden = set()
for f in glob.glob("models/*/model.yaml"):
    y = yaml.safe_load(open(f)) or {}
    if y.get("hidden"): hidden.add(y.get("name", f.split("/")[1]))
REF = {"tso_final"}
live = pd.read_parquet("results/scores.parquet").assign(phase="live")
bt = pd.read_parquet("results/backtest/scores.parquet").assign(phase="backtest")
for d in (live, bt): d["issue_date"] = d["issue_date"].astype(str)
bt = bt[~bt.issue_date.isin(set(live.issue_date))]
gm = lambda a: float(np.exp(np.mean(np.log(a)))) if len(a) and all(x > 0 for x in a) else None

def pooled(s):
    base = s[s.model == "naive_weekly"].set_index(["issue_date", "zone", "target"])[["mae", "rmse"]]
    j = s.join(base, on=["issue_date", "zone", "target"], rsuffix="_b").dropna(subset=["mae", "mae_b"])
    j["n"] = j["n"].fillna(96)
    g = j.groupby(["model", "zone", "target"]).apply(lambda x: pd.Series({
        "rel": (x.n * x.mae).sum() / (x.n * x.mae_b).sum(),
        "relr": np.sqrt((x.n * x.rmse ** 2).sum() / (x.n * x.rmse_b ** 2).sum()),
        "pin": (x.n * x.pinball).sum() / x.n[x.pinball.notna()].sum() if x.pinball.notna().any() else np.nan,
        "days": len(x)}), include_groups=False).reset_index()
    return g

def elo(s, boot=200):
    s = s[~s.model.isin(REF) & s.rel_mae.notna()]
    models = sorted(s.model.unique()); k = len(models); ix = {m: i for i, m in enumerate(models)}
    days = sorted(s.issue_date.unique())
    per = []
    for d in days:
        W = np.zeros((k, k)); N = np.zeros((k, k))
        for _, r in s[s.issue_date == d].groupby(["zone", "target"]):
            v = r.set_index("model").rel_mae
            for a in v.index:
                for b in v.index:
                    if a == b: continue
                    i, j = ix[a], ix[b]
                    N[i, j] += 1
                    W[i, j] += 1.0 if v[a] < v[b] else 0.5 if v[a] == v[b] else 0.0
        per.append((W, N))
    def fit(ids, start=None):
        W = sum(per[i][0] for i in ids); N = sum(per[i][1] for i in ids)
        p = np.ones(k) if start is None else start.copy()
        wins = 0.5 + W.sum(1)
        for _ in range(300):
            den = 1 / (p + 1) + (N / (p[:, None] + p[None, :])).sum(1)
            q = wins / den; q /= np.exp(np.mean(np.log(q)))
            delta = np.max(np.abs(np.log(q / p))); p = q
            if delta < 1e-7: break
        return p
    to = lambda p: 1000 + 400 * np.log10(p)
    allp = fit(range(len(days)))
    rng = np.random.default_rng(42); draws = []
    for _ in range(boot):
        draws.append(to(fit(rng.integers(0, len(days), len(days)), allp)))
    draws = np.array(draws)
    races = s.groupby("model").size()
    return {m: {"elo": float(to(allp[i])), "lo": float(np.percentile(draws[:, i], 2.5)), "hi": float(np.percentile(draws[:, i], 97.5)), "races": int(races[m])} for m, i in ix.items()}

out = {}
for name, s, zones in (("backtest", bt, None), ("live", live, ["BE", "NL", "FR"])):
    s = s[~s.model.isin(hidden)]
    if zones: s = s[s.zone.isin(zones)]
    p = pooled(s); e = elo(s)
    rows = {}
    for m, g in p.groupby("model"):
        r = {"rel_gm": gm(list(g.rel)), "rel_mean": float(g.rel.mean()), "rel_med": float(g.rel.median()), "relr_gm": gm(list(g.relr)),
             "nser": len(g), "days": int(g.days.max())}
        for t, gt in g.groupby("target"): r[t] = float(np.exp(np.log(gt.rel).mean()))
        r["pin_rel"] = None
        r.update(e.get(m, {}))
        rows[m] = r
    out[name] = {"days": [s.issue_date.min(), s.issue_date.max(), s.issue_date.nunique()], "zones": sorted(s.zone.unique()), "models": rows}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "preprint_results.json", "w"), indent=1)
for name in out:
    print("==", name, out[name]["days"], out[name]["zones"])
    rows = sorted(out[name]["models"].items(), key=lambda kv: -(kv[1].get("elo") or 0))
    print(f"{'model':16} {'elo':>5} {'lo':>5} {'hi':>5} {'races':>5} {'relGM':>6} {'relMn':>6} {'rRMSE':>6} {'price':>6} {'wind':>6} {'solar':>6} {'load':>6} nser")
    for m, r in rows:
        f = lambda v: f"{v:6.3f}" if isinstance(v, float) and v == v else "     -"
        print(f"{m:16} {r.get('elo', 0):5.0f} {r.get('lo', 0):5.0f} {r.get('hi', 0):5.0f} {r.get('races', 0):5d} {f(r['rel_gm'])} {f(r['rel_mean'])} {f(r['relr_gm'])} {f(r.get('price'))} {f(r.get('wind'))} {f(r.get('solar'))} {f(r.get('load'))} {r['nser']}")
