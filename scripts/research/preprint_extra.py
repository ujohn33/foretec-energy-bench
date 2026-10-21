"""Preprint extras: full field on the last four issue days, TSO pairs without DK, example-day data. Run from benchmark root."""
import glob, json, os, sys
import numpy as np, pandas as pd, yaml
sys.argv += [] ; out = sys.argv[1]
exec(open("scripts/research/preprint_all.py").read().split("ps = per_series(s)")[0].replace('out, last = sys.argv[1], sys.argv[2]', 'out, last = sys.argv[1], "2026-10-19"'))
# 1) full field, issue days 16-19
w = s[s.issue_date >= "2026-10-16"]
p = per_series(w); e = elo(w)
ff = {}
for m, g in p.groupby("model"):
    r = {"rel": gm(g.rel), "relr": gm(g.relr), "nser": len(g)}
    for t, gt in g.groupby("target"): r[t] = gm(gt.rel)
    r.update(e.get(m, {})); ff[m] = r
json.dump(ff, open(f"{out}/late_field.json", "w"), indent=1)
# 2) TSO pairs without DK
tp = {}
for ref in ["tso_cutoff", "tso_final"]:
    rr = s[(s.model == ref) & ~s.zone.isin(["DK1", "DK2"])][K]
    tp[ref] = {"series": sorted(set(rr.zone + "_" + rr.target)), "races": len(rr), "models": {}}
    for m in [ref, "chronos2_cov", "timesfm3_cov", "t0_beta_cov", "tirex2_cov", "lgbm_cov"]:
        x = s[s.model == m].merge(rr)
        pp = per_series(x)
        tp[ref]["models"][m] = {"races": len(x), **{t: gm(g.rel) for t, g in pp.groupby("target")}}
json.dump(tp, open(f"{out}/tso_pairs_nodk.json", "w"), indent=1)
# 3) example days: ratio chronos2_cov / chronos2 per race on wind and solar
a = s[s.model == "chronos2"].set_index(K).rel_mae; b = s[s.model == "chronos2_cov"].set_index(K).rel_mae
r = (b / a).dropna().reset_index(); r.columns = K + ["ratio"]
r = r[r.target.isin(["wind", "solar"])].join(a.rename("rel_hist"), on=K).join(b.rename("rel_cov"), on=K)
r.sort_values("ratio").to_csv(f"{out}/example_candidates.csv", index=False)
print(r.ratio.describe())
