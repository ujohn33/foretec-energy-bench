"""Each entry against each TSO reference on exactly the races both forecast (as-run scores, no DK for tso_final)."""
import json, os, sys
import numpy as np, pandas as pd
s = pd.read_parquet(os.environ.get("SCORES", "results/scores.parquet")); s["issue_date"] = s.issue_date.astype(str)
s = s[s.issue_date <= os.environ.get("LAST", "2026-10-19")]
K = ["issue_date", "zone", "target"]
base = s[s.model == "naive_weekly"].set_index(K)[["mae"]]
gm = lambda a: float(np.exp(np.mean(np.log(a))))
def rel(x):
    j = x.join(base, on=K, rsuffix="_b")
    return j.groupby(["zone", "target"]).apply(lambda q: (q.n * q.mae).sum() / (q.n * q.mae_b).sum(), include_groups=False)
out = {}
models = ["chronos2_cov", "timesfm3_cov", "t0_beta_cov", "t0_alpha_cov", "tirex2_cov", "lgbm_cov", "t0_beta", "chronos2", "timesfm3"]
for ref, excl in [("tso_cutoff", []), ("tso_final", ["DK1", "DK2"])]:
    r = s[(s.model == ref) & ~s.zone.isin(excl)]
    out[ref] = {}
    for m in models:
        x = s[s.model == m].merge(r[K])
        common = x[K]
        rr, rm = rel(r.merge(common)), rel(x)
        res = {}
        for t in ["wind", "solar", "load"]:
            a = rr[rr.index.get_level_values("target") == t]; b = rm[rm.index.get_level_values("target") == t]
            if len(a): res[t] = {"ref": round(gm(a), 3), "model": round(gm(b), 3), "races": int((common.target == t).sum())}
        out[ref][m] = res
print(json.dumps(out, indent=1))
