"""Preprint Table 4: each foundation model without vs with covariates on the races both forecast (live, BE/NL/FR, issue days to 2026-10-19). Run from the benchmark root."""
import numpy as np, pandas as pd
s = pd.read_parquet("results/scores.parquet"); s["issue_date"] = s.issue_date.astype(str)
s = s[s.zone.isin(["BE", "NL", "FR"]) & (s.issue_date <= "2026-10-19")]
base = s[s.model == "naive_weekly"].set_index(["issue_date", "zone", "target"])[["mae"]]
def rel(x):
    x = x.join(base, on=["issue_date", "zone", "target"], rsuffix="_b")
    per = x.groupby(["zone", "target"]).apply(lambda q: (q.n * q.mae).sum() / (q.n * q.mae_b).sum(), include_groups=False)
    return per
K = ["issue_date", "zone", "target"]
rows = []
for m in ["chronos2", "timesfm3", "timesfm25", "t0_alpha", "t0_beta", "tirex2"]:
    a, b = s[s.model == m], s[s.model == m + "_cov"]
    common = a[K].merge(b[K])
    ra, rb = rel(a.merge(common)), rel(b.merge(common))
    ratio = (rb / ra)
    gm = lambda v: float(np.exp(np.log(v).mean()))
    out = {"model": m, "races": len(common), "rel": round(gm(ra), 3), "rel_cov": round(gm(rb), 3), "change_all": round(gm(ratio) - 1, 3)}
    for t in ["price", "wind", "solar", "load"]:
        r = ratio[ratio.index.get_level_values("target") == t]
        out[t] = round(gm(r) - 1, 3) if len(r) else None
    rows.append(out)
d = pd.DataFrame(rows); print(d.to_string(index=False))
print("median change", d.change_all.median(), "mean", d.change_all.mean().round(3))
