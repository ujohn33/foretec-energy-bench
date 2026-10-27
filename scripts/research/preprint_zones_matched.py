"""Figure 5 data: per zone and target group, pooled Rel. MAE of each plotted model on the races ALL of them share
(TSO references included where they forecast that zone and group; tso_final not in DK). As-run scores via SCORES."""
import os, sys
import numpy as np, pandas as pd
out = sys.argv[1]
s = pd.read_parquet(os.environ.get("SCORES", "results/scores.parquet")); s["issue_date"] = s.issue_date.astype(str)
s = s[s.issue_date <= os.environ.get("LAST", "2026-10-19")]
K = ["issue_date", "zone", "target"]
base = s[s.model == "naive_weekly"].set_index(K).mae.rename("mae_b")
MODELS = ["naive_daily", "chronos2", "lgbm_cov", "chronos2_cov", "tso_cutoff", "tso_final"]
GROUPS = {"price": ["price"], "ws": ["wind", "solar"], "load": ["load"]}
ZONES = ["BE", "NL", "FR", "DE", "DK1", "DK2"]
for g, targets in GROUPS.items():
    rows = []
    for i, z in enumerate(ZONES):
        sub = s[(s.zone == z) & s.target.isin(targets) & s.model.isin(MODELS)]
        if z in ("DK1", "DK2"): sub = sub[sub.model != "tso_final"]
        present = [m for m in MODELS if (sub.model == m).any()]
        if not present: continue
        races = None
        for m in present:
            r = set(map(tuple, sub[sub.model == m][K].values))
            races = r if races is None else races & r
        x = sub[[tuple(v) in races for v in sub[K].values]].join(base, on=K)
        rel = x.groupby(["model", "target"]).apply(lambda q: (q.n.fillna(96) * q.mae).sum() / (q.n.fillna(96) * q.mae_b).sum(), include_groups=False)
        vals = {m: float(np.exp(np.log(rel.loc[m]).mean())) if m in rel.index.get_level_values(0) else None for m in MODELS}
        n = len({(a, b) for a, _, b in races})
        rows.append([i, z, n] + [vals[m] for m in MODELS])
        print(g, z, "races", len(races), {m: (round(v, 3) if v else None) for m, v in vals.items()})
    with open(f"{out}/zones_{g}.dat", "w") as o:
        o.write("x zone races " + " ".join(m.replace("_", "") for m in MODELS) + "\n")
        for r in rows:
            o.write(" ".join(str(v) if not isinstance(v, float) else f"{v:.3f}" for v in r[:3]) + " " +
                    " ".join("nan" if v is None else f"{v:.3f}" for v in r[3:]) + "\n")
