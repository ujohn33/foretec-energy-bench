"""As-run live scores: replace every score row of a model re-run after the gate (run log "reruns") by the score of the
forecast as it was locked before the gate (the earliest backup of that file). Read-only on the benchmark; writes OUT.
Ensembles count from 20 Oct, the day they first ran live; their earlier rows were computed after the gate and are dropped.
usage: python scripts/research/preprint_asrun_scores.py OUT.parquet LAST_ISSUE_DATE  (run from the benchmark root);
then SCORES=OUT.parquet python scripts/research/preprint_all.py DIR LAST_ISSUE_DATE and the other preprint scripts."""
import json, sys
from pathlib import Path
import pandas as pd
from foretec_live.config import load_config
from foretec_live.score import metrics

out, last = sys.argv[1], sys.argv[2]
cfg = load_config()
R = Path("results/forecasts"); A = Path("data/actuals"); B = Path("/srv/foretec")
BACKUPS = [lambda d, m: B / "rerun_backup_20261019" / d / "forecasts" / f"{m}.parquet",
           lambda d, m: B / "rerun2_backup_20261019" / d / "forecasts" / f"{m}.parquet",
           lambda d, m: B / "rerun3_backup_20261019" / d / "forecasts" / f"{m}.parquet",
           lambda d, m: B / "rerun3_backup_20261019" / "live" / d / f"{m}.parquet",
           lambda d, m: B / "rerun4_backup_20261019" / "live" / d / f"{m}.parquet"]
s = pd.read_parquet("results/scores.parquet"); s["issue_date"] = s.issue_date.astype(str)
s = s[s.issue_date <= last].copy()
log = []
for d in sorted(s.issue_date.unique()):
    rj = R / d / "_run.json"
    reruns = json.loads(rj.read_text()).get("reruns", []) if rj.exists() else []
    added = json.loads(rj.read_text()).get("ensembles_added_utc") if rj.exists() else None
    touched = {}
    for r in reruns:
        for m in r.get("models", []):
            touched.setdefault(m, set()).update(r.get("series") or ["*"])
    for m, ser in touched.items():
        orig = next((f(d, m) for f in BACKUPS if f(d, m).exists()), None)
        rows = s.index[(s.issue_date == d) & (s.model == m)]
        rows = [i for i in rows if "*" in ser or f"{s.at[i, 'zone']}_{s.at[i, 'target']}" in ser]
        if orig is None:
            log.append((d, m, "NO BACKUP", len(rows))); continue
        fc_all = pd.read_parquet(orig)
        n_ok = 0
        for i in rows:
            z, t = s.at[i, "zone"], s.at[i, "target"]
            fc = fc_all[(fc_all.zone == z) & (fc_all.target == t)]
            fa = A / str((pd.Timestamp(d) + pd.Timedelta(days=1)).date()) / f"{z}_{t}.parquet"
            if fc.empty or not fa.exists():
                s.loc[i, ["mae", "rmse", "bias", "pinball"]] = float("nan"); continue
            y = pd.read_parquet(fa)["value"]
            r = metrics(fc, y, (cfg.get("score_resolution") or {}).get(f"{z}_{t}"))
            for k in ("n", "mae", "rmse", "bias", "pinball"):
                s.at[i, k] = r[k]
            n_ok += 1
        log.append((d, m, orig.parent.parent.name if "forecasts" in str(orig) else orig.parent.parent.parent.name, n_ok))
# ensembles were computed after the gate (ensembles_added_utc): not live forecasts in this period
ens = s.model.str.startswith("ens_")
s = s[~(ens & (s.issue_date < "2026-10-20"))]
base = s[s.model == cfg["baseline"]].set_index(["issue_date", "zone", "target"]).mae.rename("mae_base")
s = s.drop(columns=["mae_base", "rel_mae"], errors="ignore").join(base, on=["issue_date", "zone", "target"])
s["rel_mae"] = s.mae / s.mae_base
s = s[s.mae.notna()]
s.to_parquet(out, index=False)
pd.DataFrame(log, columns=["issue_date", "model", "source", "rows"]).to_csv(out.replace(".parquet", "_log.csv"), index=False)
print("rows", len(s), "replaced", sum(x[3] for x in log if isinstance(x[3], int)), "no-backup", [x for x in log if x[2] == "NO BACKUP"][:10])
