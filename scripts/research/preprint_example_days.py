import sys, pandas as pd, datetime as dt
out = sys.argv[1]
for issue, zone, target in [("2026-10-07", "BE", "wind"), ("2026-10-13", "FR", "solar")]:
    delivery = (pd.Timestamp(issue) + pd.Timedelta(days=1)).date()
    a = pd.read_parquet(f"data/actuals/{delivery}/{zone}_{target}.parquet")
    print(a.columns.tolist(), a.index[:2] if len(a.columns) < 3 else a.head(2))
    if "time_utc" in a.columns: a = a.set_index("time_utc")
    y = a.iloc[:, 0] if a.shape[1] == 1 else a["value"]
    y.index = pd.to_datetime(y.index)
    frames = {"actual": y}
    for m in ["chronos2", "chronos2_cov", "tso_cutoff", "tso_final"]:
        try:
            f = pd.read_parquet(f"results/forecasts/{issue}/{m}.parquet")
        except FileNotFoundError:
            continue
        f = f[(f.zone == zone) & (f.target == target)]
        if f.empty: continue
        f = f.set_index(pd.to_datetime(f.delivery_utc))
        frames[m] = f["point"]
        if m == "chronos2_cov":
            frames["lo"] = f["q0.1"]; frames["hi"] = f["q0.9"]
    d = pd.DataFrame(frames)
    idx = d.index.tz_localize("UTC") if d.index.tz is None else d.index
    loc = idx.tz_convert("Europe/Brussels")
    d = d[(loc.date == delivery)]
    loc = loc[(loc.date == delivery)]
    d.insert(0, "h", [(t.hour + t.minute / 60) for t in loc])
    d = (d.drop(columns="h") / 1000).assign(h=d["h"].values)   # GW
    cols = ["h"] + [c for c in d.columns if c != "h"]
    d[cols].round(4).to_csv(f"{out}/example_{zone}_{target}.dat", sep=" ", index=False, na_rep="nan")
    print(issue, zone, target, len(d), d.columns.tolist())
