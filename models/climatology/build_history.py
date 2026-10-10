"""Download the climatology history once: Energy-Charts, one request per year and endpoint.

    FORETEC_HOME=/srv/foretec/foretec-live python models/climatology/build_history.py
writes data/static/climatology_history.parquet (zone, target, time_utc, value), 15-minute grid.

    ... build_history.py --targets load
adds targets to the existing file from ENTSO-E; the other rows are kept as they are.

    ... build_history.py --series BE_wind BE_solar
rebuilds single series through the configured source chain (source_by_series applies, e.g. Elia's
quarter-hours for BE wind and solar), one year per request; the other rows are kept.
"""
import argparse
import datetime as dt
import os
import time
from pathlib import Path

import pandas as pd

from foretec_live.config import load_config
from foretec_live.data.base import to_15min
from foretec_live.data.energycharts import SOLAR_NAMES, WIND_NAMES, _get, _pick, _series

START, END = dt.date(2023, 1, 1), dt.date(2026, 8, 31)

cfg = load_config()
ap = argparse.ArgumentParser()
ap.add_argument("--targets", nargs="*", help="add only these targets, from ENTSO-E, keeping the rest of the file")
ap.add_argument("--series", nargs="*", help="rebuild only these series (ZONE_target) through the source chain")
args = ap.parse_args()
if args.series:
    from foretec_live.data import get_source

    src, new = get_source(cfg), []
    for key in args.series:
        zone, target = key.split("_")
        for year in range(START.year, END.year + 1):
            a, b = max(START, dt.date(year, 1, 1)), min(END, dt.date(year, 12, 31)) + dt.timedelta(days=1)
            s = src.fetch(zone, target, pd.Timestamp(a), pd.Timestamp(b))
            expected = len(pd.date_range(pd.Timestamp(a), pd.Timestamp(b), freq="15min", inclusive="left"))
            print(key, year, int(s.notna().sum()), "of", expected, "points", getattr(src, "provenance", {}).get((zone, target)), flush=True)
            if s.notna().sum() < 0.95 * expected:   # never write a history with a hole (e.g. a source rate-limited us)
                raise SystemExit(f"{key} {year}: only {int(s.notna().sum())} of {expected} points; history not written")
            new.append(s.rename("value").to_frame().assign(zone=zone, target=target))
    new = pd.concat(new).reset_index().rename(columns={"index": "time_utc"})
    out = Path(os.environ.get("FORETEC_HOME", ".")) / "data" / "static" / "climatology_history.parquet"
    old = pd.read_parquet(out)
    drop = (old["zone"] + "_" + old["target"]).isin(args.series)
    h = pd.concat([old[~drop], new[["zone", "target", "time_utc", "value"]]], ignore_index=True)
    h.to_parquet(out, index=False)
    print(out, len(h), "rows")
    raise SystemExit(0)
if args.targets:
    from foretec_live.data.entsoe import EntsoeSource

    src, new = EntsoeSource(cfg), []
    for zone in cfg["zones"]:
        for target in args.targets:
            s = src.fetch(zone, target, pd.Timestamp(START), pd.Timestamp(END + dt.timedelta(days=1)))
            new.append(s.rename("value").to_frame().assign(zone=zone, target=target))
            print(zone, target, int(s.notna().sum()), "points", flush=True)
    new = pd.concat(new).reset_index().rename(columns={"index": "time_utc"})
    out = Path(os.environ.get("FORETEC_HOME", ".")) / "data" / "static" / "climatology_history.parquet"
    old = pd.read_parquet(out)
    h = pd.concat([old[~old["target"].isin(args.targets)], new[["zone", "target", "time_utc", "value"]]], ignore_index=True)
    h.to_parquet(out, index=False)
    print(out, len(h), "rows")
    raise SystemExit(0)
frames = []
for zone, z in cfg["zones"].items():
    for year in range(START.year, END.year + 1):
        a, b = max(START, dt.date(year, 1, 1)), min(END, dt.date(year, 12, 31))
        js = _get("/price", {"bzn": z["energycharts_bzn"], "start": a.isoformat(), "end": b.isoformat()})
        frames.append(to_15min(_series(js["unix_seconds"], js["price"])).rename("value").to_frame().assign(zone=zone, target="price"))
        js = _get("/public_power", {"country": z["energycharts_country"], "start": a.isoformat(), "end": b.isoformat()})
        for target, names in (("wind", WIND_NAMES), ("solar", SOLAR_NAMES)):
            total = None
            for t in _pick(js["production_types"], names):
                s = _series(js["unix_seconds"], t["data"])
                total = s if total is None else total.add(s, fill_value=0)
            frames.append(to_15min(total).rename("value").to_frame().assign(zone=zone, target=target))
        print(zone, year, flush=True)
h = pd.concat(frames).reset_index().rename(columns={"index": "time_utc"})
h = h[(h["time_utc"] >= pd.Timestamp(START)) & (h["time_utc"] < pd.Timestamp(END + dt.timedelta(days=1)))]
h = h.drop_duplicates(["zone", "target", "time_utc"], keep="last")
out = Path(os.environ.get("FORETEC_HOME", ".")) / "data" / "static" / "climatology_history.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
h[["zone", "target", "time_utc", "value"]].to_parquet(out, index=False)
print(out, len(h), "rows")
print(h.groupby(["zone", "target"])["time_utc"].agg(["min", "max", "count"]))
