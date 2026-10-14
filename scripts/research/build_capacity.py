"""Daily installed wind and solar capacity per zone, for capacity-scaled targets (research; experiment E).

  DE   Marktstammdatenregister: each unit's net rated power from its commissioning date to its final
       decommissioning (parse_capacity.py output de_capacity_daily.parquet); wind = onshore + offshore
  BE   Elia Open Data monitored capacity (ods031/ods086 wind, summed over parts; ods032/ods087 solar, region Belgium),
       daily averages
  DK1, Energinet CapacityPerMunicipality, monthly, municipalities assigned to DK1/DK2 by region (Wikidata); the
  DK2  offshore pseudo-municipality 1 (Thor, from 2026) is DK1
Usage: FORETEC_HOME=... python scripts/research/build_capacity.py /srv/foretec/mastr
Writes data/static/capacity/<zone>_<target>.parquet (index date, column mw).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
out = Path(os.environ.get("FORETEC_HOME", ".")) / "data" / "static" / "capacity"
out.mkdir(parents=True, exist_ok=True)
DAYS = pd.date_range("2022-01-01", "2026-10-31", freq="D")


def save(key: str, s: pd.Series) -> None:
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")].reindex(DAYS).ffill().bfill()
    s.rename("mw").to_frame().rename_axis("date").to_parquet(out / f"{key}.parquet")
    print(key, "MW:", {str(d.date()): round(s.loc[d]) for d in (DAYS[0], pd.Timestamp("2024-09-01"), pd.Timestamp("2026-09-01"))})


# DE: MaStR
de = pd.read_parquet(src / "de_capacity_daily.parquet")
save("DE_wind", de["wind_onshore"] + de["wind_offshore"])
save("DE_solar", de["solar"])

# BE: Elia monitored capacity
wind = pd.DataFrame(json.loads((src / "cap" / "be_wind_ods031.json").read_text()) + json.loads((src / "cap" / "be_wind_ods086.json").read_text()))
wind = wind.drop_duplicates(["day", "offshoreonshore", "region"], keep="last").groupby("day")["cap"].sum()
wind.index = pd.to_datetime(wind.index)
save("BE_wind", wind)
solar = pd.DataFrame(json.loads((src / "cap" / "be_solar_ods032.json").read_text()) + json.loads((src / "cap" / "be_solar_ods087.json").read_text()))
solar = solar.drop_duplicates("day", keep="last").set_index("day")["cap"]
solar.index = pd.to_datetime(solar.index)
save("BE_solar", solar)

# DK: Energinet capacity per municipality, monthly
rec = pd.DataFrame(json.loads((src / "cap" / "dk_capacity_history.json").read_text())["records"]).fillna(0)
kom = pd.DataFrame(json.loads((src / "dk_kommuner_wikidata.json").read_text()))
area = {str(int(k)): ("DK2" if r in ("Capital Region of Denmark", "Region Zealand", "Eastern Denmark") else "DK1")
        for k, r in zip(kom["kode"], kom["region"])}
area["1"] = "DK1"                                   # offshore pseudo-municipality: Thor (DK1)
rec["area"] = rec["MunicipalityNo"].map(area)
rec = rec[rec["area"].notna()]
rec["month"] = pd.to_datetime(rec["Month"].str[:10])
for a in ("DK1", "DK2"):
    g = rec[rec["area"] == a].groupby("month")[["OnshoreWindCapacity", "OffshoreWindCapacity", "SolarPowerCapacity"]].sum()
    save(f"{a}_wind", g["OnshoreWindCapacity"] + g["OffshoreWindCapacity"])
    save(f"{a}_solar", g["SolarPowerCapacity"])
