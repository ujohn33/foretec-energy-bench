#!/usr/bin/env python
"""Weather-point centroids for DE (the DE-LU bidding zone), DK1 and DK2, merged into centroids.csv.

Same buckets, cluster counts and capacity-weighted k-means as build_centroids.py (imported from it), so the
method is identical to BE, NL and FR. Rows of other zones are left exactly as they are: their NWP caches are
keyed by their own centroids and stay valid.

Sources
  DE wind, solar  Marktstammdatenregister (Bundesnetzagentur, DL-DE-BY-2.0) Gesamtdatenexport, units in operation
                  (EinheitBetriebsstatus 35), parsed by /srv/foretec/mastr/parse_units.py into de_wind_units.csv.gz
                  (one row per turbine; WindAnLandOderAufSee 889 = offshore), de_solar_cells.csv.gz (units with
                  coordinates, kW per 0.05 degree cell) and de_solar_plz.csv.gz (units without public coordinates,
                  i.e. small rooftop systems, kW per postcode, placed at the GeoNames postcode centroid)
  DK solar,       Energinet Energi Data Service, CapacityPerMunicipality (latest month), at the municipality's
  onshore wind    Wikidata coordinates (P1168/P625); DK2 = regions Hovedstaden and Sjaelland (Eastern Denmark)
  DK offshore     the operating farms and their capacity (Danish Energy Agency list), farm centres approximate;
                  grouped by site (as for NL and FR); Thor included (Energinet counts 3.7 GW offshore)
  load            GeoNames cities1000 population (DE and LU for DE-LU; DK1/DK2 split at 10.95 E, Bornholm DK2)
Run:  python build_centroids_north.py --data /srv/foretec/mastr
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_centroids import LOG, N_CLUSTERS, RAW, ROOT, add, cluster_points, grouped_centroids, log

UA = {"User-Agent": "foretec-geo-centroids/1.0 (research script)"}
ZONES = ("DE", "DK1", "DK2")

DK_OFFSHORE = [   # name, area, group, latitude, longitude, MW (operating, October 2026; farm centres approximate)
    ("Horns Rev 1", "DK1", "Horns Rev", 55.48, 7.84, 160), ("Horns Rev 2", "DK1", "Horns Rev", 55.60, 7.58, 209),
    ("Horns Rev 3", "DK1", "Horns Rev", 55.70, 7.68, 407), ("Thor", "DK1", "Thor", 56.35, 7.75, 1000),
    ("Vesterhav Syd", "DK1", "West coast", 55.97, 8.08, 170), ("Vesterhav Nord", "DK1", "West coast", 56.47, 8.07, 180),
    ("Nissum Bredning", "DK1", "West coast", 56.66, 8.25, 28),
    ("Anholt", "DK1", "Kattegat", 56.60, 11.21, 400), ("Samsø", "DK1", "Kattegat", 55.72, 10.59, 23),
    ("Kriegers Flak", "DK2", "Kriegers Flak", 55.03, 12.88, 604),
    ("Rødsand II", "DK2", "Rødsand", 54.56, 11.53, 207), ("Nysted", "DK2", "Rødsand", 54.55, 11.72, 166),
    ("Middelgrunden", "DK2", "Øresund and Great Belt", 55.69, 12.67, 40), ("Sprogø", "DK2", "Øresund and Great Belt", 55.33, 10.96, 21),
]


def cluster(pts: pd.DataFrame, k: int) -> pd.DataFrame:
    """cluster_points, robust to a few dominant points (a handful of municipalities or farms carrying most of the
    weight): the k-means initialisation samples k distinct points by weight, which pandas refuses once one point
    holds more than 1/k of it. Such points are split into identical copies sharing their weight, which leaves every
    weighted centroid unchanged; unit_count is then recounted on the original points."""
    pts = pts[pts["weight"] > 0].reset_index(drop=True)
    cap = pts["weight"].sum() / (2 * k)
    reps = np.ceil(pts["weight"] / cap).clip(lower=1).astype(int)
    split = pts.loc[pts.index.repeat(reps)].copy()
    split["weight"] = split["weight"] / reps.loc[split.index].to_numpy()
    out = cluster_points(split.reset_index(drop=True), k)
    d = ((pts["latitude"].to_numpy()[:, None] - out["latitude"].to_numpy()) ** 2
         + (pts["longitude"].to_numpy()[:, None] - out["longitude"].to_numpy()) ** 2)
    out["unit_count"] = np.bincount(d.argmin(axis=1), minlength=len(out))
    return out


def cities(data: Path, countries: tuple[str, ...]) -> pd.DataFrame:
    with zipfile.ZipFile(data / "cities1000.zip") as z:
        df = pd.read_csv(z.open("cities1000.txt"), sep="\t", header=None, usecols=[4, 5, 8, 14],
                         names=["latitude", "longitude", "cc", "population"], quoting=3, low_memory=False)
    df = df[df["cc"].isin(countries) & (df["population"] > 0)]
    return df.rename(columns={"population": "weight"})[["latitude", "longitude", "weight"]].reset_index(drop=True)


def build_de(out, summary, data: Path) -> None:
    w = pd.read_csv(data / "de_wind_units.csv.gz", dtype={"plz": str})
    w["weight"] = w["kw"] / 1000.0
    w = w.dropna(subset=["lat", "lon"]).rename(columns={"lat": "latitude", "lon": "longitude"})
    on, off = w[~w["sea"]], w[w["sea"]]
    log(f"DE wind: {len(w)} turbines with coordinates, onshore {on.weight.sum():.0f} MW, offshore {off.weight.sum():.0f} MW")
    add(out, summary, "DE", "wind_onshore", cluster(on, N_CLUSTERS["wind_onshore"]), "mastr", total=on.weight.sum())
    add(out, summary, "DE", "wind_offshore", cluster(off, 4), "mastr", total=off.weight.sum())

    cells = pd.read_csv(data / "de_solar_cells.csv.gz").rename(columns={"lat": "latitude", "lon": "longitude"})
    plz = pd.read_csv(data / "de_solar_plz.csv.gz", dtype={"plz": str})
    with zipfile.ZipFile(data / "postal_DE.zip") as z:
        pc = pd.read_csv(z.open("DE.txt"), sep="\t", header=None, usecols=[1, 9, 10], names=["plz", "latitude", "longitude"], dtype={"plz": str})
    pc = pc.groupby("plz")[["latitude", "longitude"]].mean()
    p = plz.join(pc, on="plz")
    lost = p["latitude"].isna()
    log(f"DE solar: {cells.kw.sum() / 1e6:.1f} GW with coordinates, {p.kw.sum() / 1e6:.1f} GW by postcode "
        f"({p.loc[lost, 'kw'].sum() / 1e3:.0f} MW with unknown postcodes dropped)")
    pts = pd.concat([cells.assign(weight=cells.kw / 1000.0), p[~lost].assign(weight=p.kw / 1000.0)])[["latitude", "longitude", "weight"]]
    pts = pts[pts["latitude"].between(47, 55.2) & pts["longitude"].between(5.8, 15.1)]
    add(out, summary, "DE", "solar", cluster(pts, N_CLUSTERS["solar"]), "mastr+geonames_postcodes", total=pts.weight.sum())

    pop = cities(data, ("DE", "LU"))
    log(f"DE-LU load: {len(pop)} localities, {pop.weight.sum() / 1e6:.1f} M inhabitants")
    add(out, summary, "DE", "load", cluster(pop, N_CLUSTERS["load"]), "geonames_cities1000", total=pop.weight.sum())


def eds(dataset: str, **params) -> list:
    """Energi Data Service records; waits out its rate limit (429) a few times."""
    import time
    for _ in range(6):
        r = requests.get(f"https://api.energidataservice.dk/dataset/{dataset}", params=params, headers=UA, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()["records"]
        time.sleep(30)
    raise RuntimeError(f"{dataset}: still rate-limited")


def build_dk(out, summary, data: Path) -> None:
    import json
    if (data / "dk_capacity.json").exists():   # Energi Data Service rate-limits hard: a saved copy of the same query
        cap = pd.DataFrame(json.loads((data / "dk_capacity.json").read_text())["records"])
        month = cap["Month"].max()
    else:
        month = eds("CapacityPerMunicipality", limit=1, sort="Month DESC")[0]["Month"]
        cap = pd.DataFrame(eds("CapacityPerMunicipality", limit=1000, start=month[:10],
                               end=str((pd.Timestamp(month) + pd.Timedelta(days=1)).date())))
    # municipality centres and regions from Wikidata (Danish municipality code P1168, coordinates P625); the
    # Dataforsyningen DAWA API that used to serve them is gone (HTTP 410 since 2026)
    kom = pd.DataFrame(json.loads((data / "dk_kommuner_wikidata.json").read_text()))
    kom = pd.DataFrame({"MunicipalityNo": kom["kode"].astype(int).astype(str), "region": kom["region"],
                        "longitude": kom["longitude"], "latitude": kom["latitude"]})
    cap = cap.merge(kom, on="MunicipalityNo", how="left")
    unplaced = cap["latitude"].isna()
    log(f"DK capacity {month[:7]}: {len(cap)} rows; solar {cap.SolarPowerCapacity.sum():.0f} MW, onshore "
        f"{cap.OnshoreWindCapacity.sum():.0f} MW, offshore {cap.OffshoreWindCapacity.sum():.0f} MW; not a municipality "
        f"(offshore pseudo-entries): solar {cap.loc[unplaced, 'SolarPowerCapacity'].sum():.0f} MW, "
        f"onshore {cap.loc[unplaced, 'OnshoreWindCapacity'].sum():.0f} MW")
    cap = cap[~unplaced]
    cap["area"] = np.where(cap["region"].isin(["Capital Region of Denmark", "Region Zealand", "Eastern Denmark"]), "DK2", "DK1")
    off = pd.DataFrame(DK_OFFSHORE, columns=["name", "area", "group", "latitude", "longitude", "weight"])
    pop = cities(data, ("DK",))
    pop["area"] = np.where(pop["longitude"] > 10.95, "DK2", "DK1")
    for area in ("DK1", "DK2"):
        c = cap[cap["area"] == area]
        s = c.assign(weight=c.SolarPowerCapacity)[["latitude", "longitude", "weight"]]
        o = c.assign(weight=c.OnshoreWindCapacity)[["latitude", "longitude", "weight"]]
        f = off[off["area"] == area]
        lp = pop[pop["area"] == area]
        log(f"{area}: solar {s.weight.sum():.0f} MW, onshore {o.weight.sum():.0f} MW, offshore {f.weight.sum():.0f} MW "
            f"({len(f)} farms), {lp.weight.sum() / 1e6:.2f} M inhabitants")
        add(out, summary, area, "wind_onshore", cluster(o, N_CLUSTERS["wind_onshore"]), "energinet_capacity_per_municipality", total=o.weight.sum())
        add(out, summary, area, "wind_offshore", grouped_centroids(f, "group").drop(columns="group"), "dea_offshore_farms",
            total=f.weight.sum(), raw_total=None, note=", ".join(sorted(f.group.unique())))
        add(out, summary, area, "solar", cluster(s, N_CLUSTERS["solar"]), "energinet_capacity_per_municipality", total=s.weight.sum())
        add(out, summary, area, "load", cluster(lp, N_CLUSTERS["load"]), "geonames_cities1000", total=lp.weight.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with the MaStR extracts and GeoNames files")
    args = ap.parse_args()
    out, summary = [], []
    build_de(out, summary, args.data)
    build_dk(out, summary, args.data)
    new = pd.concat(out, ignore_index=True)
    new["latitude"], new["longitude"] = new.latitude.round(4), new.longitude.round(4)
    new["weight"] = np.where(new.bucket == "load", new.weight.round(0), new.weight.round(3))
    new["unit_count"] = new.unit_count.round(0).astype(int)
    cols = ["zone", "bucket", "cluster_id", "latitude", "longitude", "weight", "unit_count", "source"]
    old = pd.read_csv(ROOT / "centroids.csv")
    border = {"wind_onshore": 0, "wind_offshore": 1, "solar": 2, "load": 3}
    new = new.sort_values(["zone", "bucket", "cluster_id"], key=lambda s: s.map(border) if s.name == "bucket" else s)
    df = pd.concat([old[~old["zone"].isin(ZONES)], new[cols]], ignore_index=True)   # existing rows first, unchanged
    df[cols].to_csv(ROOT / "centroids.csv", index=False)
    sm = pd.DataFrame(summary)
    print(sm.to_string(index=False))
    (RAW / "build_log_north.txt").write_text("\n".join(LOG) + "\n\n" + sm.to_string(index=False) + "\n")
    print(f"wrote {ROOT / 'centroids.csv'}: {len(old[~old['zone'].isin(ZONES)])} existing rows kept, {len(new)} new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
