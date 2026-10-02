#!/usr/bin/env python
"""Build capacity-/population-weighted geographic centroids for BE, NL and FR.

Buckets: wind_onshore, wind_offshore, solar (MW weights) and load (population
weights). Each point set (installation, municipality, locality or turbine) is
reduced to a handful of centroids with the weighted k-means used by the
predico project (utils/solar_clusters.py, SolarClusters.weighted_kmeans), copied
verbatim below.

All downloads are cached under geo/raw/ (delete a file to refresh it).
Run:  python build_centroids.py            (writes geo/centroids.csv)
Needs: pandas, numpy, requests, geopandas, shapely.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
UA = {"User-Agent": "foretec-geo-centroids/1.0 (research script)"}

N_CLUSTERS = {"wind_onshore": 8, "solar": 8, "load": 6}
# mainland France = the 12 continental regions (excludes Corsica "94", overseas "01"-"06" and the
# overseas collectivities). Corsica is not part of the continental grid / bidding-zone load.
MAINLAND_FR_REGIONS = {"11", "24", "27", "28", "32", "44", "52", "53", "75", "76", "84", "93"}

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# weighted k-means: verbatim copy of SolarClusters.weighted_kmeans
# (WP_predico_project/utils/solar_clusters.py)
# ---------------------------------------------------------------------------
def weighted_kmeans(X, weights, n_clusters, max_iter=100, tol=1e-4):
    """
    Perform weighted K-means clustering.

    Parameters:
    -----------
    X : pandas.DataFrame
        Data points to cluster (should contain 'latitude' and 'longitude')
    weights : pandas.Series
        Weights for each data point
    n_clusters : int
        Number of clusters to form
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)

    Returns:
    --------
    clusters : numpy.ndarray
        Cluster assignments for each data point
    centroids : numpy.ndarray
        Coordinates of cluster centroids
    """
    # Initialize centroids by sampling from data, with probability proportional to weight
    centroids = X.sample(n_clusters, weights=weights, random_state=42).to_numpy()
    X_np = X.to_numpy()
    weights_np = weights.to_numpy()

    for i in range(max_iter):
        # Compute distances between data points and centroids
        distances = np.linalg.norm(X_np[:, None] - centroids[None, :], axis=2)
        # Assign points to nearest centroid
        clusters = np.argmin(distances, axis=1)

        # Update centroids based on weighted mean of assigned points
        new_centroids = []
        for k in range(n_clusters):
            points_in_cluster = X_np[clusters == k]
            weights_in_cluster = weights_np[clusters == k]
            if len(points_in_cluster) > 0:
                weighted_centroid = np.average(points_in_cluster, axis=0, weights=weights_in_cluster)
            else:
                weighted_centroid = X_np[np.random.choice(len(X_np))]
            new_centroids.append(weighted_centroid)

        new_centroids = np.array(new_centroids)
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return clusters, centroids


def cluster_points(pts: pd.DataFrame, k: int) -> pd.DataFrame:
    """pts: latitude, longitude, weight (>0). Returns one row per cluster."""
    pts = pts[pts["weight"] > 0].reset_index(drop=True)
    if len(pts) <= k:
        out = pts[["latitude", "longitude", "weight"]].copy()
        out["unit_count"] = pts.get("unit_count", pd.Series(1, index=pts.index)).values
    else:
        np.random.seed(42)  # makes the (rare) empty-cluster re-seed branch reproducible
        X = pts[["latitude", "longitude"]]
        labels, cents = weighted_kmeans(X, pts["weight"], k)
        rows = []
        for c in range(k):
            m = labels == c
            rows.append(
                dict(latitude=cents[c, 0], longitude=cents[c, 1],
                     weight=pts.loc[m, "weight"].sum(), unit_count=int(m.sum()))
            )
        out = pd.DataFrame(rows)
        out = out[out["unit_count"] > 0]
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    out["cluster_id"] = np.arange(len(out))
    return out


def grouped_centroids(pts: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Explicit groups (offshore farm clusters): weighted mean position per group."""
    rows = []
    for g, d in pts.groupby(group_col):
        rows.append(dict(
            latitude=np.average(d.latitude, weights=d.weight),
            longitude=np.average(d.longitude, weights=d.weight),
            weight=d.weight.sum(), unit_count=len(d), group=g))
    out = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    out["cluster_id"] = np.arange(len(out))
    return out


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------
def fetch(url: str, fname: str, params: dict | None = None, timeout: int = 300) -> Path:
    path = RAW / fname
    if path.exists() and path.stat().st_size > 0:
        return path
    log(f"download {url} -> raw/{fname}")
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


OVERPASS_MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def overpass(query: str, fname: str) -> Path | None:
    path = RAW / fname
    if path.exists() and path.stat().st_size > 0:
        return path
    for attempt in range(3):
        for url in OVERPASS_MIRRORS:
            try:
                r = requests.post(url, data={"data": query}, headers=UA, timeout=200)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    json.loads(r.text)
                    path.write_text(r.text)
                    log(f"overpass ok via {url} -> raw/{fname}")
                    return path
                log(f"overpass {url}: HTTP {r.status_code}")
            except Exception as e:  # noqa: BLE001
                log(f"overpass {url}: {type(e).__name__}")
        time.sleep(20)
    return None


def ods_export_csv(base: str, dataset: str, fname: str, **params) -> pd.DataFrame:
    """Opendatasoft v2.1 CSV export (semicolon separated)."""
    url = f"{base}/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
    p = fetch(url, fname, params={"delimiter": ";", **params})
    return pd.read_csv(p, sep=";", dtype=str)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_output(v) -> float:
    """OSM generator:output:electricity -> MW (NaN if unknown)."""
    if not isinstance(v, str):
        return np.nan
    m = re.match(r"\s*([0-9]+(?:[.,][0-9]+)?)\s*([kKmMgG]?W)?", v)
    if not m:
        return np.nan
    x = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "").upper()
    if unit == "W":
        return x / 1e6
    if unit == "KW":
        return x / 1e3
    if unit == "MW":
        return x
    if unit == "GW":
        return x * 1e3
    # no unit: OSM convention is ambiguous; large numbers are kW, small ones MW
    return x / 1e3 if x >= 50 else (x if x < 20 else np.nan)


# ---------------------------------------------------------------------------
# shared reference data
# ---------------------------------------------------------------------------
def be_municipalities() -> gpd.GeoDataFrame:
    p = fetch(
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "georef-belgium-municipality/exports/geojson",
        "be_municipalities.geojson",
        params={"select": "geo_point_2d,geo_shape,mun_code,mun_name_fr,mun_name_nl,prov_code,"
                          "prov_name_fr,prov_name_nl,reg_code,reg_name_nl,year"},
    )
    g = gpd.read_file(p)

    def first(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return x[0] if len(x) else None
        if isinstance(x, str) and x.startswith("["):
            try:
                v = json.loads(x)
                return v[0] if v else None
            except Exception:  # noqa: BLE001
                return x.strip("[]'\" ")
        return x

    for c in ["mun_code", "mun_name_fr", "mun_name_nl", "prov_code", "prov_name_fr",
              "prov_name_nl", "reg_code", "reg_name_nl"]:
        g[c] = g[c].map(first)
    pt = g["geo_point_2d"].map(lambda d: d if isinstance(d, dict) else json.loads(d))
    g["latitude"] = pt.map(lambda d: d["lat"])
    g["longitude"] = pt.map(lambda d: d["lon"])
    g["region"] = g["reg_name_nl"].map(
        {"Vlaams Gewest": "Flanders", "Waals Gewest": "Wallonia",
         "Brussels Hoofdstedelijk Gewest": "Brussels"})
    # Elia ods087 province names
    prov_map = {
        "Provincie Antwerpen": "Antwerp", "Provincie Oost-Vlaanderen": "East-Flanders",
        "Provincie Vlaams-Brabant": "Flemish-Brabant", "Provincie Limburg": "Limburg",
        "Provincie West-Vlaanderen": "West-Flanders", "Provincie Henegouwen": "Hainaut",
        "Provincie Luik": "Liège", "Provincie Luxemburg": "Luxembourg",
        "Provincie Namen": "Namur", "Provincie Waals-Brabant": "Walloon-Brabant",
    }
    g["elia_region"] = g["prov_name_nl"].map(prov_map)
    g.loc[g.region == "Brussels", "elia_region"] = "Brussels"
    return g


def be_population() -> pd.DataFrame:
    url = ("https://statbel.fgov.be/sites/default/files/files/opendata/"
           "bevolking%20naar%20woonplaats%2C%20nationaliteit%20burgelijke%20staat%20%2C"
           "%20leeftijd%20en%20geslacht/TF_SOC_POP_STRUCT_2026.zip")
    p = fetch(url, "pop_be_2026.zip")
    with zipfile.ZipFile(p) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        df = pd.read_csv(z.open(name), sep="|", usecols=["CD_REFNIS", "MS_POPULATION"],
                         dtype={"CD_REFNIS": str}, encoding="utf-8-sig")
    return df.groupby("CD_REFNIS", as_index=False)["MS_POPULATION"].sum().rename(
        columns={"CD_REFNIS": "mun_code", "MS_POPULATION": "population"})


def elia_latest(dataset: str, fname: str) -> pd.DataFrame:
    url = f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset}/records"
    p = fetch(url, fname, params={"order_by": "datetime desc", "limit": 100})
    df = pd.DataFrame(json.loads(p.read_text())["results"])
    latest = df["datetime"].max()
    return df[df["datetime"] == latest].copy()


def nl_gemeenten(year: int) -> gpd.GeoDataFrame:
    url = ("https://api.pdok.nl/cbs/gebiedsindelingen/ogc/v1/collections/"
           "gemeente_gegeneraliseerd/items")
    p = fetch(url, f"nl_gemeenten_{year}.geojson",
              params={"f": "json", "jaarcode": year, "limit": 1000})
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    gp = g.to_crs(28992)  # RD New for an area-correct centroid
    c = gp.geometry.centroid
    inside = gp.geometry.contains(c)
    c = c.where(inside, gp.geometry.representative_point())
    c = gpd.GeoSeries(c, crs=28992).to_crs(4326)
    g["latitude"], g["longitude"] = c.y.values, c.x.values
    return g[["statcode", "statnaam", "latitude", "longitude", "geometry"]]


def cbs_odata(table: str, fname: str, flt: str, select: str | None = None) -> pd.DataFrame:
    url = f"https://opendata.cbs.nl/ODataApi/odata/{table}/TypedDataSet"
    params = {"$filter": flt}
    if select:
        params["$select"] = select
    p = fetch(url, fname, params=params)
    df = pd.DataFrame(json.loads(p.read_text())["value"])
    df["RegioS"] = df["RegioS"].str.strip()
    return df


def fr_communes() -> pd.DataFrame:
    p = fetch("https://geo.api.gouv.fr/communes", "fr_communes.json",
              params={"fields": "code,nom,centre,population,codeRegion,codeDepartement",
                      "format": "json"})
    d = json.loads(p.read_text())
    df = pd.DataFrame(d)
    df["longitude"] = df["centre"].map(lambda c: c["coordinates"][0] if isinstance(c, dict) else np.nan)
    df["latitude"] = df["centre"].map(lambda c: c["coordinates"][1] if isinstance(c, dict) else np.nan)
    return df.drop(columns="centre")


def fr_communes_deleguees() -> pd.DataFrame:
    """Former communes (associées/déléguées) -> parent commune code, for old INSEE codes."""
    p = fetch("https://geo.api.gouv.fr/communes_associees_deleguees",
              "fr_communes_deleguees.json",
              params={"fields": "code,chefLieu,centre", "format": "json"})
    df = pd.DataFrame(json.loads(p.read_text()))
    if "centre" in df:
        df["longitude"] = df["centre"].map(lambda c: c["coordinates"][0] if isinstance(c, dict) else np.nan)
        df["latitude"] = df["centre"].map(lambda c: c["coordinates"][1] if isinstance(c, dict) else np.nan)
    return df


def odre_registry() -> pd.DataFrame:
    cols = ("nominstallation,codeinseecommune,codeinseecommuneimplantation,commune,coderegion,"
            "codedepartement,filiere,technologie,puismaxinstallee,nbinstallations,"
            "datemiseenservice,datederaccordement,regime")
    return ods_export_csv(
        "https://odre.opendatasoft.com",
        "registre-national-installation-production-stockage-electricite-agrege",
        "fr_odre_registre_wind_solar.csv",
        select=cols, where='filiere="Solaire" OR filiere="Eolien"',
    )


def land_mask() -> gpd.GeoDataFrame:
    p = fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip", "ne_10m_land.zip")
    g = gpd.read_file(f"zip://{p}")
    return g.cx[-6:10, 41:55].to_crs(3035)


def rivm_turbines() -> pd.DataFrame:
    p = fetch("https://data.rivm.nl/geo/alo/wfs", "nl_rivm_windturbines.json",
              params={"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                      "typeNames": "alo:rivm_windturbines_vermogen_actueel",
                      "outputFormat": "application/json", "srsName": "EPSG:4326"})
    d = json.loads(p.read_bytes().decode("utf-8"))
    df = pd.DataFrame([dict(f["properties"], longitude=f["geometry"]["coordinates"][0],
                            latitude=f["geometry"]["coordinates"][1]) for f in d["features"]])
    # some builds of this WFS return latin-1 mojibake in the country field
    df["land"] = df["land"].map(lambda s: s.encode("latin1").decode("utf8") if "Ã" in str(s) else s)
    df["weight"] = df["kw"] / 1e3
    return df


# ---------------------------------------------------------------------------
# BELGIUM
# ---------------------------------------------------------------------------
def build_be(out: list, summary: list) -> None:
    mun = be_municipalities()
    pop = be_population()
    mun = mun.merge(pop, on="mun_code", how="left")
    miss = mun.population.isna().sum()
    log(f"BE: {len(mun)} municipalities, {miss} without Statbel population")

    # ---- load ----------------------------------------------------------------
    pts = mun[["latitude", "longitude"]].assign(weight=mun.population.fillna(0))
    add(out, summary, "BE", "load", cluster_points(pts, N_CLUSTERS["load"]),
        "statbel_pop2026+ngi", total=pts.weight.sum())

    # ---- wind onshore: OSM turbines inside BE municipalities, calibrated to Elia
    q = ('[out:json][timeout:170];(node["power"="generator"]["generator:source"="wind"]'
         '(49.45,2.5,51.52,6.45);way["power"="generator"]["generator:source"="wind"]'
         '(49.45,2.5,51.52,6.45););out center tags;')
    p = overpass(q, "osm_be_wind.json")
    if p is None:
        raise RuntimeError("Overpass unavailable for BE wind; rerun later")
    meta = json.loads(p.read_text())
    osm_ts = meta["osm3s"]["timestamp_osm_base"]
    el = meta["elements"]
    w = pd.DataFrame([dict(
        latitude=e.get("lat", e.get("center", {}).get("lat")),
        longitude=e.get("lon", e.get("center", {}).get("lon")),
        out=e.get("tags", {}).get("generator:output:electricity")) for e in el])
    w["mw"] = w["out"].map(parse_output)
    gw = gpd.GeoDataFrame(w, geometry=gpd.points_from_xy(w.longitude, w.latitude), crs=4326)
    gw = gpd.sjoin(gw, mun[["region", "geometry"]], how="inner", predicate="within")
    n_tag = gw.mw.notna().sum()
    fill = gw.mw.median()
    gw["weight"] = gw.mw.fillna(fill)
    raw_by_region = gw.groupby("region").weight.sum()
    elia_w = elia_latest("ods086", "be_elia_ods086_latest.json")
    elia_on = elia_w[elia_w.offshoreonshore == "Onshore"].groupby("region").monitoredcapacity.sum()  # Elia- + DSO-connected
    scale = (elia_on / raw_by_region).reindex(raw_by_region.index).fillna(1.0)
    gw["weight"] = gw["weight"] * gw["region"].map(scale)
    log(f"BE wind_onshore: OSM {osm_ts}, {len(gw)} turbines in BE, {n_tag} with output tag, "
        f"untagged filled with median {fill:.2f} MW; raw OSM MW by region "
        f"{raw_by_region.round(0).to_dict()}; Elia ods086 monitored onshore "
        f"{elia_on.round(0).to_dict()} (at {elia_w.datetime.iloc[0]}); scale {scale.round(3).to_dict()}")
    add(out, summary, "BE", "wind_onshore", cluster_points(gw, N_CLUSTERS["wind_onshore"]),
        "osm+elia", total=gw.weight.sum(), raw_total=raw_by_region.sum())

    # ---- wind offshore: RIVM turbine positions (Belgian zone), 2 farm clusters
    t = rivm_turbines()
    off = t[(t.land == "België") & (t.ondergrond == "zee")].copy()
    log(f"BE wind_offshore: RIVM {len(off)} turbines, {off.weight.sum():.0f} MW; Elia monitored "
        f"offshore {elia_w[elia_w.offshoreonshore == 'Offshore'].monitoredcapacity.sum():.0f} MW")
    add(out, summary, "BE", "wind_offshore", cluster_points(off, 2), "rivm", total=off.weight.sum())

    # ---- solar ---------------------------------------------------------------
    # Flanders: Fluvius per municipality (kVA, inverter); Wallonia: ORES per locality (MVA);
    # both then scaled per province to Elia's monitored capacity (ods087). Communes not served
    # by ORES (RESA, AIEG, AIESH, REW) and Brussels (Sibelga) get the remaining Elia province
    # capacity distributed by population.
    flu = ods_export_csv("https://opendata.fluvius.be", "1_33-lp-open-data-fluvius",
                         "be_fluvius_lokale_productie.csv")
    flu["kva"] = to_num(flu["geinstalleerdvermogen_kva"])
    flu_peil = flu["peildatum"].dropna().max()
    fs = flu[flu.technologie == "ZONNE-ENERGIE"].groupby("hoofdgemeente", as_index=False).kva.sum()
    vl = mun[mun.region == "Flanders"].copy()

    def norm(s):
        s = str(s).upper()
        s = re.sub(r"[^A-Z0-9]", "", s.encode("ascii", "ignore").decode())
        return s

    vl["key"] = vl.mun_name_nl.map(norm)
    fs["key"] = fs.hoofdgemeente.map(norm)
    alias = {"SINTGENESIUSRODE": "SINTGENESIUSRODE", "VOERSTREEK": "VOEREN"}
    fs["key"] = fs["key"].replace(alias)
    fsj = fs.merge(vl[["key", "mun_code", "latitude", "longitude", "elia_region", "population"]],
                   on="key", how="left")
    unm = fsj[fsj.mun_code.isna()]
    log(f"BE solar Fluvius ({flu_peil}): {len(fs)} municipalities, {fs.kva.sum()/1e3:.0f} MVA; "
        f"unmatched names: {unm.hoofdgemeente.tolist()} ({unm.kva.sum()/1e3:.1f} MVA dropped)")
    fsj = fsj.dropna(subset=["mun_code"])
    fl_pts = fsj[["latitude", "longitude", "elia_region"]].assign(weight=fsj.kva / 1e3)

    ores = ods_export_csv(
        "https://www.odwb.be",
        "ores-electricite-puissance-des-productions-decentralisees-par-localite-annuel",
        "be_ores_productions_decentralisees.csv",
        select="annee,type_de_production,categorie_producteur,code_postal,localite_lc,"
               "nom_commune,centroid,province,puissance_installation_mva")
    ores["mva"] = to_num(ores["puissance_installation_mva"])
    ores["year"] = ores["annee"].str[:4]
    ores_year = ores.loc[ores.type_de_production == "Photovoltaïque", "year"].max()
    o = ores[(ores.type_de_production == "Photovoltaïque") & (ores.year == ores_year)].copy()
    ll = o["centroid"].str.split(",", expand=True)
    o["latitude"], o["longitude"] = to_num(ll[0]), to_num(ll[1])
    o = o.dropna(subset=["latitude", "longitude", "mva"])
    go = gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o.longitude, o.latitude), crs=4326)
    go = gpd.sjoin(go, mun[["mun_code", "elia_region", "region", "geometry"]], how="inner",
                   predicate="within")
    go = go[go.region == "Wallonia"]
    log(f"BE solar ORES ({ores_year}): {len(go)} locality x category rows, "
        f"{go.mva.sum():.0f} MVA in {go.mun_code.nunique()} Walloon communes")
    wa_pts = go[["latitude", "longitude", "elia_region"]].assign(weight=go.mva)

    elia_s = elia_latest("ods087", "be_elia_ods087_latest.json").groupby("region").monitoredcapacity.sum()
    log(f"BE solar Elia ods087 monitored capacity: Belgium {elia_s.get('Belgium'):.0f} MW")
    parts = []
    for reg, d in pd.concat([fl_pts, wa_pts]).groupby("elia_region"):
        target = elia_s.get(reg, np.nan)
        cov = d.weight.sum()
        if reg in ("Antwerp", "East-Flanders", "Flemish-Brabant", "Limburg", "West-Flanders"):
            parts.append(d.assign(weight=d.weight * target / cov))
            log(f"  {reg}: Fluvius {cov:.0f} -> Elia {target:.0f} (x{target/cov:.3f})")
        else:
            covered = set(go.loc[go.elia_region == reg, "mun_code"])
            gap_mun = mun[(mun.elia_region == reg) & ~mun.mun_code.isin(covered)]
            gap = max(target - cov, 0.0)
            if gap > 0 and len(gap_mun):
                gm = gap_mun[["latitude", "longitude", "elia_region"]].assign(
                    weight=gap * gap_mun.population / gap_mun.population.sum())
                parts.append(d)
                parts.append(gm)
                log(f"  {reg}: ORES {cov:.0f} MVA in {len(covered)} communes; Elia {target:.0f}; "
                    f"gap {gap:.0f} MW spread by population over {len(gap_mun)} non-ORES communes")
            else:
                parts.append(d.assign(weight=d.weight * target / cov))
                log(f"  {reg}: ORES {cov:.0f} -> Elia {target:.0f} (x{target/cov:.3f}, no gap communes)")
    bxl = mun[mun.region == "Brussels"]
    parts.append(bxl[["latitude", "longitude", "elia_region"]].assign(
        weight=elia_s["Brussels"] * bxl.population / bxl.population.sum()))
    log(f"  Brussels: Elia {elia_s['Brussels']:.0f} MW spread by population over 19 communes")
    sol = pd.concat(parts, ignore_index=True)
    add(out, summary, "BE", "solar", cluster_points(sol, N_CLUSTERS["solar"]),
        "fluvius+ores+elia", total=sol.weight.sum(),
        raw_total=fl_pts.weight.sum() + wa_pts.weight.sum())


# ---------------------------------------------------------------------------
# NETHERLANDS
# ---------------------------------------------------------------------------
NL_OFFSHORE_GROUPS = {
    "Borssele I": "Borssele", "Borssele II": "Borssele", "Borssele III": "Borssele",
    "Borssele IV": "Borssele", "Borssele V": "Borssele",
    "Windpark Hollandse Kust Zuid": "Hollandse Kust Zuid + Luchterduinen",
    "Luchterduinen": "Hollandse Kust Zuid + Luchterduinen",
    "HKN_5": "Hollandse Kust Noord + Prinses Amalia + OWEZ",
    "Prinses Amalia Windparken": "Hollandse Kust Noord + Prinses Amalia + OWEZ",
    "NSW Offshore windpark Egmond aan Zee": "Hollandse Kust Noord + Prinses Amalia + OWEZ",
    "Gemini I": "Gemini", "Gemini II": "Gemini",
}


def build_nl(out: list, summary: list) -> None:
    t = rivm_turbines()
    nl = t[t.land == "Nederland"]
    on = nl[nl.ondergrond.isin(["land", "binnenwater"])].copy()
    log(f"NL wind_onshore: RIVM {len(on)} turbines (land {int((on.ondergrond=='land').sum())}, "
        f"inland water {int((on.ondergrond=='binnenwater').sum())}), {on.weight.sum():.0f} MW; "
        f"data dates {sorted(t.datum.str[:10].unique())}")
    add(out, summary, "NL", "wind_onshore", cluster_points(on, N_CLUSTERS["wind_onshore"]),
        "rivm", total=on.weight.sum())

    off = nl[nl.ondergrond == "zee"].copy()
    off["group"] = off["naam"].map(NL_OFFSHORE_GROUPS)
    if off.group.isna().any():
        log(f"NL offshore: unmapped farm names {off.loc[off.group.isna(), 'naam'].unique()}")
        off = off.dropna(subset=["group"])
    g = grouped_centroids(off, "group")
    log("NL wind_offshore groups: " + "; ".join(
        f"{r.group} {r.weight:.0f} MW ({r.unit_count} turbines)" for r in g.itertuples()))
    add(out, summary, "NL", "wind_offshore", g, "rivm", total=off.weight.sum(),
        note="; ".join(g.group))

    # ---- solar: CBS 85005NED, latest year, all sectors, panel capacity (kWp) per municipality
    per = json.loads(fetch("https://opendata.cbs.nl/ODataApi/odata/85005NED/Perioden",
                           "nl_cbs_85005_perioden.json").read_text())["value"]
    last = sorted(r["Key"] for r in per)[-1]
    year = int(last[:4])
    s = cbs_odata("85005NED", f"nl_cbs_85005_{last}.json",
                  f"Perioden eq '{last}' and SectorEnVermogensklasse eq 'E007161'")
    nat = s.loc[s.RegioS == "NL01", "OpgesteldVermogenVanZonnepanelen_2"].sum() / 1e3
    s = s[s.RegioS.str.startswith("GM")]
    gm = nl_gemeenten(year)
    sj = s.merge(gm, left_on="RegioS", right_on="statcode", how="left")
    n_nocoord = sj.latitude.isna().sum()
    n_null = sj.OpgesteldVermogenVanZonnepanelen_2.isna().sum()
    sj = sj.dropna(subset=["latitude", "OpgesteldVermogenVanZonnepanelen_2"])
    pts = sj[["latitude", "longitude"]].assign(weight=sj.OpgesteldVermogenVanZonnepanelen_2 / 1e3)
    log(f"NL solar: CBS 85005NED {last}, {len(s)} municipalities ({n_nocoord} without PDOK {year} "
        f"geometry, {n_null} suppressed/null); sum {pts.weight.sum():.0f} MWp vs national {nat:.0f} MWp")
    add(out, summary, "NL", "solar", cluster_points(pts, N_CLUSTERS["solar"]),
        "cbs85005+pdok", total=pts.weight.sum(), raw_total=nat)

    # ---- load: CBS 37230ned population at start of latest month per municipality
    per = json.loads(fetch("https://opendata.cbs.nl/ODataApi/odata/37230ned/Perioden",
                           "nl_cbs_37230_perioden.json").read_text())["value"]
    last = sorted(r["Key"] for r in per if "MM" in r["Key"])[-1]
    p = cbs_odata("37230ned", f"nl_cbs_37230_{last}.json",
                  f"Perioden eq '{last}' and substring(RegioS,0,2) eq 'GM'",
                  select="RegioS,Perioden,BevolkingAanHetBeginVanDePeriode_1")
    gm = nl_gemeenten(int(last[:4]))
    pj = p.merge(gm, left_on="RegioS", right_on="statcode", how="left")
    log(f"NL load: CBS 37230ned {last}, {len(p)} municipalities, {pj.latitude.isna().sum()} "
        f"without geometry; population {pj.BevolkingAanHetBeginVanDePeriode_1.sum():,.0f}")
    pj = pj.dropna(subset=["latitude"])
    pts = pj[["latitude", "longitude"]].assign(weight=pj.BevolkingAanHetBeginVanDePeriode_1)
    add(out, summary, "NL", "load", cluster_points(pts, N_CLUSTERS["load"]),
        "cbs37230+pdok", total=pts.weight.sum())


# ---------------------------------------------------------------------------
# FRANCE
# ---------------------------------------------------------------------------
FR_OFFSHORE_FARMS = {
    # keyword in nomInstallation -> (farm, cluster group, anchor lat, lon). The anchor is the
    # OSM search centre; every OSM turbine at sea is assigned to its nearest anchor and the farm
    # position is the mean of its turbines. Only when OSM has no turbines for a farm is the
    # anchor itself used (Dieppe-Le Tréport: approximate centre derived from the published
    # distances of ~17 km to Dieppe and ~15.5 km to Le Tréport; uncertainty ~5 km).
    "BANC-DE-GUERANDE": ("Saint-Nazaire (Banc de Guérande)", "Loire-Vendée Atlantic", 47.15, -2.58),
    "VENT-DES-ILES": ("Yeu-Noirmoutier (Vent des Îles)", "Loire-Vendée Atlantic", 46.86, -2.38),
    "OPEN C": ("Floatgen / SEM-REV (Le Croisic)", "Loire-Vendée Atlantic", 47.24, -2.78),
    "BAIE-DE-ST-BRIEUC": ("Saint-Brieuc", "Brittany North", 48.85, -2.55),
    "HAUTES - FALAISES": ("Fécamp (Hautes Falaises)", "Normandy Channel", 49.87, 0.25),
    "RIDENS-DE-DIEPPE": ("Dieppe-Le Tréport (Ridens de Dieppe)", "Normandy Channel", 50.09, 1.15),
    "FARAMAN": ("Provence Grand Large (Faraman)", "Mediterranean", 43.25, 4.80),
    "GRUISSAN": ("EolMed (Gruissan)", "Mediterranean", 43.02, 3.25),
    "LEUCATE": ("EFGL (Leucate)", "Mediterranean", 42.85, 3.20),
}


def build_fr(out: list, summary: list) -> None:
    com = fr_communes()
    main = com[com.codeRegion.isin(MAINLAND_FR_REGIONS)].dropna(subset=["latitude"])
    log(f"FR: {len(com)} communes, {len(main)} in mainland (excl. Corsica + overseas), "
        f"population {main.population.sum():,.0f}")

    # ---- load
    pts = main[["latitude", "longitude"]].assign(weight=main.population.fillna(0))
    add(out, summary, "FR", "load", cluster_points(pts, N_CLUSTERS["load"]),
        "insee_pop+geoapi", total=pts.weight.sum())

    # ---- registry
    r = odre_registry()
    r["mw"] = to_num(r["puismaxinstallee"]) / 1e3
    r["n"] = to_num(r["nbinstallations"]).fillna(1)
    r = r[r.datederaccordement.isna()]
    r = r[r.coderegion.isin(MAINLAND_FR_REGIONS) | (r.coderegion.isna() & r.codedepartement.isna())]
    cc = main.set_index("code")[["latitude", "longitude"]]
    dele = fr_communes_deleguees()
    dele_cc = dele.dropna(subset=["latitude"]).set_index("code")[["latitude", "longitude"]] \
        if "latitude" in dele else pd.DataFrame(columns=["latitude", "longitude"])

    def locate(df: pd.DataFrame, label: str) -> pd.DataFrame:
        code = df.codeinseecommune.fillna(df.codeinseecommuneimplantation).copy()
        # municipal arrondissements of Paris / Lyon / Marseille -> parent commune
        code = code.where(~code.str.match(r"^751\d\d$", na=False), "75056")
        code = code.where(~code.str.match(r"^6938\d$", na=False), "69123")
        code = code.where(~code.str.match(r"^132(0[1-9]|1[0-6])$", na=False), "13055")
        lat = code.map(cc.latitude)
        lon = code.map(cc.longitude)
        m = lat.isna()
        lat[m] = code[m].map(dele_cc.latitude)  # former (associated / delegated) communes
        lon[m] = code[m].map(dele_cc.longitude)
        df = df.assign(latitude=lat, longitude=lon)
        # rows aggregated at department level only (e.g. "Agrégation des installations de moins
        # de 36kW" without commune): put them at the capacity-weighted centroid of the located
        # rows of the same department
        loc = df.dropna(subset=["latitude"])
        dep_c = loc.groupby("codedepartement").apply(
            lambda d: pd.Series({"lat": np.average(d.latitude, weights=d.mw.fillna(0).clip(lower=1e-9)),
                                 "lon": np.average(d.longitude, weights=d.mw.fillna(0).clip(lower=1e-9))}),
            include_groups=False)
        m = df.latitude.isna() & df.codedepartement.isin(dep_c.index)
        dep_mw = df.mw[m].sum()
        df.loc[m, "latitude"] = df.loc[m, "codedepartement"].map(dep_c.lat)
        df.loc[m, "longitude"] = df.loc[m, "codedepartement"].map(dep_c.lon)
        miss = df.latitude.isna()
        log(f"FR {label}: {len(df)} registry rows, {df.mw.sum():.0f} MW; {int(m.sum())} rows "
            f"({dep_mw:.1f} MW) only known by department -> department centroid; "
            f"{int(miss.sum())} rows ({df.mw[miss].sum():.1f} MW) not locatable -> dropped")
        if miss.any():
            dropped = df[miss].assign(dep=df.codedepartement.fillna("?"), com=df.codeinseecommune.fillna("?"))
            log("   dropped by department/commune: " + str(
                dropped.groupby(["dep", "com"]).mw.sum().sort_values(ascending=False).head(8).round(1).to_dict()))
        return df[~miss]

    on = r[(r.filiere == "Eolien") & (~r.technologie.fillna("").str.startswith("En mer"))]
    on = locate(on, "wind_onshore")
    pts = on[["latitude", "longitude"]].assign(weight=on.mw)
    add(out, summary, "FR", "wind_onshore", cluster_points(pts, N_CLUSTERS["wind_onshore"]),
        "odre_registre", total=pts.weight.sum())

    sol = r[(r.filiere == "Solaire") & (~r.technologie.isin(["Batterie", "Thermodynamique"]))]
    sol = locate(sol, "solar")
    pts = sol[["latitude", "longitude"]].assign(weight=sol.mw, unit_count=sol.n)
    log(f"FR solar: {int(sol.n.sum()):,} installations (small ones aggregated per commune in the registry)")
    add(out, summary, "FR", "solar", cluster_points(pts, N_CLUSTERS["solar"]),
        "odre_registre", total=pts.weight.sum())

    # ---- offshore: capacity from the registry, position from OSM turbines at sea
    off = r[(r.filiere == "Eolien") & (r.technologie.fillna("").str.startswith("En mer"))].copy()
    def farm_of(name):
        for k, v in FR_OFFSHORE_FARMS.items():
            if k in str(name).upper():
                return v
        return None
    off["farm_info"] = off.nominstallation.map(farm_of)
    if off.farm_info.isna().any():
        log(f"FR offshore unmapped registry rows: {off.loc[off.farm_info.isna(), 'nominstallation'].tolist()}")
        off = off.dropna(subset=["farm_info"])
    farms = off.groupby(off.farm_info.map(lambda v: v[0])).mw.sum()

    positions = {}
    q = "[out:json][timeout:170];("
    for k, (farm, grp, la, lo) in FR_OFFSHORE_FARMS.items():
        q += f'node["power"="generator"]["generator:source"="wind"](around:25000,{la},{lo});'
    q += ");out;"
    p = overpass(q, "osm_fr_offshore_candidates.json")
    if p is not None:
        el = json.loads(p.read_text())["elements"]
        cand = pd.DataFrame([dict(latitude=e["lat"], longitude=e["lon"]) for e in el if "lat" in e])
        land = land_mask()
        gc = gpd.GeoDataFrame(cand, geometry=gpd.points_from_xy(cand.longitude, cand.latitude),
                              crs=4326).to_crs(3035)
        dist = gc.geometry.apply(lambda pt: land.distance(pt).min())
        sea = cand[(dist > 3000).values]
        anchors = list(FR_OFFSHORE_FARMS.values())
        dd = np.stack([np.hypot(sea.latitude.values - la,
                                (sea.longitude.values - lo) * np.cos(np.radians(la)))
                       for _, _, la, lo in anchors], axis=1)
        nearest, dmin = dd.argmin(axis=1), dd.min(axis=1)
        for i, (farm, grp, la, lo) in enumerate(anchors):
            near = sea[(nearest == i) & (dmin < 0.2)]
            if len(near):
                positions[farm] = (near.latitude.mean(), near.longitude.mean(), len(near), "osm")
    rows = []
    for farm, mw in farms.items():
        grp, la, lo = next((g, a, b) for f, g, a, b in FR_OFFSHORE_FARMS.values() if f == farm)
        lat, lon, nt, src = positions.get(farm, (la, lo, 0, "approx"))
        rows.append(dict(farm=farm, group=grp, latitude=lat, longitude=lon, weight=mw,
                         turbines_osm=nt, pos_source=src))
    fo = pd.DataFrame(rows)
    log("FR offshore farms:\n" + fo.round(4).to_string(index=False))
    fo.to_csv(RAW / "fr_offshore_farms_resolved.csv", index=False)
    g = grouped_centroids(fo.assign(unit_count=1), "group")
    g["unit_count"] = fo.groupby("group").size().reindex(g.group).values  # farms per group
    add(out, summary, "FR", "wind_offshore", g, "odre_registre+osm", total=fo.weight.sum(),
        note="; ".join(g.group))


# ---------------------------------------------------------------------------
def add(out, summary, zone, bucket, cl, source, total, raw_total=None, note=None):
    cl = cl.copy()
    cl["zone"], cl["bucket"], cl["source"] = zone, bucket, source
    out.append(cl)
    summary.append(dict(zone=zone, bucket=bucket, n_clusters=len(cl), total_weight=total,
                        raw_total=raw_total, source=source, note=note))


def main() -> int:
    out, summary = [], []
    build_be(out, summary)
    build_nl(out, summary)
    build_fr(out, summary)
    df = pd.concat(out, ignore_index=True)
    df["latitude"] = df.latitude.round(4)
    df["longitude"] = df.longitude.round(4)
    df["weight"] = np.where(df.bucket == "load", df.weight.round(0), df.weight.round(3))
    df["unit_count"] = df.unit_count.round(0).astype(int)
    order = {"BE": 0, "NL": 1, "FR": 2}
    border = {"wind_onshore": 0, "wind_offshore": 1, "solar": 2, "load": 3}
    df = df.sort_values(["zone", "bucket", "cluster_id"],
                        key=lambda s: s.map(order) if s.name == "zone" else
                        (s.map(border) if s.name == "bucket" else s))
    cols = ["zone", "bucket", "cluster_id", "latitude", "longitude", "weight", "unit_count", "source"]
    df[cols].to_csv(ROOT / "centroids.csv", index=False)
    sm = pd.DataFrame(summary)
    print(sm.to_string(index=False))
    (RAW / "build_log.txt").write_text("\n".join(LOG) + "\n\n" + sm.to_string(index=False) + "\n")
    print(f"wrote {ROOT / 'centroids.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
