# Weather-point centroids for BE, NL and FR

`centroids.csv` lists the points where we fetch numerical weather prediction (NWP)
forecasts for Belgium (BE), the Netherlands (NL) and France (FR). For each zone and
bucket we average the forecasts at these points, weighted by `weight`. The results
are features for day-ahead forecasts of national wind and solar generation and of
day-ahead power prices.

| bucket | what the points represent | weight |
|---|---|---|
| `wind_onshore` | where onshore wind capacity is installed | MW installed |
| `wind_offshore` | operating offshore wind farm clusters | MW installed |
| `solar` | where PV capacity is installed, rooftop included | MW (MWp or MVA, see below) |
| `load` | where people live, used as a proxy for temperature-driven demand | inhabitants |

Columns: `zone, bucket, cluster_id, latitude, longitude, weight, unit_count, source`.
Coordinates are WGS84 decimal degrees, rounded to 4 decimals. `cluster_id` is ordered
by descending weight. `unit_count` is the number of input records in the cluster. What
a record is depends on the source (turbine, municipality, locality, registry line or
farm), and the tables below say which. Only the **relative** weights within a
zone and bucket matter for the aggregation.

`build_centroids.py` rebuilds the file from the public sources. It caches downloads
in `raw/`, which is git-ignored. It needs pandas, numpy, requests, geopandas and
shapely. A run also writes a detailed log to `raw/build_log.txt`.

## Method

Each set of input points becomes a few centroids through the **capacity-weighted
k-means** of the predico project (`utils/solar_clusters.py`,
`SolarClusters.weighted_kmeans`). The function is copied verbatim into the script:

1. The initial centroids are drawn from the points with probability proportional
   to weight (`DataFrame.sample(..., random_state=42)`).
2. Each point is assigned to the nearest centroid, using Euclidean distance in
   plain latitude/longitude degrees. This is the same as the original and is not
   an equal-area metric.
3. Each centroid is recomputed as the weighted mean of its members.
4. Steps 2 and 3 repeat until the shift falls below 1e-4 or 100 iterations pass.

A cluster's weight is the sum of its members' weights. The only addition to the
original is `np.random.seed(42)` before each call, which makes the empty-cluster
reseed branch reproducible.

Cluster counts:

- **wind_onshore:** 8 clusters per zone.
- **solar:** 8 clusters per zone.
- **load:** 6 clusters per zone.
- **wind_offshore:** at most 4 clusters per zone, one per group of nearby farms.
  - **BE:** k-means with k=2 on turbine positions, since the Belgian zone is compact.
  - **NL and FR:** explicit farm groups. Each centroid is the capacity-weighted mean
    position of its farms or turbines.

## Sources and results per zone × bucket

Totals are those of the points that went into the clustering, compared with the
rough 2026 sanity targets in the brief.

### Belgium

| bucket | clusters | total | sanity target | verdict |
|---|---|---|---|---|
| wind_onshore | 8 | 3,731 MW (calibrated; raw OSM 3,476 MW) | ~3.5 GW | OK |
| wind_offshore | 2 | 2,213 MW (Elia monitored: 2,262 MW) | ~2.3 GW | OK |
| solar | 8 | 12,128 MW (calibrated to Elia) | 10–11 GW | **Above target.** This is Elia's own monitored capacity for Oct 2026, so the target is probably out of date. |
| load | 6 | 11,867,630 inhabitants | — | — |

**wind_onshore**

- **Positions:** OpenStreetMap `power=generator` + `generator:source=wind`, nodes and
  ways, fetched through the Overpass API (`maps.mail.ru` mirror). The OSM snapshot is
  2026-10-02T13:26Z. Licence: ODbL, © OpenStreetMap contributors.
  - We fetched a bounding box and kept the turbines that fall inside Belgian
    municipality polygons. That gives 1,394 turbines and leaves out offshore turbines.
- **Capacity:**
  - 760 of the 1,394 turbines have a `generator:output:electricity` tag.
  - **Assumption:** the 634 untagged turbines get the median of the tagged ones,
    2.35 MW.
  - The sums were then scaled per region to Elia's monitored onshore capacity in
    Elia Open Data **ods086** ("Wind power production estimation and forecast on
    Belgian grid (Near real-time)", Elia Open Data Licence). That capacity includes
    both Elia- and DSO-connected turbines.
    - Flanders: 2,057 MW (×1.076).
    - Wallonia: 1,672 MW (×1.070).
    - Brussels: one 2 MW turbine, left unscaled.
  - The scaling only fixes the Flanders/Wallonia split. Within each region the
    distribution still depends on how complete OSM is.
- **unit_count:** turbines.

**wind_offshore**

- **Source:** turbine positions and rated power from **RIVM "Windturbines –
  vermogen"** (WFS layer `alo:rivm_windturbines_vermogen_actueel`,
  https://data.rivm.nl/geo/alo/wfs). The layer also covers Belgian turbines.
  - Licence: Public Domain Mark, according to the RIVM metadata in the Nationaal
    Georegister.
  - Data date: 2026-01-11.
- **Contents:** 392 Belgian sea turbines, 2,213 MW.
  - Elia ods086 monitors 2,262 MW offshore. The ~2% gap does not change the positions.
  - RIVM notes that its offshore positions are less precise than its onshore ones.
    We judge that accurate enough for NWP grid cells.
- **Clusters:** two, splitting the Belgian offshore zone into a south-east (inshore)
  half and a north-west half.
- **unit_count:** turbines.

**solar** (no single national register is openly downloadable, so three sources
are combined)

- **Flanders:** **Fluvius "Lokale productie-installaties per gemeente"**
  (https://opendata.fluvius.be/explore/dataset/1_33-lp-open-data-fluvius/).
  - Status: 2026-08. Licence: "Open data license – FLUVIUS"
    (https://opendata.fluvius.be/p/licentieopendatafluvius).
  - Coverage: all 285 Flemish municipalities. The weight is installed inverter
    capacity in kVA for technology `ZONNE-ENERGIE`, so all sizes including rooftop.
  - Location: each municipality sits at its centroid from NGI AdminVector (see load).
- **Wallonia:** **ORES "024 – Puissance des productions décentralisées par
  localité (Annuel)"**, published on the ODWB portal (https://www.odwb.be).
  - Year: 2025. Licence: CC0.
  - Contents: 1,892 locality × size-class rows of photovoltaic MVA, placed at the
    locality centroid. They cover 197 of the 261 Walloon communes.
  - **Gap:** ORES does not serve the RESA area around Liège or the small DSOs (AIEG,
    AIESH, REW).
    - For each province, the difference between Elia's monitored capacity and the
      ORES total goes to the communes that have no ORES data, split **by population**.
    - Liège: 572 MW over 53 communes.
    - Hainaut: 152 MW over 6 communes.
    - Luxembourg: 101 MW over 1 commune.
    - Namur: 68 MW over 4 communes.
    - **Assumption:** in those communes, PV density follows population.
- **Brussels:** no municipality-level data was found. **Assumption:** Elia's
  Brussels capacity (331 MW) is split over the 19 communes by population.
- **Calibration:** province totals are scaled to Elia Open Data **ods087**
  ("Photovoltaic power production estimation and forecast on Belgian grid (Near
  real-time)", Elia Open Data Licence), using its monitored capacity per province.
  - Belgium total: 12,128 MW.
  - Scale factors for the Flemish provinces: ×1.07 to ×1.17. The likely causes are
    kVA versus kWp and transmission-connected plants.
  - Walloon-Brabant: ×1.24.
- **Elia reference file not reused:** `WP_predico_project/data/elia_solar_cluster_points.csv`
  has 10 clusters snapped to a 0.25° grid, and its source register is not openly
  downloadable. The sources above are newer, municipality- or locality-level and
  reproducible.
- **unit_count:** municipality or locality rows.

**load**

- **Population:** **Statbel "Population by place of residence, nationality, marital
  status, age and sex"** on 1 January 2026 (`TF_SOC_POP_STRUCT_2026`), summed per
  municipality code.
  - Licence: Statbel open data. Reuse is allowed with attribution. We have not
    reviewed the exact licence text again for this build.
- **Boundaries:** **NGI-IGN AdminVector** municipalities (2025 boundaries, 565
  municipalities), taken from the Opendatasoft "georef-belgium-municipality" dataset.
  - Licence: NGI standard open licence.
  - The municipality centroid comes from the dataset's `geo_point_2d`.
- **unit_count:** municipalities.

### Netherlands

| bucket | clusters | total | sanity target | verdict |
|---|---|---|---|---|
| wind_onshore | 8 | 7,064 MW | ~7 GW | OK |
| wind_offshore | 4 | 4,759 MW | 4.7–6 GW | OK. Hollandse Kust West VI is not included, see below. |
| solar | 8 | 28,981 MWp (national 29,426 MWp) | 25–30 GW | OK. Data is end of 2025. |
| load | 6 | 18,153,271 inhabitants | — | — |

**wind_onshore**

- **Source:** RIVM "Windturbines – vermogen" (as above). All Dutch turbines with
  `ondergrond` = `land` (3,426) or `binnenwater` (162, e.g. Windpark Fryslân and the
  IJsselmeer parks), each weighted by rated kW.
- **Data dates:** 2026-01-11, plus 20 turbines added on 2026-09-14.
- **unit_count:** turbines.

**wind_offshore**

- **Source:** RIVM `ondergrond` = `zee`, 671 turbines. Farms are grouped by RIVM farm
  name:
  1. **Hollandse Kust Zuid + Luchterduinen:** 1,658 MW.
  2. **Borssele I–V:** 1,502.5 MW.
  3. **Hollandse Kust Noord + Prinses Amalia + Egmond aan Zee (OWEZ):** 998 MW.
  4. **Gemini I+II:** 600 MW.
- **Not included:** Hollandse Kust West VI (Ecowende, 760 MW). It produced first power
  in July 2026 and is due to be complete by the end of 2026, but it is not yet in the
  RIVM layer. Once it is complete it should join group 3. That would move group 3's
  centroid west and roughly double its weight.
- **unit_count:** turbines.

**solar**

- **Capacity:** **CBS StatLine 85005NED** "Zonnestroom; vermogen en vermogensklasse,
  bedrijven en woningen, regio".
  - Period: 2025 (installed capacity at the end of 2025). The table was last modified
    on 2026-06-12. Licence: CC BY 4.0.
  - Weight: panel capacity (kWp) for all sectors, businesses plus dwellings, per
    municipality. That covers small rooftop through ground-mounted systems.
  - 446 MWp is coded "GM0000, niet in te delen" (cannot be assigned) and has no
    location, so it is not included. 18 historical municipality codes have no values.
- **Boundaries:** **PDOK "CBS Gebiedsindelingen"**, `gemeente_gegeneraliseerd` for
  2025 (342 municipalities). Licence: CC BY 4.0.
  - Each municipality's point is its area centroid in RD New. If the centroid falls
    outside the polygon, a point inside the polygon is used instead.
- **unit_count:** municipalities.

**load**

- **Population:** **CBS StatLine 37230ned** "Bevolkingsontwikkeling; regio per
  maand", population at the start of August 2026 per municipality. Licence: CC BY 4.0.
- **Boundaries:** PDOK CBS Gebiedsindelingen 2026, with centroids computed as for solar.
- **unit_count:** municipalities.

### France (mainland only: the 12 continental regions; Corsica and overseas excluded)

| bucket | clusters | total | sanity target | verdict |
|---|---|---|---|---|
| wind_onshore | 8 | 24,503 MW | 23–25 GW | OK |
| wind_offshore | 4 | 2,296 MW | 1.5–2.5 GW | OK |
| solar | 8 | 33,943 MW | 25–30 GW | **Above target.** The registry is newer (31/07/2026) and includes all small installations. |
| load | 6 | 65,810,329 inhabitants | — | — |

**Registry used for wind_onshore, wind_offshore and solar**

- **Source:** **ODRÉ "Registre national des installations de production et de
  stockage d'électricité (au 31/07/2026)"**, published by RTE with all French DSOs
  (https://odre.opendatasoft.com/explore/dataset/registre-national-installation-production-stockage-electricite-agrege/).
  - The dataset was modified on 2026-09-10. Licence: Licence Ouverte v2.0 (Etalab).
- **Filters:**
  - Rows that have a disconnection date (`dateDeraccordement`) are removed.
  - Only the 12 continental regions are kept. Corsica (region 94) is not part of the
    continental system. Overseas regions (01–06) are excluded.
- **Placement:** each row goes to the centre of its INSEE commune.
  - Municipal arrondissements of Paris, Lyon and Marseille map to the parent commune.
  - Former (delegated or associated) communes use their own centre.
  - Rows known only by department are placed at the capacity-weighted centroid of
    that department's located rows. This affects 961 solar rows (405 MW) and 1 wind
    row (16 MW).
  - Rows with no location at all are dropped: 1.8 MW of solar and 5.7 MW of wind.

**wind_onshore**

- **Selection:** `filiere = Eolien` with technology `Terrestre`, empty or `Autre`.
  The empty-technology mainland rows are onshore parks in Centre-Val de Loire.
- **unit_count:** registry rows. A row is roughly one wind farm.

**solar**

- **Selection:** `filiere = Solaire`, excluding the `Batterie` and `Thermodynamique`
  technologies.
- **Small systems:** installations under 36 kW are aggregated per commune in the
  registry, so rooftop PV is fully represented. The 128,630 rows cover about 1.26
  million installations.
- **unit_count:** registry rows, not installations.

**wind_offshore**

- **Capacity:** taken from the same registry (technology `En mer posé` / `En mer
  flottant`, 17 rows), summed per farm.
- **Positions:** the mean of the OpenStreetMap turbines at sea (more than 3 km from the
  Natural Earth 10m land polygon, which is public domain) nearest to each farm. The
  OSM snapshot is 2026-10-02.

| farm | MW in registry | position source |
|---|---|---|
| Saint-Nazaire (Banc de Guérande) | 480 | OSM, 76 turbines |
| Yeu-Noirmoutier (Vent des Îles) | 500 | OSM, 61 turbines |
| Floatgen / SEM-REV (Le Croisic) | 10 | OSM, 6 turbines nearest its anchor, probably including edge turbines of Saint-Nazaire. With 10 MW in the same group, the effect is negligible. |
| Saint-Brieuc | 496 | OSM, 62 turbines |
| Fécamp | 497 | OSM, 71 turbines |
| Dieppe-Le Tréport (Ridens de Dieppe) | 248 (only the first of two units is registered) | **approximate**: 50.09 N, 1.15 E |
| Provence Grand Large (Faraman), floating | 25.2 | OSM |
| EolMed (Gruissan) | 30 | OSM |
| EFGL (Leucate) | 10 | OSM |

- **Dieppe-Le Tréport:** OSM has no turbines for this farm yet. Its position was
  derived from the published distances to the coast (about 17 km from Dieppe and
  15.5 km from Le Tréport). It may be off by about 5 km.
- **Not included:** Centre Manche / Courseulles-sur-Mer (Calvados) is not in the
  registry as of 31/07/2026.
- **Groups:**
  1. **Loire-Vendée Atlantic** (Saint-Nazaire, Yeu-Noirmoutier, Floatgen): 990 MW.
  2. **Normandy Channel** (Fécamp, Dieppe-Le Tréport): 745 MW.
  3. **Brittany North** (Saint-Brieuc): 496 MW.
  4. **Mediterranean** (Provence Grand Large, EolMed, EFGL): 65 MW.
- **Caveat for group 4:** its centroid lies at sea between the Rhône-delta site and
  the Gruissan/Leucate sites, which are about 120 km apart. It is a compromise, and it
  carries 2.8% of French offshore capacity.
- **unit_count:** farms.

**load**

- **Source:** **geo.api.gouv.fr** `/communes` (Etalab, Licence Ouverte v2.0, data from
  INSEE and IGN): the commune centre and the latest INSEE legal municipal population.
- **Coverage:** 34,386 mainland communes. Accessed on 2026-10-02.
- **unit_count:** communes.

## Known limitations

- **Temperature-only load proxy:** the load weights are residential population only.
  They do not account for industrial or tertiary demand, or for electric heating
  share, which matters in France.
- **Mixed vintages:** the inputs range from end of 2025 (NL solar, BE-Wallonia
  ORES) to Oct 2026 (OSM, Elia monitored capacity). Fast-growing PV regions may be
  slightly under-weighted.
- **BE onshore relies on OSM completeness:** about 45% of BE onshore turbines lack a
  capacity tag. Region-level calibration to Elia corrects the Flanders/Wallonia
  split, but not the distribution within each region.
- **Population-based fills in BE solar:** the Liège/RESA area (572 MW) and Brussels
  (331 MW) are spread by population, not by observed installations.
- **Coarse distance metric:** the k-means uses degrees as a planar metric. At about
  50° N, one degree of longitude is about 0.64 of a degree of latitude, so clusters
  stretch slightly east–west. This keeps it identical to the existing predico centroids.
