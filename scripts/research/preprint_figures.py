"""Matplotlib figures for the methodology preprint (docs/preprint).

usage: python scripts/research/preprint_figures.py RESULTS_DIR OUT_DIR
  RESULTS_DIR  output of preprint_all.py (perseries.csv)
  OUT_DIR      docs/preprint
Needs matplotlib, pandas, geopandas (any env with them; the benchmark venv does not have matplotlib).
Writes fig_map.pdf (weather sampling points) and fig_heatmap.pdf (pooled Rel. MAE per series).
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

res, out = Path(sys.argv[1]), Path(sys.argv[2])
root = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
ZONES = ["BE", "NL", "FR", "DE", "DK1", "DK2"]
ZLAB = {"DE": "DE-LU"}

# ---------------------------------------------------------------- weather sampling points
c = pd.read_csv(root / "geo" / "centroids.csv")
land = gpd.read_file(f"zip://{root / 'geo' / 'raw' / 'ne_10m_land.zip'}")
fig, ax = plt.subplots(figsize=(6.3, 5.0))
land.clip((-6, 41, 17, 58.5)).plot(ax=ax, color="#EEF0F3", edgecolor="#B9BEC7", linewidth=0.4)
COL = {"BE": "#C8102E", "NL": "#E07B00", "FR": "#2E6DB4", "DE": "#3A3A3A", "DK1": "#2A9D8F", "DK2": "#7B5EA7"}
MARK = {"wind_onshore": "^", "wind_offshore": "v", "solar": "o", "load": "s"}
for (z, b), g in c.groupby(["zone", "bucket"]):
    w = g["weight"] / g["weight"].sum()                      # share of the zone's bucket at each point
    ax.scatter(g.longitude, g.latitude, s=8 + 160 * w, marker=MARK.get(b, "o"), facecolor=COL[z],
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
for z, (x, y) in {"BE": (3.2, 50.15), "NL": (4.3, 53.15), "FR": (1.0, 45.5), "DE": (10.6, 50.0),
                  "DK1": (8.2, 56.75), "DK2": (12.6, 55.9)}.items():
    ax.text(x, y, ZLAB.get(z, z), color=COL[z], fontsize=8, fontweight="bold", ha="center", zorder=4)
ax.set_xlim(-5.5, 16.5); ax.set_ylim(41.5, 58.0); ax.set_aspect(1 / np.cos(np.radians(50)))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#B9BEC7")
ax.legend(handles=[Line2D([], [], marker=m, ls="", color="#555", markersize=5, label=b.replace("_", " "))
                   for b, m in MARK.items()], loc="upper left", frameon=False, title="bucket", title_fontsize=7)
fig.tight_layout(); fig.savefig(out / "fig_map.pdf", bbox_inches="tight"); plt.close(fig)
print("map:", len(c), "points")

# ---------------------------------------------------------------- Rel. MAE per series
p = pd.read_csv(res / "perseries.csv")
MODELS = [("tso_final", "tso_final (ref.)"), ("tso_cutoff", "tso_cutoff"), ("chronos2_cov", "chronos2_cov"),
          ("timesfm3_cov", "timesfm3_cov"), ("t0_beta_cov", "t0_beta_cov"), ("tirex2_cov", "tirex2_cov"),
          ("lgbm_cov", "lgbm_cov"), ("chronos2", "chronos2"), ("timesfm3", "timesfm3"), ("naive_daily", "naive_daily")]
TARGETS = ["price", "wind", "solar", "load"]
cols = [(t, z) for t in TARGETS for z in ZONES if ((p.zone == z) & (p.target == t) & (p.model == "naive_weekly")).any()]
M = np.full((len(MODELS), len(cols)), np.nan)
for i, (m, _) in enumerate(MODELS):
    for j, (t, z) in enumerate(cols):
        v = p[(p.model == m) & (p.zone == z) & (p.target == t)].rel
        if len(v) and not (m == "tso_final" and z in ("DK1", "DK2")):   # DK final forecasts: different scope
            M[i, j] = float(v.iloc[0])
fig, ax = plt.subplots(figsize=(6.6, 3.1))
norm = TwoSlopeNorm(vmin=np.log2(0.1), vcenter=0.0, vmax=np.log2(2.5))
im = ax.imshow(np.log2(M), cmap="RdBu_r", norm=norm, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if (v < 0.22 or v > 1.9) else "#1b1b1b")
ax.set_xticks(range(len(cols)), [ZLAB.get(z, z) for _, z in cols], rotation=90, fontsize=6)
ax.set_yticks(range(len(MODELS)), [lab for _, lab in MODELS], fontsize=6.5, family="DejaVu Sans Mono")
ax.tick_params(length=0)
start = 0
for t in TARGETS:
    n = sum(1 for tt, _ in cols if tt == t)
    ax.text(start + (n - 1) / 2, -0.9, t, ha="center", va="bottom", fontsize=7, fontweight="bold")
    if start: ax.axvline(start - 0.5, color="white", lw=1.5)
    start += n
ax.axhline(1.5, color="white", lw=1.5); ax.axhline(6.5, color="white", lw=1.5)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=np.log2([0.125, 0.25, 0.5, 1, 2]))
cb.ax.set_yticklabels(["0.125", "0.25", "0.5", "1", "2"], fontsize=6); cb.outline.set_visible(False)
cb.set_label("Rel. MAE (log scale)", fontsize=6.5)
fig.tight_layout(); fig.savefig(out / "fig_heatmap.pdf", bbox_inches="tight"); plt.close(fig)
print("heatmap:", M.shape)

# ---------------------------------------------------------------- accuracy vs parameter count
pf = out / "params.dat"          # model params(M) hist cov label: Rel. MAE on 16-19 Oct, every model on the same races
if pf.exists():
    d = pd.read_csv(pf, sep=" ")
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for _, r in d.iterrows():
        if np.isfinite(r["cov"]):
            ax.plot([r.params, r.params], [r["hist"], r["cov"]], color="#C9D6E8", lw=1.2, zorder=1)
    ax.scatter(d.params, d["hist"], s=26, facecolor="white", edgecolor="#8A8F98", lw=1.2, zorder=3, label="history only")
    c = d.dropna(subset=["cov"])
    ax.scatter(c.params, c["cov"], s=28, color="#2E6DB4", zorder=4, label="with covariates")
    # small models: label next to the point; the crowded 90-350M group: a label column on the right with leaders
    for _, r in d[d.params < 90].iterrows():
        ax.annotate(f"{r.label.replace('~', ' ')} ({r.params:.1f}M)" if r.params < 10 else f"{r.label.replace('~', ' ')} ({r.params:.0f}M)",
                    (r.params, r["hist"]), xytext=(5, 0), textcoords="offset points",
                    fontsize=5.8, color="#4A4F58", va="center")
    big = d[d.params >= 90].sort_values("hist", ascending=False).reset_index(drop=True)
    ys = np.linspace(0.875, 0.615, len(big))
    for (_, r), y in zip(big.iterrows(), ys):
        ax.plot([r.params, 560], [r["hist"], y], color="#C3C7CE", lw=0.4, zorder=2)
        ax.text(590, y, f"{r.label.replace('~', ' ')} ({r.params:.0f}M)", fontsize=5.8, color="#4A4F58", va="center")
    ax.axhline(1.0, color="#555", lw=0.6, ls="--"); ax.text(2.6, 1.0, "seasonal naive", fontsize=5.8, va="bottom", color="#555")
    ax.set_xscale("log"); ax.set_xlim(2.2, 1700); ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("parameters (millions, log scale)"); ax.set_ylabel("Rel. MAE, 16–19 October")
    ax.grid(True, which="major", color="#E6E8EC", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig_params.pdf", bbox_inches="tight"); plt.close(fig)
    print("params:", len(d))
