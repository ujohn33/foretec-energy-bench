/* Foretec Transparency leaderboard. Reads data/summary.json and data/days/<issue_date>.json. */
(() => {
  "use strict";

  const METRICS = [
    { k: "rel_mae", label: "Rel. MAE", d: 3, unitless: true },
    { k: "mae", label: "MAE", d: 1 },
    { k: "rmse", label: "RMSE", d: 1 },
    { k: "pinball", label: "Pinball", d: 2 },
  ];
  const TARGET_LABEL = { price: "Price", wind: "Wind", solar: "Solar" };
  const S = {
    zone: "all", target: "all", metric: "rel_mae", phase: null, view: "rank", topN: 10, covFilter: "all",
    hidden: new Set(), charts: {}, days: {}, x: { day: null, zone: "BE", target: "price", band: null },
  };
  let D = null;        // summary.json
  let ROWS = [];       // score rows as objects
  let MODELS = [];     // model names in fixed colour order
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const metricDef = () => METRICS.find((m) => m.k === S.metric);
  const fmt = (v, d) => (v == null || !isFinite(v) ? "–" : v.toFixed(d));
  const shortDate = (d) => new Date(d + "T12:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  const addDays = (d, n) => { const t = new Date(d + "T12:00:00Z"); t.setUTCDate(t.getUTCDate() + n); return t.toISOString().slice(0, 10); };

  // ---------- colours: fixed per model, never by rank ----------
  // Colour follows the model family, never its rank: one palette slot per foundation-model family,
  // lighter shades for variants within a family, navy ink shades for the classical baselines.
  const FAMILY_ORDER = ["chronos", "timesfm", "moirai", "tirex", "toto", "sundial", "ttm"];
  let FAM = {};
  function color(m) {
    const fam = FAM[m] || m;
    const members = MODELS.filter((x) => (FAM[x] || x) === fam);
    const k = Math.max(0, members.indexOf(m));
    let base;
    if (fam === "baseline") base = css("--ink");
    else {
      const i = FAMILY_ORDER.indexOf(fam);
      base = i >= 0 ? css(`--s${i + 1}`) : css("--s8");
    }
    return k === 0 ? base : mix(base, css("--surface"), Math.min(0.22 * k, 0.66));
  }
  function hexToRgb(h) {
    if (h.startsWith("rgb")) return h.match(/[\d.]+/g).slice(0, 3).map(Number);
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mix(a, b, w) { // w = weight of b
    const A = hexToRgb(a), B = hexToRgb(b);
    return "rgb(" + A.map((v, i) => Math.round(v * (1 - w) + B[i] * w)).join(",") + ")";
  }
  function alpha(h, a) { const [r, g, b] = hexToRgb(h); return `rgba(${r},${g},${b},${a})`; }
  const btColor = (m) => mix(color(m), css("--bt"), 0.62);

  // ---------- data ----------
  // a model "uses covariates" when it takes anything beyond its own history and the calendar
  const COVARIATE_INPUTS = ["nwp", "load_forecast", "fuel"];
  let USES_COV = {};
  const inField = (m) => S.covFilter === "all" || (S.covFilter === "with") === !!USES_COV[m];
  function filteredRows(metric = S.metric) {
    return ROWS.filter((r) => inField(r.model) && (S.zone === "all" || r.zone === S.zone) && (S.target === "all" || r.target === S.target) && r[metric] != null);
  }
  function phaseOf(d) { return (D.days && D.days[d]) || (ROWS.find((r) => r.issue_date === d) || {}).phase || "backtest"; }

  /** Per issue day: average rank across the selected series, raw mean, and championship points. */
  function dayTable(rows) {
    const byDay = new Map();
    for (const r of rows) {
      if (!byDay.has(r.issue_date)) byDay.set(r.issue_date, new Map());
      const ser = byDay.get(r.issue_date);
      const key = r.zone + "|" + r.target;
      if (!ser.has(key)) ser.set(key, []);
      ser.get(key).push(r);
    }
    const out = [];
    for (const [d, ser] of [...byDay.entries()].sort()) {
      const ranks = {}, vals = {};
      for (const list of ser.values()) {
        const sorted = [...list].sort((a, b) => a[S.metric] - b[S.metric]);
        sorted.forEach((r, i) => {
          // ties share the average position
          const tied = sorted.filter((x) => x[S.metric] === r[S.metric]).map((x) => sorted.indexOf(x) + 1);
          (ranks[r.model] ||= []).push(mean(tied));
          (vals[r.model] ||= []).push(r[S.metric]);
        });
      }
      const avgRank = {}, raw = {}, points = {};
      for (const m in ranks) { avgRank[m] = mean(ranks[m]); raw[m] = mean(vals[m]); }
      const order = Object.keys(avgRank).sort((a, b) => avgRank[a] - avgRank[b]);
      order.forEach((m, i) => { points[m] = order.length - i; });
      out.push({ d, phase: phaseOf(d), avgRank, raw, points, n: order.length });
    }
    return out;
  }

  function standings(days, phase) {
    const sel = days.filter((x) => phase === "all" || x.phase === phase);
    const set = new Set(sel.map((x) => x.d));
    const relRows = filteredRows("rel_mae").filter((r) => set.has(r.issue_date));
    const metRows = filteredRows().filter((r) => set.has(r.issue_date));
    const res = {};
    for (const x of sel) {
      for (const m in x.points) {
        const s = (res[m] ||= { model: m, pts: 0, ranks: [], ptsList: [] });
        s.pts += x.points[m]; s.ranks.push(x.avgRank[m]); s.ptsList.push(x.points[m]);
      }
    }
    return Object.values(res).map((s) => ({
      ...s,
      avgRank: mean(s.ranks),
      best: Math.max(...s.ptsList), worst: Math.min(...s.ptsList), days: s.ptsList.length,
      rel: mean(relRows.filter((r) => r.model === s.model).map((r) => r.rel_mae)),
      metric: mean(metRows.filter((r) => r.model === s.model).map((r) => r[S.metric])),
    })).sort((a, b) => b.pts - a.pts || a.avgRank - b.avgRank);
  }

  const narrow = () => window.innerWidth < 640;
  const singleSeries = () => S.zone !== "all" && S.target !== "all";
  const rawAllowed = () => S.metric === "rel_mae" || singleSeries();
  function unitSuffix() {
    if (metricDef().unitless || S.target === "all") return "";
    return " (" + D.config.targets[S.target].unit + ")";
  }

  // ---------- controls ----------
  function seg(el, items, current, onPick) {
    el.innerHTML = items.map((it) =>
      `<button type="button" role="radio" data-k="${it.k}" aria-checked="${it.k === current}" ${it.disabled ? "disabled" : ""} ${it.title ? `title="${esc(it.title)}"` : ""}>${esc(it.label)}</button>`).join("");
    el.onclick = (e) => { const b = e.target.closest("button"); if (b && !b.disabled) onPick(b.dataset.k); };
  }
  function options(el, items, current) {
    el.innerHTML = items.map((it) => `<option value="${esc(it.k)}" ${it.k === current ? "selected" : ""}>${esc(it.label)}</option>`).join("");
  }

  function renderControls() {
    options($("f-zone"), [{ k: "all", label: "All zones" }, ...D.config.zones.map((z) => ({ k: z, label: z }))], S.zone);
    options($("f-target"), [{ k: "all", label: "All targets" }, ...Object.keys(D.config.targets).map((t) => ({ k: t, label: TARGET_LABEL[t] || t }))], S.target);
    seg($("f-metric"), METRICS.map((m) => ({ k: m.k, label: m.label })), S.metric, (k) => { S.metric = k; if (!rawAllowed()) S.view = "rank"; renderAll(); });
    const hasLive = ROWS.some((r) => r.phase === "live");
    seg($("f-phase"), [
      { k: "live", label: "Live", disabled: !hasLive, title: hasLive ? "" : "No live forecasts scored yet" },
      { k: "backtest", label: "Backtest" },
      { k: "all", label: "All" },
    ], S.phase, (k) => { S.phase = k; renderAll(); });
    seg($("f-cov"), [
      { k: "all", label: "All" },
      { k: "with", label: "With", title: "Models that also use NWP weather, the ENTSO-E load forecast or fuel cost" },
      { k: "without", label: "Without", title: "Models that only see the target's own history" },
    ], S.covFilter, (k) => { S.covFilter = k; renderAll(); renderExplorerChart(); });
    seg($("f-view"), [
      { k: "rank", label: "Avg rank" },
      { k: "value", label: "Value", disabled: !rawAllowed(), title: rawAllowed() ? "" : "Pick one zone and one target to compare raw values across days" },
    ], S.view, (k) => { S.view = k; renderAll(); });
  }

  // ---------- chart plumbing ----------
  const backtestBand = {
    id: "btBand",
    beforeDatasetsDraw(chart, _a, opts) {
      if (!opts || opts.from == null) return;
      const { ctx, chartArea: a, scales: { x } } = chart;
      const step = x.getPixelForValue(1) - x.getPixelForValue(0) || 0;
      const x0 = Math.max(a.left, x.getPixelForValue(opts.from) - step / 2);
      const x1 = Math.min(a.right, x.getPixelForValue(opts.to) + step / 2);
      ctx.save();
      ctx.fillStyle = css("--band");
      ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      ctx.fillStyle = css("--bt-ink");
      ctx.font = "10px 'Geist Mono', monospace";
      ctx.fillText("BACKTEST", x0 + 8, a.top + 14);
      if (x1 < a.right - 40) ctx.fillText("LIVE", x1 + 8, a.top + 14);
      ctx.restore();
    },
  };
  const endLabels = {
    id: "endLabels",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.on) return;
      const { ctx } = chart;
      const placed = [];
      ctx.save();
      ctx.font = "11px 'Geist Mono', monospace";
      ctx.textBaseline = "middle";
      chart.data.datasets.forEach((ds, i) => {
        if (!ds.endLabel || !chart.isDatasetVisible(i)) return;
        const meta = chart.getDatasetMeta(i);
        let k = ds.data.length - 1;
        while (k >= 0 && ds.data[k] == null) k--;
        if (k < 0) return;
        const p = meta.data[k];
        let y = p.y;
        while (placed.some((q) => Math.abs(q - y) < 13)) y += 13;
        placed.push(y);
        ctx.fillStyle = ds.endColor;
        ctx.beginPath(); ctx.arc(p.x + 9, y, 3.5, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = css("--ink2");
        ctx.fillText(ds.endLabel, p.x + 16, y);
      });
      ctx.restore();
    },
  };

  function baseOptions(extra = {}) {
    const ink2 = css("--ink2"), grid = css("--grid"), axis = css("--rule");
    return {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      layout: { padding: { right: narrow() ? 6 : extra.rightPad ?? 120, top: 4 } },
      scales: {
        x: { grid: { color: grid, drawTicks: false }, border: { color: axis }, ticks: { color: ink2, font: { family: "Geist Mono", size: 11 }, maxRotation: 0, autoSkipPadding: 14, padding: 6 } },
        y: { grid: { color: grid, drawTicks: false }, border: { display: false }, ticks: { color: ink2, font: { family: "Geist Mono", size: 11 }, padding: 8 }, title: { display: !!extra.yTitle, text: extra.yTitle, color: ink2, font: { family: "Geist Mono", size: 10 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: css("--surface"), titleColor: css("--ink"), bodyColor: css("--ink2"), borderColor: css("--rule"), borderWidth: 1,
          padding: 10, boxPadding: 4, usePointStyle: true, titleFont: { family: "Archivo", weight: 600 }, bodyFont: { family: "Geist Mono", size: 11 },
          itemSort: extra.itemSort,
          callbacks: extra.tooltip || {},
        },
        btBand: extra.band || {},
        endLabels: { on: extra.endLabels !== false && !narrow() },
      },
    };
  }
  function draw(id, cfg) {
    if (S.charts[id]) S.charts[id].destroy();
    S.charts[id] = new Chart($(id).getContext("2d"), { ...cfg, plugins: [backtestBand, endLabels] });
  }
  function legend(el, models, extra = "") {
    el.innerHTML = extra + models.map((m) =>
      `<button type="button" data-m="${esc(m)}" aria-pressed="${!S.hidden.has(m)}"><i style="border-top-color:${color(m)}"></i>${esc(m)}</button>`).join("");
    el.onclick = (e) => {
      const b = e.target.closest("button[data-m]"); if (!b) return;
      const m = b.dataset.m; S.hidden.has(m) ? S.hidden.delete(m) : S.hidden.add(m);
      renderCharts(); renderExplorerChart();
    };
  }
  function bandRange(days) {
    const idx = days.map((x, i) => (x.phase === "backtest" ? i : -1)).filter((i) => i >= 0);
    return idx.length ? { from: idx[0], to: idx[idx.length - 1] } : {};
  }
  function lineDataset(m, data, days) {
    const isBt = (i) => days[i] && days[i].phase === "backtest";
    const c = color(m), g = btColor(m);
    return {
      label: m, data, hidden: S.hidden.has(m), endLabel: m, endColor: c,
      borderColor: c, backgroundColor: c, borderWidth: 2, tension: 0.25, spanGaps: true,
      pointRadius: 3, pointHoverRadius: 5, pointBorderWidth: 2, pointBorderColor: css("--surface"),
      pointBackgroundColor: data.map((_, i) => (isBt(i) ? g : c)),
      segment: { borderColor: (ctx) => (isBt(ctx.p1DataIndex) ? g : c), borderDash: (ctx) => (isBt(ctx.p1DataIndex) ? [5, 4] : undefined) },
    };
  }

  // ---------- sections ----------
  let DAYS = [];

  function renderStandings() {
    const st = standings(DAYS, S.phase).filter((x) => shown(x.model));
    const bt = S.phase === "backtest";
    const md = metricDef();
    $("standings").classList.toggle("is-bt", bt);
    const top = st.slice(0, 3);
    $("podium").innerHTML = top.length ? top.map((s, i) => `
      <div class="pod ${i === 0 ? "lead" : ""}">
        <div class="pod-top"><span class="place">${["1st", "2nd", "3rd"][i]} place</span>${bt ? '<span class="tag bt">Backtest</span>' : '<span class="tag live">Live</span>'}</div>
        <div class="name"><i style="background:${color(s.model)}"></i>${esc(s.model)}</div>
        <div class="stats">
          <div class="stat"><span>Points</span><b>${s.pts}</b></div>
          <div class="stat"><span>Avg rank</span><b>${fmt(s.avgRank, 2)}</b></div>
          <div class="stat"><span>${esc(md.label)}</span><b>${fmt(s.metric, md.d)}</b></div>
        </div>
      </div>`).join("") : `<div class="pod" style="grid-column:1/-1;color:var(--ink2)">No scored days for this selection yet.</div>`;

    const maxPts = Math.max(1, ...st.map((s) => s.pts));
    const t = $("standings-table");
    t.innerHTML = `<thead><tr><th>#</th><th>Model</th><th class="r">Total pts</th><th class="r">Avg rank</th><th class="r">Rel. MAE</th>
      ${S.metric !== "rel_mae" ? `<th class="r">${esc(md.label)}${esc(unitSuffix())}</th>` : ""}
      <th class="r">Best day</th><th class="r">Worst day</th><th class="r">Days</th><th>Points</th></tr></thead><tbody>` +
      st.map((s, i) => `<tr class="${bt ? "is-bt" : ""}">
        <td class="pos">${String(i + 1).padStart(2, "0")}</td>
        <td class="model"><i style="background:${color(s.model)}"></i>${esc(s.model)}</td>
        <td class="r v">${s.pts}</td><td class="r v">${fmt(s.avgRank, 2)}</td><td class="r v">${fmt(s.rel, 3)}</td>
        ${S.metric !== "rel_mae" ? `<td class="r v">${fmt(s.metric, md.d)}</td>` : ""}
        <td class="r v">${s.best}</td><td class="r v">${s.worst}</td><td class="r v">${s.days}</td>
        <td><div class="bar"><span style="width:${Math.round(140 * s.pts / maxPts)}px;background:${bt ? btColor(s.model) : color(s.model)}"></span><b>${s.pts}</b></div></td>
      </tr>`).join("") + "</tbody>";
    if (!st.length) t.innerHTML = "";
  }

  function renderCharts() {
    const labels = DAYS.map((x) => shortDate(x.d));
    const band = bandRange(DAYS);
    const md = metricDef();
    const rank = S.view === "rank";
    const models = MODELS.filter((m) => shown(m) && DAYS.some((x) => x.avgRank[m] != null));
    const phaseNote = (i) => (DAYS[i] && DAYS[i].phase === "backtest" ? " · backtest" : " · live");
    $("perf-sub").textContent = rank
      ? "Average rank per issue day across the selected series; lower is better (1 = best)."
      : `Mean ${md.label}${unitSuffix()} per issue day; lower is better.`;

    draw("c-perf", {
      type: "line",
      data: { labels, datasets: models.map((m) => lineDataset(m, DAYS.map((x) => (rank ? x.avgRank[m] : x.raw[m]) ?? null), DAYS)) },
      options: (() => {
        const o = baseOptions({
          band, yTitle: rank ? "Average rank" : md.label + unitSuffix(),
          itemSort: (a, b) => a.raw - b.raw,
          tooltip: { title: (it) => DAYS[it[0].dataIndex].d + phaseNote(it[0].dataIndex), label: (c) => ` ${c.dataset.label}: ${fmt(c.raw, rank ? 2 : md.d)}` },
        });
        if (rank) { o.scales.y.reverse = true; o.scales.y.min = 1; o.scales.y.ticks.stepSize = 1; }
        return o;
      })(),
    });
    legend($("l-perf"), models);

    const race = models.map((m) => { let c = 0; return lineDataset(m, DAYS.map((x) => { c += x.points[m] || 0; return c; }), DAYS); });
    draw("c-race", {
      type: "line", data: { labels, datasets: race },
      options: baseOptions({
        band, yTitle: "Cumulative points", itemSort: (a, b) => b.raw - a.raw,
        tooltip: { title: (it) => DAYS[it[0].dataIndex].d + phaseNote(it[0].dataIndex), label: (c) => ` ${c.dataset.label}: ${c.raw} pts` },
      }),
    });
    legend($("l-race"), models);
  }

  function renderHeatmap() {
    const st = standings(DAYS, "all").filter((x) => shown(x.model));
    const t = $("heatmap");
    if (!DAYS.length) { t.innerHTML = ""; return; }
    const lerp = (a, b, w) => mix(a, b, w);
    const head = `<thead><tr><th>Model</th>${DAYS.map((x) => `<th class="${x.phase === "backtest" ? "is-bt" : "live"}" title="${x.d} · ${x.phase}">${esc(shortDate(x.d))}</th>`).join("")}<th>Avg</th></tr></thead>`;
    const body = st.map((s) => {
      const cells = DAYS.map((x) => {
        const r = x.avgRank[s.model];
        if (r == null) return `<td class="cell">–</td>`;
        const w = x.n > 1 ? (x.n - r) / (x.n - 1) : 1;
        const bt = x.phase === "backtest";
        const bg = lerp(css(bt ? "--heat-bt-lo" : "--heat-live-lo"), css(bt ? "--heat-bt-hi" : "--heat-live-hi"), w);
        const dark = matchMedia("(prefers-color-scheme: dark)").matches;
        const fg = (dark ? w < 0.55 : w > 0.55) ? "#FFFFFF" : "#0B1F3A";
        return `<td class="cell" style="background:${bg};color:${fg}" title="${esc(s.model)} · ${x.d} (${x.phase}): avg rank ${fmt(r, 2)} of ${x.n}">${fmt(r, 1)}</td>`;
      }).join("");
      return `<tr><td class="model"><i style="background:${color(s.model)}"></i>${esc(s.model)}</td>${cells}<td class="avg">${fmt(s.avgRank, 1)}</td></tr>`;
    }).join("");
    t.innerHTML = head + "<tbody>" + body + "</tbody>";
  }

  function renderDetails() {
    const md = metricDef();
    const set = new Set(DAYS.filter((x) => S.phase === "all" || x.phase === S.phase).map((x) => x.d));
    const rows = filteredRows().filter((r) => set.has(r.issue_date));
    const models = MODELS.filter((m) => shown(m) && rows.some((r) => r.model === m));
    const series = [];
    for (const t of Object.keys(D.config.targets)) for (const z of D.config.zones)
      if ((S.zone === "all" || S.zone === z) && (S.target === "all" || S.target === t)) series.push([z, t]);
    $("details-sub").textContent = `${md.label}, mean over ${set.size} issue day${set.size === 1 ? "" : "s"} (${S.phase === "all" ? "live and backtest" : S.phase}). Best per row in bold.`;
    const bt = S.phase === "backtest";
    $("details-table").innerHTML = `<thead><tr><th>Series</th><th>Unit</th>${models.map((m) => `<th class="r">${esc(m)}</th>`).join("")}</tr></thead><tbody>` +
      series.map(([z, t]) => {
        const vals = models.map((m) => mean(rows.filter((r) => r.model === m && r.zone === z && r.target === t).map((r) => r[S.metric])));
        const best = Math.min(...vals.filter((v) => v != null));
        return `<tr class="${bt ? "is-bt" : ""}"><td class="model">${z} ${TARGET_LABEL[t] || t}</td><td>${md.unitless ? "ratio" : esc(D.config.targets[t].unit)}</td>` +
          vals.map((v) => `<td class="num ${v === best ? "best" : ""}">${fmt(v, md.d)}</td>`).join("") + "</tr>";
      }).join("") + "</tbody>";
  }

  function renderModels() {
    const live = new Set(ROWS.filter((r) => r.phase === "live").map((r) => r.model));
    $("models-table").innerHTML = `<thead><tr><th>Model</th><th>Author</th><th>Description</th><th>Inputs</th><th>Implementation</th><th>Status</th></tr></thead><tbody>` +
      D.models.map((m) => `<tr><td class="model"><i style="background:${color(m.name)}"></i>${esc(m.name)}</td><td>${esc(m.author)}</td><td class="desc">${esc(m.description)}</td>
        <td class="wrap">${inputChips(m.inputs)}</td><td class="kind">${esc(m.kind)}</td><td class="${m.enabled === false ? "wrap" : ""}">${m.enabled === false ? `<span class="tag bt">Retired</span><br><small>${esc(m.retired || "")}</small>` : live.has(m.name) ? '<span class="tag live">Live</span>' : '<span class="tag bt">Backtest only</span>'}</td></tr>`).join("") + "</tbody>";
  }

  const INPUT_LABEL = { history: "Own history", nwp: "NWP ensemble", load_forecast: "ENTSO-E load fc", fuel: "Fuel cost", calendar: "Calendar" };
  function inputChips(inputs) {
    return (inputs || ["history"]).map((k) => `<span class="chip ${k === "history" ? "" : "on"}">${esc(INPUT_LABEL[k] || k)}</span>`).join("");
  }

  // ---------- inputs: what the models knew ----------
  const NWP_STYLE = { icon_eu: [], gfs_seamless: [6, 4], ecmwf_ifs: [2, 3] };
  function fmtUtc(s) { return s ? String(s).replace("T", " ").slice(0, 16) + " UTC" : "–"; }
  const SRC_NAME = { entsoe: "ENTSO-E", energycharts: "Energy-Charts" };
  function provenance(p) {
    if (!p) return null;
    const parts = Object.entries(p).filter(([k, n]) => k !== "missing" && n > 0);
    return parts.length === 1 ? SRC_NAME[parts[0][0]] || parts[0][0] : parts.map(([k, n]) => `${SRC_NAME[k] || k} ${n.toLocaleString("en-GB")}`).join(" + ");
  }
  function historyRow(day) {
    const hs = day.history_sources;
    const lines = Object.keys(D.config.targets).map((t) => {
      const p = provenance(hs && hs[`${S.x.zone}_${t}`]);
      return `${TARGET_LABEL[t] || t}: ${p ? esc(p) + (p.includes("+") ? " quarter-hours" : "") : day.phase === "backtest" || !hs ? "Energy-Charts" : "–"}`;
    }).join("<br>");
    const note = hs ? "ENTSO-E Transparency Platform (A44 prices, A75 generation per type); gaps from Energy-Charts"
                    : "Energy-Charts (issue days up to 3 Oct 2026, and backtests, which do not store their snapshots)";
    return `<tr><td>Target history</td><td class="wrap">${note}</td><td class="wrap">prices through the end of D-1; wind and solar through the cut-off minus 1 h</td><td class="wrap">${lines}</td></tr>`;
  }
  function renderInputs(day) {
    const inp = day && day.inputs;
    const t = $("inputs-table");
    $("cov-zone").textContent = S.x.zone;
    const head = `<thead><tr><th>Input</th><th>Vintage used</th><th>Why it is admissible</th><th>Fetched / source</th></tr></thead>`;
    if (!inp) {
      t.innerHTML = head + `<tbody>${day ? historyRow(day) : ""}<tr><td colspan="4" class="empty">No covariate model ran for this issue day, so there are no frozen covariates to show.</td></tr></tbody>`;
      $("cov-card").style.display = "none";
      return;
    }
    $("cov-card").style.display = "";
    const bt = day.phase === "backtest";
    const lagNote = (m) => `init ≤ cut-off − ${D.covariates.nwp_models[m].lag_hours} h`;
    const rows = [
      ...inp.runs.map((r) => [inp.nwp_models[r.model].label, `run ${fmtUtc(r.run_utc)}`, r.status === "ok" ? lagNote(r.model) : `<b>${esc(r.status)}</b>`,
        r.fetched_utc ? fmtUtc(r.fetched_utc) + (bt ? " (archive)" : "") : "–"]),
      ["Load forecast", esc(inp.load_forecast), bt ? "backtest: currently published version" : "fetched at the cut-off", "ENTSO-E A65"],
      ["Fuel cost", esc(inp.fuel), "quotes count from the end of their date", "oilpriceapi"],
    ];
    t.innerHTML = head + `<tbody>` +
      `<tr><td>Cut-off / gate</td><td class="wrap">${fmtUtc(inp.cutoff_utc)} · gate ${esc(D.config.gate)} Brussels</td><td class="wrap">${inp.n_columns} covariate columns frozen at the cut-off</td>` +
      `<td class="wrap"><a href="${inp.download.covariates}">covariates (CSV)</a> · <a href="${inp.download.meta}">run log (JSON)</a></td></tr>` +
      historyRow(day) +
      rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "wrap" : ""}">${c}</td>`).join("")}</tr>`).join("") + "</tbody>";

    const shown = inp.shown[S.x.zone] || {};
    const keys = Object.keys(shown);
    if (!keys.includes(S.cov)) S.cov = keys[0];
    options($("cov-var"), keys.map((k) => ({ k, label: shown[k].label })), S.cov);
    const v = shown[S.cov];
    if (!v) return;
    const t0 = new Date(day.t0);
    const tf = new Intl.DateTimeFormat("en-GB", { timeZone: D.config.timezone, hour: "2-digit", minute: "2-digit" });
    const labels = Array.from({ length: day.n }, (_, i) => tf.format(new Date(t0.getTime() + i * 900e3)));
    const members = Object.keys(v.members);
    const ink = css("--ink"), ink2 = css("--ink2");
    const datasets = [];
    if (members.length > 1) {
      const lo = labels.map((_, i) => Math.min(...members.map((m) => v.members[m][i] ?? Infinity)));
      const hi = labels.map((_, i) => Math.max(...members.map((m) => v.members[m][i] ?? -Infinity)));
      datasets.push({ label: "_lo", data: lo, borderWidth: 0, pointRadius: 0, fill: false });
      datasets.push({ label: "_hi", data: hi, borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: alpha(css("--blue"), 0.14) });
      const mean = labels.map((_, i) => { const xs = members.map((m) => v.members[m][i]).filter((x) => x != null); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; });
      datasets.push({ label: "Ensemble mean", data: mean, borderColor: ink, borderWidth: 2.6, pointRadius: 0, tension: 0.15, endLabel: "mean", endColor: ink });
    }
    for (const m of members) {
      const lab = (inp.nwp_models[m] && inp.nwp_models[m].label) || (m === "entsoe" ? "ENTSO-E" : m);
      datasets.push({ label: lab, data: v.members[m], borderColor: css("--blue-ink"), borderDash: NWP_STYLE[m] || [], borderWidth: 1.5,
        pointRadius: 0, tension: 0.15, endLabel: lab.split(" ")[0], endColor: css("--blue-ink") });
    }
    const o = baseOptions({ yTitle: `${v.label} (${v.unit})`, tooltip: { label: (c) => (c.dataset.label.startsWith("_") ? null : ` ${c.dataset.label}: ${fmt(c.raw, 1)} ${v.unit}`) } });
    o.plugins.tooltip.filter = (it) => !it.dataset.label.startsWith("_");
    o.scales.x.ticks.maxTicksLimit = 13;
    draw("c-cov", { type: "line", data: { labels, datasets }, options: o });
    $("l-cov").innerHTML = (members.length > 1 ? `<span class="actual" style="display:flex;align-items:center;gap:7px"><i></i>Ensemble mean</span>
      <span style="display:flex;align-items:center;gap:7px"><i style="border-top:8px solid ${alpha(css("--blue"), 0.25)}"></i>Min–max across models (disagreement)</span>` : "") +
      members.map((m) => `<span style="display:flex;align-items:center;gap:7px"><i style="border-top:2px ${m === "gfs_seamless" ? "dashed" : m === "ecmwf_ifs" ? "dotted" : "solid"} ${css("--blue-ink")}"></i>${esc((inp.nwp_models[m] && inp.nwp_models[m].label) || m)}</span>`).join("");
    $("cov-note").textContent = `Delivery day ${addDays(day.issue_date, 1)}, ${S.x.zone}, capacity-weighted over the centroids below. Values exactly as frozen at the cut-off.`;
  }

  const BUCKET_LABEL = { wind_onshore: "Wind onshore", wind_offshore: "Wind offshore", solar: "Solar", load: "Load centres (population)" };
  const BUCKET_SLOT = { wind_onshore: "--s1", wind_offshore: "--s3", solar: "--s4", load: "--ink2" };
  let MAP = null;
  function renderCentroids() {
    const C = D.covariates && D.covariates.centroids;
    if (!C || !C.length) {
      $("map").outerHTML = `<p class="empty">Centroids are not published yet.</p>`;
      return;
    }
    const groups = {};
    for (const c of C) { const k = c.zone + "|" + c.bucket; (groups[k] ||= { zone: c.zone, bucket: c.bucket, n: 0, w: 0, units: 0, src: new Set() }); const g = groups[k]; g.n++; g.w += c.weight; g.units += c.n; g.src.add(c.source); }
    $("centroid-table").innerHTML = `<thead><tr><th>Zone</th><th>Bucket</th><th class="r">Centroids</th><th class="r">Total weight</th><th class="r">Units</th><th>Source</th></tr></thead><tbody>` +
      Object.values(groups).map((g) => `<tr><td>${g.zone}</td><td>${BUCKET_LABEL[g.bucket] || g.bucket}</td><td class="r v">${g.n}</td>
        <td class="r v">${g.bucket === "load" ? (g.w / 1e6).toFixed(1) + " M people" : (g.w / 1000).toFixed(1) + " GW"}</td><td class="r v">${g.units.toLocaleString("en-GB")}</td><td class="kind">${esc([...g.src].join(", "))}</td></tr>`).join("") + "</tbody>";
    $("l-map").innerHTML = Object.keys(BUCKET_LABEL).map((b) => `<span style="display:flex;align-items:center;gap:7px"><i class="dot" style="background:${css(BUCKET_SLOT[b])}"></i>${BUCKET_LABEL[b]}</span>`).join("");
    if (typeof L === "undefined") return;
    if (!MAP) {
      MAP = L.map("map", { scrollWheelZoom: false, attributionControl: true });
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 10, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(MAP);
    }
    const maxW = {};
    for (const c of C) maxW[c.bucket] = Math.max(maxW[c.bucket] || 0, c.weight);
    const layer = L.layerGroup().addTo(MAP);
    for (const c of C) {
      L.circleMarker([c.lat, c.lon], { radius: 4 + 14 * Math.sqrt(c.weight / maxW[c.bucket]), color: css(BUCKET_SLOT[c.bucket]), weight: 1.5, fillOpacity: 0.35 })
        .bindTooltip(`${c.zone} · ${BUCKET_LABEL[c.bucket]}<br>${c.bucket === "load" ? Math.round(c.weight).toLocaleString("en-GB") + " people" : Math.round(c.weight) + " MW"} · ${c.n} units<br>${esc(c.source)}`)
        .addTo(layer);
    }
    MAP.fitBounds(C.map((c) => [c.lat, c.lon]), { padding: [20, 20] });
  }

  function actualSource(day, key) {
    const p = provenance(day.actual_sources && day.actual_sources[key]);
    return p ? p + (p.includes("+") ? " quarter-hours" : "") : "Energy-Charts";
  }

  // ---------- pipeline: timeline diagram ----------
  function hm(t) { const [h, m] = String(t).split(":").map(Number); return h + m / 60; }
  function brusselsOffset() {
    const now = new Date();
    const loc = new Date(now.toLocaleString("en-US", { timeZone: D.config.timezone }));
    const utc = new Date(now.toLocaleString("en-US", { timeZone: "UTC" }));
    return Math.round((loc - utc) / 36e5);
  }
  function renderTimeline() {
    const T = D.timeline;
    if (!T) return;
    // hours relative to D-1 00:00 Brussels; the D-1 morning around cut-off and gate gets most of the width
    const H = [-24, 0, 7, 16, 24, 48, 72, 96], F = [0, 0.07, 0.11, 0.56, 0.62, 0.80, 0.90, 1];
    const W = 1000, L = 128, R = 18, top = 54, laneH = 64;
    const x = (h) => { let i = 0; while (i < H.length - 2 && h > H[i + 1]) i++; const f = F[i] + (F[i + 1] - F[i]) * (Math.min(Math.max(h, H[i]), H[i + 1]) - H[i]) / (H[i + 1] - H[i]); return L + f * (W - L - R); };
    const lanes = ["Price", "Wind", "Solar", "Inputs"];
    const y = (i) => top + i * laneH;
    const Hh = top + lanes.length * laneH + 8;
    const cut = hm(T.issue_time), gate = hm(T.gate), off = brusselsOffset();
    const sched = T.schedule || {};
    const scores = (sched.score || ["14:45"]).map(hm);
    const out = [];
    const txt = (xx, yy, t, cls = "", anchor = "middle") => out.push(`<text x="${xx.toFixed(1)}" y="${yy}" text-anchor="${anchor}" class="${cls}">${esc(t)}</text>`);
    // day bands
    [[-24, 0, "D−2"], [0, 24, "D−1 · issue day"], [24, 48, "D · delivery"], [48, 72, "D+1"], [72, 96, "D+2"]].forEach(([a, b, lab], i) => {
      out.push(`<rect class="band${a === 24 ? " delivery" : ""}" x="${x(a)}" y="${top - 6}" width="${x(b) - x(a)}" height="${Hh - top}" opacity="${a === 24 ? 1 : i % 2 ? 0.55 : 0}"/>`);
      txt((x(a) + x(b)) / 2, 18, lab, "day");
    });
    // zoomed hours on D-1
    [8, 10, 12, 14].forEach((h) => { out.push(`<line class="grid" x1="${x(h)}" x2="${x(h)}" y1="${top - 6}" y2="${Hh}" opacity=".5"/>`); txt(x(h), top - 12, `${String(h).padStart(2, "0")}:00`); });
    [7, 16].forEach((h) => out.push(`<line class="break" x1="${x(h)}" x2="${x(h)}" y1="${top - 6}" y2="${Hh}"/>`));
    lanes.forEach((lab, i) => { txt(16, y(i) + laneH / 2 + 4, lab, "lane", "start"); out.push(`<line class="grid" x1="${L}" x2="${W - R}" y1="${y(i) + laneH}" y2="${y(i) + laneH}"/>`); });
    const bar = (i, a, b, cls, label, dy = 0) => {
      const yy = y(i) + 14 + dy;
      out.push(`<rect class="${cls}" x="${x(a)}" y="${yy}" width="${Math.max(2, x(b) - x(a))}" height="12" rx="1"/>`);
      if (label) txt(x(a) + 6, yy + 25, label, "", "start");
    };
    const mark = (i, h, cls, label, dy = 0, shape = "circle") => {
      const yy = y(i) + 20 + dy, xx = x(h);
      out.push(shape === "diamond" ? `<path class="mk ${cls}" d="M${xx} ${yy - 6} L${xx + 6} ${yy} L${xx} ${yy + 6} L${xx - 6} ${yy} Z"/>` : `<circle class="mk ${cls}" cx="${xx}" cy="${yy}" r="5"/>`);
      if (label) txt(xx, yy + 22, label);
    };
    // price: everything up to the end of D-1 is known (cleared on D-2); D's prices clear just after the gate
    bar(0, -24, 24, "hist", "history: prices to the end of D−1 (cleared on D−2)");
    mark(0, 12.9, "pub", "D prices published ~12:55", -2);
    scores.forEach((h, k) => mark(0, h, "score", k ? "" : `scored ${sched.score.join(" / ")}`, -2, "diamond"));
    // wind and solar: measured up to the cut-off minus the publication lag; actuals arrive during D
    ["wind", "solar"].forEach((t, k) => {
      const lag = (T.lag_hours[t] || 0), days = T.score_after_days[t] || 2;
      bar(k + 1, -24, cut - lag, "hist", `history: metered to ${String(Math.floor(cut - lag)).padStart(2, "0")}:${String(Math.round(((cut - lag) % 1) * 60)).padStart(2, "0")} (cut-off − ${lag} h)`);
      bar(k + 1, 24 + lag, 48 + lag, "act", `actuals metered during D (~${lag} h lag)`);
      mark(k + 1, 24 * (days + 1) + scores[0], "score", `scored D+${days} ${sched.score ? sched.score[0] : ""}`, -2, "diamond");
    });
    // inputs: the runs the cut-off rule admits, the ENTSO-E load forecast and the fuel quote
    const runs = {};
    for (const [m, v] of Object.entries(T.nwp)) {
      const initUtc = Math.floor((cut - off - v.lag_hours) / v.cycle_hours) * v.cycle_hours;
      (runs[initUtc] ||= []).push(v.label.split(" ")[0]);
    }
    Object.entries(runs).forEach(([u, labs], k) => mark(3, Number(u) + off, "in", `${labs.join(", ")} ${String(u).padStart(2, "0")} UTC`, k % 2 ? 20 : -2));
    mark(3, gate - 2, "in", "ENTSO-E load fc ≤ " + `${String(gate - 2).padStart(2, "0")}:00`, 20);
    mark(3, -1, "in", "fuel: quote dated ≤ D−2", -2);
    // cut-off and gate on top of everything
    out.push(`<line class="cut" x1="${x(cut)}" x2="${x(cut)}" y1="${top - 30}" y2="${Hh}"/>`);
    out.push(`<line class="gate" x1="${x(gate)}" x2="${x(gate)}" y1="${top - 30}" y2="${Hh}"/>`);
    txt(x(cut) - 4, top - 26, `cut-off ${T.issue_time}`, "", "end");
    txt(x(gate) + 4, top - 26, `gate ${T.gate}`, "", "start");
    const svg = $("timeline");
    svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
    svg.innerHTML = out.join("");
    $("l-timeline").innerHTML = [
      `<span><i class="sw" style="background:${alpha(css("--blue"), 0.4)}"></i>history a model may use</span>`,
      `<span><i class="sw" style="background:${alpha(css("--green"), 0.6)}"></i>actuals being metered</span>`,
      `<span><i style="border-top:2px dashed ${css("--blue-ink")}"></i>data cut-off, models run</span>`,
      `<span><i style="border-top:2px solid ${css("--amber")}"></i>day-ahead gate closure</span>`,
      `<span>○ input available · ● actuals published · ◆ scored</span>`,
      `<span>Dashed grey lines: the axis is stretched between them. Times are Brussels local.</span>`,
    ].join("");
  }

  // ---------- pipeline: run tracker ----------
  function cellStatus(c, day) {
    if (!c) return null;
    if (c.err || c.skip) return { k: "err", icon: "✕", label: `error: ${c.err} failed, ${c.skip} skipped` };
    if (day.phase === "live" && day.on_time === false) return { k: "late", icon: "⏱", label: "finished after the gate, not scored" };
    if (c.due === 0) return { k: "wait", icon: "…", label: "forecasts generated, scoring not due yet" };
    if (c.scored < c.due) return { k: "miss", icon: "!", label: `scored ${c.scored} of ${c.due} due series (awaiting actuals)` };
    if (c.due < c.ok) return { k: "part", icon: "◐", label: `scored ${c.scored} of ${c.ok}; the rest is not due yet` };
    return { k: "ok", icon: "✓", label: `all ${c.ok} series forecast and scored` };
  }
  const localTime = (u) => (u ? new Date(String(u).replace(" ", "T").replace(/\+00:00$/, "Z").replace(/(\d)$/, "$1Z"))
    .toLocaleTimeString("en-GB", { timeZone: D.config.timezone, hour: "2-digit", minute: "2-digit" }) : "–");
  function renderTracker() {
    const T = D.tracker;
    if (!T || !T.days.length) { $("tracker").innerHTML = `<tbody><tr><td class="empty">No runs yet.</td></tr></tbody>`; return; }
    const hasLive = T.days.some((d) => d.phase === "live");
    if (!S.trk) S.trk = hasLive ? "live" : "backtest";
    seg($("trk-phase"), [{ k: "live", label: "Live", disabled: !hasLive }, { k: "backtest", label: "Backtest" }], S.trk,
      (k) => { S.trk = k; S.trkSel = null; renderTracker(); });
    const days = T.days.filter((d) => d.phase === S.trk).slice(-21);
    const models = MODELS.filter((m) => days.some((d) => T.cells[d.d] && T.cells[d.d][m]));
    const head = `<thead><tr><th>Model</th>${days.map((d) => {
      const b = d.phase !== "live" ? "" : d.on_time === false ? `<small class="late">late ${localTime(d.finished)}</small>` : d.on_time ? `<small class="ok">✓ ${localTime(d.finished)}</small>` : "<small>–</small>";
      return `<th title="${d.d} · run ${localTime(d.started)}–${localTime(d.finished)} Brussels">${esc(shortDate(d.d))}${b}</th>`;
    }).join("")}</tr></thead>`;
    const body = models.map((m) => `<tr><td class="model"><i style="background:${color(m)}"></i>${esc(m)}</td>${days.map((d) => {
      const st = cellStatus(T.cells[d.d] && T.cells[d.d][m], d);
      if (!st) return `<td></td>`;
      const sel = S.trkSel && S.trkSel.d === d.d && S.trkSel.m === m;
      return `<td><button type="button" class="trk ${st.k}" data-d="${d.d}" data-m="${esc(m)}" aria-pressed="${!!sel}" title="${esc(m)} · ${d.d}: ${esc(st.label)}">${st.icon}</button></td>`;
    }).join("")}</tr>`).join("");
    $("tracker").innerHTML = head + `<tbody>${body}</tbody>`;
    $("tracker").onclick = (e) => { const b = e.target.closest("button.trk"); if (!b) return; S.trkSel = { d: b.dataset.d, m: b.dataset.m }; renderTracker(); };
    $("l-tracker").innerHTML = [["ok", "✓", "forecast and fully scored"], ["part", "◐", "scored so far, rest not due"], ["wait", "…", "forecast, scoring not due"],
      ["miss", "!", "due but actuals missing"], ["late", "⏱", "after the gate"], ["err", "✕", "error or skipped"]]
      .map(([k, i, l]) => `<span style="display:flex;align-items:center;gap:7px"><span class="trk ${k}" style="display:inline-grid;place-items:center;cursor:default">${i}</span>${l}</span>`).join("");
    const det = $("trk-detail");
    if (!S.trkSel) { det.hidden = true; return; }
    const d = T.days.find((x) => x.d === S.trkSel.d), c = T.cells[S.trkSel.d] && T.cells[S.trkSel.d][S.trkSel.m];
    if (!d || !c) { det.hidden = true; return; }
    const st = cellStatus(c, d);
    det.hidden = false;
    det.innerHTML = `<h4>${esc(S.trkSel.m)} · issue ${d.d} → delivery ${addDays(d.d, 1)} <span class="tag ${d.phase === "live" ? "live" : "bt"}">${d.phase}</span></h4>
      <p>${st.icon} ${esc(st.label)}.</p>
      <p class="mono">run ${localTime(d.started)}–${localTime(d.finished)} Brussels${d.phase === "live" ? ` · gate ${esc(D.config.gate)} · ${d.on_time ? "on time" : "LATE"}` : ""} ·
        forecasts ${c.ok}/${d.n_series} ok${c.err ? `, ${c.err} failed` : ""}${c.skip ? `, ${c.skip} skipped` : ""} · scored ${c.scored} of ${c.due} due · model time ${c.sec} s</p>
      ${c.msgs.length ? `<ul>${c.msgs.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}`;
  }

  // ---------- forecast explorer ----------
  async function loadDay(d) {
    if (!S.days[d]) S.days[d] = fetch(`data/days/${d}.json`, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    return S.days[d];
  }
  function renderExplorerControls() {
    const ds = Object.keys(D.days || {}).sort().reverse();
    if (!S.x.day) S.x.day = ds[0];
    options($("x-day"), ds.map((d) => ({ k: d, label: `${d} → ${addDays(d, 1)} · ${D.days[d]}` })), S.x.day);
    options($("x-zone"), D.config.zones.map((z) => ({ k: z, label: z })), S.x.zone);
    options($("x-target"), Object.keys(D.config.targets).map((t) => ({ k: t, label: TARGET_LABEL[t] || t })), S.x.target);
  }
  async function renderExplorerChart() {
    const day = await loadDay(S.x.day);
    const card = $("x-card");
    if (!day) { $("x-meta").innerHTML = "No forecasts stored for this day."; return; }
    const key = `${S.x.zone}_${S.x.target}`;
    const ser = day.series[key] || { actual: null, models: {} };
    const unit = D.config.targets[S.x.target].unit;
    const bt = day.phase === "backtest";
    card.classList.toggle("is-bt", bt);
    const models = MODELS.filter((m) => shown(m) && inField(m) && ser.models[m]);
    const bandModels = models.filter((m) => ser.models[m].lo);
    if (!bandModels.includes(S.x.band)) S.x.band = bandModels[0] || "";
    options($("x-band"), [{ k: "", label: "No band" }, ...bandModels.map((m) => ({ k: m, label: `${m} 10–90%` }))], S.x.band);

    const t0 = new Date(day.t0);
    const tf = new Intl.DateTimeFormat("en-GB", { timeZone: D.config.timezone, hour: "2-digit", minute: "2-digit" });
    const labels = Array.from({ length: day.n }, (_, i) => tf.format(new Date(t0.getTime() + i * 900e3)));
    const datasets = [];
    if (S.x.band && ser.models[S.x.band]) {
      const c = color(S.x.band);
      datasets.push({ label: "_lo", data: ser.models[S.x.band].lo, borderWidth: 0, pointRadius: 0, fill: false });
      datasets.push({ label: "_hi", data: ser.models[S.x.band].hi, borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: alpha(c, 0.16) });
    }
    if (ser.actual) datasets.push({ label: "Actual", data: ser.actual, borderColor: css("--ink"), backgroundColor: css("--ink"), borderWidth: 2.5, pointRadius: 0, tension: 0.15, endLabel: "Actual", endColor: css("--ink"), order: -1 });
    for (const m of models) {
      const c = color(m);
      datasets.push({ label: m, data: ser.models[m].p, hidden: S.hidden.has(m), borderColor: c, backgroundColor: c, borderWidth: 1.6, pointRadius: 0, pointHoverRadius: 4, tension: 0.15, endLabel: m, endColor: c });
    }
    const o = baseOptions({
      yTitle: `${TARGET_LABEL[S.x.target]} (${unit})`,
      itemSort: (a, b) => (b.dataset.label === "Actual") - (a.dataset.label === "Actual"),
      tooltip: { label: (c) => (c.dataset.label.startsWith("_") ? null : ` ${c.dataset.label}: ${fmt(c.raw, 1)} ${unit}`) },
    });
    o.plugins.tooltip.filter = (it) => !it.dataset.label.startsWith("_");
    o.scales.x.ticks.maxTicksLimit = 13;
    draw("c-day", { type: "line", data: { labels, datasets }, options: o });

    const delivered = addDays(day.issue_date, 1);
    $("x-meta").innerHTML = `<span>Locked <b>${day.issue_date} ${esc(D.config.issue_time)}</b></span><span>Gate <b>${esc(D.config.gate || D.config.issue_time)}</b></span><span>Delivery <b>${delivered}</b></span>
      <span class="tag ${bt ? "bt" : "live"}">${bt ? "Backtest" : "Live"}</span>
      ${ser.actual ? "" : '<span class="tag pending">Actuals not published yet</span>'}
      ${ser.actual ? `<span>Actuals <b>${esc(actualSource(day, key))}</b></span>` : ""}`;
    legend($("l-day"), models, ser.actual ? `<span class="actual" style="display:flex;align-items:center;gap:7px"><i></i>Actual</span>` : "");

    renderInputs(day);
    const sc = ROWS.filter((r) => r.issue_date === day.issue_date && r.zone === S.x.zone && r.target === S.x.target);
    const tb = $("x-table");
    if (!sc.length) { tb.innerHTML = `<tbody><tr><td class="empty">Not scored yet. ${S.x.target === "price" ? "Prices are scored the afternoon of the issue day." : "Wind and solar are scored two days after delivery."}</td></tr></tbody>`; return; }
    const best = (k) => Math.min(...sc.map((r) => (k === "bias" ? Math.abs(r[k]) : r[k])).filter((v) => v != null));
    tb.innerHTML = `<thead><tr><th>Model</th><th class="r">MAE</th><th class="r">RMSE</th><th class="r">Bias</th><th class="r">Pinball</th><th class="r">Rel. MAE</th></tr></thead><tbody>` +
      MODELS.filter(shown).map((m) => sc.find((r) => r.model === m)).filter(Boolean).map((r) => `<tr class="${bt ? "is-bt" : ""}">
        <td class="model"><i style="background:${color(r.model)}"></i>${esc(r.model)}</td>
        ${["mae", "rmse", "bias", "pinball", "rel_mae"].map((k) => `<td class="num ${(k === "bias" ? Math.abs(r[k]) : r[k]) === best(k) ? "best" : ""}">${fmt(r[k], k === "rel_mae" ? 3 : 1)}</td>`).join("")}
      </tr>`).join("") + `</tbody>`;
  }

  // ---------- wiring ----------
  let SHOWN = new Set();
  const shown = (m) => SHOWN.has(m);
  function updateShown() {
    let st = standings(DAYS, S.phase);
    if (!st.length) st = standings(DAYS, "all");
    const ranked = st.map((x) => x.model);
    const rest = MODELS.filter((m) => !ranked.includes(m) && inField(m));
    SHOWN = new Set([...ranked, ...rest].slice(0, S.topN));
    const total = new Set([...ranked, ...rest]).size;
    const el = $("f-top");
    el.max = String(total);
    el.value = String(Math.min(S.topN, total));
    $("f-top-val").textContent = S.topN >= total ? `All ${total}` : `Top ${S.topN} of ${total}`;
  }

  function renderAll() {
    DAYS = dayTable(filteredRows());
    updateShown();
    renderControls();
    renderStandings();
    renderCharts();
    renderHeatmap();
    renderDetails();
  }

  async function init() {
    try {
      D = await (await fetch("data/summary.json", { cache: "no-cache" })).json();
    } catch (e) {
      document.querySelector("#standings .wrap").insertAdjacentHTML("afterbegin", '<p class="lede">Leaderboard data is not available yet.</p>');
      return;
    }
    const cols = D.score_columns;
    ROWS = D.scores.map((a) => Object.fromEntries(cols.map((c, i) => [c, a[i]])));
    MODELS = D.models.map((m) => m.name);
    FAM = Object.fromEntries(D.models.map((m) => [m.name, m.family || m.name]));
    USES_COV = Object.fromEntries(D.models.map((m) => [m.name, (m.inputs || []).some((k) => COVARIATE_INPUTS.includes(k))]));
    const famRank = (m) => { const f = FAM[m] || m; const i = FAMILY_ORDER.indexOf(f); return f === "baseline" ? 99 : i < 0 ? 50 : i; };
    MODELS.sort((a, b) => famRank(a) - famRank(b) || a.localeCompare(b));
    for (const r of ROWS) if (!MODELS.includes(r.model)) MODELS.push(r.model);
    S.phase = ROWS.some((r) => r.phase === "live") ? "live" : "backtest";

    const up = new Date(D.generated_utc.replace("Z", ":00Z"));
    $("updated").textContent = up.toLocaleString("en-GB", { timeZone: D.config.timezone, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    document.querySelectorAll("[data-k]").forEach((el) => { if (el.tagName === "SPAN" && D.config[el.dataset.k] != null) el.textContent = D.config[el.dataset.k]; });

    $("f-zone").onchange = (e) => { S.zone = e.target.value; if (!rawAllowed()) S.view = "rank"; if (S.zone !== "all") S.x.zone = S.zone; renderAll(); renderExplorerControls(); renderExplorerChart(); };
    $("f-target").onchange = (e) => { S.target = e.target.value; if (!rawAllowed()) S.view = "rank"; if (S.target !== "all") S.x.target = S.target; renderAll(); renderExplorerControls(); renderExplorerChart(); };
    $("f-top").oninput = (e) => { S.topN = +e.target.value; renderAll(); renderExplorerChart(); };
    $("x-day").onchange = (e) => { S.x.day = e.target.value; renderExplorerChart(); };
    $("x-zone").onchange = (e) => { S.x.zone = e.target.value; renderExplorerChart(); };
    $("x-target").onchange = (e) => { S.x.target = e.target.value; renderExplorerChart(); };
    $("x-band").onchange = (e) => { S.x.band = e.target.value; renderExplorerChart(); };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderAll(); renderModels(); renderExplorerChart(); renderTimeline(); renderTracker(); });
    let wasNarrow = narrow();
    window.addEventListener("resize", () => { if (narrow() !== wasNarrow) { wasNarrow = narrow(); renderCharts(); renderExplorerChart(); } });

    Chart.defaults.font.family = "Archivo, -apple-system, sans-serif";
    renderAll();
    renderModels();
    renderCentroids();
    renderTimeline();
    renderTracker();
    $("cov-var").onchange = (e) => { S.cov = e.target.value; renderExplorerChart(); };
    renderExplorerControls();
    renderExplorerChart();
    // content renders after load, so redo the jump to a #section link
    if (location.hash.length > 1) { const el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); }
  }

  document.fonts && document.fonts.ready ? document.fonts.ready.then(init) : init();
})();
