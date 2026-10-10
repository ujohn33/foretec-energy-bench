/* Foretec Energy Bench leaderboard. Reads data/summary.json and data/days/<issue_date>.json. */
(() => {
  "use strict";

  const METRICS = [
    { k: "rel_mae", label: "Rel. MAE", d: 3, unitless: true },
    { k: "mae", label: "MAE", d: 1 },
    { k: "rmse", label: "RMSE", d: 1 },
    { k: "pinball", label: "Pinball", d: 2 },
  ];
  const TARGET_LABEL = { price: "Day-ahead price", wind: "Wind", solar: "Solar", load: "Load" };
  const S = {
    zone: "all", target: "all", metric: "rel_mae", phase: null, view: "rank", topN: 10, covFilter: "all", xTop: 10, sort: { k: "pos", dir: 1 },
    hidden: new Set(), charts: {}, days: {}, x: { day: null, zone: "BE", target: "price", band: null },
  };
  let D = null;        // summary.json
  let ROWS = [];       // score rows as objects
  let MODELS = [];     // model names in fixed colour order
  const $ = (id) => document.getElementById(id);
  const PAGE = document.body.dataset.page || "leaderboard";
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
  // slots 1-8 are the validated categorical palette; slot 9 (t0) was chosen with the palette validator to clear
  // every existing slot (light: normal-vision dE >= 19, CVD dE >= 13; dark step likewise). New families get grey.
  const FAMILY_ORDER = ["chronos", "timesfm", "moirai", "tirex", "toto", "sundial", "ttm", "nwp", "t0"];
  let FAM = {};
  // One colour per model, on every page: it depends only on the model's family and its place in the
  // full model list (D.models), never on filters, rank or page. Full-strength ink is reserved for the
  // observed values (Actual), so no model ever uses it.
  const actualColor = () => css("--ink");
  function color(m) {
    const fam = FAM[m] || m;
    const members = MODELS.filter((x) => (FAM[x] || x) === fam);
    const k = Math.max(0, members.indexOf(m));
    let base;
    if (fam === "baseline") return mix(css("--ink"), css("--surface"), Math.min(0.32 + 0.09 * k, 0.7));
    if (fam === "tso") base = css("--ink2");
    else {
      const i = FAMILY_ORDER.indexOf(fam);
      base = i >= 0 ? css(`--s${i + 1}`) : css("--muted");
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
  let REF = new Set();   // reference entries (published after the gate): shown, never ranked
  let CAVEAT = {};       // model -> caveat from model.yaml: the name gets a star that explains it
  const nm = (m) => esc(m) + (CAVEAT[m] ? `<sup class="caveat" data-tip="caveat:${esc(m)}" tabindex="0">*</sup>` : "");
  // benchmarks (baselines and the grid operators' forecasts) stay in the field under every covariate filter
  const isBench = (m) => FAM[m] === "baseline" || FAM[m] === "tso";
  const inField = (m) => REF.has(m) || S.covFilter === "all" || isBench(m) || (S.covFilter === "with") === !!USES_COV[m];
  function filteredRows(metric = S.metric, refs = false) {
    return ROWS.filter((r) => REF.has(r.model) === refs && inField(r.model) && (S.zone === "all" || r.zone === S.zone) && (S.target === "all" || r.target === S.target) && r[metric] != null);
  }
  // the races are run over the whole field: the covariate filter only chooses which models are shown, so a
  // model's score is the same under All, With and Without (zone and target choose which races count)
  function raceRows(metric = S.metric) {
    return ROWS.filter((r) => !REF.has(r.model) && (S.zone === "all" || r.zone === S.zone) && (S.target === "all" || r.target === S.target) && r[metric] != null);
  }
  function phaseOf(d) { return (D.days && D.days[d]) || (ROWS.find((r) => r.issue_date === d) || {}).phase || "backtest"; }

  // ---------- race points ----------
  // A race is one issue day x one series. Its field is the models with a scored forecast in it. With n in the
  // field, a model at rank r scores 100 (n - r) / (n - 1), ties sharing the average rank: the winner gets 100
  // whatever the field size. A model whose run failed or was skipped is not in that race at all (no 0), so the
  // standing is its average over the races it finished; the run tracker still shows the failures.
  const MIN_RACES = 24;

  /** Per issue day: race points, average rank and raw mean across the selected series. */
  function dayTable(rows) {
    const byDay = new Map();
    for (const r of rows) {
      if (!byDay.has(r.issue_date)) byDay.set(r.issue_date, new Map());
      const ser = byDay.get(r.issue_date);
      const key = r.zone + "_" + r.target;
      if (!ser.has(key)) ser.set(key, []);
      ser.get(key).push(r);
    }
    const out = [];
    for (const [d, ser] of [...byDay.entries()].sort()) {
      const ranks = {}, vals = {}, pts = {};
      for (const [key, list] of ser) {
        const sorted = [...list].sort((a, b) => a[S.metric] - b[S.metric]);
        const n = sorted.length;
        if (n < 2) continue;
        sorted.forEach((r) => {
          const tied = sorted.filter((x) => x[S.metric] === r[S.metric]).map((x) => sorted.indexOf(x) + 1);
          const rk = mean(tied);
          (ranks[r.model] ||= []).push(rk);
          (vals[r.model] ||= []).push(r[S.metric]);
          (pts[r.model] ||= []).push(100 * (n - rk) / (n - 1));
        });
      }
      const avgRank = {}, raw = {}, points = {};
      for (const m in ranks) { avgRank[m] = mean(ranks[m]); raw[m] = mean(vals[m]); }
      for (const m in pts) points[m] = mean(pts[m]);
      out.push({ d, phase: phaseOf(d), avgRank, raw, points, pts, n: Object.keys(pts).length });
    }
    return out;
  }

  function standings(days, phase) {
    const sel = days.filter((x) => phase === "all" || x.phase === phase);
    const set = new Set(sel.map((x) => x.d));
    const byMetric = Object.fromEntries(METRICS.map((m) => [m.k, filteredRows(m.k).filter((r) => set.has(r.issue_date))]));
    const meanOf = (k, model) => mean(byMetric[k].filter((r) => r.model === model).map((r) => r[k]));
    const res = {};
    for (const x of sel) {
      for (const m in x.pts) {
        const s = (res[m] ||= { model: m, sum: 0, races: 0, ranks: [], dayScores: [] });
        s.sum += x.pts[m].reduce((a, b) => a + b, 0); s.races += x.pts[m].length;
        s.dayScores.push(x.points[m]);
        if (x.avgRank[m] != null) s.ranks.push(x.avgRank[m]);
      }
    }
    const list = Object.values(res);
    const need = Math.min(MIN_RACES, Math.ceil(0.5 * Math.max(0, ...list.map((s) => s.races))));
    return list.map((s) => ({
      ...s,
      score: s.sum / s.races,
      provisional: s.races < need,
      avgRank: mean(s.ranks),
      best: Math.max(...s.dayScores), worst: Math.min(...s.dayScores), days: s.dayScores.length,
      rel: meanOf("rel_mae", s.model), mae: meanOf("mae", s.model), rmse: meanOf("rmse", s.model), pinball: meanOf("pinball", s.model),
    })).sort((a, b) => (a.provisional - b.provisional) || b.score - a.score || (a.avgRank ?? 99) - (b.avgRank ?? 99));
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
      `<button type="button" role="radio" data-k="${it.k}" aria-checked="${it.k === current}" ${it.disabled ? "disabled" : ""} ${it.title ? `title="${esc(it.title)}"` : ""} ${it.tip ? `data-tip="${it.tip}"` : ""}>${esc(it.label)}</button>`).join("");
    el.onclick = (e) => { const b = e.target.closest("button"); if (b && !b.disabled) onPick(b.dataset.k); };
  }
  function options(el, items, current) {
    el.innerHTML = items.map((it) => `<option value="${esc(it.k)}" ${it.k === current ? "selected" : ""}>${esc(it.label)}</option>`).join("");
  }

  function renderControls() {
    options($("f-zone"), [{ k: "all", label: "All zones" }, ...D.config.zones.map((z) => ({ k: z, label: z }))], S.zone);
    options($("f-target"), [{ k: "all", label: "All targets" }, ...Object.keys(D.config.targets).map((t) => ({ k: t, label: TARGET_LABEL[t] || t }))], S.target);
    const hasLive = ROWS.some((r) => r.phase === "live");
    seg($("f-phase"), [
      { k: "live", label: "Live", disabled: !hasLive, title: hasLive ? "" : "No live forecasts scored yet" },
      { k: "backtest", label: "Backtest" },
      { k: "all", label: "All" },
    ], S.phase, (k) => { S.phase = k; renderAll(); });
    seg($("f-cov"), [
      { k: "all", label: "All" },
      { k: "with", label: "With", title: "Models that also use NWP weather, the ENTSO-E load forecast or fuel cost, plus the benchmarks (baselines and the grid operators' forecasts)" },
      { k: "without", label: "Without", title: "Models that only see the target's own history, plus the benchmarks (baselines and the grid operators' forecasts)" },
    ], S.covFilter, (k) => { S.covFilter = k; renderAll(); renderExplorerChart(); });
    seg($("f-view"), [
      { k: "rank", label: "Avg rank", tip: "rank" },
      { k: "value", label: "Value", tip: "value", disabled: !rawAllowed(), title: rawAllowed() ? "" : "Pick one zone and one target to compare raw values across days" },
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
      ctx.font = "600 11px 'Source Sans 3', sans-serif";
      ctx.fillText("Backtest", x0 + 8, a.top + 15);
      if (x1 < a.right - 40) ctx.fillText("Live", x1 + 8, a.top + 15);
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
      ctx.font = "12px 'Source Sans 3', sans-serif";
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
        x: { grid: { color: grid, drawTicks: false }, border: { color: axis }, ticks: { color: ink2, font: { size: 12 }, maxRotation: 0, autoSkipPadding: 14, padding: 6 } },
        y: { grid: { color: grid, drawTicks: false }, border: { display: false }, ticks: { color: ink2, font: { size: 12 }, padding: 8 }, title: { display: !!extra.yTitle, text: extra.yTitle, color: ink2, font: { size: 12, weight: 600 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: css("--surface"), titleColor: css("--ink"), bodyColor: css("--ink2"), borderColor: css("--rule"), borderWidth: 1,
          padding: 10, boxPadding: 4, usePointStyle: true, titleFont: { weight: 700, size: 13 }, bodyFont: { size: 12.5 },
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
      `<button type="button" data-m="${esc(m)}" aria-pressed="${!S.hidden.has(m)}"><i style="border-top-color:${color(m)}"></i>${nm(m)}</button>`).join("");
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
      label: m, data, hidden: S.hidden.has(m), endLabel: CAVEAT[m] ? m + "*" : m, endColor: c,
      borderColor: c, backgroundColor: c, borderWidth: 2, tension: 0, spanGaps: true,
      pointRadius: 3, pointHoverRadius: 5, pointBorderWidth: 2, pointBorderColor: css("--surface"),
      pointBackgroundColor: data.map((_, i) => (isBt(i) ? g : c)),
      segment: { borderColor: (ctx) => (isBt(ctx.p1DataIndex) ? g : c), borderDash: (ctx) => (isBt(ctx.p1DataIndex) ? [5, 4] : undefined) },
    };
  }


  // ---------- explanations: shown on hover (mouse), focus (keyboard) or tap (touch) wherever a metric is named ----------
  const units = () => [...new Set(Object.values(D.config.targets).map((t) => t.unit))].join(" or ");
  const mixedUnits = () => (S.target === "all" ? " With all targets selected, values in different units are averaged together: pick one target to compare them." : "");
  const fieldSize = () => Math.max(0, ...DAYS.map((x) => x.n));
  const GLOSS = {
    pos: () => ["Position", "Place in the standings: highest score first, equal scores split by average rank; provisional models (too few races yet) follow the ranked ones. It stays the same when you sort the table by another column."],
    pts: () => ["Score", `Average race points. A race is one issue day and one series; every model with a scored forecast takes part. With n in the race, a model at rank r scores 100 × (n − r) / (n − 1): the winner 100, the last 0. A failed run is left out of the race rather than counted as 0. The score averages a model's races, so new models and different field sizes compare fairly. Ranked by ${metricDef().label}.`],
    races: () => ["Races", `Races the model was due in during the period (issue day × series). A model with fewer than ${MIN_RACES} races, or half of the most any model has, is provisional and listed after the ranked ones.`],
    rank: () => ["Average rank", `In every selected series (zone and target) the models are ranked by ${metricDef().label} for the day: 1 is best, and tied models share the average place. Ranks are averaged over the series, then over the days. Lower is better.`],
    rel_mae: () => ["Rel. MAE", `A model's MAE divided by the MAE of ${D.config.baseline} (the same quarter-hour one week earlier) on the same day and series. Below 1 beats that baseline; 0.5 halves its error. It has no unit, so prices, wind and solar can be averaged together. Lower is better.`],
    mae: () => ["MAE", `Mean absolute error: the average size of the miss over the day's quarter-hours, in the series' own unit (${units()}). Lower is better.${mixedUnits()}`],
    rmse: () => ["RMSE", `Root mean squared error: like MAE, but errors are squared before averaging, so a few large misses (price spikes, wind ramps) cost more than many small ones. Lower is better.${mixedUnits()}`],
    pinball: () => ["Pinball loss", `Quantile loss averaged over the 10th to 90th percentiles. It scores the whole forecast band: narrow bands score well only if the outcome falls inside them as often as they claim. Only models that publish quantiles have one. Lower is better.${mixedUnits()}`],
    bias: () => ["Bias", "Mean of forecast minus actual. Positive means the model forecasts too high on average, negative too low. Closest to 0 is best."],
    best: () => ["Best day", "The model's highest average race points on a single issue day."],
    worst: () => ["Worst day", "The model's lowest average race points on a single issue day."],
    days: () => ["Days", `Issue days on which the model took part in at least one race in the selected period. The score averages its races (day × series), so fewer days does not lower it; a model with too few races is provisional.`],
    bar: () => ["Score", "The score as a bar on its 0–100 scale."],
    metric: () => GLOSS[S.metric](),
    value: () => ["Value", `The mean ${metricDef().label} per issue day, in the series' own unit. Needs one zone and one target.`],
    heat: () => ["Average rank", "Each cell is the model's average rank on that issue day across the selected series (1 = best); darker is better. The last column averages over all days."],
  };
  let tipAnchor = null, tipTimer = null, lastPointer = "mouse";
  function showTip(el) {
    const key = el && el.dataset.tip, cav = key && key.startsWith("caveat:") && key.slice(7);
    const tip = $("tip"), g = el && (GLOSS[key] || (cav && CAVEAT[cav] ? () => [`${cav} *`, CAVEAT[cav]] : null));
    if (!tip || !g || !D) return;
    const [title, body] = g();
    tip.innerHTML = `<b>${esc(title)}</b>${esc(body)}`;
    tip.dataset.key = el.dataset.tip;
    tip.dataset.scope = (el.closest("[id]") || {}).id || "";
    tip.classList.add("on");
    tipAnchor = el;
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), document.documentElement.clientWidth - w - 8);
    const below = r.bottom + 8, above = r.top - h - 8;
    tip.style.left = x + "px";
    tip.style.top = (below + h > window.innerHeight - 8 && above > 8 ? above : below) + "px";
  }
  function hideTip() { const tip = $("tip"); if (tip) tip.classList.remove("on"); tipAnchor = null; clearTimeout(tipTimer); }
  // a re-render (e.g. after sorting) replaces the header the tip points at: follow its successor
  function refreshTip() {
    const tip = $("tip");
    if (!tipAnchor || tipAnchor.isConnected) return;
    const scope = tip.dataset.scope && $(tip.dataset.scope);
    const el = scope && scope.querySelector(`[data-tip="${tip.dataset.key}"]`);
    el ? showTip(el) : hideTip();
  }
  document.addEventListener("pointerdown", (e) => { lastPointer = e.pointerType; }, true);
  document.addEventListener("keydown", () => { lastPointer = "keyboard"; }, true);
  document.addEventListener("pointerover", (e) => {
    if (e.pointerType !== "mouse") return;
    const el = e.target.closest("[data-tip]");
    if (el && el !== tipAnchor) showTip(el);
  });
  document.addEventListener("pointerout", (e) => {
    if (e.pointerType === "mouse" && tipAnchor && tipAnchor.contains(e.target) && !tipAnchor.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener("focusin", (e) => { const el = e.target.closest("[data-tip]"); if (el && el.matches(":focus-visible")) showTip(el); });
  document.addEventListener("focusout", (e) => { if (e.target === tipAnchor) hideTip(); });
  // touch has no hover: a tap shows the explanation for a few seconds (capture phase, before a sort re-renders the header)
  document.addEventListener("click", (e) => {
    if (lastPointer !== "touch" && lastPointer !== "pen") return;
    const el = e.target.closest("[data-tip]");
    if (!el) { hideTip(); return; }
    showTip(el);
    tipTimer = setTimeout(hideTip, 6000);
  }, true);
  window.addEventListener("scroll", hideTip, true);
  window.addEventListener("resize", hideTip);

  // ---------- standings sort ----------
  const SORT_COLS = {
    pos: { get: (s) => s.pos, dir: 1 },
    model: { get: (s) => s.model, dir: 1 },
    pts: { get: (s) => s.score, dir: -1 },
    races: { get: (s) => s.races, dir: -1 },
    rank: { get: (s) => s.avgRank, dir: 1 },
    rel: { get: (s) => s.rel, dir: 1 },
    mae: { get: (s) => s.mae, dir: 1 },
    rmse: { get: (s) => s.rmse, dir: 1 },
    pinball: { get: (s) => s.pinball, dir: 1 },
    days: { get: (s) => s.days, dir: -1 },
    best: { get: (s) => s.best, dir: -1 },
    worst: { get: (s) => s.worst, dir: -1 },
  };
  function sortKey() { const k = S.sort && S.sort.k; return SORT_COLS[k] ? k : "pos"; }
  function sortRows(st) {
    const k = sortKey(), dir = k === S.sort.k ? S.sort.dir : 1, get = SORT_COLS[k].get;
    return [...st].sort((a, b) => {
      const x = get(a), y = get(b);
      if (x == null || !isFinite(x) && typeof x === "number") return 1;      // missing values always last
      if (y == null || !isFinite(y) && typeof y === "number") return -1;
      const c = typeof x === "string" ? x.localeCompare(y) : x - y;
      return c * dir || a.pos - b.pos;
    });
  }
  function sortTh(key, label, tip, right = true) {
    const on = sortKey() === key, dir = on && key === S.sort.k ? S.sort.dir : 1;
    const aria = on ? (dir < 0 ? "descending" : "ascending") : "none";
    return `<th class="${right ? "r" : ""}" aria-sort="${aria}"><button type="button" class="sort" data-sort="${key}" ${tip ? `data-tip="${tip}"` : ""}>` +
      `<span class="${tip ? "lbl" : ""}">${esc(label)}</span><i aria-hidden="true">${on ? (dir < 0 ? "▼" : "▲") : "↕"}</i></button></th>`;
  }

  // ---------- sections ----------
  let DAYS = [];

  function renderStandings() {
    const full = standings(DAYS, S.phase);
    // place in the whole field, so the number does not change with the covariate filter or the top-N slider
    const place = Object.fromEntries(full.filter((x) => !x.provisional).map((x, i) => [x.model, i + 1]));
    const st = full.filter((x) => shown(x.model));
    const bt = S.phase === "backtest";
    const md = metricDef();
    $("standings").classList.toggle("is-bt", bt);
    const top = st.filter((x) => !x.provisional).slice(0, 3);
    $("podium").innerHTML = top.length ? top.map((s, i) => `
      <div class="pod ${place[s.model] === 1 ? "lead" : ""}">
        <div class="pod-top"><span class="place" aria-label="place ${place[s.model]}">${place[s.model]}</span>
          <div class="name"><i style="background:${color(s.model)}"></i>${nm(s.model)}</div>
          ${bt ? '<span class="tag bt">Backtest</span>' : '<span class="tag live">Live</span>'}</div>
        <div class="stats">
          <div class="stat"><span data-tip="pts">Score</span><b>${fmt(s.score, 1)}</b></div>
          <div class="stat"><span data-tip="rank">Avg rank</span><b>${fmt(s.avgRank, 2)}</b></div>
          <div class="stat"><span data-tip="${S.metric}">${esc(md.label)}</span><b>${fmt(s.metric, md.d)}</b></div>
        </div>
      </div>`).join("") : `<div class="pod" style="grid-column:1/-1;color:var(--ink2)">No scored days for this selection yet.</div>`;

    const t = $("standings-table");
    const unit = S.target === "all" ? "" : ` (${D.config.targets[S.target].unit})`;
    t.innerHTML = `<thead><tr>${sortTh("pos", "#", "pos", false)}${sortTh("model", "Model", "", false)}${sortTh("pts", "Score", "pts", false)}${sortTh("days", "Days", "days")}${sortTh("rank", "Avg rank", "rank")}
      ${sortTh("rel", "Rel. MAE", "rel_mae")}${sortTh("mae", "MAE" + unit, "mae")}${sortTh("rmse", "RMSE" + unit, "rmse")}${sortTh("pinball", "Pinball" + unit, "pinball")}
      ${sortTh("best", "Best day", "best")}${sortTh("worst", "Worst day", "worst")}</tr></thead><tbody>` +
      sortRows(st.map((s) => ({ ...s, pos: place[s.model] ?? 1e3 }))).map((s) => `<tr class="${bt ? "is-bt" : ""}${s.provisional ? " prov" : ""}">
        <td class="pos">${s.provisional ? `<span data-tip="races">prov.</span>` : s.pos}</td>
        <td class="model"><i style="background:${color(s.model)}"></i>${nm(s.model)}</td>
        <td><div class="bar"><span style="width:${Math.round(0.9 * Math.max(0, s.score))}px;background:${bt ? btColor(s.model) : color(s.model)}"></span><b>${fmt(s.score, 1)}</b></div></td>
        <td class="r v">${s.days}</td><td class="r v">${fmt(s.avgRank, 2)}</td>
        <td class="r v">${fmt(s.rel, 3)}</td><td class="r v">${fmt(s.mae, 1)}</td><td class="r v">${fmt(s.rmse, 1)}</td><td class="r v">${fmt(s.pinball, 1)}</td>
        <td class="r v">${fmt(s.best, 0)}</td><td class="r v">${fmt(s.worst, 0)}</td>
      </tr>`).join("") + "</tbody>";
    if (!st.length) t.innerHTML = "";
    const starred = st.filter((x) => CAVEAT[x.model]);
    $("st-notes").innerHTML = starred.map((x) => `<span>${esc(x.model)}*</span> ${esc(CAVEAT[x.model])}`).join("<br>");
    t.onclick = (e) => {
      const b = e.target.closest("button[data-sort]");
      if (!b) return;
      const k = b.dataset.sort;
      S.sort = sortKey() === k && S.sort.k === k ? { k, dir: -S.sort.dir } : { k, dir: SORT_COLS[k].dir };
      saveFilters();
      renderStandings();
      if (e.detail === 0) { const nb = t.querySelector(`button[data-sort="${k}"]`); if (nb) nb.focus(); }   // keyboard: keep focus on the header
      refreshTip();
    };
    const set = new Set(DAYS.filter((x) => S.phase === "all" || x.phase === S.phase).map((x) => x.d));
    const refRel = filteredRows("rel_mae", true).filter((r) => set.has(r.issue_date));
    const refMet = filteredRows(S.metric, true).filter((r) => set.has(r.issue_date));
    const refs = [...REF].filter((m) => refRel.some((r) => r.model === m));
    if (refs.length && st.length) {
      const only = S.target === "all" ? " (wind and solar only)" : "";
      t.querySelector("tbody").insertAdjacentHTML("beforeend", refs.map((m) => {
        const mine = refRel.filter((r) => r.model === m);
        const mm = (k) => mean(filteredRows(k, true).filter((r) => set.has(r.issue_date) && r.model === m).map((r) => r[k]));
        return `<tr class="ref"><td class="pos">ref.</td><td class="model"><i style="background:${color(m)}"></i>${nm(m)}</td>
          <td class="wrap">Reference, not ranked: published after the gate${only}</td><td class="r v">${new Set(mine.map((r) => r.issue_date)).size}</td><td class="r v">–</td>
          <td class="r v">${fmt(mean(mine.map((r) => r.rel_mae)), 3)}</td><td class="r v">${fmt(mm("mae"), 1)}</td><td class="r v">${fmt(mm("rmse"), 1)}</td><td class="r v">${fmt(mm("pinball"), 1)}</td>
          <td class="r v">–</td><td class="r v">–</td></tr>`;
      }).join(""));
    }
  }

  function renderCharts() {
    const labels = DAYS.map((x) => shortDate(x.d));
    const band = bandRange(DAYS);
    const md = metricDef();
    const rank = S.view === "rank";
    const models = MODELS.filter((m) => shown(m) && DAYS.some((x) => x.pts[m] != null));
    const phaseNote = (i) => (DAYS[i] && DAYS[i].phase === "backtest" ? " · backtest" : " · live");

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

    // running average race points: every model on the same 0-100 scale, however long it has been running
    const race = models.map((m) => {
      let sum = 0, n = 0;
      return lineDataset(m, DAYS.map((x) => { const p = x.pts[m]; if (p) { sum += p.reduce((a, b) => a + b, 0); n += p.length; } return n ? sum / n : null; }), DAYS);
    });
    draw("c-race", {
      type: "line", data: { labels, datasets: race },
      options: baseOptions({
        band, yTitle: "Average race points (running)", itemSort: (a, b) => b.raw - a.raw,
        tooltip: { title: (it) => DAYS[it[0].dataIndex].d + phaseNote(it[0].dataIndex), label: (c) => ` ${c.dataset.label}: ${fmt(c.raw, 1)}` },
      }),
    });
    legend($("l-race"), models);
  }

  function renderHeatmap() {
    const st = standings(DAYS, "all").filter((x) => shown(x.model));
    const t = $("heatmap");
    if (!DAYS.length) { t.innerHTML = ""; return; }
    const lerp = (a, b, w) => mix(a, b, w);
    const head = `<thead><tr><th>Model</th>${DAYS.map((x) => `<th class="${x.phase === "backtest" ? "is-bt" : "live"}" title="${x.d} · ${x.phase}">${esc(shortDate(x.d))}</th>`).join("")}<th><span class="lbl" data-tip="heat">Avg</span></th></tr></thead>`;
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
      return `<tr><td class="model"><i style="background:${color(s.model)}"></i>${nm(s.model)}</td>${cells}<td class="avg">${fmt(s.avgRank, 1)}</td></tr>`;
    }).join("");
    t.innerHTML = head + "<tbody>" + body + "</tbody>";
  }

  function renderModels() {
    if (!$("models-table")) return;
    const live = new Set(ROWS.filter((r) => r.phase === "live").map((r) => r.model));
    $("models-table").innerHTML = `<thead><tr><th>Model</th><th>Author</th><th>Description</th><th>Inputs</th><th>Implementation</th><th>Status</th></tr></thead><tbody>` +
      D.models.map((m) => `<tr><td class="model"><i style="background:${color(m.name)}"></i>${nm(m.name)}</td><td>${esc(m.author)}</td><td class="desc">${esc(m.description)}${m.caveat ? `<br><small>* ${esc(m.caveat)}</small>` : ""}</td>
        <td class="wrap">${inputChips(m.inputs)}</td><td class="kind">${esc(m.kind)}</td><td class="${m.enabled === false ? "wrap" : ""}">${m.enabled === false ? `<span class="tag bt">Retired</span><br><small>${esc(m.retired || "")}</small>` : m.reference ? '<span class="tag bt">Reference, not ranked</span>' : m.live_only ? '<span class="tag live">Live only</span>' : live.has(m.name) ? '<span class="tag live">Live</span>' : '<span class="tag bt">Backtest only</span>'}</td></tr>`).join("") + "</tbody>";
  }

  const INPUT_LABEL = { history: "Own history", nwp: "NWP ensemble", load_forecast: "Load forecast", fuel: "Fuel cost", calendar: "Calendar" };
  function inputChips(inputs) {
    return (inputs || ["history"]).map((k) => `<span class="chip ${k === "history" ? "" : "on"}">${esc(INPUT_LABEL[k] || k)}</span>`).join("");
  }

  // ---------- inputs: what the models knew ----------
  const NWP_STYLE = { icon_eu: [], gfs_seamless: [6, 4], ecmwf_ifs: [2, 3] };
  function fmtUtc(s) { return s ? String(s).replace("T", " ").slice(0, 16) + " UTC" : "–"; }
  const SRC_NAME = { entsoe: "ENTSO-E", elia: "Elia", rte: "RTE", energycharts: "Energy-Charts" };
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
    const note = hs ? "ENTSO-E Transparency Platform (A44 prices, A75 generation per type); gaps from the TSOs' own open data (Elia, RTE), then Energy-Charts"
                    : "Energy-Charts (issue days up to 3 Oct 2026, and backtests, which do not store their snapshots)";
    return `<tr><td>Target history</td><td class="wrap">${note}</td><td class="wrap">prices through the end of D-1; wind and solar through the cut-off minus 1 h</td><td class="wrap">${lines}</td></tr>`;
  }
  function renderInputs(day) {
    if (!$("inputs-table")) return;
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
      datasets.push({ label: "Ensemble mean", data: mean, borderColor: ink, borderWidth: 2.6, pointRadius: 0, tension: 0, endLabel: "mean", endColor: ink });
    }
    for (const m of members) {
      const lab = (inp.nwp_models[m] && inp.nwp_models[m].label) || (m === "entsoe" ? "ENTSO-E" : m);
      datasets.push({ label: lab, data: v.members[m], borderColor: css("--blue-ink"), borderDash: NWP_STYLE[m] || [], borderWidth: 1.5,
        pointRadius: 0, tension: 0, endLabel: lab.split(" ")[0], endColor: css("--blue-ink") });
    }
    const o = baseOptions({ yTitle: `${v.label} (${v.unit})`, tooltip: { label: (c) => (c.dataset.label.startsWith("_") ? null : ` ${c.dataset.label}: ${fmt(c.raw, 1)} ${v.unit}`) } });
    o.plugins.tooltip.filter = (it) => !it.dataset.label.startsWith("_");
    o.scales.x.ticks.maxTicksLimit = 13;
    draw("c-cov", { type: "line", data: { labels, datasets }, options: o });
    $("l-cov").innerHTML = (members.length > 1 ? `<span class="actual" style="display:flex;align-items:center;gap:7px"><i></i>Ensemble mean</span>
      <span style="display:flex;align-items:center;gap:7px"><i style="border-top:8px solid ${alpha(css("--blue"), 0.25)}"></i>Min–max across models (disagreement)</span>` : "") +
      members.map((m) => `<span style="display:flex;align-items:center;gap:7px"><i style="border-top:2px ${m === "gfs_seamless" ? "dashed" : m === "ecmwf_ifs" ? "dotted" : "solid"} ${css("--blue-ink")}"></i>${esc((inp.nwp_models[m] && inp.nwp_models[m].label) || m)}</span>`).join("");
  }

  const BUCKET_LABEL = { wind_onshore: "Wind onshore", wind_offshore: "Wind offshore", solar: "Solar", load: "Load centres (population)" };
  const BUCKET_SLOT = { wind_onshore: "--s1", wind_offshore: "--s3", solar: "--s4", load: "--ink2" };
  let MAP = null;
  function renderCentroids() {
    if (!$("centroid-table")) return;
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

  function tsoNote(models, day) {
    if (models.some((m) => FAM[m] === "tso")) return "";
    const why = S.x.target === "price" ? "TSOs do not forecast prices"
      : S.x.zone === "FR" ? "RTE publishes its next-day wind and solar forecast at 16:15 on D-1, after the gate"
      : day.phase === "live" ? "not published yet" : "not available for this day";
    return `<span class="tag pending">TSO forecast: ${esc(why)}</span>`;
  }

  // end of the last published quarter-hour of a provisional actual, Brussels time
  function meteredTo(day, actual) {
    let i = actual.length - 1;
    while (i >= 0 && actual[i] == null) i--;
    const t = new Date(new Date(day.t0).getTime() + (i + 1) * 15 * 60000);
    return t.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "Europe/Brussels" });
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
    if (!$("timeline")) return;
    const T = D.timeline;
    if (!T) return;
    // hours relative to D-1 00:00 Brussels; all of D-1 up to 16:00 is zoomed in (runs, cut-off, gate, auction)
    const H = [-24, 0, 16, 24, 48, 72, 96], F = [0, 0.06, 0.58, 0.64, 0.80, 0.90, 1];
    const W = 1000, L = 128, R = 14, top = 74, laneH = 70;
    const x = (h) => { let i = 0; while (i < H.length - 2 && h > H[i + 1]) i++; const f = F[i] + (F[i + 1] - F[i]) * (Math.min(Math.max(h, H[i]), H[i + 1]) - H[i]) / (H[i + 1] - H[i]); return L + f * (W - L - R); };
    const lanes = ["Day-ahead price", "Wind", "Solar", "Load", "Inputs"];
    const y = (i) => top + i * laneH;
    const Hh = top + lanes.length * laneH + 6;
    const cut = hm(T.issue_time), gate = hm(T.gate), off = brusselsOffset();
    const sched = T.schedule || {};
    const scores = (sched.score || ["14:45"]).map(hm);
    const hh = (h) => `${String(Math.floor(h)).padStart(2, "0")}:${String(Math.round((h % 1) * 60)).padStart(2, "0")}`;
    const out = [];
    const txt = (xx, yy, t, cls = "", anchor = "middle") => out.push(`<text x="${xx.toFixed(1)}" y="${yy}" text-anchor="${anchor}" class="${cls}">${esc(t)}</text>`);
    // day bands and labels (row 1)
    [[-24, 0, "D−2"], [0, 24, "D−1 · issue day"], [24, 48, "D · delivery"], [48, 72, "D+1"], [72, 96, "D+2"]].forEach(([a, b, lab], i) => {
      out.push(`<rect class="band${a === 24 ? " delivery" : ""}" x="${x(a)}" y="${top - 8}" width="${x(b) - x(a)}" height="${Hh - top + 8}" opacity="${a === 24 ? 1 : i % 2 ? 0.55 : 0}"/>`);
      txt((x(a) + x(b)) / 2, 16, lab, "day");
    });
    // hour ticks inside the zoom (row 3)
    [0, 4, 8, 12, 16].forEach((h) => { out.push(`<line class="grid" x1="${x(h)}" x2="${x(h)}" y1="${top - 8}" y2="${Hh}" opacity=".45"/>`); txt(x(h), top - 14, hh(h)); });
    out.push(`<line class="break" x1="${x(16)}" x2="${x(16)}" y1="${top - 8}" y2="${Hh}"/>`);
    lanes.forEach((lab, i) => { txt(14, y(i) + laneH / 2 + 4, lab, "lane", "start"); out.push(`<line class="grid" x1="${L}" x2="${W - R}" y1="${y(i) + laneH}" y2="${y(i) + laneH}"/>`); });
    const bar = (i, a, b, cls, label) => {
      out.push(`<rect class="${cls}" x="${x(a)}" y="${y(i) + 10}" width="${Math.max(2, x(b) - x(a))}" height="11" rx="1"/>`);
      if (label) txt(x(a) + 4, y(i) + 35, label, "", "start");
    };
    // markers on row A (y+50) or B (y+20 for inputs); label beside the marker
    const mark = (i, h, cls, label, { row = 50, side = "right", shape = "circle" } = {}) => {
      const yy = y(i) + row, xx = x(h);
      out.push(shape === "diamond" ? `<path class="mk ${cls}" d="M${xx} ${yy - 6} L${xx + 6} ${yy} L${xx} ${yy + 6} L${xx - 6} ${yy} Z"/>` : `<circle class="mk ${cls}" cx="${xx}" cy="${yy}" r="5"/>`);
      if (label) txt(side === "right" ? xx + 10 : xx - 10, yy + 4, label, "", side === "right" ? "start" : "end");
    };
    // price: everything up to the end of D-1 is known (cleared on D-2); D's prices clear just after the gate
    // The x-axis is clock time. Up to the cut-off the model knows every price for delivery until 23:45 on D-1,
    // because that auction cleared on D-2: solid bar to the cut-off, hatched for the later hours already published.
    bar(0, -24, cut, "hist", "");
    out.push(`<rect class="hist-pub" x="${x(cut)}" y="${y(0) + 10}" width="${x(24) - x(cut)}" height="11" rx="1" fill="url(#hatch)"/>`);
    txt(x(-24) + 4, y(0) + 35, "known at the cut-off: every price to 23:45 on D−1, cleared on D−2 ●", "", "start");
    out.push(`<circle class="mk pub" cx="${x(12.9 - 24)}" cy="${y(0) + 15.5}" r="4"/>`);
    mark(0, 12.9, "pub", "D prices published ~12:55", { side: "left" });
    const priceRuns = (sched.score || ["14:45"]).filter((t) => !T.price_score_after || hm(t) >= hm(T.price_score_after));
    priceRuns.forEach((t, k) => mark(0, hm(t), "score", k === priceRuns.length - 1 ? `scored ${priceRuns.join(" and ")}` : "", { shape: "diamond" }));
    // wind, solar and load: metered up to the cut-off minus the publication lag; actuals arrive during D
    ["wind", "solar", "load"].forEach((t, k) => {
      const lag = T.lag_hours[t] || 0, days = T.score_after_days[t] || 2;
      bar(k + 1, -24, cut - lag, "hist", `history: metered to ${hh(cut - lag)} (cut-off − ${lag} h)`);
      bar(k + 1, 24 + lag, 48 + lag, "act", `actuals metered during D, ~${lag} h lag`);
      const sc = 24 * (days + 1) + scores[0];
      out.push(`<rect class="recheck" x="${x(sc)}" y="${y(k + 1) + 47}" width="${x(96) - x(sc)}" height="6" rx="1"/>`);
      mark(k + 1, sc, "score", `scored D+${days} ${sched.score ? sched.score[0] : ""} if complete`, { side: "left", shape: "diamond" });
      txt(x(96) - 2, y(k + 1) + 66, "re-checked to D+3 →", "", "end");
    });
    // inputs: the runs the cut-off rule admits, the ENTSO-E load forecast and the fuel quote
    const runs = {};
    for (const v of Object.values(T.nwp)) {
      const initUtc = Math.floor((cut - off - v.lag_hours) / v.cycle_hours) * v.cycle_hours;
      (runs[initUtc] ||= []).push(v.label.split(" ")[0]);
    }
    const order = Object.keys(runs).map(Number).sort((a, b) => a - b);
    order.forEach((u, k) => mark(4, u + off, "in", `${runs[u].join(" + ")} ${String(u).padStart(2, "0")} UTC run`, { row: k % 2 ? 50 : 22, side: k % 2 ? "left" : "right" }));
    mark(4, gate - 2, "in", `ENTSO-E load forecast, by ${hh(gate - 2)}`, { row: 50, side: "right" });
    mark(4, -0.5, "in", "fuel quote ≤ D−2", { row: 22, side: "left" });
    // cut-off and gate over everything (row 2 labels)
    out.push(`<line class="cut" x1="${x(cut)}" x2="${x(cut)}" y1="${top - 34}" y2="${Hh}"/>`);
    out.push(`<line class="gate" x1="${x(gate)}" x2="${x(gate)}" y1="${top - 34}" y2="${Hh}"/>`);
    txt(x(cut) - 5, top - 34, `cut-off ${T.issue_time}`, "", "end");
    txt(x(gate) + 5, top - 34, `gate ${T.gate}`, "", "start");
    const svg = $("timeline");
    svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
    const hatch = `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="6" height="6" fill="${alpha(css("--blue"), 0.12)}"/><line x1="0" y1="0" x2="0" y2="6" stroke="${alpha(css("--blue"), 0.55)}" stroke-width="2"/></pattern></defs>`;
    svg.innerHTML = hatch + out.join("");
    $("l-timeline").innerHTML = [
      `<span><i class="sw" style="background:${alpha(css("--blue"), 0.4)}"></i>history a model may use, up to the cut-off</span>`,
      `<span><i class="sw" style="background:repeating-linear-gradient(45deg, ${alpha(css("--blue"), 0.55)} 0 2px, ${alpha(css("--blue"), 0.12)} 2px 6px)"></i>prices for later hours of D−1, already published on D−2</span>`,
      `<span><i class="sw" style="background:${alpha(css("--green"), 0.6)}"></i>actuals being metered</span>`,
      `<span><i class="sw" style="background:${alpha(css("--green"), 0.25)}"></i>actuals re-checked, re-scored if revised</span>`,
      `<span><i style="border-top:2px dashed ${css("--blue-ink")}"></i>data cut-off, models run</span>`,
      `<span><i style="border-top:2px solid ${css("--amber")}"></i>day-ahead gate closure</span>`,
      `<span>○ input available · ● actuals published · ◆ scored</span>`,
      `<span>D−1 00:00–16:00 is stretched (dashed line); times are Brussels local.</span>`,
    ].join("");
  }

  // ---------- pipeline: run tracker ----------
  // one series: S scored, M due but actuals missing, W forecast but not due, F failed, K skipped
  const SER = {
    S: { k: "ok", icon: "✓", label: "forecast and scored" }, W: { k: "wait", icon: "…", label: "forecast, scoring not due yet" },
    M: { k: "miss", icon: "!", label: "due, actuals missing" }, F: { k: "err", icon: "✕", label: "forecast failed" },
    K: { k: "err", icon: "✕", label: "skipped" }, L: { k: "late", icon: "⏱", label: "finished after the gate, not scored" },
  };
  const trkZones = () => (S.zone === "all" ? D.config.zones : [S.zone]);
  const trkTargets = () => (S.target === "all" ? Object.keys(D.config.targets) : [S.target]);
  const serCode = (c, day, key) => { const x = c && c.s && c.s[key]; return x && day.phase === "live" && day.on_time === false && x !== "F" && x !== "K" ? "L" : x; };
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
    if (!$("tracker")) return;
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
      const rv = (d.revised || []).length ? `<small title="actuals revised and re-scored: ${esc(d.revised.join(", "))}">↻ ${d.revised.length} revised</small>` : "";
      return `<th title="${d.d} · ${d.started ? `run ${localTime(d.started)}–` : "finished "}${localTime(d.finished)} Brussels">${esc(shortDate(d.d))}${b}${rv}</th>`;
    }).join("")}</tr></thead>`;
    const zs = trkZones(), ts = trkTargets(), keys = zs.flatMap((z) => ts.map((t) => `${z}_${t}`));
    const cell = (c, d, m) => {
      if (!c) return null;
      const codes = keys.map((k) => serCode(c, d, k));
      const sel = S.trkSel && S.trkSel.d === d.d && S.trkSel.m === m;
      if (!codes.some(Boolean)) return `<span class="trk-na" title="${esc(m)} does not forecast this selection">·</span>`;
      if (keys.length === 1) {
        const st = SER[codes[0]];
        return `<button type="button" class="trk ${st.k}" data-d="${d.d}" data-m="${esc(m)}" aria-pressed="${sel}" title="${esc(m)} · ${keys[0].replace("_", " ")} · ${d.d}: ${esc(st.label)}">${st.icon}</button>`;
      }
      // several series: one small square per series, zones in rows and targets in columns
      const counts = {};
      codes.forEach((x) => { if (x) counts[SER[x].label] = (counts[SER[x].label] || 0) + 1; });
      const tip = Object.entries(counts).map(([l, n]) => `${n} ${l}`).join(", ");
      return `<button type="button" class="trk grid" style="--c:${ts.length}" data-d="${d.d}" data-m="${esc(m)}" aria-pressed="${sel}" title="${esc(m)} · ${d.d}: ${esc(tip)}">` +
        codes.map((x, i) => `<i class="${x ? SER[x].k : "na"}" title="${keys[i].replace("_", " ")}: ${x ? SER[x].label : "not forecast by this model"}"></i>`).join("") + "</button>";
    };
    const body = models.map((m) => `<tr><td class="model"><i style="background:${color(m)}"></i>${nm(m)}</td>${days.map((d) => `<td>${cell(T.cells[d.d] && T.cells[d.d][m], d, m) || ""}</td>`).join("")}</tr>`).join("");
    $("tracker").innerHTML = head + `<tbody>${body}</tbody>`;
    $("tracker").onclick = (e) => { const b = e.target.closest("button.trk"); if (!b) return; S.trkSel = { d: b.dataset.d, m: b.dataset.m }; renderTracker(); };
    $("l-tracker").innerHTML = [["ok", "✓", "forecast and scored"], ["wait", "…", "forecast, scoring not due"],
      ["miss", "!", "due, actuals missing"], ["late", "⏱", "after the gate"], ["err", "✕", "failed or skipped"]]
      .map(([k, i, l]) => `<span style="display:flex;align-items:center;gap:7px"><span class="trk ${k}" style="display:inline-grid;place-items:center;cursor:default">${i}</span>${l}</span>`).join("") +
      (keys.length > 1 ? `<span style="display:flex;align-items:center;gap:7px"><span class="trk grid key" style="--c:${ts.length};cursor:default">${keys.map(() => "<i class=\"wait\"></i>").join("")}</span>one square per series: rows ${esc(zs.join(", "))}, columns ${esc(ts.map((t) => TARGET_LABEL[t] || t).join(", "))}</span>` : "");
    const det = $("trk-detail");
    if (!S.trkSel) { det.hidden = true; return; }
    const d = T.days.find((x) => x.d === S.trkSel.d), c = T.cells[S.trkSel.d] && T.cells[S.trkSel.d][S.trkSel.m];
    if (!d || !c) { det.hidden = true; return; }
    const st = cellStatus(c, d);
    det.hidden = false;
    det.innerHTML = `<h4>${esc(S.trkSel.m)} · issue ${d.d} → delivery ${addDays(d.d, 1)} <span class="tag ${d.phase === "live" ? "live" : "bt"}">${d.phase === "live" ? "Live" : "Backtest"}</span></h4>
      <p>${st.icon} ${esc(st.label)}.</p>
      <p class="mono">${d.started ? `run ${localTime(d.started)}–${localTime(d.finished)}` : `finished ${localTime(d.finished)}`} Brussels${d.phase === "live" ? ` · gate ${esc(D.config.gate)} · ${d.on_time ? "on time" : "LATE"}` : ""} ·
        forecasts ${c.ok}/${d.n_series} ok${c.err ? `, ${c.err} failed` : ""}${c.skip ? `, ${c.skip} skipped` : ""} · scored ${c.scored} of ${c.due} due · model time ${c.sec} s</p>
      <div class="tablewrap"><table class="trk-series"><thead><tr><th></th>${Object.keys(D.config.targets).map((t) => `<th>${esc(TARGET_LABEL[t] || t)}</th>`).join("")}</tr></thead><tbody>${D.config.zones.map((z) => `<tr><td class="model">${z}</td>${Object.keys(D.config.targets).map((t) => {
        const key = `${z}_${t}`, x = serCode(c, d, key), a = (d.act || {})[key];
        if (!x) return `<td class="na">not forecast</td>`;
        const st = SER[x];
        const actual = a === "frozen" ? "actuals in" : a === "prov" ? "actuals provisional" : x === "W" ? "" : "no actuals yet";
        return `<td><span class="trk ${st.k} mini">${st.icon}</span> ${esc(st.label)}${actual ? `<br><small>${actual}</small>` : ""}${c.why && c.why[key] ? `<br><small class="why">${esc(c.why[key])}</small>` : ""}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table></div>
      ${(d.revised || []).length ? `<p>Actuals revised after first scoring, re-scored: ${esc(d.revised.join(", "))}.</p>` : ""}
      <p><a href="explorer.html?day=${d.d}">Open ${d.d} in the Forecast Explorer →</a></p>`;
  }

  // ---------- forecast explorer ----------
  async function loadDay(d) {
    if (!S.days[d]) S.days[d] = fetch(`data/days/${d}.json`, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    return S.days[d];
  }
  // explorer ranking: points over every scored day (live and backtest) for the explorer's zone and target
  function explorerRanking() {
    const rows = ROWS.filter((r) => !REF.has(r.model) && r.zone === S.x.zone && r.target === S.x.target && r[S.metric] != null
      && (S.phase === "all" || r.phase === S.phase));
    const sum = {}, n = {};
    for (const x of dayTable(rows)) for (const m in x.pts) { sum[m] = (sum[m] || 0) + x.pts[m].reduce((a, b) => a + b, 0); n[m] = (n[m] || 0) + x.pts[m].length; }
    return Object.keys(sum).sort((a, b) => sum[b] / n[b] - sum[a] / n[a]);
  }

  function renderExplorerControls() {
    if (!$("x-day")) return;
    let ds = Object.keys(D.days || {}).sort().reverse();
    if (S.phase !== "all") ds = ds.filter((d) => D.days[d] === S.phase);
    if (!ds.includes(S.x.day)) S.x.day = ds[0];
    options($("x-day"), ds.map((d) => ({ k: d, label: `${d} → ${addDays(d, 1)} · ${D.days[d]}` })), S.x.day);
    options($("x-zone"), D.config.zones.map((z) => ({ k: z, label: z })), S.x.zone);
    options($("x-target"), Object.keys(D.config.targets).map((t) => ({ k: t, label: TARGET_LABEL[t] || t })), S.x.target);
  }
  async function renderExplorerChart() {
    if (!$("x-day")) return;
    const day = await loadDay(S.x.day);
    const card = $("x-card");
    if (!day) { $("x-meta").innerHTML = "No forecasts stored for this day."; return; }
    const key = `${S.x.zone}_${S.x.target}`;
    const ser = day.series[key] || { actual: null, models: {} };
    const unit = D.config.targets[S.x.target].unit;
    const bt = day.phase === "backtest";
    card.classList.toggle("is-bt", bt);
    const ranked = explorerRanking();
    const cands = MODELS.filter((m) => !REF.has(m) && inField(m) && ser.models[m])
      .sort((a, b) => (ranked.indexOf(a) + 1 || 1e3) - (ranked.indexOf(b) + 1 || 1e3));
    const top = new Set(cands.slice(0, S.xTop));
    const models = MODELS.filter((m) => ser.models[m] && (top.has(m) || REF.has(m) || FAM[m] === "tso"));   // TSO: the yardstick
    const xt = $("x-top");
    if (xt) { xt.max = String(Math.max(1, cands.length)); xt.value = String(Math.min(S.xTop, cands.length));
      $("x-top-val").textContent = S.xTop >= cands.length ? `All ${cands.length}` : `Top ${S.xTop} of ${cands.length}`; }
    const bandModels = models.filter((m) => ser.models[m].lo);
    if (S.x.band !== "" && !bandModels.includes(S.x.band)) S.x.band = bandModels[0] || "";
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
    const actualLabel = ser.provisional ? "Actual so far" : "Actual";
    if (ser.actual) datasets.push({ label: actualLabel, data: ser.actual, borderColor: actualColor(), backgroundColor: actualColor(), borderWidth: 3, pointRadius: 0, tension: 0, endLabel: actualLabel, endColor: actualColor(), order: -1 });
    for (const m of models) {
      const c = color(m);
      datasets.push({ label: m, data: ser.models[m].p, hidden: S.hidden.has(m), borderColor: c, backgroundColor: c, borderWidth: REF.has(m) ? 2 : 1.6,
        borderDash: REF.has(m) ? [6, 4] : undefined, pointRadius: 0, pointHoverRadius: 4, tension: 0, endLabel: REF.has(m) ? `${m} (ref)` : CAVEAT[m] ? m + "*" : m, endColor: c });
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
      ${ser.actual ? (ser.provisional ? `<span class="tag pending" title="${day.provisional_fetched_utc ? `fetched ${esc(localTime(day.provisional_fetched_utc))} Brussels` : ""}">Provisional actuals to ${esc(meteredTo(day, ser.actual))}, not scored yet</span>` : "") : '<span class="tag pending">Actuals not published yet</span>'}
      ${ser.actual ? `<span>Actuals <b>${esc(actualSource(day, key))}</b></span>` : ""}
      ${(day.revisions || []).filter((r) => r.series === key).map((r) => `<span class="tag pending" title="re-fetched ${esc(String(r.checked_utc).slice(0, 16))} UTC">Actuals revised ${(100 * r.rel_change).toFixed(2)}%, re-scored</span>`).join("")}
      ${tsoNote(models, day)}`;
    legend($("l-day"), models, ser.actual ? `<span class="actual" style="display:flex;align-items:center;gap:7px"><i></i>${actualLabel}</span>` : "");

    renderInputs(day);
    const sc = ROWS.filter((r) => r.issue_date === day.issue_date && r.zone === S.x.zone && r.target === S.x.target);
    const tb = $("x-table");
    if (!sc.length) { tb.innerHTML = `<tbody><tr><td class="empty">Not scored yet. ${S.x.target === "price" ? "Prices are scored the afternoon of the issue day." : "Wind, solar and load are scored at 07:00 the day after delivery, once the whole day is published."}</td></tr></tbody>`; return; }
    const best = (k) => Math.min(...sc.map((r) => (k === "bias" ? Math.abs(r[k]) : r[k])).filter((v) => v != null));
    const xth = (k, l) => `<th class="r"><span class="lbl" data-tip="${k}">${l}</span></th>`;
    tb.innerHTML = `<thead><tr><th>Model</th>${xth("mae", "MAE")}${xth("rmse", "RMSE")}${xth("bias", "Bias")}${xth("pinball", "Pinball")}${xth("rel_mae", "Rel. MAE")}</tr></thead><tbody>` +
      models.map((m) => sc.find((r) => r.model === m)).filter(Boolean).map((r) => `<tr class="${bt ? "is-bt" : ""}">
        <td class="model"><i style="background:${color(r.model)}"></i>${nm(r.model)}</td>
        ${["mae", "rmse", "bias", "pinball", "rel_mae"].map((k) => `<td class="num ${(k === "bias" ? Math.abs(r[k]) : r[k]) === best(k) ? "best" : ""}">${fmt(r[k], k === "rel_mae" ? 3 : 1)}</td>`).join("")}
      </tr>`).join("") + `</tbody>`;
  }

  // ---------- wiring ----------
  let SHOWN = new Set();
  // the top-N slider lives on the leaderboard; elsewhere every model is shown
  const shown = (m) => PAGE !== "leaderboard" || SHOWN.has(m);
  function updateShown() {
    let st = standings(DAYS, S.phase).filter((x) => inField(x.model));
    if (!st.length) st = standings(DAYS, "all").filter((x) => inField(x.model));
    const ranked = st.map((x) => x.model);
    const rest = MODELS.filter((m) => !ranked.includes(m) && inField(m));
    // the grid operators' forecasts are the yardstick: always shown, whatever the top-N slider says
    SHOWN = new Set([...[...ranked, ...rest].slice(0, S.topN), ...MODELS.filter((m) => FAM[m] === "tso" && inField(m))]);
    const total = new Set([...ranked, ...rest]).size;
    const el = $("f-top");
    el.max = String(total);
    el.value = String(Math.min(S.topN, total));
    $("f-top-val").textContent = S.topN >= total ? `All ${total}` : `Top ${S.topN} of ${total}`;
  }

  function renderAll() {
    if (!$("standings")) return;
    saveFilters();
    DAYS = dayTable(raceRows());
    updateShown();
    renderControls();
    renderStandings();
    renderCharts();
    renderHeatmap();
    renderTracker();
    refreshTip();
  }

  // the metric is no longer a filter: races rank by Rel. MAE (within a race the same order as MAE), and the
  // standings show all four metrics as sortable columns
  const FILTER_KEYS = ["zone", "target", "phase", "covFilter", "topN", "sort"];
  function loadFilters() {
    try { const f = JSON.parse(localStorage.getItem("ft-filters") || "{}"); for (const k of FILTER_KEYS) if (f[k] != null) S[k] = f[k]; return f; }
    catch (e) { return {}; }
  }
  function saveFilters() {
    try { localStorage.setItem("ft-filters", JSON.stringify(Object.fromEntries(FILTER_KEYS.map((k) => [k, S[k]])))); } catch (e) { /* storage unavailable */ }
  }

  async function init() {
    try {
      D = await (await fetch("data/summary.json", { cache: "no-cache" })).json();
    } catch (e) {
      const host = document.querySelector("main section .wrap");
      if (host) host.insertAdjacentHTML("afterbegin", '<p class="lede">Data is not available yet.</p>');
      return;
    }
    const cols = D.score_columns;
    ROWS = D.scores.map((a) => Object.fromEntries(cols.map((c, i) => [c, a[i]])));
    MODELS = D.models.map((m) => m.name);
    FAM = Object.fromEntries(D.models.map((m) => [m.name, m.family || m.name]));
    USES_COV = Object.fromEntries(D.models.map((m) => [m.name, (m.inputs || []).some((k) => COVARIATE_INPUTS.includes(k))]));
    REF = new Set(D.models.filter((m) => m.reference).map((m) => m.name));
    CAVEAT = Object.fromEntries(D.models.filter((m) => m.caveat).map((m) => [m.name, m.caveat]));
    const famRank = (m) => { const f = FAM[m] || m; const i = FAMILY_ORDER.indexOf(f); return f === "tso" ? 98 : f === "baseline" ? 99 : i < 0 ? 50 : i; };
    MODELS.sort((a, b) => famRank(a) - famRank(b) || a.localeCompare(b));
    for (const r of ROWS) if (!MODELS.includes(r.model)) MODELS.push(r.model);
    const hasLive = ROWS.some((r) => r.phase === "live");
    S.phase = hasLive ? "live" : "backtest";
    loadFilters();
    if (!["live", "backtest", "all"].includes(S.phase) || (S.phase === "live" && !hasLive)) S.phase = hasLive ? "live" : "backtest";
    if (!METRICS.some((m) => m.k === S.metric)) S.metric = "rel_mae";

    document.querySelectorAll("[data-k]").forEach((el) => { if (el.tagName === "SPAN" && D.config[el.dataset.k] != null) el.textContent = D.config[el.dataset.k]; });
    const on = (id, ev, fn) => { const el = $(id); if (el) el[ev] = fn; };
    if (typeof Chart !== "undefined") Chart.defaults.font.family = "'Source Sans 3', -apple-system, sans-serif";
    const rerender = () => { renderAll(); renderModels(); renderExplorerChart(); renderTimeline(); renderTracker(); };
    afterFonts = rerender;
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
    let wasNarrow = narrow();
    window.addEventListener("resize", () => { if (narrow() !== wasNarrow) { wasNarrow = narrow(); if ($("standings")) renderCharts(); renderExplorerChart(); } });

    if ($("updated")) {
      const up = new Date(D.generated_utc.replace("Z", ":00Z"));
      $("updated").textContent = up.toLocaleString("en-GB", { timeZone: D.config.timezone, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    }
    if (PAGE === "leaderboard") {
      on("f-zone", "onchange", (e) => { S.zone = e.target.value; if (!rawAllowed()) S.view = "rank"; renderAll(); });
      on("f-target", "onchange", (e) => { S.target = e.target.value; if (!rawAllowed()) S.view = "rank"; renderAll(); });
      on("f-top", "oninput", (e) => { S.topN = +e.target.value; renderAll(); });
      renderAll();
      renderTracker();
    }
    if (PAGE === "explorer") {
      if (S.zone !== "all" && D.config.zones.includes(S.zone)) S.x.zone = S.zone;
      if (S.target !== "all" && S.target in D.config.targets) S.x.target = S.target;
      S.xTop = S.topN;
      const q = new URLSearchParams(location.search);   // deep links: explorer.html?day=2026-10-03&zone=BE&target=wind
      if (q.get("day") && D.days && D.days[q.get("day")]) { S.x.day = q.get("day"); if (S.phase !== "all" && D.days[S.x.day] !== S.phase) S.phase = "all"; }
      if (D.config.zones.includes(q.get("zone"))) S.x.zone = q.get("zone");
      if (q.get("target") in D.config.targets) S.x.target = q.get("target");
      const changed = () => { S.zone = S.x.zone; S.target = S.x.target; S.topN = S.xTop; saveFilters();
        history.replaceState(null, "", `?day=${S.x.day}&zone=${S.x.zone}&target=${S.x.target}`); };
      const segs = () => {
        seg($("x-phase"), [{ k: "live", label: "Live", disabled: !hasLive }, { k: "backtest", label: "Backtest" }, { k: "all", label: "All" }], S.phase,
          (k) => { S.phase = k; segs(); renderExplorerControls(); changed(); renderExplorerChart(); });
        seg($("x-cov"), [{ k: "all", label: "All" }, { k: "with", label: "With" }, { k: "without", label: "Without" }], S.covFilter,
          (k) => { S.covFilter = k; segs(); changed(); renderExplorerChart(); });
      };
      segs();
      on("x-day", "onchange", (e) => { S.x.day = e.target.value; changed(); renderExplorerChart(); });
      on("x-zone", "onchange", (e) => { S.x.zone = e.target.value; changed(); renderExplorerChart(); });
      on("x-target", "onchange", (e) => { S.x.target = e.target.value; changed(); renderExplorerChart(); });
      on("x-band", "onchange", (e) => { S.x.band = e.target.value; renderExplorerChart(); });
      on("x-top", "oninput", (e) => { S.xTop = +e.target.value; changed(); renderExplorerChart(); });
      on("cov-var", "onchange", (e) => { S.cov = e.target.value; renderExplorerChart(); });
      renderExplorerControls();
      changed();
      renderExplorerChart();
    }
    if (PAGE === "methodology") {
      renderTimeline();
      renderCentroids();
      renderModels();
    }
    // content renders after load, so redo the jump to a #section link
    if (location.hash.length > 1) { const el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); }
  }

  // links to sections that moved off the one-page site
  const MOVED = { forecasts: "explorer.html", inputs: "explorer.html", pipeline: "methodology.html", method: "methodology.html", submit: "methodology.html", models: "methodology.html", centroids: "methodology.html" };
  if (PAGE === "leaderboard" && MOVED[location.hash.slice(1)]) location.replace(MOVED[location.hash.slice(1)] + location.hash);

  // draw at once; when the web fonts arrive, draw the canvases again with the right text metrics
  let afterFonts = null;
  init().then(() => { if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => afterFonts && afterFonts()); });
})();
