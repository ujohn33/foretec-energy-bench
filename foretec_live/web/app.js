/* Foretec Energy Bench leaderboard. Reads data/summary.json and data/days/<issue_date>.json. */
(() => {
  "use strict";

  const METRICS = [
    { k: "rel_mae", label: "Rel. MAE", d: 3, unitless: true },
    { k: "mae", label: "MAE", d: 1 },
    { k: "rmse", label: "RMSE", d: 1 },
    { k: "pinball", label: "Pinball", d: 2 },
  ];
  const zoneLabel = (z) => (D.config && D.config.zone_labels && D.config.zone_labels[z]) || z;   // DE -> DE-LU
  const TARGET_LABEL = { price: "Day-ahead price", wind: "Wind", solar: "Solar", load: "Load" };
  const S = {
    zone: "all", target: "all", metric: "rel_mae", phase: null, win: "all", agg: "mean", view: "rank", topN: 10, covFilter: "all", xTop: 10, xPin: new Set(), pin: new Set(), details: false, sort: { k: "pos", dir: 1 },
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
  // every existing slot (light: normal-vision dE >= 19, CVD dE >= 13; dark step likewise); slots 10 (tabby, brown) and
  // 11 (timestral, cyan) clear every slot by dE >= 32. New families get grey.
  const FAMILY_ORDER = ["chronos", "timesfm", "moirai", "tirex", "toto", "sundial", "ttm", "nwp", "t0", "tabby", "timestral"];
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

  // ---------- races (Elo score, see below) ----------
  // A race is one issue day x one series. Its field is the models with a scored forecast in it. With n in the
  // field, a model at rank r scores 100 (n - r) / (n - 1), ties sharing the average rank: the winner gets 100
  // whatever the field size. A model whose run failed or was skipped is not in that race at all (no 0), so the
  // standing is its average over the races it finished; the run tracker still shows the failures.
  const MIN_RACES = 24;

  // ---------- Diebold-Mariano ----------
  // Loss = Rel. MAE of each race, so series of different scale pool. The loss differential of A against B
  // is averaged per issue day over the series both delivered, and the test runs on that daily series:
  // Newey-West variance (Bartlett, floor(T^1/3) lags, since weather regimes make days autocorrelated),
  // Harvey-Leybourne-Newbold small-sample correction, Student t with T - 1 degrees of freedom, two-sided.
  const DM_MIN_DAYS = 5;
  const SHOW_DM = false;   // the column is hidden for now (too cluttered for what it tells); the test stays available
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5, ser = 1.000000000190015;
    t -= (x + 0.5) * Math.log(t);
    for (const k of c) ser += k / ++y;
    return -t + Math.log(2.5066282746310005 * ser / x);
  }
  function betacf(a, b, x) {
    const TINY = 1e-30; let c = 1, d = 1 - (a + b) * x / (a + 1);
    d = 1 / (Math.abs(d) < TINY ? TINY : d); let h = d;
    for (let m = 1; m <= 200; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
      d = 1 + aa * d; d = 1 / (Math.abs(d) < TINY ? TINY : d); c = 1 + aa / c; c = Math.abs(c) < TINY ? TINY : c; h *= d * c;
      aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
      d = 1 + aa * d; d = 1 / (Math.abs(d) < TINY ? TINY : d); c = 1 + aa / c; c = Math.abs(c) < TINY ? TINY : c;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-12) break;
    }
    return h;
  }
  function ibeta(a, b, x) {   // regularized incomplete beta I_x(a, b)
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  const tTwoSided = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));
  /** Per-day loss of a model: issue day -> (series -> Rel. MAE). */
  function dailyLoss(rows, model) {
    const out = new Map();
    for (const r of rows) if (r.model === model && r.rel_mae != null) {
      if (!out.has(r.issue_date)) out.set(r.issue_date, new Map());
      out.get(r.issue_date).set(r.zone + "_" + r.target, r.rel_mae);
    }
    return out;
  }
  /** DM test of loss map a against b: mean differential (a - b, below 0 = a better), p-value, days. */
  function dmTest(a, b) {
    const d = [];
    for (const [day, la] of [...a.entries()].sort()) {
      const lb = b.get(day); if (!lb) continue;
      const diffs = [...la].filter(([k]) => lb.has(k)).map(([k, v]) => v - lb.get(k));
      if (diffs.length) d.push(mean(diffs));
    }
    const T = d.length;
    if (T < DM_MIN_DAYS) return { T, p: null, diff: T ? mean(d) : null };
    const m = mean(d), e = d.map((x) => x - m);
    const gamma = (k) => { let s = 0; for (let i = k; i < T; i++) s += e[i] * e[i - k]; return s / T; };
    const L = Math.floor(Math.cbrt(T));
    let v = gamma(0);
    for (let k = 1; k <= L; k++) v += 2 * (1 - k / (L + 1)) * gamma(k);
    if (!(v > 0)) return { T, p: m === 0 ? 1 : 0, diff: m };
    const t = m / Math.sqrt(v / T) * Math.sqrt((T - 1) / T);   // HLN with h = 1
    return { T, p: tTwoSided(t, T - 1), diff: m };
  }
  const fmtP = (p) => (p == null ? "–" : p < 0.001 ? "<0.001" : p.toFixed(p < 0.01 ? 3 : 2));
  function dmCell(r, isLeader) {
    if (isLeader) return `<td class="r v dm lead">#1</td>`;
    if (!r || r.p == null) return `<td class="r v dm">–</td>`;
    const sig = r.p < 0.05, cls = !sig ? "tie" : r.diff < 0 ? "better" : "worse";
    const mark = !sig ? "≈ " : r.diff < 0 ? "▲ " : "";
    return `<td class="r v dm ${cls}" title="Rel. MAE ${r.diff >= 0 ? "+" : ""}${r.diff.toFixed(3)} per day vs #1 over ${r.T} days">${mark}${fmtP(r.p)}</td>`;
  }

  /** Per issue day: the head-to-head games, each model's share of games won (in %), average rank and raw mean across the selected series. */
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
      const ranks = {}, vals = {}, pts = {}, pairs = new Map();
      for (const [key, list] of ser) {
        const sorted = [...list].sort((a, b) => a[S.metric] - b[S.metric]);
        const n = sorted.length;
        if (n < 2) continue;
        // every pair of models in the race is one game: the lower error wins, equal errors split it
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
          const a = sorted[i], b = sorted[j];
          const [x, y, w] = a.model < b.model ? [a.model, b.model, 1] : [b.model, a.model, 0];
          const g = pairs.get(x + "\t" + y) || { wx: 0, n: 0 };
          g.wx += a[S.metric] === b[S.metric] ? 0.5 : w;   // sorted: a is never worse than b
          g.n += 1;
          pairs.set(x + "\t" + y, g);
        }
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
      out.push({ d, phase: phaseOf(d), avgRank, raw, points, pts, pairs, n: Object.keys(pts).length });
    }
    return out;
  }

  // ---------- Elo score: Bradley-Terry on the Elo scale ----------
  // All pairwise games of the selected races are fitted at once (Bradley-Terry, MM algorithm of Hunter 2004), so the
  // rating does not depend on the order of the races, unlike a running Elo. One virtual drawn game against an average
  // opponent keeps a model that wins every game finite. Scale: 1000 = the field's average, +400 = 10:1 odds of winning.
  // 95% interval: the fit repeated on issue days resampled with replacement (days, not races, are independent).
  function btMatrices(days) {
    const models = [...new Set(days.flatMap((x) => Object.keys(x.pts)))];
    const ix = Object.fromEntries(models.map((m, i) => [m, i]));
    const k = models.length;
    const perDay = days.map((x) => {
      const W = new Float64Array(k * k), N = new Float64Array(k * k);
      for (const [key, g] of x.pairs) {
        const [a, b] = key.split("\t"), i = ix[a], j = ix[b];
        if (i == null || j == null) continue;
        W[i * k + j] += g.wx; W[j * k + i] += g.n - g.wx;
        N[i * k + j] += g.n; N[j * k + i] += g.n;
      }
      return { W, N };
    });
    return { models, k, perDay };
  }
  function btFit(k, W, N, start) {
    let p = start ? Float64Array.from(start) : new Float64Array(k).fill(1);
    const wins = new Float64Array(k);
    for (let i = 0; i < k; i++) { let s = 0.5; for (let j = 0; j < k; j++) s += W[i * k + j]; wins[i] = s; }
    for (let it = 0; it < 300; it++) {
      const q = new Float64Array(k);
      let delta = 0;
      for (let i = 0; i < k; i++) {
        let den = 1 / (p[i] + 1);                       // the virtual game against an average opponent (strength 1)
        for (let j = 0; j < k; j++) { const n = N[i * k + j]; if (n) den += n / (p[i] + p[j]); }
        q[i] = wins[i] / den;
      }
      let g = 0; for (let i = 0; i < k; i++) g += Math.log(q[i]);
      g = Math.exp(g / k);                              // geometric mean 1: the average model rates 1000
      for (let i = 0; i < k; i++) { q[i] /= g; delta = Math.max(delta, Math.abs(Math.log(q[i] / p[i]))); }
      p = q;
      if (delta < 1e-7) break;
    }
    return p;
  }
  const toElo = (p) => 1000 + 400 * Math.log10(p);
  function eloScores(days, boot = 0) {
    const { models, k, perDay } = btMatrices(days);
    if (k < 2) return {};
    const sum = (ids) => {
      const W = new Float64Array(k * k), N = new Float64Array(k * k);
      for (const d of ids) { const m = perDay[d]; for (let t = 0; t < k * k; t++) { W[t] += m.W[t]; N[t] += m.N[t]; } }
      return [W, N];
    };
    const all = btFit(k, ...sum(perDay.map((_, i) => i)));
    const out = Object.fromEntries(models.map((m, i) => [m, { elo: toElo(all[i]) }]));
    if (boot && perDay.length > 1) {
      const draws = models.map(() => []);
      let seed = 42;
      const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);   // reproducible
      for (let b = 0; b < boot; b++) {
        const ids = perDay.map(() => Math.floor(rnd() * perDay.length));
        const p = btFit(k, ...sum(ids), all);
        p.forEach((v, i) => draws[i].push(toElo(v)));
      }
      models.forEach((m, i) => {
        const s = draws[i].sort((a, b) => a - b);
        out[m].lo = s[Math.floor(0.025 * (s.length - 1))]; out[m].hi = s[Math.ceil(0.975 * (s.length - 1))];
      });
    }
    return out;
  }

  // the rating window: all time, or the last N calendar days up to the latest scored issue day of the period
  const WINDOWS = { all: null, d30: 30, d7: 7 };
  function periodDays(days, phase) {
    const sel = days.filter((x) => phase === "all" || x.phase === phase);
    const n = WINDOWS[S.win];
    if (!n || !sel.length) return sel;
    const from = addDays(sel[sel.length - 1].d, 1 - n);
    return sel.filter((x) => x.d >= from);
  }

  // ---------- relative errors: pooled per series, then aggregated across series ----------
  // Per series (zone x target), every quarter-hour of every day in the period is pooled first and only then
  // divided by the baseline (naive_weekly) on the same days: Rel. MAE = sum |e| / sum |e_base|, Rel. RMSE =
  // sqrt(sum e^2 / sum e_base^2). The series the zone and target filters select are then combined by the mean,
  // the median or the geometric mean (S.agg) of those per-series values.
  const median = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), h = s.length >> 1; return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2; };
  const AGGS = { mean, median, gmean: (a) => (a.length && a.every((x) => x > 0) ? Math.exp(mean(a.map(Math.log))) : null) };
  function pooledStats(days, refs) {
    const inSel = (r) => days.has(r.issue_date) && (S.zone === "all" || r.zone === S.zone) && (S.target === "all" || r.target === S.target) && r.mae != null;
    const base = new Map(ROWS.filter((r) => r.model === D.config.baseline && inSel(r)).map((r) => [`${r.issue_date}|${r.zone}_${r.target}`, r]));
    const acc = {};
    for (const r of ROWS) {
      if (REF.has(r.model) !== refs || !inSel(r)) continue;
      const key = `${r.zone}_${r.target}`, b = base.get(`${r.issue_date}|${key}`), n = r.n || 96;
      const a = ((acc[r.model] ||= {})[key] ||= { ae: 0, se: 0, bae: 0, bse: 0, n: 0, pin: 0, npin: 0, wis: 0, wink: 0, cov: 0, nw: 0, cw: 0, bw: 0 });
      a.n += n;
      a.ae += n * r.mae; a.se += n * r.rmse * r.rmse;
      if (r.pinball != null) { a.pin += n * r.pinball; a.npin += n; }
      if (r.wis != null) {
        a.wis += n * r.wis; a.wink += n * r.winkler; a.cov += n * r.cover; a.nw += n;
        if (b && b.wis != null) { a.cw += n * r.wis; a.bw += n * b.wis; }
      }
      if (b) { a.bae += n * b.mae; a.bse += n * b.rmse * b.rmse; a.both = (a.both || 0) + 1; a.cae = (a.cae || 0) + n * r.mae; a.cse = (a.cse || 0) + n * r.rmse * r.rmse; }
    }
    const out = {};
    for (const m in acc) {
      const ser = Object.values(acc[m]);
      const rel = ser.filter((a) => a.bae > 0).map((a) => a.cae / a.bae), relr = ser.filter((a) => a.bse > 0).map((a) => Math.sqrt(a.cse / a.bse));
      const n = ser.reduce((x, a) => x + a.n, 0), npin = ser.reduce((x, a) => x + a.npin, 0), nw = ser.reduce((x, a) => x + a.nw, 0);
      const relw = ser.filter((a) => a.bw > 0).map((a) => a.cw / a.bw);
      const sum = (k) => ser.reduce((x, a) => x + a[k], 0);
      out[m] = {
        rel: AGGS[S.agg](rel), relr: AGGS[S.agg](relr), nser: ser.length,
        mae: ser.reduce((x, a) => x + a.ae, 0) / n, rmse: Math.sqrt(ser.reduce((x, a) => x + a.se, 0) / n),
        pinball: npin ? ser.reduce((x, a) => x + a.pin, 0) / npin : null,
        relw: relw.length ? AGGS[S.agg](relw) : null,
        wis: nw ? sum("wis") / nw : null, winkler: nw ? sum("wink") / nw : null, cover: nw ? sum("cov") / nw : null,
      };
    }
    return out;
  }

  function standings(days, phase) {
    const sel = periodDays(days, phase);
    const set = new Set(sel.map((x) => x.d));
    const ps = pooledStats(set, false);
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
    const elo = eloScores(sel, 200);
    return list.map((s) => ({
      ...s,
      score: elo[s.model] ? elo[s.model].elo : null, eloLo: elo[s.model] && elo[s.model].lo, eloHi: elo[s.model] && elo[s.model].hi,
      provisional: s.races < need,
      avgRank: mean(s.ranks),
      best: Math.max(...s.dayScores), worst: Math.min(...s.dayScores), days: s.dayScores.length,
      ...(ps[s.model] || {}),
    })).sort((a, b) => (a.provisional - b.provisional) || b.score - a.score || (a.avgRank ?? 99) - (b.avgRank ?? 99));
  }

  const narrow = () => window.innerWidth < 640;
  const aggText = () => (singleSeries() ? "" : `Across the selected series: the ${{ mean: "mean", median: "median", gmean: "geometric mean" }[S.agg]} of the per-series values.`);
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
    options($("f-zone"), [{ k: "all", label: "All zones" }, ...D.config.zones.map((z) => ({ k: z, label: zoneLabel(z) }))], S.zone);
    options($("f-target"), [{ k: "all", label: "All targets" }, ...Object.keys(D.config.targets).map((t) => ({ k: t, label: TARGET_LABEL[t] || t }))], S.target);
    const hasLive = ROWS.some((r) => r.phase === "live");
    // "Rated over": the live runs of the last 7 or 30 days or all of them, or the backtest (all its days)
    seg($("f-win"), [
      { k: "d7", label: "Last 7 days", disabled: !hasLive, title: "Live runs of the last 7 days up to the latest scored issue day" },
      { k: "d30", label: "Last 30 days", disabled: !hasLive, title: "Live runs of the last 30 days up to the latest scored issue day" },
      { k: "all", label: "All time", disabled: !hasLive, title: "Every live run" },
      { k: "bt", label: "Backtest", title: "The backtest: every model re-run on past issue days with the data public at their cut-off" },
    ], S.phase === "backtest" ? "bt" : S.win, (k) => {
      if (k === "bt") { S.phase = "backtest"; S.win = "all"; } else { S.phase = "live"; S.win = k; }
      renderAll();
    });
    seg($("f-agg"), [
      { k: "mean", label: "Mean" }, { k: "median", label: "Median" }, { k: "gmean", label: "Geo. mean" },
    ], S.agg, (k) => { S.agg = k; renderAll(); });
    // one zone and one target: a single series, so mean, median and geometric mean are the same number
    const one = singleSeries();
    $("f-agg").classList.toggle("off", one);
    $("f-agg").querySelectorAll("button").forEach((x) => { x.disabled = one; });
    $("f-agg").title = one ? "One series selected (one zone, one target): nothing to aggregate. Pick all zones or all targets." : "";
    seg($("f-cov"), [
      { k: "all", label: "All" },
      { k: "with", label: "With", title: "Models that also use NWP weather, the ENTSO-E load forecast or fuel cost, plus the benchmarks (baselines and the grid operators' forecasts)" },
      { k: "without", label: "Without", title: "Models that only see the target's own history, plus the benchmarks (baselines and the grid operators' forecasts)" },
    ], S.covFilter, (k) => { S.covFilter = k; renderAll(); renderExplorerChart(); });
    if ($("f-cols")) seg($("f-cols"), [
      { k: "key", label: "Key columns" },
      { k: "all", label: "All columns", title: "Also the 95% interval, MAE, RMSE, WIS, Winkler 80%, coverage 80%, best and worst day" },
    ], S.details ? "all" : "key", (k) => { S.details = k === "all"; renderAll(); });
    updateFilterSummary();
    seg($("f-view"), [
      { k: "rank", label: "Avg rank", tip: "rank" },
      { k: "value", label: "Value", tip: "value", disabled: !rawAllowed(), title: rawAllowed() ? "" : "Pick one zone and one target to compare raw values across days" },
    ], S.view, (k) => { S.view = k; renderAll(); });
  }

  // the "More" button lists the secondary filters' state; the compact bar (while scrolling) summarises all of them
  function updateFilterSummary() {
    if (!$("f-sum-text")) return;
    if (PAGE === "explorer") return updateExplorerSummary();
    if (PAGE === "value") return updateValueSummary();
    const win = { d7: "last 7 days", d30: "last 30 days", all: "all time" }[S.win];
    const cov = { all: "", with: "with covariates", without: "without covariates" }[S.covFilter];
    const top = $("f-top-val").textContent;
    const agg = singleSeries() ? "" : { mean: "", median: "median", gmean: "geo. mean" }[S.agg];
    $("more-sum").textContent = "· " + [cov || "all models", top, agg].filter(Boolean).join(" · ");
    const zone = S.zone === "all" ? "All zones" : zoneLabel(S.zone), target = S.target === "all" ? "All targets" : TARGET_LABEL[S.target] || S.target;
    const rated = S.phase === "backtest" ? "Backtest" : win[0].toUpperCase() + win.slice(1);
    $("f-sum-text").innerHTML = [zone, target, rated, cov, top, agg].filter(Boolean).map((x, i) => i < 3 ? `<b>${esc(x)}</b>` : esc(x)).join(" · ");
  }
  function updateValueSummary() {
    const phase = { live: "live", backtest: "backtest", all: "live and backtest" }[S.phase];
    const mode = S.v.mode === "imb" ? "remainder at imbalance price" : "remainder as error cost";
    $("more-sum").textContent = `· ${phase} · ${mode}`;
    const zone = S.v.zone === "all" ? "All zones" : zoneLabel(S.v.zone);
    $("f-sum-text").innerHTML = [`<b>${esc(zone)}</b>`, `<b>${esc(TARGET_LABEL[S.v.target] || S.v.target)}</b>`, `<b>${esc(S.v.model || "")}</b>`, `p${S.v.p}`, phase, mode].join(" · ");
  }
  function updateExplorerSummary() {
    const cov = { all: "all models", with: "with covariates", without: "without covariates" }[S.covFilter];
    const band = S.x.band ? `band ${S.x.band}` : "no band";
    $("more-sum").textContent = "· " + [cov, $("x-top-val").textContent, band].join(" · ");
    const zone = zoneLabel(S.x.zone), target = TARGET_LABEL[S.x.target] || S.x.target;
    $("f-sum-text").innerHTML = [`<b>${esc(zone)}</b>`, `<b>${esc(target)}</b>`, `<b>${esc(S.x.day)} → ${esc(addDays(S.x.day, 1))}</b>`, esc(cov), esc(band)].join(" · ");
  }
  function setupFilterBar() {
    const fb = document.querySelector(".filterbar"), more = $("f-more"), panel = $("more-panel");
    if (!fb || !more) return;
    more.onclick = () => {
      const open = panel.hidden;
      panel.hidden = !open;
      more.setAttribute("aria-expanded", String(open));
      more.querySelector(".more-label").textContent = open ? "Less" : "More";
    };
    // while scrolling down the page the bar shrinks to one summary line; a click on it opens the filters again until the
    // reader scrolls on. Hysteresis and a scroll correction keep the page from jumping when the bar changes height.
    let openAt = null;
    const update = () => {
      if (getComputedStyle(fb).position !== "sticky") { fb.classList.remove("compact"); return; }
      const y = scrollY, on = fb.classList.contains("compact");
      if (openAt != null && Math.abs(y - openAt) > 240) openAt = null;
      const want = openAt == null && (on ? y > 140 : y > 340);
      if (want === on) return;
      const h = fb.offsetHeight;
      fb.classList.toggle("compact", want);
      if (y > 0) window.scrollBy(0, fb.offsetHeight - h);
    };
    addEventListener("scroll", update, { passive: true });
    $("f-sum").onclick = () => { openAt = scrollY; update(); openAt = scrollY; };
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
    pts: () => ["Elo", `Elo score: a Bradley–Terry rating on the Elo scale. A race is one issue day and one series; in each race every pair of models is one game, won by the lower ${metricDef().label} (equal errors split it). All games of the selected period are fitted at once, so the order of the races does not matter. 1000 is the field's average; 400 points more means 10:1 odds of winning a game. Beating a strong model counts more than beating a weak one. A failed run is left out of the race.`],
    ci: () => ["95% interval", "How much the Elo score would move if the set of issue days had been different: the 2.5th to 97.5th percentile of 200 refits, each on the period's issue days drawn at random with replacement (a bootstrap over issue days; whole days, because all races of a day share the same weather). Overlapping intervals mean the order of two models is not settled yet. Sorting uses the lower end."],
    races: () => ["N evaluations", `Scored forecasts in the period, one per issue day and series (a race). A model with fewer than ${MIN_RACES}, or half of the most any model has, is provisional and listed after the ranked ones.`],
    dm: () => ["Diebold–Mariano test vs #1", `p-value of the Diebold–Mariano test that the model and the leader are equally accurate. Loss is Rel. MAE, differenced per race and averaged per issue day over the series both delivered; Newey–West variance, Harvey–Leybourne–Newbold correction, two-sided. ≈ marks no significant difference at 5% (statistically tied with #1), ▲ significantly more accurate than #1 on Rel. MAE, a bare p significantly less accurate. Needs ${DM_MIN_DAYS} common days.`],
    rank: () => ["Average rank", `In every selected series (zone and target) the models are ranked by ${metricDef().label} for the day: 1 is best, and tied models share the average place. Ranks are averaged over the series, then over the days. Lower is better.`],
    rel: () => ["Rel. MAE", `Per series (zone and target), all of the model's absolute errors in the period are pooled and divided by those of ${D.config.baseline} (the same quarter-hour one week earlier) on the same days; below 1 beats that baseline. ${aggText()} Lower is better.`],
    relw: () => ["Rel. WIS", `The weighted interval score of the model's quantiles, relative to ${D.config.baseline} on the same days and series: it rewards a forecast distribution that is both sharp and calibrated. Below 1 beats the baseline. Only models that publish quantiles have one. ${aggText()} Lower is better.`],
    wis: () => ["WIS", `Weighted interval score: the score of the 10-90%, 20-80%, 30-70% and 40-60% intervals and the median together, in the series' unit (${units()}); equal to twice the average pinball loss. Lower is better.${mixedUnits()}`],
    winkler: () => ["Winkler 80%", `Interval score of the 10-90% interval: its width, plus 10 times the distance by which the actual falls outside it. Lower is better.${mixedUnits()}`],
    cover: () => ["Coverage 80%", "Share of quarter-hours with the actual inside the 10-90% interval. A calibrated forecast covers 80%; less is overconfident, more is too wide. Sorted by the distance from 80%."],
    relr: () => ["Rel. RMSE", `Like Rel. MAE with squared errors: per series sqrt(sum of squared errors / the baseline's), on the same days. Large misses weigh more. ${aggText()} Lower is better.`],
    nser: () => ["Series", "How many of the selected series (zone × target) the model forecasts. Relative errors of models with different series are averaged over different sets, so compare them with care."],
    agg: () => ["Across series", "How the per-series Rel. MAE and Rel. RMSE are combined when several zones or targets are selected."],
    rel_mae: () => ["Rel. MAE", `A model's MAE divided by the MAE of ${D.config.baseline} (the same quarter-hour one week earlier) on the same day and series. Below 1 beats that baseline; 0.5 halves its error. It has no unit, so prices, wind and solar can be averaged together. Lower is better.`],
    mae: () => ["MAE", `Mean absolute error: the average size of the miss over the day's quarter-hours, in the series' own unit (${units()}). Lower is better.${mixedUnits()}`],
    rmse: () => ["RMSE", `Root mean squared error: like MAE, but errors are squared before averaging, so a few large misses (price spikes, wind ramps) cost more than many small ones. Lower is better.${mixedUnits()}`],
    pinball: () => ["Pinball loss", `Quantile loss averaged over the 10th to 90th percentiles. It scores the whole forecast band: narrow bands score well only if the outcome falls inside them as often as they claim. Only models that publish quantiles have one. Lower is better.${mixedUnits()}`],
    bias: () => ["Bias", "Mean of forecast minus actual. Positive means the model forecasts too high on average, negative too low. Closest to 0 is best."],
    best: () => ["Best day", "The model's best issue day: the share of its head-to-head games it won that day, in % (averaged over that day's races)."],
    worst: () => ["Worst day", "The model's worst issue day: the share of its head-to-head games it won that day, in % (averaged over that day's races)."],
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
    ci: { get: (s) => s.eloLo, dir: -1 },
    races: { get: (s) => s.races, dir: -1 },
    rank: { get: (s) => s.avgRank, dir: 1 },
    rel: { get: (s) => s.rel, dir: 1 },
    relr: { get: (s) => s.relr, dir: 1 },
    nser: { get: (s) => s.nser, dir: -1 },
    mae: { get: (s) => s.mae, dir: 1 },
    rmse: { get: (s) => s.rmse, dir: 1 },
    pinball: { get: (s) => s.pinball, dir: 1 },
    relw: { get: (s) => s.relw, dir: 1 },
    wis: { get: (s) => s.wis, dir: 1 },
    winkler: { get: (s) => s.winkler, dir: 1 },
    cover: { get: (s) => (s.cover == null ? null : Math.abs(s.cover - 0.8)), dir: 1 },   // closest to 80% first
    dm: { get: (s) => s.dm, dir: -1 },
    best: { get: (s) => s.best, dir: -1 },
    worst: { get: (s) => s.worst, dir: -1 },
  };
  const DETAIL_COLS = ["ci", "dm", "nser", "mae", "rmse", "pinball", "wis", "winkler", "cover", "best", "worst"];   // shown with "All columns" only
  function sortKey() { const k = S.sort && S.sort.k; return SORT_COLS[k] && (S.details || !DETAIL_COLS.includes(k)) ? k : "pos"; }
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
  function sortTh(key, label, tip, right = true, cls = "") {
    const on = sortKey() === key, dir = on && key === S.sort.k ? S.sort.dir : 1;
    const aria = on ? (dir < 0 ? "descending" : "ascending") : "none";
    return `<th class="${[right ? "r" : "", cls].join(" ").trim()}" aria-sort="${aria}"><button type="button" class="sort" data-sort="${key}" ${tip ? `data-tip="${tip}"` : ""}>` +
      `<span class="${tip ? "lbl" : ""}">${esc(label)}</span><i aria-hidden="true">${on ? (dir < 0 ? "▼" : "▲") : "↕"}</i></button></th>`;
  }

  const pctCell = (v) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(0) + "%");

  // ---------- sections ----------
  let DAYS = [];

  function renderStandings() {
    const full = standings(DAYS, S.phase);
    // place in the whole field, so the number does not change with the covariate filter or the top-N slider
    const place = Object.fromEntries(full.filter((x) => !x.provisional).map((x, i) => [x.model, i + 1]));
    const st = full.filter((x) => shown(x.model));
    const bt = S.phase === "backtest";
    // Diebold-Mariano against #1 of the whole field, on the races of the selected period
    const inPeriod = new Set(periodDays(DAYS, S.phase).map((x) => x.d));
    const leader = Object.keys(place).find((m) => place[m] === 1);
    const dmRows = raceRows("rel_mae").filter((r) => inPeriod.has(r.issue_date));
    const leadLoss = SHOW_DM && leader ? dailyLoss(dmRows, leader) : null;
    const dm = Object.fromEntries(st.map((s) => [s.model, leadLoss && s.model !== leader ? dmTest(dailyLoss(dmRows, s.model), leadLoss) : null]));
    $("standings").classList.toggle("is-bt", bt);

    const t = $("standings-table");
    const unit = S.target === "all" ? "" : ` (${D.config.targets[S.target].unit})`;
    const det = S.details;
    const multi = det && !singleSeries();   // the series count only matters when more than one series is combined
    const dmOn = det && SHOW_DM;
    // the Elo bar spans 700..1300; the 95% interval is drawn on it as a whisker
    const eloX = (v) => Math.min(100, Math.max(0, (v - 700) / 6));
    t.innerHTML = `<thead><tr>${sortTh("pos", "#", "pos", false)}${sortTh("model", "Model", "", false)}${sortTh("pts", "Elo", "pts", false, "elo")}${det ? sortTh("ci", "95% interval", "ci", true, "hide-sm") : ""}${dmOn ? sortTh("dm", "DM p vs #1", "dm") : ""}${sortTh("rank", "Avg rank", "rank")}
      ${sortTh("rel", "Rel. MAE", "rel")}${sortTh("relr", "Rel. RMSE", "relr")}${sortTh("relw", "Rel. WIS", "relw")}${multi ? sortTh("nser", "Series", "nser") : ""}${det ? sortTh("mae", "MAE" + unit, "mae") + sortTh("rmse", "RMSE" + unit, "rmse") + sortTh("wis", "WIS" + unit, "wis") + sortTh("winkler", "Winkler 80%" + unit, "winkler") + sortTh("cover", "Coverage 80%", "cover") : ""}
      ${det ? sortTh("best", "Best day", "best") + sortTh("worst", "Worst day", "worst") : ""}${sortTh("races", "N evaluations", "races")}</tr></thead><tbody>` +
      sortRows(st.map((s) => ({ ...s, pos: place[s.model] ?? 1e3, dm: s.model === leader ? 2 : dm[s.model] && dm[s.model].p }))).map((s) => { const medal = !s.provisional && s.pos <= 3 ? ` medal m${s.pos}` : ""; return `<tr class="${bt ? "is-bt" : ""}${s.provisional ? " prov" : ""}${medal}">
        <td class="pos">${s.provisional ? `<span data-tip="races">prov.</span>` : medal ? `<span>${s.pos}</span>` : s.pos}</td>
        <td class="model"><i style="background:${color(s.model)}"></i>${nm(s.model)}</td>
        <td class="elo" ${s.eloLo != null ? `title="95% interval ${fmt(s.eloLo, 0)}–${fmt(s.eloHi, 0)}"` : ""}><div class="bar"><em><span style="width:${Math.round(eloX(s.score))}%;background:${bt ? btColor(s.model) : color(s.model)}"></span>${s.eloLo != null ? `<i class="ci" style="left:${eloX(s.eloLo).toFixed(1)}%;width:${(eloX(s.eloHi) - eloX(s.eloLo)).toFixed(1)}%"></i>` : ""}</em><b>${fmt(s.score, 0)}</b></div></td>
        ${det ? `<td class="r v hide-sm">${s.eloLo != null ? `${fmt(s.eloLo, 0)}–${fmt(s.eloHi, 0)}` : "–"}</td>` : ""}${dmOn ? dmCell(dm[s.model], s.model === leader) : ""}<td class="r v">${fmt(s.avgRank, 2)}</td>
        <td class="r v">${fmt(s.rel, 3)}</td><td class="r v">${fmt(s.relr, 3)}</td><td class="r v">${fmt(s.relw, 3)}</td>${multi ? `<td class="r v">${s.nser ?? "–"}</td>` : ""}${det ? `<td class="r v">${fmt(s.mae, 1)}</td><td class="r v">${fmt(s.rmse, 1)}</td><td class="r v">${fmt(s.wis, 1)}</td><td class="r v">${fmt(s.winkler, 1)}</td><td class="r v">${pctCell(s.cover)}</td>
        <td class="r v">${fmt(s.best, 0)}</td><td class="r v">${fmt(s.worst, 0)}</td>` : ""}<td class="r v">${s.races}</td>
      </tr>`; }).join("") + "</tbody>";
    if (!st.length) t.innerHTML = `<tbody><tr><td class="empty">No scored days for this selection yet.</td></tr></tbody>`;
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
    const refRel = filteredRows("rel_mae", true).filter((r) => inPeriod.has(r.issue_date));
    const refs = [...REF].filter((m) => refRel.some((r) => r.model === m));
    const refStats = pooledStats(inPeriod, true);
    if (refs.length && st.length) {
      const only = S.target === "all" ? " (wind and solar only)" : "";
      t.querySelector("tbody").insertAdjacentHTML("beforeend", refs.map((m) => {
        const mine = refRel.filter((r) => r.model === m), q = refStats[m] || {};
        return `<tr class="ref"><td class="pos">ref.</td><td class="model"><i style="background:${color(m)}"></i>${nm(m)}</td>
          <td class="elo note">Reference, not ranked: published after the gate${only}</td>${det ? '<td class="r v hide-sm">–</td>' : ""}${dmOn ? dmCell(leadLoss && dmTest(dailyLoss(mine, m), leadLoss), false) : ""}<td class="r v">–</td>
          <td class="r v">${fmt(q.rel, 3)}</td><td class="r v">${fmt(q.relr, 3)}</td><td class="r v">${fmt(q.relw, 3)}</td>${multi ? `<td class="r v">${q.nser ?? "–"}</td>` : ""}${det ? `<td class="r v">${fmt(q.mae, 1)}</td><td class="r v">${fmt(q.rmse, 1)}</td><td class="r v">${fmt(q.wis, 1)}</td><td class="r v">${fmt(q.winkler, 1)}</td><td class="r v">${pctCell(q.cover)}</td>
          <td class="r v">–</td><td class="r v">–</td>` : ""}<td class="r v">${mine.length}</td></tr>`;
      }).join(""));
    }
  }

  // the issue days the time charts show: the Period filter's phase only (All shows both)
  const phaseDays = () => (S.phase === "all" ? DAYS : DAYS.filter((x) => x.phase === S.phase));
  const SMOOTH_DAYS = 7, SMOOTH_MIN = 3, EMPHASIS = 3;

  // focus helpers shared by the time charts: lines drawn in full (the leaders, plus models clicked in the legend) are
  // bold and labelled, the rest faint and unlabelled; the tooltip lists only the full lines and counts the others
  function styleFocus(ds, m, isFull, faintA = 0.3) {
    Object.assign(ds, { pointRadius: 0, tension: 0.25 });
    if (isFull) return Object.assign(ds, { borderWidth: 3, pointHoverRadius: 4, order: 0 });
    const faint = alpha(color(m), faintA);
    return Object.assign(ds, { borderWidth: 1.1, pointHoverRadius: 0, endLabel: null, borderColor: faint, segment: { ...ds.segment, borderColor: () => faint }, order: 1 });
  }
  function focusTooltip(o, models, isFull) {
    const n = models.filter((m) => !isFull(m) && !S.hidden.has(m)).length;
    o.plugins.tooltip.filter = (it) => isFull(it.dataset.label);
    o.plugins.tooltip.callbacks.footer = () => (n ? `+ ${n} more: click one in the legend to highlight it` : "");
    Object.assign(o.plugins.tooltip, { footerColor: css("--ink2"), footerFont: { size: 11.5, weight: 400 } });
    return o;
  }
  /** legend: a faint model toggles in `pins` (drawn in full or not); a leading one is hidden or shown */
  function focusLegend(el, models, lead, pins, isFull, rerender, extra = "") {
    el.innerHTML = extra + models.map((m) => `<button type="button" data-m="${esc(m)}" class="${isFull(m) ? "" : "faint"}" aria-pressed="${!S.hidden.has(m)}"
      title="${lead(m) ? "click to hide or show" : pins.has(m) ? "click to fade" : "click to highlight"}"><i style="border-top-color:${color(m)}"></i>${nm(m)}</button>`).join("");
    el.onclick = (e) => {
      const b = e.target.closest("button[data-m]"); if (!b) return;
      const m = b.dataset.m;
      if (lead(m)) S.hidden.has(m) ? S.hidden.delete(m) : S.hidden.add(m);
      else pins.has(m) ? pins.delete(m) : pins.add(m);
      rerender();
    };
  }

  function renderCharts() {
    const days = phaseDays();
    const labels = days.map((x) => shortDate(x.d));
    const band = bandRange(days);
    const md = metricDef();
    const rank = S.view === "rank";
    const models = MODELS.filter((m) => shown(m) && days.some((x) => x.pts[m] != null));
    const phaseNote = (i) => (S.phase !== "all" ? "" : days[i] && days[i].phase === "backtest" ? " · backtest" : " · live");

    // trailing 7-issue-day mean of the daily value (from 3 days on), so day-to-day noise does not hide the trend;
    // the podium's top three (and models clicked in the legend) are drawn in full, the rest faint
    const top = new Set(standings(DAYS, S.phase).filter((x) => shown(x.model) && !x.provisional).slice(0, EMPHASIS).map((x) => x.model));
    const isLead = (m) => top.has(m), isFull = (m) => top.has(m) || S.pin.has(m);
    const faintA = models.length > 8 ? 0.2 : 0.3;
    const smooth = (vals) => vals.map((_, i) => {
      const w = vals.slice(Math.max(0, i - SMOOTH_DAYS + 1), i + 1).filter((v) => v != null);
      return vals[i] != null && w.length >= SMOOTH_MIN ? w.reduce((a, b) => a + b, 0) / w.length : null;
    });
    const perf = models.map((m) => {
      return styleFocus(lineDataset(m, smooth(days.map((x) => (rank ? x.avgRank[m] : x.raw[m]) ?? null)), days), m, isFull(m), faintA);
    });
    draw("c-perf", {
      type: "line",
      data: { labels, datasets: perf },
      options: (() => {
        const o = baseOptions({
          band, yTitle: (rank ? "Average rank" : md.label + unitSuffix()) + `, ${SMOOTH_DAYS}-day average`,
          itemSort: (a, b) => a.raw - b.raw,
          tooltip: { title: (it) => `${SMOOTH_DAYS} issue days to ${days[it[0].dataIndex].d}${phaseNote(it[0].dataIndex)}`, label: (c) => ` ${c.dataset.label}: ${fmt(c.raw, rank ? 2 : md.d)}` },
        });
        if (rank) { o.scales.y.reverse = true; o.scales.y.min = 1; o.scales.y.ticks.stepSize = 1; }
        return focusTooltip(o, models, isFull);
      })(),
    });
    focusLegend($("l-perf"), models, isLead, S.pin, isFull, renderCharts);

    // running Elo over the live days only (the backtest is a separate exercise): the Bradley-Terry fit on the live
    // races up to each issue day. A model's line starts once it has raced on MIN_ELO_DAYS live days, so its first
    // day or two do not swing it to extremes.
    const MIN_ELO_DAYS = 3;
    const liveDays = DAYS.filter((x) => x.phase === (S.phase === "backtest" ? "backtest" : "live"));
    const running = liveDays.map((_, i) => eloScores(liveDays.slice(0, i + 1)));
    const race = models.map((m) => {
      let n = 0;
      const ds = lineDataset(m, running.map((e, i) => { if (liveDays[i].pts[m]) n += 1; return n >= MIN_ELO_DAYS && e[m] ? e[m].elo : null; }), liveDays);
      return styleFocus(ds, m, isFull(m), faintA);
    });
    draw("c-race", {
      type: "line", data: { labels: liveDays.map((x) => shortDate(x.d)), datasets: race },
      options: focusTooltip(baseOptions({
        band: bandRange(liveDays), yTitle: `Elo (running, ${S.phase === "backtest" ? "backtest" : "live"} days)`, itemSort: (a, b) => b.raw - a.raw,
        tooltip: { title: (it) => liveDays[it[0].dataIndex].d, label: (c) => ` ${c.dataset.label}: ${fmt(c.raw, 0)}` },
      }), models, isFull),
    });
    focusLegend($("l-race"), models, isLead, S.pin, isFull, renderCharts);
  }

  function renderHeatmap() {
    const days = phaseDays();
    const st = standings(DAYS, S.phase).filter((x) => shown(x.model));
    const t = $("heatmap");
    if (!days.length) { t.innerHTML = ""; return; }
    const lerp = (a, b, w) => mix(a, b, w);
    const head = `<thead><tr><th>Model</th>${days.map((x) => `<th class="${x.phase === "backtest" ? "is-bt" : "live"}" title="${x.d} · ${x.phase}">${esc(shortDate(x.d))}</th>`).join("")}<th><span class="lbl" data-tip="heat">Avg</span></th></tr></thead>`;
    const body = st.map((s) => {
      const cells = days.map((x) => {
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
      ["Fuel cost", esc(inp.fuel), "daily closes count from the end of their date", "Yahoo Finance: TTF=F, CO2.L"],
    ];
    t.innerHTML = head + `<tbody>` +
      `<tr><td>Cut-off / gate</td><td class="wrap">${fmtUtc(inp.cutoff_utc)} · gate ${esc(D.config.gate)} CET/CEST</td><td class="wrap">${inp.n_columns} covariate columns frozen at the cut-off</td>` +
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
      : ["DE", "DK1", "DK2"].includes(S.x.zone) && S.x.target !== "load" ? "published at 18:00 on D-1, after the gate"
      : day.phase === "live" ? "not published yet" : "not available for this day";
    return `<span class="tag pending">TSO forecast: ${esc(why)}</span>`;
  }

  // end of the last published quarter-hour of a provisional actual, CET/CEST
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
    // hours relative to D-1 00:00 CET/CEST; all of D-1 up to 16:00 is zoomed in (runs, cut-off, gate, auction)
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
      `<span>D−1 00:00–16:00 is stretched (dashed line); times are CET/CEST.</span>`,
    ].join("");
  }

  // ---------- pipeline: run tracker ----------
  // one series: S scored, M due but actuals missing, W forecast but not due, F failed, K skipped, V voided
  const SER = {
    S: { k: "ok", icon: "✓", label: "forecast and scored" }, W: { k: "wait", icon: "…", label: "forecast, scoring not due yet" },
    M: { k: "miss", icon: "!", label: "due, actuals missing" }, F: { k: "err", icon: "✕", label: "forecast failed" },
    K: { k: "err", icon: "✕", label: "skipped" }, L: { k: "late", icon: "⏱", label: "finished after the gate, not scored" },
    V: { k: "void", icon: "⊘", label: "voided, not scored" },
  };
  const trkZones = () => (S.zone === "all" ? D.config.zones : [S.zone]);
  const trkTargets = () => (S.target === "all" ? Object.keys(D.config.targets) : [S.target]);
  const serCode = (c, day, key) => { const x = c && c.s && c.s[key]; return x && day.phase === "live" && day.on_time === false && x !== "F" && x !== "K" ? "L" : x; };
  function cellStatus(c, day) {
    if (!c) return null;
    if (c.err || c.skip) return { k: "err", icon: "✕", label: `error: ${c.err} failed, ${c.skip} skipped` };
    if (day.phase === "live" && day.on_time === false) return { k: "late", icon: "⏱", label: "finished after the gate, not scored" };
    const nv = c.void || 0, vtxt = nv ? `; ${nv} voided` : "";
    if (c.due === 0 && nv === c.ok) return { k: "void", icon: "⊘", label: `${nv} series voided, not scored` };
    if (c.due === 0) return { k: "wait", icon: "…", label: `forecasts generated, scoring not due yet${vtxt}` };
    if (c.scored < c.due) return { k: "miss", icon: "!", label: `scored ${c.scored} of ${c.due} due series (awaiting actuals)${vtxt}` };
    if (c.due + nv < c.ok) return { k: "part", icon: "◐", label: `scored ${c.scored} of ${c.ok}; the rest is not due yet${vtxt}` };
    return { k: "ok", icon: "✓", label: nv ? `scored ${c.scored} of ${c.ok}${vtxt}` : `all ${c.ok} series forecast and scored` };
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
      return `<th title="${d.d} · ${d.started ? `run ${localTime(d.started)}–` : "finished "}${localTime(d.finished)} CET/CEST">${esc(shortDate(d.d))}${b}${rv}</th>`;
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
      <p class="mono">${d.started ? `run ${localTime(d.started)}–${localTime(d.finished)}` : `finished ${localTime(d.finished)}`} CET/CEST${d.phase === "live" ? ` · gate ${esc(D.config.gate)} · ${d.on_time ? "on time" : "LATE"}` : ""} ·
        forecasts ${c.ok}/${d.n_series} ok${c.err ? `, ${c.err} failed` : ""}${c.skip ? `, ${c.skip} skipped` : ""} · scored ${c.scored} of ${c.due} due · model time ${c.sec} s</p>
      <div class="tablewrap"><table class="trk-series"><thead><tr><th></th>${Object.keys(D.config.targets).map((t) => `<th>${esc(TARGET_LABEL[t] || t)}</th>`).join("")}</tr></thead><tbody>${D.config.zones.map((z) => `<tr><td class="model">${z}</td>${Object.keys(D.config.targets).map((t) => {
        const key = `${z}_${t}`, x = serCode(c, d, key), a = (d.act || {})[key];
        if (!x) return `<td class="na">not forecast</td>`;
        const st = SER[x];
        const actual = a === "frozen" ? "actuals in" : a === "prov" ? "actuals provisional" : x === "W" ? "" : "no actuals yet";
        return `<td><span class="trk ${st.k} mini">${st.icon}</span> ${esc(st.label)}${actual ? `<br><small>${actual}</small>` : ""}${c.why && c.why[key] ? `<br><small class="why">${esc(c.why[key])}</small>` : ""}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table></div>
      ${(d.revised || []).length ? `<p>Actuals revised after first scoring, re-scored: ${esc(d.revised.join(", "))}.</p>` : ""}
      ${d.phase === "live" ? `<p><a href="explorer.html?day=${d.d}">Open ${d.d} in the Forecast Explorer →</a></p>` : ""}`;
  }

  // ---------- forecast explorer ----------
  async function loadDay(d) {
    if (!S.days[d]) S.days[d] = fetch(`data/days/${d}.json`, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    return S.days[d];
  }
  // explorer ranking: Elo over every scored day (live and backtest) for the explorer's zone and target
  const xPhase = () => (ROWS.some((r) => r.phase === "live") ? "live" : "all");
  function explorerRanking() {
    const rows = ROWS.filter((r) => !REF.has(r.model) && r.zone === S.x.zone && r.target === S.x.target && r[S.metric] != null
      && (xPhase() === "all" || r.phase === xPhase()));
    const e = eloScores(dayTable(rows));
    return Object.keys(e).sort((a, b) => e[b].elo - e[a].elo);
  }

  function renderExplorerControls() {
    if (!$("x-day")) return;
    let ds = Object.keys(D.days || {}).sort().reverse();
    if (xPhase() !== "all") ds = ds.filter((d) => D.days[d] === xPhase());
    if (!ds.includes(S.x.day)) S.x.day = ds[0];
    options($("x-day"), ds.map((d) => ({ k: d, label: `${d} → ${addDays(d, 1)} · ${D.days[d]}` })), S.x.day);
    options($("x-zone"), D.config.zones.map((z) => ({ k: z, label: zoneLabel(z) })), S.x.zone);
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
    // the band: by default the best-ranked model that publishes quantiles
    const bandModels = cands.filter((m) => top.has(m) && ser.models[m].lo);
    if (S.x.band !== "" && !bandModels.includes(S.x.band)) S.x.band = bandModels[0] || "";
    options($("x-band"), [{ k: "", label: "No band" }, ...bandModels.map((m) => ({ k: m, label: `${m} 10–90%` }))], S.x.band);

    const t0 = new Date(day.t0);
    const tf = new Intl.DateTimeFormat("en-GB", { timeZone: D.config.timezone, hour: "2-digit", minute: "2-digit" });
    const labels = Array.from({ length: day.n }, (_, i) => tf.format(new Date(t0.getTime() + i * 900e3)));
    const datasets = [];
    if (S.x.band && ser.models[S.x.band]) {
      const c = color(S.x.band);
      datasets.push({ label: "_lo", data: ser.models[S.x.band].lo, borderWidth: 0, pointRadius: 0, fill: false });
      datasets.push({ label: "_hi", data: ser.models[S.x.band].hi, borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: alpha(c, 0.1) });
    }
    const actualLabel = ser.provisional ? "Actual so far" : "Actual";
    if (ser.actual) datasets.push({ label: actualLabel, data: ser.actual, borderColor: actualColor(), backgroundColor: actualColor(), borderWidth: 3, pointRadius: 0, tension: 0, endLabel: actualLabel, endColor: actualColor(), order: -1 });
    // the top EMPHASIS models by Elo (and the TSO yardstick) are drawn in full and labelled, the rest faint and
    // unlabelled; every model stays in the tooltip and the legend
    // models clicked in the legend (S.xPin) are drawn in full too; the faint lines fade further when there are many
    const lead = new Set(cands.slice(0, EMPHASIS));
    const isTso = (m) => REF.has(m) || FAM[m] === "tso";
    const full = (m) => lead.has(m) || isTso(m) || S.xPin.has(m);
    const faintA = models.length > 8 ? 0.16 : 0.28;
    for (const m of models) {
      const c = color(m), tso = isTso(m), f = full(m);
      datasets.push({ label: m, data: ser.models[m].p, hidden: S.hidden.has(m), borderColor: f ? c : alpha(c, faintA), backgroundColor: c,
        borderWidth: f ? 2.2 : 1, borderDash: tso ? [6, 4] : undefined, pointRadius: 0, pointHoverRadius: f ? 4 : 0, tension: 0, order: f ? 0 : 1,
        endLabel: !f ? null : REF.has(m) ? `${m} (ref)` : CAVEAT[m] ? m + "*" : m, endColor: c });
    }
    const o = baseOptions({
      yTitle: `${TARGET_LABEL[S.x.target]} (${unit})`,
      itemSort: (a, b) => (b.dataset.label === "Actual") - (a.dataset.label === "Actual"),
      tooltip: { label: (c) => (c.dataset.label.startsWith("_") ? null : ` ${c.dataset.label}: ${fmt(c.raw, 1)} ${unit}`) },
    });
    // the tooltip lists the actual and the lines drawn in full; the faint ones are counted
    const nFaint = models.filter((m) => !full(m) && !S.hidden.has(m)).length;
    o.plugins.tooltip.filter = (it) => !it.dataset.label.startsWith("_") && (!ser.models[it.dataset.label] || full(it.dataset.label));
    o.plugins.tooltip.callbacks.footer = () => (nFaint ? `+ ${nFaint} faint model${nFaint > 1 ? "s" : ""}: click one in the legend to highlight it` : "");
    Object.assign(o.plugins.tooltip, { footerColor: css("--ink2"), footerFont: { size: 11.5, weight: 400 } });
    o.scales.x.ticks.maxTicksLimit = 13;
    draw("c-day", { type: "line", data: { labels, datasets }, options: o });

    const delivered = addDays(day.issue_date, 1);
    $("x-meta").innerHTML = `<span>Locked <b>${day.issue_date} ${esc(D.config.issue_time)}</b></span><span>Gate <b>${esc(D.config.gate || D.config.issue_time)}</b></span><span>Delivery <b>${delivered}</b></span>
      <span class="tag ${bt ? "bt" : "live"}">${bt ? "Backtest" : "Live"}</span>
      ${ser.actual ? (ser.provisional ? `<span class="tag pending" title="${day.provisional_fetched_utc ? `fetched ${esc(localTime(day.provisional_fetched_utc))} CET/CEST` : ""}">Provisional actuals to ${esc(meteredTo(day, ser.actual))}, not scored yet</span>` : "") : '<span class="tag pending">Actuals not published yet</span>'}
      ${ser.actual ? `<span>Actuals <b>${esc(actualSource(day, key))}</b></span>` : ""}
      ${(day.revisions || []).filter((r) => r.series === key).map((r) => `<span class="tag pending" title="re-fetched ${esc(String(r.checked_utc).slice(0, 16))} UTC">Actuals revised ${(100 * r.rel_change).toFixed(2)}%, re-scored</span>`).join("")}
      ${tsoNote(models, day)}`;
    // explorer legend: click a faint model to draw it in full (again to fade it); click a leading or TSO model to hide it
    updateFilterSummary();
    const ld = $("l-day");
    ld.innerHTML = (ser.actual ? `<span class="actual" style="display:flex;align-items:center;gap:7px"><i></i>${actualLabel}</span>` : "") +
      models.map((m) => `<button type="button" data-m="${esc(m)}" class="${full(m) ? "" : "faint"}" aria-pressed="${!S.hidden.has(m)}"
        title="${lead.has(m) || isTso(m) ? "click to hide or show" : S.xPin.has(m) ? "click to fade" : "click to highlight"}"><i style="border-top-color:${color(m)}"></i>${nm(m)}</button>`).join("");
    ld.onclick = (e) => {
      const b = e.target.closest("button[data-m]"); if (!b) return;
      const m = b.dataset.m;
      if (lead.has(m) || isTso(m)) S.hidden.has(m) ? S.hidden.delete(m) : S.hidden.add(m);
      else S.xPin.has(m) ? S.xPin.delete(m) : S.xPin.add(m);
      renderExplorerChart();
    };

    renderInputs(day);
    const sc = ROWS.filter((r) => r.issue_date === day.issue_date && r.zone === S.x.zone && r.target === S.x.target);
    const tb = $("x-table");
    if (!sc.length) { tb.innerHTML = `<tbody><tr><td class="empty">Not scored yet. ${S.x.target === "price" ? "Prices are scored the afternoon of the issue day." : "Wind, solar and load are scored at 07:00 the day after delivery, once the whole day is published."}</td></tr></tbody>`; return; }
    // best per column: lowest, except bias (closest to 0) and coverage (closest to 80%)
    const dev = (k, v) => (v == null ? null : k === "bias" ? Math.abs(v) : k === "cover" ? Math.abs(v - 0.8) : v);
    const best = (k) => Math.min(...sc.map((r) => dev(k, r[k])).filter((v) => v != null));
    const xth = (k, l) => `<th class="r"><span class="lbl" data-tip="${k}">${l}</span></th>`;
    const COLS = ["mae", "rmse", "bias", "wis", "winkler", "cover", "rel_mae"];
    tb.innerHTML = `<thead><tr><th>Model</th>${xth("mae", "MAE")}${xth("rmse", "RMSE")}${xth("bias", "Bias")}${xth("wis", "WIS")}${xth("winkler", "Winkler 80%")}${xth("cover", "Coverage 80%")}${xth("rel_mae", "Rel. MAE")}</tr></thead><tbody>` +
      models.map((m) => sc.find((r) => r.model === m)).filter(Boolean).sort((a, b) => (a.mae ?? 1e18) - (b.mae ?? 1e18)).map((r) => `<tr class="${bt ? "is-bt" : ""}">
        <td class="model"><i style="background:${color(r.model)}"></i>${nm(r.model)}</td>
        ${COLS.map((k) => `<td class="num ${r[k] != null && dev(k, r[k]) === best(k) ? "best" : ""}">${k === "cover" ? pctCell(r[k]) : fmt(r[k], k === "rel_mae" ? 3 : 1)}</td>`).join("")}
      </tr>`).join("") + `</tbody>`;
  }

  // ---------- forecast value (value.html) ----------
  // data/value.json: per series (zone_target) and issue day, the perfect-foresight revenue and each model's revenue
  // for every strategy (sell the point forecast, or the p10..p90 percentile of its distribution) in EUR. A model's
  // score over a selection is sum(revenue) / sum(perfect-foresight revenue) over the series-days it delivered.
  let VAL = null;
  const VTARGETS = ["wind", "solar"];
  // settlement of the deviations: "err" = error cost only (|deviation| x |imbalance - day-ahead|, never a gain; the
  // default, it scores the forecast), "imb" = the zone's own settlement: its long and short imbalance prices, two
  // prices where the TSO publishes two (NL, FR), one elsewhere; what the asset would have been paid, gains included
  const VSFX = () => (S.v.mode === "imb" ? "" : "e");
  const costLabel = () => (S.v.mode === "imb" ? "Imbalance cost" : "Error cost");
  const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%");
  function valueSeries() {
    return Object.entries(VAL.series).filter(([k]) => {
      const [z, t] = [k.slice(0, k.lastIndexOf("_")), k.slice(k.lastIndexOf("_") + 1)];
      return t === S.v.target && (S.v.zone === "all" || z === S.v.zone);
    }).map(([, s]) => s);
  }
  /** model -> strategy -> {r, perfect, energy, n} summed over the selected series and days */
  function valueTotals() {
    const out = {};
    for (const s of valueSeries()) {
      s.days.forEach((d, i) => {
        if (S.phase !== "all" && s.phase[i] !== S.phase) return;
        for (const [m, strats] of Object.entries(s.models)) {
          for (const [k, arr] of Object.entries(strats)) {
            if (arr[i] == null) continue;
            const t = ((out[m] = out[m] || {})[k] = out[m][k] || { r: 0, perfect: 0, energy: 0, n: 0 });
            t.r += arr[i]; t.perfect += s.perfect[i]; t.energy += s.energy[i]; t.n += 1;
          }
        }
      });
    }
    const share = (t) => (t && t.perfect > 0 ? t.r / t.perfect : null);
    const cost = (t) => (t && t.energy > 0 ? (t.perfect - t.r) / t.energy : null);   // EUR per MWh produced
    const res = {};
    for (const [m, st] of Object.entries(out)) {
      const curve = VAL.percentiles.map((p) => share(st["p" + p + VSFX()]));
      const ok = curve.filter((v) => v != null);
      const bestI = ok.length ? curve.indexOf(Math.max(...ok)) : -1;
      res[m] = { st, curve, share, cost, quant: ok.length > 0, bestP: bestI >= 0 ? VAL.percentiles[bestI] : null, best: bestI >= 0 ? curve[bestI] : null,
        point: share(st["point" + VSFX()]), n: (st.point || {}).n || 0 };
    }
    return res;
  }
  const vCursor = {
    id: "vCursor",
    afterDatasetsDraw(chart, _a, o) {
      if (!o || o.p == null) return;
      const { ctx, chartArea: a, scales: { x } } = chart;
      const line = (p, dash, col, text, below) => {
        const px = x.getPixelForValue(p);
        ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = dash ? 1 : 1.5; ctx.setLineDash(dash ? [3, 3] : []);
        ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = css("--ink2"); ctx.font = "12px 'Source Code Pro', monospace";
        ctx.textAlign = px > a.right - 80 ? "right" : "left";
        ctx.fillText(text, px + (ctx.textAlign === "right" ? -6 : 6), below ? a.bottom - 8 : a.top + 12);
        ctx.restore();
      };
      if (o.best != null && o.best !== o.p) line(o.best, true, css("--ink2"), `best p${o.best}`, false);
      line(o.p, false, css("--ink"), `p${o.p}`, true);
    },
  };
  function renderValueControls(res) {
    options($("v-zone"), [{ k: "all", label: "All zones" }, ...D.config.zones.map((z) => ({ k: z, label: zoneLabel(z) }))], S.v.zone);
    seg($("v-mode"), [
      { k: "err", label: "Remainder as error cost", title: "The committed quantity is sold at the day-ahead price; the remainder (output minus committed) costs |remainder| x |imbalance price - day-ahead price|, so an error never earns" },
      { k: "imb", label: "Remainder at imbalance price", title: "The committed quantity is sold at the day-ahead price; the remainder is settled at the zone's imbalance price as it would have been paid: separate long and short prices where the TSO publishes two (NL, FR), one price in BE, DE-LU, DK1 and DK2" },
    ], S.v.mode, (k) => { S.v.mode = k; S.v.model = null; renderValue(); });
    seg($("v-target"), VTARGETS.map((t) => ({ k: t, label: TARGET_LABEL[t] })), S.v.target, (k) => { S.v.target = k; S.v.model = null; renderValue(); });
    const hasLive = Object.values(VAL.series).some((s) => s.phase.includes("live"));
    seg($("v-phase"), [{ k: "live", label: "Live", disabled: !hasLive }, { k: "backtest", label: "Backtest" }, { k: "all", label: "All" }], S.phase,
      (k) => { S.phase = k; renderValue(); });
    const ms = MODELS.filter((m) => res[m] && res[m].quant);
    options($("v-model"), ms.map((m) => ({ k: m, label: REF.has(m) ? `${m} (ref)` : m })), S.v.model);
  }
  // explainer diagram: one stylised day of a 100 MW asset. The forecast band (p10-p90), the percentile sold day-ahead
  // (it follows the slider), the actual output, and the gaps that are settled at the imbalance price.
  const VZ = { 10: -1.282, 15: -1.036, 20: -0.842, 25: -0.674, 30: -0.524, 35: -0.385, 40: -0.253, 45: -0.126, 50: 0, 55: 0.126, 60: 0.253,
    65: 0.385, 70: 0.524, 75: 0.674, 80: 0.842, 85: 1.036, 90: 1.282 };
  const VHOURS = Array.from({ length: 25 }, (_, t) => t);
  const vMed = VHOURS.map((t) => 46 + 18 * Math.sin((2 * Math.PI * (t - 5)) / 24) + 7 * Math.sin((2 * Math.PI * t) / 9));
  const vAct = VHOURS.map((t, i) => Math.max(2, vMed[i] + 13 * Math.sin((2 * Math.PI * t) / 7 + 1) + 6 * Math.sin((2 * Math.PI * t) / 3.3)));
  const vSd = vMed.map((m) => 7 + 0.18 * m);
  let vAnimated = false;
  function renderExplainer() {
    const el = $("v-explain");
    if (!el) return;
    const W = 440, H = 196, x0 = 34, x1 = 420, y0 = 20, y1 = 174;
    const X = (t) => x0 + ((x1 - x0) * t) / 24, Y = (v) => y1 - ((y1 - y0) * Math.min(100, Math.max(0, v))) / 100;
    const line = (vals) => vals.map((v, t) => `${t ? "L" : "M"}${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    const lo = vMed.map((m, i) => m + VZ[10] * vSd[i]), hi = vMed.map((m, i) => m + VZ[90] * vSd[i]);
    const sold = vMed.map((m, i) => m + VZ[S.v.p] * vSd[i]);
    const band = line(hi) + VHOURS.slice().reverse().map((t) => `L${X(t).toFixed(1)},${Y(lo[t]).toFixed(1)}`).join("") + "Z";
    const gap = line(vAct) + VHOURS.slice().reverse().map((t) => `L${X(t).toFixed(1)},${Y(sold[t]).toFixed(1)}`).join("") + "Z";
    const above = line(sold) + `L${x1},${y0 - 4}L${x0},${y0 - 4}Z`, below = line(sold) + `L${x1},${y1}L${x0},${y1}Z`;
    const imb = S.v.mode === "imb";
    const mw = (VAL && VAL.asset_mw) || 100;
    el.classList.toggle("anim", !vAnimated);
    vAnimated = true;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="vx-svg" aria-hidden="true">
        <defs><clipPath id="vx-up"><path d="${above}"/></clipPath><clipPath id="vx-dn"><path d="${below}"/></clipPath></defs>
        ${[0, 50, 100].map((v) => `<line x1="${x0}" x2="${x1}" y1="${Y(v)}" y2="${Y(v)}" class="vx-grid"/><text x="${x0 - 6}" y="${Y(v) + 4}" class="vx-ax" text-anchor="end">${v}</text>`).join("")}
        ${[0, 6, 12, 18, 24].map((t) => `<text x="${X(t)}" y="${H - 6}" class="vx-ax" text-anchor="middle">${String(t).padStart(2, "0")}:00</text>`).join("")}
        <text x="${x0 - 6}" y="9" class="vx-ax" text-anchor="end">MW</text>
        <path d="${band}" class="vx-band"/>
        <path d="${gap}" class="vx-gap vx-short" clip-path="url(#vx-dn)"/>
        <path d="${gap}" class="vx-gap vx-surplus" clip-path="url(#vx-up)"/>
        <path d="${line(sold)}" class="vx-sold"/>
        <path d="${line(vAct)}" class="vx-act" pathLength="1"/>
      </svg>
      <ol class="vx-steps">
        <li><i class="sw sw-band"></i><i class="sw sw-sold"></i><div><b>Sell p${S.v.p} of the forecast</b><span>before the 12:00 gate, at the day-ahead price</span></div></li>
        <li><i class="sw sw-act"></i><div><b>Produce what the weather gives</b><span>a <span id="v-mw">${mw}</span> MW asset that runs like its zone's fleet</span></div></li>
        <li><i class="sw sw-short"></i><i class="sw sw-surplus"></i><div><b>${imb ? "Settle only the remainder" : "Pay for the remainder"}</b><span>${imb ? "what you produce above or below what you sold, at the zone's long or short imbalance price" : "what you produce above or below what you sold, × |imbalance − day-ahead price|"}</span></div></li>
        <li class="vx-score"><div><b>Score = revenue kept ÷ perfect foresight</b><span>move the slider to see how the percentile changes the gaps</span></div></li>
      </ol>`;
  }
  function renderValue() {
    if (!VAL) return;
    const res = valueTotals();
    const at = (m) => res[m].curve[VAL.percentiles.indexOf(S.v.p)];
    const ranked = Object.keys(res).filter((m) => !REF.has(m)).sort((a, b) =>
      (res[b].quant - res[a].quant) || ((res[a].quant ? at(b) - at(a) : res[b].point - res[a].point) || 0));
    // default: the best model at this percentile that is not a naive baseline (a baseline can win a short window by luck)
    if (!S.v.model || !res[S.v.model] || !res[S.v.model].quant) S.v.model = ranked.find((m) => res[m].quant && FAM[m] !== "baseline") || ranked.find((m) => res[m].quant) || null;
    renderValueControls(res);
    $("v-p").value = String(S.v.p);
    $("v-p-h").textContent = "p" + S.v.p;
    renderExplainer();
    updateFilterSummary();
    const m = S.v.model, r = m && res[m];
    if (!r) {
      $("v-table").innerHTML = '<tbody><tr><td class="empty">No priced and scored days for this selection yet.</td></tr></tbody>';
      ["v-r1", "v-r2", "v-r3"].forEach((id) => { $(id).textContent = "–"; $(id + "-s").textContent = ""; });
      if (S.charts["c-value"]) { S.charts["c-value"].destroy(); delete S.charts["c-value"]; }
      return;
    }
    // chart: the chosen model bold, the four best others at this percentile thin, the grid operator's point forecast flat
    const others = ranked.filter((x) => x !== m && res[x].quant).slice(0, 4);
    const tso = Object.keys(res).filter((x) => FAM[x] === "tso" && res[x].point != null);
    const xy = (curve) => VAL.percentiles.map((p, i) => ({ x: p, y: curve[i] == null ? null : 100 * curve[i] }));
    const datasets = [
      { label: m, data: xy(r.curve), borderColor: color(m), backgroundColor: color(m), borderWidth: 3, pointRadius: 0, pointHoverRadius: 4, tension: 0.25, order: -1 },
      { label: `${m} point forecast`, data: VAL.percentiles.map((p) => ({ x: p, y: 100 * r.point })), borderColor: color(m), borderWidth: 1.2, borderDash: [2, 3], pointRadius: 0, pointHoverRadius: 0 },
      ...others.map((x) => ({ label: x, data: xy(res[x].curve), borderColor: alpha(color(x), 0.75), backgroundColor: color(x), borderWidth: 1.3, pointRadius: 0, pointHoverRadius: 3, tension: 0.25 })),
      ...tso.map((x) => ({ label: `${x} (point)`, data: VAL.percentiles.map((p) => ({ x: p, y: 100 * res[x].point })), borderColor: color(x), borderWidth: 1.4, borderDash: [6, 4], pointRadius: 0, pointHoverRadius: 0 })),
    ];
    const o = baseOptions({ yTitle: "Share of perfect-foresight revenue (%)", rightPad: 12, endLabels: false,
      tooltip: { title: (it) => `committed p${it[0].raw.x}`, label: (c) => ` ${c.dataset.label}: ${fmt(c.raw.y, 1)}%` } });
    o.scales.x.type = "linear"; o.scales.x.min = VAL.percentiles[0]; o.scales.x.max = VAL.percentiles[VAL.percentiles.length - 1];
    o.scales.x.ticks.stepSize = 10; o.scales.x.ticks.callback = (v) => "p" + v;
    o.scales.y.ticks.callback = (v) => v + "%";
    o.plugins.vCursor = { p: S.v.p, best: r.bestP };
    o.onClick = (e, _el, chart) => { const p = Math.round(chart.scales.x.getValueForPixel(e.x) / 5) * 5; if (p >= 10 && p <= 90) { S.v.p = p; renderValue(); } };
    if (S.charts["c-value"]) S.charts["c-value"].destroy();
    S.charts["c-value"] = new Chart($("c-value").getContext("2d"), { type: "line", data: { datasets }, options: o, plugins: [vCursor] });
    $("l-value").innerHTML = datasets.map((ds, i) => `<span style="display:inline-flex;align-items:center;gap:7px;margin-right:16px"><i style="display:inline-block;width:18px;border-top:${i === 0 ? 3 : 2}px ${ds.borderDash ? "dashed" : "solid"} ${ds.borderColor}"></i>${esc(ds.label)}</span>`).join("");

    const t = r.st["p" + S.v.p + VSFX()];
    const zoneTxt = S.v.zone === "all" ? "all zones" : zoneLabel(S.v.zone);
    $("v-r1-l").textContent = `Committed p${S.v.p} · ${m}`;
    $("v-r1").textContent = pct(at(m));
    const eur = (v) => (Math.abs(v) >= 1e6 ? `€${(v / 1e6).toFixed(2)}m` : `€${(v / 1e3).toFixed(0)}k`);
    $("v-r1-s").textContent = t ? `${eur(t.r)} of ${eur(t.perfect)} · point forecast ${pct(r.point)}` : "";
    $("v-r2").textContent = r.bestP != null ? "p" + r.bestP : "–";
    $("v-r2-s").textContent = r.best != null ? `${pct(r.best)} · ${r.bestP === S.v.p ? "you are on it" : `${((r.best - at(m)) * 100).toFixed(1)} pts above p${S.v.p}`}` : "";
    const c = r.cost(t);
    $("v-r3-l").textContent = costLabel();
    $("v-r3").textContent = c == null ? "–" : `€${c.toFixed(2)}`;
    const nz = valueSeries().filter((x) => x.phase.some((ph, i) => (S.phase === "all" || ph === S.phase) && (x.models[m] || {})["p" + S.v.p]?.[i] != null)).length;   // zones in the sum
    $("v-r3-s").textContent = t ? `per MWh produced, ${(t.energy / 1e3).toFixed(1)} GWh from ${nz > 1 ? `${nz} assets of ${VAL.asset_mw} MW` : `${VAL.asset_mw} MW`} in ${zoneTxt}` : "";

    // three groups: models with percentiles (ranked at the slider's percentile), point-only models (ranked on their
    // point forecast) and the reference entries published after the gate
    const withQ = ranked.filter((x) => res[x].quant);
    const pointOnly = ranked.filter((x) => !res[x].quant).sort((a, b) => (res[b].point ?? -1) - (res[a].point ?? -1));
    const refs = Object.keys(res).filter((x) => REF.has(x));
    const bestAt = Math.max(...withQ.map(at).filter((v) => v != null));
    const bestPoint = Math.max(...ranked.map((x) => res[x].point).filter((v) => v != null));
    const costHead = costLabel();
    const row = (x, rank) => {
      const q = res[x], a = q.quant ? at(x) : null;
      const tq = q.quant ? q.st["p" + S.v.p + VSFX()] : null, cx = tq ? q.cost(tq) : null;
      const nq = tq ? tq.n : 0;   // evaluations with quantiles; the point forecast counts every evaluation
      const nCell = q.quant && nq < q.n
        ? `<td class="r num" title="Percentile columns over the ${nq} evaluations with quantiles; point forecast over all ${q.n}">${nq} / ${q.n}</td>`
        : `<td class="r num">${q.n}</td>`;
      const qCells = q.quant
        ? `<td class="r v ${a === bestAt ? "best" : ""}">${pct(a)}</td><td class="r num">${q.bestP != null ? "p" + q.bestP : "–"}</td><td class="r num">${pct(q.best)}</td>`
        : `<td class="r po" colspan="3">point only</td>`;
      return `<tr class="${x === m ? "on" : ""}" data-m="${esc(x)}" style="cursor:${q.quant ? "pointer" : "default"}">
        <td class="r num">${rank}</td><td class="model"><i style="background:${color(x)}"></i>${nm(x)}</td>${qCells}
        <td class="r v ${q.point === bestPoint ? "best" : ""}">${pct(q.point)}</td><td class="r num">${cx == null ? "–" : "€" + cx.toFixed(2)}</td>${nCell}</tr>`;
    };
    const group = (label) => `<tr class="grp"><td colspan="8">${label}</td></tr>`;
    $("v-table").innerHTML = `<thead><tr><th class="r">#</th><th>Model</th><th class="r">At p${S.v.p}</th><th class="r">Best percentile</th><th class="r">At best</th><th class="r">Point forecast</th><th class="r">${costHead} at p${S.v.p}</th><th class="r">N evaluations</th></tr></thead><tbody>` +
      withQ.map((x, i) => row(x, i + 1)).join("") +
      (pointOnly.length ? group("Point forecast only: no quantiles to commit a percentile from, ranked on the point forecast") + pointOnly.map((x, i) => row(x, i + 1)).join("") : "") +
      (refs.length ? group("Reference: published after the gate, not ranked") + refs.map((x) => row(x, "ref")).join("") : "") + "</tbody>";
    $("v-table").onclick = (e) => { const tr = e.target.closest("tr[data-m]"); if (tr && res[tr.dataset.m].quant) { S.v.model = tr.dataset.m; renderValue(); } };
    $("v-notes").textContent = `The asset is ${VAL.asset_mw} MW in every zone: its output is the zone's actual output times ${VAL.asset_mw} MW over the installed capacity, and the forecasts are scaled the same way, so with All zones every zone counts equally. An evaluation is one zone on one delivery day with a published day-ahead price, imbalance price and final ${S.v.target} actual. ` +
      (S.v.mode === "imb"
        ? "Remainder at imbalance price: the committed quantity is sold at the day-ahead price and only the remainder is settled at the zone's imbalance price, as the asset would have been paid: a surplus at the long price, a shortfall bought back at the short price. NL publishes two prices in about half the quarter-hours (when both directions were activated; median gap EUR 66) and FR in nearly all (median gap EUR 17); BE, DE-LU, DK1 and DK2 have one price. A deviation can earn when the price is on its side of the day-ahead price, so a share can exceed 100% and the imbalance cost can be negative; over a few weeks this rests on a handful of price spikes, not on the forecast. "
        : "Remainder as error cost: the committed quantity is sold at the day-ahead price and the remainder costs its size times the gap between the imbalance and day-ahead prices, whichever way it goes, so no forecast error ever earns and the share is at most 100%. Over a few weeks a handful of price spikes still weigh heavily, because a large error in a spike is expensive. ") +
      "Models without quantiles are ranked separately on their point forecast; a model that published quantiles on only some days shows both counts (with quantiles / all). Click a row to draw its curve.";
  }
  const q0 = (k) => new URLSearchParams(location.search).get(k);
  async function initValue() {
    try { VAL = await (await fetch("data/value.json", { cache: "no-cache" })).json(); }
    catch (e) { $("v-table").innerHTML = '<tbody><tr><td class="empty">Value data is not available yet.</td></tr></tbody>'; return; }
    S.v = { zone: S.zone !== "all" && D.config.zones.includes(S.zone) ? S.zone : "all", target: VTARGETS.includes(S.target) ? S.target : "wind", model: null, p: 50, mode: "err" };
    if (["imb", "dual"].includes(q0("mode"))) S.v.mode = "imb";   // dual: an earlier, stylised mode, now the remainder at imbalance price
    const q = new URLSearchParams(location.search);
    if (q.get("zone") === "all" || D.config.zones.includes(q.get("zone"))) S.v.zone = q.get("zone");
    if (VTARGETS.includes(q.get("target"))) S.v.target = q.get("target");
    if (q.get("model")) S.v.model = q.get("model");
    if (+q.get("p") >= 10 && +q.get("p") <= 90) S.v.p = Math.round(+q.get("p") / 5) * 5;
    $("v-zone").onchange = (e) => { S.v.zone = e.target.value; renderValue(); };
    $("v-model").onchange = (e) => { S.v.model = e.target.value; renderValue(); };
    $("v-p").oninput = (e) => { S.v.p = +e.target.value; renderValue(); };
    setupFilterBar();
    renderValue();
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
  const FILTER_KEYS = ["zone", "target", "phase", "win", "agg", "covFilter", "topN", "sort", "details"];
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
    // "live and backtest together" is no longer offered: the leaderboard rates live runs or the backtest
    const phases = PAGE === "leaderboard" ? ["live", "backtest"] : ["live", "backtest", "all"];   // Forecast Value keeps "All"
    if (!phases.includes(S.phase) || (S.phase === "live" && !hasLive)) S.phase = hasLive ? "live" : "backtest";
    if (!METRICS.some((m) => m.k === S.metric)) S.metric = "rel_mae";
    if (!(S.win in WINDOWS)) S.win = "all";
    if (!(S.agg in AGGS)) S.agg = "mean";

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
      setupFilterBar();
      renderAll();
      renderTracker();
    }
    if (PAGE === "explorer") {
      if (S.zone !== "all" && D.config.zones.includes(S.zone)) S.x.zone = S.zone;
      if (S.target !== "all" && S.target in D.config.targets) S.x.target = S.target;
      S.xTop = S.topN;
      const q = new URLSearchParams(location.search);   // deep links: explorer.html?day=2026-10-03&zone=BE&target=wind
      if (q.get("day") && D.days && (xPhase() === "all" || D.days[q.get("day")] === xPhase())) S.x.day = q.get("day");   // live days only
      if (D.config.zones.includes(q.get("zone"))) S.x.zone = q.get("zone");
      if (q.get("target") in D.config.targets) S.x.target = q.get("target");
      const changed = () => { S.zone = S.x.zone; S.target = S.x.target; S.topN = S.xTop; saveFilters();
        history.replaceState(null, "", `?day=${S.x.day}&zone=${S.x.zone}&target=${S.x.target}`); };
      const segs = () => {
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
      setupFilterBar();
      renderExplorerControls();
      changed();
      renderExplorerChart();
    }
    if (PAGE === "value") {
      await initValue();
      afterFonts = () => renderValue();
      matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderValue);
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
