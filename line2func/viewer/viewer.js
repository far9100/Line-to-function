// Canvas viewer for line2func results, shared by the app (python -m line2func) and the plain
// result viewer (python -m line2func.serve). It draws with Canvas2D from a few cached Path2D
// objects, finds the curve under the pointer with a spatial grid and renders only the visible
// rows of the equation list, so results with 10,000+ curves stay smooth.
import { t } from "./i18n.js";

export const PALETTE = ["#e6194b", "#0082c8", "#3cb44b", "#f58230", "#911eb4", "#00a0a0", "#f032e6", "#808000"];
export const DESMOS_LIMIT = 5000; // = line2func.export.DESMOS_CURVE_LIMIT (tests check it)
const ROW = 68, CELL = 32;

// ---------- line colors (must match line2func/export.py; see tests/test_viewer_assets.py) ----------
export const LINE_COLOR_MODES = ["bw", "palette", "random"]; // the page's choices; "measured" is the SVG's own
export const RANDOM_BUCKETS = 64; // = line2func.export.RANDOM_BUCKETS: hues repeat every 64 strokes
export const LINE_WIDTH_MODES = ["measured", "uniform"]; // = line2func.export.LINE_WIDTH_MODES
export const WIDTH_STEP = 0.1; // measured widths are drawn rounded to this, to keep the number of paths small
const MIN_SCREEN_WIDTH = 0.75; // a thin stroke stays at least this many screen pixels wide, whatever the zoom
export const BW_COLOR = "#000";

// A hue in 0..359 from seed and bucket; the 32-bit steps mirror line2func.export._hue.
function hue(seed, bucket) {
  let x = (seed + Math.imul(bucket, 0x9e3779b1)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x % 360;
}

/** The color for stroke number `stroke` in line color `mode` (= line2func.export.stroke_color). */
export function strokeColor(mode, stroke, seed = 0) {
  if (mode === "bw" || mode === "measured") return BW_COLOR;
  const n = mode === "palette" ? PALETTE.length : RANDOM_BUCKETS;
  const bucket = ((stroke % n) + n) % n; // stroke numbers are never negative, but keep the index in range
  return mode === "palette" ? PALETTE[bucket] : `hsl(${hue(seed, bucket)} 70% 45%)`;
}

// ---------- math (must match line2func/export.py byte for byte; see tests/test_viewer_assets.py) ----------
export function power(p0, p1, p2, p3) { // [a, b, c, d] of a t^3 + b t^2 + c t + d
  return [-p0 + 3 * p1 - 3 * p2 + p3, 3 * p0 - 6 * p1 + 3 * p2, -3 * p0 + 3 * p1, p0];
}
export function fmt(v) { let s = v.toFixed(2); if (parseFloat(s) === 0) s = (0).toFixed(2); return s; }
const SUP = ["t³", "t²", "t", ""], TEX = ["t^{3}", "t^{2}", "t", ""];
export function pretty(c) {
  return c.map((v, k) => {
    const s = fmt(v), neg = s.startsWith("-"), body = (neg ? s.slice(1) : s) + SUP[k];
    return k === 0 ? (neg ? "−" : "") + body : (neg ? " − " : " + ") + body;
  }).join("");
}
export function latex(c) {
  return c.map((v, k) => { const s = fmt(v) + TEX[k]; return k === 0 || s.startsWith("-") ? s : "+" + s; }).join("");
}
// One Desmos parametric expression; p = [x0, y0, ..., x3, y3] in image pixels (y down), H = image height.
export function desmosLine(p, H) {
  const cx = power(p[0], p[2], p[4], p[6]), cy = power(H - p[1], H - p[3], H - p[5], H - p[7]);
  return `\\left(${latex(cx)},\\ ${latex(cy)}\\right)`;
}
// Readable form of a named Desmos equation (y = mx + c, (x-h)^2 + (y-k)^2 = r^2) or of a function
// (y = a + b(x-c) + d(x-c)^2 + e(x-c)^3).
export function namedText(tex) {
  return tex.replace(/\\left\\\{/g, t("viewer.where")).replace(/\\right\\\}/g, "")
    .replace(/\\left\(/g, "(").replace(/\\right\)/g, ")").replace(/\^\{2\}/g, "²").replace(/\^\{3\}/g, "³")
    .replace(/\\le /g, " ≤ ").replace(/\\ge /g, " ≥ ").replace(/-/g, "−").replace(/=/g, " = ");
}
function bezier(p, s) {
  const m = 1 - s, a = m * m * m, b = 3 * m * m * s, c = 3 * m * s * s, d = s * s * s;
  return [a * p[0] + b * p[2] + c * p[4] + d * p[6], a * p[1] + b * p[3] + c * p[5] + d * p[7]];
}
function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  let u = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  u = u < 0 ? 0 : u > 1 ? 1 : u;
  const ex = ax + u * dx - px, ey = ay + u * dy - py;
  return ex * ex + ey * ey;
}
export function esc(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
}
const tagName = (tag) => t("tag." + tag, {}, tag);
const shapeName = (type) => t("shape." + type, {}, type);

// el: {stage, canvas, tip, message, list, spacer, detail, stats, fit, originalOnly, viewMode, bgAlpha,
//      convertBar, listToggleLabel}
// Display modes (el.viewMode): "lines" (the curves on paper), "original" (on the original image) and "missed"
// (on the quality check's map of missed detail). el.originalOnly shows the original image alone.
export function createViewer(el) {
  const { stage, canvas, tip, message, list, spacer, detail, stats } = el;
  const onLineColor = el.onLineColor || (() => {}); // the app persists the choice and retitles the downloads
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, curves = [], groups = [], grid = new Map();
  let status = "empty", failMsg = "", quality = null; // status: empty, loading, failed, preview (an image alone), ready
  let functionCount = null; // equations y = f(x) / x = g(y) in the result, when it has them
  let desmosWarning = true; // the app shows its own banner instead
  let view = { s: 1, tx: 0, ty: 0 }, dpr = window.devicePixelRatio || 1;
  let hover = -1, selected = -1, dirty = true, fitted = false, generation = 0;
  // followFit: refit when the stage changes size until the user pans or zooms (a banner or a wrapping
  // toolbar can appear after the first fit)
  let followFit = false;
  let mode = "original", originalOnly = false; // the chosen display; a mode whose image is missing shows "lines"
  let lineColor = "palette", colorSeed = 0; // the chosen line color (LINE_COLOR_MODES) and the random mode's seed
  let lineWidthMode = "measured"; // LINE_WIDTH_MODES: each stroke's own width, or one width for all
  let widthGroups = null; // paths grouped by (color bucket, rounded stroke width), built when first drawn

  let docWidth = 2; // the result's own line width (meta.line_width), the one "uniform" uses
  const bg = { images: {}, alpha: { original: 0.35, missed: 0.9 } }; // images: original, quality
  const pointers = new Map();
  let drag = null, pinch = null;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- data ----------
  function reset() {
    W = 0; H = 0; curves = []; groups = []; grid = new Map();
    hover = -1; selected = -1; quality = null; fitted = false; functionCount = null; followFit = false;
    widthGroups = null;
    pointers.clear(); drag = null; pinch = null;
    canvas.classList.remove("dragging", "over");
    tip.style.display = "none";
    list.scrollTop = 0; spacer.innerHTML = ""; spacer.style.height = "0px";
    bg.images = {};
    dirty = true;
  }

  function prepare(doc) {
    W = doc.image.width; H = doc.image.height;
    docWidth = Number(doc.meta?.line_width) || 2; // = the SVG <g> stroke-width, used where a stroke has none
    curves = doc.curves.map((c, i) => {
      const p = c.ctrl.flat();
      const xs = [p[0], p[2], p[4], p[6]], ys = [p[1], p[3], p[5], p[7]];
      const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      const chord = Math.hypot(p[2] - p[0], p[3] - p[1]) + Math.hypot(p[4] - p[2], p[5] - p[3]) + Math.hypot(p[6] - p[4], p[7] - p[5]);
      const n = Math.max(8, Math.min(96, Math.ceil(chord / 3)));
      const poly = new Float64Array(2 * (n + 1));
      for (let k = 0; k <= n; k++) { const q = bezier(p, k / n); poly[2 * k] = q[0]; poly[2 * k + 1] = q[1]; }
      const cx = power(p[0], p[2], p[4], p[6]), cy = power(H - p[1], H - p[3], H - p[5], H - p[7]);
      return { i, stroke: c.stroke, conf: c.confidence, tags: c.tags || [], p, box, poly, cx, cy,
               width: c.width ?? null, color: c.color ?? null, tone: c.tone ?? null,
               shape: c.shape ?? null, functions: c.functions ?? null };
    });
    functionCount = curves.some((c) => c.functions) ? curves.reduce((n, c) => n + (c.functions ? c.functions.length : 0), 0) : null;
    // one path per random-color bucket; PALETTE.length divides RANDOM_BUCKETS, so bucket % 8 is the palette hue
    groups = Array.from({ length: RANDOM_BUCKETS }, () => new Path2D());
    for (const c of curves) {
      const p = c.p, path = groups[((c.stroke % RANDOM_BUCKETS) + RANDOM_BUCKETS) % RANDOM_BUCKETS];
      path.moveTo(p[0], p[1]); path.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
      const x0 = Math.floor(c.box[0] / CELL), x1 = Math.floor(c.box[2] / CELL);
      const y0 = Math.floor(c.box[1] / CELL), y1 = Math.floor(c.box[3] / CELL);
      for (let gy = y0; gy <= y1; gy++) for (let gx = x0; gx <= x1; gx++) {
        const cellKey = gx + "," + gy;
        let cell = grid.get(cellKey);
        if (!cell) grid.set(cellKey, (cell = []));
        cell.push(c.i);
      }
    }
    spacer.style.height = curves.length * ROW + "px";
  }

  // Paths for the measured-width mode, grouped by (color bucket, rounded width) because one stroke() call
  // draws a single color at a single width. A stroke's pieces all take the stroke's median width, which is
  // what line2func.export.to_svg writes, so the canvas and a downloaded SVG agree.
  function widthBuckets() {
    if (widthGroups) return widthGroups;
    const fallback = docWidth;
    const perStroke = new Map(); // stroke -> its pieces' widths
    for (const c of curves) {
      if (c.width == null) continue;
      let w = perStroke.get(c.stroke);
      if (!w) perStroke.set(c.stroke, (w = []));
      w.push(c.width);
    }
    const median = new Map();
    for (const [stroke, w] of perStroke) {
      w.sort((a, b) => a - b);
      median.set(stroke, w.length % 2 ? w[(w.length - 1) / 2] : 0.5 * (w[w.length / 2 - 1] + w[w.length / 2]));
    }
    const map = new Map();
    for (const c of curves) {
      const bucket = ((c.stroke % RANDOM_BUCKETS) + RANDOM_BUCKETS) % RANDOM_BUCKETS;
      const width = Math.max(WIDTH_STEP, Math.round((median.get(c.stroke) ?? fallback) / WIDTH_STEP) * WIDTH_STEP);
      const key = bucket + "|" + width.toFixed(2);
      let g = map.get(key);
      if (!g) map.set(key, (g = { bucket, width, path: new Path2D() }));
      const p = c.p;
      g.path.moveTo(p[0], p[1]);
      g.path.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
    }
    widthGroups = [...map.values()];
    return widthGroups;
  }

  // what is drawn: the chosen mode when its image is there, else the curves alone
  function shownMode() {
    if (mode === "original" && bg.images.original) return "original";
    return mode === "missed" && bg.images.quality ? "missed" : "lines";
  }
  function imageOnly() { return status === "preview" || (status === "ready" && originalOnly && !!bg.images.original); }
  function updateControls() {
    const ready = status === "ready", shown = shownMode();
    const has = { lines: true, original: !!bg.images.original, missed: !!bg.images.quality };
    for (const option of el.viewMode.options) option.disabled = !has[option.value];
    el.viewMode.value = shown;
    el.viewMode.disabled = !ready || originalOnly;
    el.originalOnly.disabled = !ready || !has.original;
    el.bgAlpha.disabled = !ready || originalOnly || shown === "lines";
    if (shown !== "lines") el.bgAlpha.value = String(Math.round(bg.alpha[shown] * 100));
    dirty = true;
  }

  // the label of the phone layout's collapsed equation list (the button is display:none elsewhere)
  function renderListToggle() {
    if (!el.listToggleLabel) return;
    el.listToggleLabel.textContent = status === "ready" ? t("count.curves", { n: curves.length }) : "";
  }

  function renderStats() {
    renderListToggle();
    if (status === "loading") { stats.textContent = t("viewer.loadingShort"); stats.title = ""; return; }
    if (status === "failed") { stats.textContent = t("viewer.failedShort"); stats.title = ""; return; }
    if (status !== "ready") { stats.textContent = ""; stats.title = ""; return; }
    const strokes = new Set(curves.map((c) => c.stroke)).size;
    let html = `${esc(t("count.curves", { n: curves.length }))} · ${esc(t("count.strokes", { n: strokes }))}`;
    if (functionCount !== null) html += ` · ${esc(t("count.functions", { n: functionCount }))}`;
    html += ` · ${W}×${H}`;
    const equations = functionCount ?? curves.length; // what "Copy all for Desmos" pastes
    if (desmosWarning && equations > DESMOS_LIMIT) html += ` · <span class="warn">${esc(t("stats.desmosWarn", { limit: DESMOS_LIMIT }))}</span>`;
    stats.title = "";
    if (quality) {
      const pct = (v) => (v === null || v === undefined ? t("common.na") : (100 * v).toFixed(1) + "%");
      const kept = quality.recall?.line?.within_2px, faint = quality.recall?.with_faint?.within_2px;
      const stray = quality.flags ? quality.flags.stray_curves : (quality.distance?.curve_to_any_ink?.max ?? 0) > 10;
      html += ` · ${esc(t("stats.kept", { kept: pct(kept), faint: pct(faint) }))}`;
      if (stray) html += ` · <span class="warn">${esc(t("stats.stray"))}</span>`;
      stats.title = t("stats.qualityTitle");
    }
    stats.innerHTML = html;
  }

  function renderMessage() {
    let text = "";
    if (status === "loading") text = t("viewer.loading");
    else if (status === "failed") text = t("viewer.loadFailed", { msg: failMsg });
    else if (status === "ready" && !curves.length) text = t("viewer.empty");
    message.textContent = text;
    message.style.display = text ? "flex" : "none";
  }

  function hitTest(x, y, radius) {
    const r2 = radius * radius, seen = new Set();
    let best = -1, bestD = r2;
    const gx0 = Math.floor((x - radius) / CELL), gx1 = Math.floor((x + radius) / CELL);
    const gy0 = Math.floor((y - radius) / CELL), gy1 = Math.floor((y + radius) / CELL);
    for (let gy = gy0; gy <= gy1; gy++) for (let gx = gx0; gx <= gx1; gx++) {
      const cell = grid.get(gx + "," + gy);
      if (!cell) continue;
      for (const i of cell) {
        if (seen.has(i)) continue;
        seen.add(i);
        const c = curves[i];
        if (!c) continue;
        const b = c.box;
        if (x < b[0] - radius || x > b[2] + radius || y < b[1] - radius || y > b[3] + radius) continue;
        const q = c.poly;
        for (let k = 0; k + 3 < q.length; k += 2) {
          const d = segDist2(x, y, q[k], q[k + 1], q[k + 2], q[k + 3]);
          if (d < bestD) { bestD = d; best = i; }
        }
      }
    }
    return best;
  }

  // ---------- drawing ----------
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const w = stage.clientWidth, h = stage.clientHeight;
    canvas.width = Math.max(1, Math.round(w * dpr)); canvas.height = Math.max(1, Math.round(h * dpr));
    if ((!fitted || followFit) && W && w && h) fitted = fit();
    dirty = true;
  }
  // Stage pixels the convert bar covers: the overlap of the two boxes, measured on every fit, so a
  // language change, a rotation, a rewrapped bar and a bar that sits below the stage are all handled
  // without being told. With the bar at bottom:16px this is its height + 24, which is the number
  // app.js used to pass in. Capped so fit() always has somewhere to put the image.
  function bottomInset() {
    const bar = el.convertBar;
    if (!bar || bar.hidden || !bar.offsetParent) return 0;
    const box = stage.getBoundingClientRect(), b = bar.getBoundingClientRect();
    const overlap = box.bottom - Math.max(b.top, box.top);
    if (!(overlap > 0)) return 0; // shown below the stage, not over it
    return Math.min(overlap + 8, Math.max(0, box.height - 96));
  }
  function fit() {
    const w = stage.clientWidth, h = stage.clientHeight - bottomInset(), pad = 24;
    if (!W || !w || h <= 0) return false; // hidden: fit once the stage has a size
    view.s = Math.max(1e-3, Math.min((w - 2 * pad) / W, (h - 2 * pad) / H));
    view.tx = (w - W * view.s) / 2; view.ty = (h - H * view.s) / 2; dirty = true;
    followFit = true;
    return true;
  }
  function zoomAt(px, py, f) {
    const s = Math.min(200, Math.max(0.02, view.s * f)), ix = (px - view.tx) / view.s, iy = (py - view.ty) / view.s;
    view.s = s; view.tx = px - ix * s; view.ty = py - iy * s; dirty = true; followFit = false;
  }
  function trace(c) { ctx.beginPath(); const p = c.p; ctx.moveTo(p[0], p[1]); ctx.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]); }
  function highlight(i, color, px) {
    const c = curves[i], s = view.s;
    if (!c) return;
    trace(c); ctx.strokeStyle = css("--halo"); ctx.lineWidth = (px + 3) / s; ctx.stroke();
    trace(c); ctx.strokeStyle = color; ctx.lineWidth = px / s; ctx.stroke();
  }
  function draw() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = css("--canvas"); ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!W) return;
    ctx.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.tx, dpr * view.ty);
    ctx.fillStyle = css("--paper"); ctx.fillRect(0, 0, W, H);
    const alone = imageOnly(), shown = shownMode();
    const img = alone || shown === "original" ? bg.images.original : shown === "missed" ? bg.images.quality : null;
    if (img) {
      ctx.globalAlpha = alone ? 1 : bg.alpha[shown]; ctx.imageSmoothingEnabled = view.s < 4;
      ctx.drawImage(img, 0, 0, W, H); ctx.globalAlpha = 1;
    }
    if (alone) return; // the original image by itself: no curves
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = 1.5 / view.s;
    // the missed-detail map is a diagnostic: one quiet gray at one width, so its red, orange and blue stand out
    const even = lineWidthMode === "uniform" || shown === "missed";
    if (shown === "missed") {
      ctx.strokeStyle = "rgba(70, 70, 70, 0.55)";
      groups.forEach((path) => ctx.stroke(path));
    } else if (even && lineColor === "bw") {
      ctx.strokeStyle = BW_COLOR; // one color at one width for every stroke: set them once
      groups.forEach((path) => ctx.stroke(path));
    } else if (even) {
      groups.forEach((path, g) => { ctx.strokeStyle = strokeColor(lineColor, g, colorSeed); ctx.stroke(path); });
    } else {
      // each stroke at its measured width, as the SVG writes it, but never thinner than a visible hairline
      const floor = MIN_SCREEN_WIDTH / view.s;
      for (const g of widthBuckets()) {
        ctx.strokeStyle = strokeColor(lineColor, g.bucket, colorSeed);
        ctx.lineWidth = Math.max(g.width, floor);
        ctx.stroke(g.path);
      }
    }
    if (hover >= 0 && hover !== selected) highlight(hover, css("--hover"), 3.5);
    if (selected >= 0 && curves[selected]) {
      highlight(selected, css("--accent"), 4.5);
      const p = curves[selected].p, r = 4 / view.s;
      ctx.fillStyle = css("--accent");
      for (const k of [0, 6]) { ctx.beginPath(); ctx.arc(p[k], p[k + 1], r, 0, 2 * Math.PI); ctx.fill(); }
    }
  }
  (function loop() {
    requestAnimationFrame(loop); // schedule first: an error in one frame must not stop the animation
    if ((window.devicePixelRatio || 1) !== dpr) resize(); // moved to a screen with another scale
    if (!dirty) return;
    dirty = false;
    try { draw(); } catch (err) { console.error(err); }
  })();

  // ---------- panels ----------
  function eqHTML(c) { return `x(t) = ${pretty(c.cx)}<br>y(t) = ${pretty(c.cy)}`; }
  function swatch(c) { return strokeColor(lineColor, c.stroke, colorSeed); }
  function showDetail() {
    const c = curves[selected];
    if (!c) {
      detail.innerHTML = status === "ready" ? `<span class="hint">${esc(t("viewer.hint"))}</span>` : "";
      return;
    }
    const tags = c.tags.map((tag) => `<span class="tag">${esc(tagName(tag))}</span>`).join(" ");
    const named = c.shape && c.shape.desmos
      ? `<div class="eq named"><span class="tag">${esc(shapeName(c.shape.type))}</span> ${esc(namedText(c.shape.desmos))}</div>` : "";
    const fns = c.functions && c.functions.length
      ? `<div class="eq fns"><span class="tag">${esc(t("viewer.functions"))}</span> ${c.functions.map((f) => esc(namedText(f))).join("<br>")}</div>` : "";
    const style = [
      c.width !== null ? esc(t("viewer.width", { w: c.width.toFixed(2) })) : "",
      c.color ? `${esc(t("viewer.color"))} <span class="swatch" style="background:${esc(c.color)}"></span> ${esc(c.color)}` : "",
    ].filter(Boolean).join(" · ");
    detail.innerHTML = `<h2>${esc(t("viewer.curve", { i: String(c.i) }))} <span class="hint">· ${esc(t("viewer.stroke", { s: String(c.stroke) }))}</span></h2>
      <div class="eq">${eqHTML(c)}</div>${named}${fns}
      <div class="meta">${esc(t("viewer.meta", { c: c.conf.toFixed(2) }))} ${tags}${style ? "<br>" + style : ""}</div>
      <div class="actions"><button type="button" data-copy="desmos">${esc(t("viewer.copy"))}</button>${named ? `<button type="button" data-copy="named">${esc(t("viewer.copyNamed"))}</button>` : ""}${fns ? `<button type="button" data-copy="functions">${esc(t("viewer.copyFunctions"))}</button>` : ""}<button type="button" data-center>${esc(t("viewer.center"))}</button></div>`;
    const copy = (text, label) => (e) => {
      const button = e.currentTarget;
      navigator.clipboard.writeText(text).then(
        () => { button.textContent = t("viewer.copied"); },
        () => { button.textContent = t("viewer.copyFailed"); });
      setTimeout(() => { button.textContent = label; }, 1500);
    };
    detail.querySelector("[data-copy=desmos]").onclick = copy(desmosLine(c.p, H), t("viewer.copy"));
    if (named) detail.querySelector("[data-copy=named]").onclick = copy(c.shape.desmos, t("viewer.copyNamed"));
    if (fns) detail.querySelector("[data-copy=functions]").onclick = copy(c.functions.join("\n"), t("viewer.copyFunctions"));
    detail.querySelector("[data-center]").onclick = () => centerOn(c.i);
  }
  function renderList() {
    const first = Math.max(0, Math.floor(list.scrollTop / ROW) - 4);
    const last = Math.min(curves.length, first + Math.ceil(list.clientHeight / ROW) + 8);
    let html = "";
    for (let i = first; i < last; i++) {
      const c = curves[i];
      html += `<div class="row${i === selected ? " active" : ""}" data-i="${i}" style="top:${i * ROW}px">
        <div class="row-head"><span class="swatch" style="background:${swatch(c)}"></span>#${i} · ${esc(t("viewer.stroke", { s: String(c.stroke) }))}
        ${c.tags.map((tag) => `<span class="tag">${esc(tagName(tag))}</span>`).join(" ")}${c.shape ? `<span class="tag">${esc(shapeName(c.shape.type))}</span>` : ""}</div>
        <div class="eq">x(t) = ${pretty(c.cx)}</div><div class="eq">y(t) = ${pretty(c.cy)}</div></div>`;
    }
    spacer.innerHTML = html;
  }
  function select(i, scrollList) {
    selected = i; dirty = true; showDetail(); renderList();
    if (i >= 0 && scrollList) {
      const top = i * ROW;
      if (top < list.scrollTop || top + ROW > list.scrollTop + list.clientHeight) list.scrollTop = top - list.clientHeight / 2 + ROW / 2;
    }
  }
  function centerOn(i) {
    const c = curves[i];
    if (!c) return;
    const b = c.box, w = stage.clientWidth, h = stage.clientHeight;
    const bw = Math.max(b[2] - b[0], 1), bh = Math.max(b[3] - b[1], 1);
    if (bw * view.s > w * 0.8 || bh * view.s > h * 0.8) view.s = Math.min((w * 0.6) / bw, (h * 0.6) / bh);
    view.tx = w / 2 - ((b[0] + b[2]) / 2) * view.s; view.ty = h / 2 - ((b[1] + b[3]) / 2) * view.s; dirty = true;
    followFit = false;
  }
  function showTip(i, x, y) {
    if (i < 0 || !curves[i]) { tip.style.display = "none"; return; }
    tip.innerHTML = `#${i} · ${esc(t("viewer.stroke", { s: String(curves[i].stroke) }))}<br>${eqHTML(curves[i])}`;
    tip.style.display = "block";
    const pad = 14, tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(x + pad, stage.clientWidth - tw - 4) + "px";
    tip.style.top = (y + pad + th > stage.clientHeight ? y - th - pad : y + pad) + "px";
  }

  // ---------- pointer input ----------
  const local = (e) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  function setHover(i) { if (i !== hover) { hover = i; dirty = true; canvas.classList.toggle("over", i >= 0); } }
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, local(e));
    if (pointers.size === 1) {
      const p = local(e);
      drag = { x: p.x, y: p.y, tx: view.tx, ty: view.ty, moved: false };
      canvas.classList.add("dragging");
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2, s: view.s, tx: view.tx, ty: view.ty };
      if (drag) drag.moved = true;
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = local(e);
    if (!pointers.has(e.pointerId)) {
      if (e.pointerType === "mouse") {
        const i = imageOnly() ? -1 : hitTest((p.x - view.tx) / view.s, (p.y - view.ty) / view.s, 6 / view.s);
        setHover(i); showTip(i, p.x, p.y);
      }
      return;
    }
    pointers.set(e.pointerId, p);
    if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const s = Math.min(200, Math.max(0.02, pinch.s * d / Math.max(pinch.d, 1)));
      const ix = (pinch.mx - pinch.tx) / pinch.s, iy = (pinch.my - pinch.ty) / pinch.s;
      view.s = s; view.tx = mx - ix * s; view.ty = my - iy * s; dirty = true; followFit = false;
    } else if (drag) {
      const dx = p.x - drag.x, dy = p.y - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; setHover(-1); showTip(-1); }
      if (drag.moved) { view.tx = drag.tx + dx; view.ty = drag.ty + dy; dirty = true; followFit = false; }
    }
  });
  function endPointer(e) {
    if (!pointers.has(e.pointerId)) return;
    const p = local(e);
    if (pointers.size === 1 && drag && !drag.moved && e.type === "pointerup") {
      const radius = (e.pointerType === "mouse" ? 6 : 14) / view.s;
      select(imageOnly() ? -1 : hitTest((p.x - view.tx) / view.s, (p.y - view.ty) / view.s, radius), true);
    }
    pointers.delete(e.pointerId);
    pinch = null;
    if (pointers.size === 1) {
      const q = [...pointers.values()][0];
      drag = { x: q.x, y: q.y, tx: view.tx, ty: view.ty, moved: true };
    } else if (pointers.size === 0) { drag = null; canvas.classList.remove("dragging"); }
  }
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse" && !pointers.size) { setHover(-1); showTip(-1); } });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const p = local(e), unit = e.deltaMode === 1 ? 0.05 : 0.0015;
    zoomAt(p.x, p.y, Math.exp(-e.deltaY * unit)); showTip(-1);
  }, { passive: false });
  canvas.addEventListener("dblclick", fit);
  list.addEventListener("scroll", renderList);
  list.addEventListener("click", (e) => { const r = e.target.closest(".row"); if (r) { const i = +r.dataset.i; select(i, false); centerOn(i); } });
  list.addEventListener("mouseover", (e) => { const r = e.target.closest(".row"); setHover(r ? +r.dataset.i : -1); });
  list.addEventListener("mouseleave", () => setHover(-1));
  el.fit.addEventListener("click", fit);
  el.viewMode.addEventListener("change", (e) => { mode = e.target.value; updateControls(); });
  if (el.lineColor) {
    el.lineColor.addEventListener("change", (e) => { setLineColor(e.target.value); onLineColor(lineColorState()); });
  }
  if (el.lineWidth) {
    el.lineWidth.addEventListener("change", (e) => { setLineWidth(e.target.value); onLineColor(lineColorState()); });
  }
  if (el.colorReroll) {
    el.colorReroll.addEventListener("click", () => {
      // a fresh seed only means anything for the random mode; switch to it so the click is never a no-op
      setLineColor("random", (Math.random() * 0x100000000) >>> 0);
      onLineColor(lineColorState());
    });
  }
  el.originalOnly.addEventListener("change", (e) => {
    originalOnly = e.target.checked;
    if (imageOnly()) { setHover(-1); showTip(-1); }
    updateControls();
  });
  el.bgAlpha.addEventListener("input", (e) => {
    const shown = shownMode();
    if (shown !== "lines") { bg.alpha[shown] = e.target.value / 100; dirty = true; }
  });
  new ResizeObserver(() => { resize(); renderList(); }).observe(stage);
  new ResizeObserver(renderList).observe(list);
  // the bar's height changes with the language and with how its rows wrap; refit under it while the
  // view is still the fitted one
  if (el.convertBar) new ResizeObserver(() => { if (followFit) fitted = fit(); }).observe(el.convertBar);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dirty = true; });

  // ---------- public ----------
  // Show a result. sources: {original, quality, qualityJson} URLs (all optional).
  function load(doc, sources = {}) {
    const mine = ++generation;
    reset();
    prepare(doc);
    status = "ready";
    originalOnly = false; el.originalOnly.checked = false; // a new result shows its curves
    resize(); renderList(); showDetail(); renderStats(); renderMessage(); updateControls();
    for (const kind of ["original", "quality"]) {
      if (!sources[kind]) continue;
      const img = new Image();
      img.onload = () => { if (mine !== generation) return; bg.images[kind] = img; updateControls(); };
      img.src = sources[kind];
    }
    if (sources.qualityJson) {
      fetch(sources.qualityJson, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((q) => { if (q && mine === generation) { quality = q; renderStats(); } })
        .catch(() => {});
    }
  }
  // Show an image alone, e.g. a drawing waiting to be converted; w, h: its size in pixels; bottom: stage
  // pixels at the bottom that something covers (it is fitted above them).
  function preview(url, w, h) {
    const mine = ++generation;
    reset();
    W = w; H = h; status = "preview";
    resize(); renderList(); showDetail(); renderStats(); renderMessage(); updateControls();
    const img = new Image();
    img.onload = () => { if (mine !== generation) return; bg.images.original = img; dirty = true; };
    img.src = url;
  }
  function setLoading() { generation++; reset(); status = "loading"; renderStats(); renderMessage(); showDetail(); updateControls(); }
  function fail(msg) { generation++; reset(); status = "failed"; failMsg = msg; renderStats(); renderMessage(); showDetail(); updateControls(); }
  function unload() { generation++; reset(); status = "empty"; renderStats(); renderMessage(); showDetail(); updateControls(); }
  function rerender() { renderStats(); renderMessage(); showDetail(); renderList(); }
  function setDesmosWarning(on) { desmosWarning = !!on; renderStats(); }
  // Keyboard shortcuts of the result screen; returns true when the key was used.
  function handleKey(e) {
    if (e.key === "Escape" && selected >= 0) { select(-1); return true; }
    if (e.key === "f" || e.key === "F") { fit(); return true; }
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && curves.length) {
      const n = curves.length, i = selected < 0 ? 0 : (selected + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
      select(i, true);
      return true;
    }
    return false;
  }
  function debug() {
    return { status, curves: curves.length, gridCells: grid.size, selected, hover, width: W, height: H,
             backgrounds: Object.keys(bg.images), mode: shownMode(), imageOnly: imageOnly(), view: { ...view } };
  }
  // The chosen line color. `seed` only matters for "random"; it is kept so a re-roll can be reproduced.
  function lineColorState() { return { mode: lineColor, seed: colorSeed, width: lineWidthMode }; }

  /** Choose the line color. Redraws the canvas and the list's swatches. */
  function setLineColor(next, seed) {
    if (!LINE_COLOR_MODES.includes(next)) return;
    lineColor = next;
    if (seed !== undefined) colorSeed = seed >>> 0;
    if (el.lineColor) el.lineColor.value = lineColor;
    if (el.colorReroll) el.colorReroll.disabled = lineColor !== "random";
    dirty = true;
    renderList(); showDetail();
  }

  /** Choose how thick the lines are drawn (LINE_WIDTH_MODES). */
  function setLineWidth(next) {
    if (!LINE_WIDTH_MODES.includes(next)) return;
    lineWidthMode = next;
    if (el.lineWidth) el.lineWidth.value = lineWidthMode;
    dirty = true;
  }

  return { load, preview, setLoading, fail, unload, rerender, setDesmosWarning, fit, handleKey, debug,
           setLineColor, lineColorState, setLineWidth, refreshList: renderList, curveCount: () => curves.length };
}
