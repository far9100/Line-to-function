// Canvas viewer for line2func results, shared by the app (python -m line2func) and the plain
// result viewer (python -m line2func.serve). It draws with Canvas2D from a few cached Path2D
// objects, finds the curve under the pointer with a spatial grid and renders only the visible
// rows of the equation list, so results with 10,000+ curves stay smooth.
import { t } from "./i18n.js";

export const PALETTE = ["#e6194b", "#0082c8", "#3cb44b", "#f58230", "#911eb4", "#00a0a0", "#f032e6", "#808000"];
export const DESMOS_LIMIT = 5000; // = line2func.export.DESMOS_CURVE_LIMIT (tests check it)
const ROW = 68, CELL = 32;
const FILLED_TAGS = ["outline", "fill_outline"]; // closed outlines, filled in the measured style

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

// el: {stage, canvas, tip, message, list, spacer, detail, stats, fit, styleLabel, styleMeasured, bgShow, bgKind, bgAlpha}
export function createViewer(el) {
  const { stage, canvas, tip, message, list, spacer, detail, stats } = el;
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, curves = [], groups = [], grid = new Map(), styled = new Map(), fills = new Map();
  let measuredStyle = false, lineWidthMeta = 2, status = "empty", failMsg = "", quality = null;
  let functionCount = null; // equations y = f(x) / x = g(y) in the result, when it has them
  let desmosWarning = true; // the app shows its own banner instead
  let view = { s: 1, tx: 0, ty: 0 }, dpr = window.devicePixelRatio || 1;
  let hover = -1, selected = -1, dirty = true, fitted = false, generation = 0;
  const bg = { img: null, show: true, alpha: 0.35, kind: "original", images: {} };
  const pointers = new Map();
  let drag = null, pinch = null;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- data ----------
  function reset() {
    W = 0; H = 0; curves = []; groups = []; grid = new Map(); styled = new Map(); fills = new Map();
    hover = -1; selected = -1; quality = null; fitted = false; functionCount = null;
    pointers.clear(); drag = null; pinch = null;
    canvas.classList.remove("dragging", "over");
    tip.style.display = "none";
    list.scrollTop = 0; spacer.innerHTML = ""; spacer.style.height = "0px";
    bg.images = {}; bg.img = null;
    updateBackgrounds();
    dirty = true;
  }

  function prepare(doc) {
    W = doc.image.width; H = doc.image.height;
    lineWidthMeta = (doc.meta && doc.meta.line_width) || 2;
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
               width: c.width ?? null, color: c.color ?? null, shape: c.shape ?? null, functions: c.functions ?? null };
    });
    functionCount = curves.some((c) => c.functions) ? curves.reduce((n, c) => n + (c.functions ? c.functions.length : 0), 0) : null;
    groups = PALETTE.map(() => new Path2D());
    const filledStrokes = new Set(curves.filter((c) => c.tags.some((tag) => FILLED_TAGS.includes(tag))).map((c) => c.stroke));
    for (const c of curves) {
      const p = c.p;
      groups[((c.stroke % PALETTE.length) + PALETTE.length) % PALETTE.length].moveTo(p[0], p[1]);
      groups[((c.stroke % PALETTE.length) + PALETTE.length) % PALETTE.length].bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
      const key = `${c.color || "#000000"}|${(Math.round((c.width || lineWidthMeta) * 4) / 4).toFixed(2)}`;
      let sp = styled.get(key);
      if (!sp) styled.set(key, (sp = new Path2D()));
      sp.moveTo(p[0], p[1]); sp.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
      const x0 = Math.floor(c.box[0] / CELL), x1 = Math.floor(c.box[2] / CELL);
      const y0 = Math.floor(c.box[1] / CELL), y1 = Math.floor(c.box[3] / CELL);
      for (let gy = y0; gy <= y1; gy++) for (let gx = x0; gx <= x1; gx++) {
        const cellKey = gx + "," + gy;
        let cell = grid.get(cellKey);
        if (!cell) grid.set(cellKey, (cell = []));
        cell.push(c.i);
      }
    }
    // closed outlines (thick strokes, filled areas): one even-odd fill per color, so holes stay open
    const byStroke = new Map();
    for (const c of curves) if (filledStrokes.has(c.stroke)) {
      if (!byStroke.has(c.stroke)) byStroke.set(c.stroke, []);
      byStroke.get(c.stroke).push(c);
    }
    for (const pieces of byStroke.values()) {
      const color = pieces.find((c) => c.color)?.color || "#000000";
      let path = fills.get(color);
      if (!path) fills.set(color, (path = new Path2D()));
      pieces.forEach((c, k) => {
        const p = c.p;
        if (k === 0) path.moveTo(p[0], p[1]);
        path.bezierCurveTo(p[2], p[3], p[4], p[5], p[6], p[7]);
      });
      path.closePath();
    }
    el.styleLabel.hidden = !curves.some((c) => c.width !== null || c.color !== null);
    spacer.style.height = curves.length * ROW + "px";
  }

  function updateBackgrounds() {
    for (const option of el.bgKind.options) option.disabled = !bg.images[option.value];
    if (!bg.images[bg.kind] && bg.images.original) bg.kind = "original";
    el.bgKind.value = bg.kind;
    bg.img = bg.images[bg.kind] || null;
    const any = Object.keys(bg.images).length > 0;
    el.bgShow.disabled = !any; el.bgAlpha.disabled = !any; el.bgKind.disabled = !any;
    dirty = true;
  }

  function renderStats() {
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
    if (!fitted && W && w && h) fitted = fit();
    dirty = true;
  }
  function fit() {
    const w = stage.clientWidth, h = stage.clientHeight, pad = 24;
    if (!W || !w || !h) return false; // hidden: fit once the stage has a size
    view.s = Math.max(1e-3, Math.min((w - 2 * pad) / W, (h - 2 * pad) / H));
    view.tx = (w - W * view.s) / 2; view.ty = (h - H * view.s) / 2; dirty = true;
    return true;
  }
  function zoomAt(px, py, f) {
    const s = Math.min(200, Math.max(0.02, view.s * f)), ix = (px - view.tx) / view.s, iy = (py - view.ty) / view.s;
    view.s = s; view.tx = px - ix * s; view.ty = py - iy * s; dirty = true;
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
    ctx.fillStyle = measuredStyle ? "#ffffff" : css("--paper"); // the measured style shows real ink on paper
    ctx.fillRect(0, 0, W, H);
    if (bg.img && bg.show && bg.alpha > 0) {
      ctx.globalAlpha = bg.alpha; ctx.imageSmoothingEnabled = view.s < 4;
      ctx.drawImage(bg.img, 0, 0, W, H); ctx.globalAlpha = 1;
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    if (measuredStyle) {
      // true to scale: fills for closed outlines, measured width in image pixels, measured ink color
      fills.forEach((path, color) => { ctx.fillStyle = color; ctx.fill(path, "evenodd"); });
      styled.forEach((path, key) => {
        const [color, w] = key.split("|");
        ctx.strokeStyle = color; ctx.lineWidth = parseFloat(w); ctx.stroke(path);
      });
    } else {
      ctx.lineWidth = 1.5 / view.s;
      groups.forEach((path, g) => { ctx.strokeStyle = PALETTE[g]; ctx.stroke(path); });
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
  function swatch(c) { return PALETTE[((c.stroke % PALETTE.length) + PALETTE.length) % PALETTE.length]; }
  function showDetail() {
    const c = curves[selected];
    if (!c) {
      detail.innerHTML = `<span class="hint">${esc(t("viewer.hint"))}</span>`;
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
        const i = hitTest((p.x - view.tx) / view.s, (p.y - view.ty) / view.s, 6 / view.s);
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
      view.s = s; view.tx = mx - ix * s; view.ty = my - iy * s; dirty = true;
    } else if (drag) {
      const dx = p.x - drag.x, dy = p.y - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; setHover(-1); showTip(-1); }
      if (drag.moved) { view.tx = drag.tx + dx; view.ty = drag.ty + dy; dirty = true; }
    }
  });
  function endPointer(e) {
    if (!pointers.has(e.pointerId)) return;
    const p = local(e);
    if (pointers.size === 1 && drag && !drag.moved && e.type === "pointerup") {
      const radius = (e.pointerType === "mouse" ? 6 : 14) / view.s;
      select(hitTest((p.x - view.tx) / view.s, (p.y - view.ty) / view.s, radius), true);
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
  el.bgShow.addEventListener("change", (e) => { bg.show = e.target.checked; dirty = true; });
  el.bgKind.addEventListener("change", (e) => {
    bg.kind = e.target.value; bg.img = bg.images[bg.kind] || null;
    if (bg.kind === "quality" && bg.alpha < 0.8) { bg.alpha = 0.9; el.bgAlpha.value = 90; }
    dirty = true;
  });
  el.bgAlpha.addEventListener("input", (e) => { bg.alpha = e.target.value / 100; dirty = true; });
  el.styleMeasured.addEventListener("change", (e) => { measuredStyle = e.target.checked; dirty = true; });
  new ResizeObserver(() => { resize(); renderList(); }).observe(stage);
  new ResizeObserver(renderList).observe(list);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dirty = true; });

  // ---------- public ----------
  // Show a result. sources: {original, lineart, quality, qualityJson} URLs (all optional).
  function load(doc, sources = {}) {
    const mine = ++generation;
    reset();
    prepare(doc);
    status = "ready";
    resize(); renderList(); showDetail(); renderStats(); renderMessage();
    for (const kind of ["original", "lineart", "quality"]) {
      if (!sources[kind]) continue;
      const img = new Image();
      img.onload = () => { if (mine !== generation) return; bg.images[kind] = img; updateBackgrounds(); };
      img.src = sources[kind];
    }
    if (sources.qualityJson) {
      fetch(sources.qualityJson, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((q) => { if (q && mine === generation) { quality = q; renderStats(); } })
        .catch(() => {});
    }
  }
  function setLoading() { generation++; reset(); status = "loading"; renderStats(); renderMessage(); showDetail(); }
  function fail(msg) { generation++; reset(); status = "failed"; failMsg = msg; renderStats(); renderMessage(); showDetail(); }
  function unload() { generation++; reset(); status = "empty"; renderStats(); renderMessage(); showDetail(); }
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
             backgrounds: Object.keys(bg.images), background: bg.kind, fills: fills.size, view: { ...view } };
  }
  return { load, setLoading, fail, unload, rerender, setDesmosWarning, fit, handleKey, debug };
}
