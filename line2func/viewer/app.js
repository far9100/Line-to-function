// line2func app: import an image (drag and drop, file picker or paste) -> configure (line art
// traced directly; photos get a line-art preview next to the original) -> convert -> result.
// Served by `python -m line2func`. When `python -m line2func.serve out/` serves it, the same page is
// the plain result viewer ("static" mode: /api/info answers {mode: "static"} or nothing).
import { t, setLang, getLang, detectLang, onLangChange } from "./i18n.js";
import { createViewer, DESMOS_LIMIT } from "./viewer.js";

// ---------- input guards and handlers: registered first, before anything can fail ----------
let dragTimer = 0, internalDrag = false;
addEventListener("dragstart", () => { internalDrag = true; });
addEventListener("dragend", () => { internalDrag = false; });
addEventListener("dragover", onDragOver);
addEventListener("dragleave", onDragLeave);
addEventListener("drop", onDrop);
addEventListener("paste", onPaste);
addEventListener("keydown", onKey);

const $ = (s) => document.querySelector(s);
const viewer = createViewer({
  stage: $("#stage"), canvas: $("#canvas"), tip: $("#tip"), message: $("#message"), list: $("#list"),
  spacer: $("#spacer"), detail: $("#detail"), stats: $("#stats"), fit: $("#fit"), styleLabel: $("#style-label"),
  styleMeasured: $("#style-measured"), bgShow: $("#bg-show"), bgKind: $("#bg-kind"), bgAlpha: $("#bg-alpha"),
});

const DEFAULTS = { method: "informative", scale: "auto", tolerance: 1, threshold: null, refine: true, faint: true,
                   upscale: "auto", named: false, shape_tolerance: 0.5, quality: false };
const PHOTO_METHODS = ["informative", "informative-coarse", "canny", "xdog"];
// the server's stages in the order they run (pipeline.trace, app.run_job) -> the step shown, and typical cost
const STEP_OF = { resize: "prepare", load_model: "prepare", lineart: "prepare", upscale: "prepare",
                  vectorize: "trace", refine: "trace", measure: "finish", outline: "finish", residual: "finish",
                  fill: "finish", count: "finish", optimize: "finish", shapes: "finish", export: "finish", quality: "quality" };
const WEIGHTS = { resize: 1, load_model: 6, lineart: 6, upscale: 3, vectorize: 45, refine: 20, measure: 8,
                  outline: 5, residual: 8, fill: 3, count: 10, optimize: 10, shapes: 6, export: 3, quality: 14 };

const S = {
  mode: "boot", screen: "boot", info: null, gone: false,
  image: null, type: "lineart", opts: { ...DEFAULTS },
  previews: new Map(), previewKey: "", jobs: new Map(),
  trace: null, result: null, ticket: 0, copyArmed: false,
};

// ---------- small helpers ----------
function show(screen) {
  S.screen = screen;
  for (const id of ["import", "configure", "result"]) $("#" + id).hidden = id !== screen;
  renderFileName();
  saveSession();
}

let toastTimer = 0;
function toast(text, kind = "", sticky = false) {
  const el = $("#toast");
  el.textContent = text;
  el.className = kind;
  el.hidden = false;
  clearTimeout(toastTimer);
  if (!sticky) toastTimer = setTimeout(() => { el.hidden = true; }, kind === "error" ? 7000 : 3500);
}

function apiError(code, info) {
  const err = new Error(code);
  err.code = code;
  err.info = info || {};
  return err;
}

function errorText(err) {
  const code = err?.code || "internal";
  const info = err?.info || {};
  const vars = {
    code, detail: info.detail || err?.message || "", field: info.field || "?",
    cmd: S.info?.weights_command || "python -m line2func.weights fetch informative",
    mb: Math.round((S.info?.limits?.max_bytes || 67108864) / 1048576),
  };
  return t("error." + code, vars, t("error.generic", vars));
}

async function api(method, path, body, headers = {}) {
  const init = { method, headers: { ...headers }, cache: "no-store" };
  if (method === "POST") {
    init.headers["X-Line2func-Token"] = S.info?.token || "";
    if (body instanceof Blob) init.body = body;
    else { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body ?? {}); }
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    checkServer();
    throw apiError("server_gone");
  }
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) throw apiError(data?.error?.code || `http_${response.status}`, data?.error);
  return data;
}

function isFormField(el) {
  return !!el?.closest?.("input, select, textarea, summary, [contenteditable='true'], [contenteditable='']");
}

// ---------- boot ----------
boot();

async function boot() {
  let info = null;
  try {
    const response = await fetch("api/info", { cache: "no-store" });
    if (response.ok) info = await response.json();
  } catch { /* the plain viewer may not answer at all */ }
  if (info && info.mode === "app") startApp(info);
  else startStatic(info);
}

function initLang(saved) {
  for (const button of document.querySelectorAll("[data-lang]")) button.addEventListener("click", () => chooseLang(button.dataset.lang));
  onLangChange(renderAll);
  setLang(detectLang(saved));
}

function chooseLang(lang) {
  setLang(lang);
  if (S.mode === "app") api("POST", "api/settings", { lang }).catch(() => {});
  else try { localStorage.setItem("line2func.lang", lang); } catch { /* storage may be blocked */ }
}

function renderAll() {
  document.title = t(S.mode === "static" ? "static.title" : "app.title");
  for (const button of document.querySelectorAll("[data-lang]")) button.setAttribute("aria-pressed", String(button.dataset.lang === getLang()));
  viewer.rerender();
  if (S.mode !== "app") return;
  $("#import-limits").textContent = t("import.limits", { mb: Math.round(S.info.limits.max_bytes / 1048576) });
  renderFileName();
  if (S.image) { renderScaleOptions(); renderMethods(); renderTypeNote(); renderQualityNote(); renderPreview(); }
  if (S.trace) renderRunning();
  renderBanner();
}

// ---------- static mode: the plain result viewer ----------
async function startStatic(info) {
  S.mode = "static";
  document.documentElement.dataset.mode = "static";
  let saved = null;
  try { saved = localStorage.getItem("line2func.lang"); } catch { /* storage may be blocked */ }
  initLang(saved);
  setupResult();
  show("result");
  viewer.setLoading();
  // the server lists the optional files it has; an older server doesn't, so try them all
  const has = (name) => !Array.isArray(info?.files) || info.files.includes(name);
  const source = (name) => (has(name) ? "data/" + name : null);
  try {
    const response = await fetch("data/curves.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const doc = await response.json();
    viewer.load(doc, { original: source("source.png"), lineart: source("lineart.png"), quality: source("quality.png"),
                       qualityJson: source("quality.json") });
    S.result = { base: "data/", doc, name: doc.meta?.source || "line2func", desmos: null, snap: null };
    prefetchDesmos(S.result);
  } catch (err) {
    viewer.fail(err.message);
  }
}

// ---------- app mode ----------
async function startApp(info) {
  S.mode = "app";
  S.info = info;
  document.documentElement.dataset.mode = "app";
  viewer.setDesmosWarning(false); // the result banner says it, with an action
  if (info.settings?.options) Object.assign(S.opts, sanitizeOptions(info.settings.options));
  initLang(info.settings?.lang);
  setupImport();
  setupConfigure();
  setupResult();
  $("#quit").addEventListener("click", quit);
  connectEvents();
  if (!(await restoreSession())) show("import");
  window.__l2f = {
    state: () => S.screen, mode: () => S.mode, debug: () => viewer.debug(), info: () => S.info,
    image: () => S.image, result: () => S.result && { job: S.result.snap, name: S.result.name },
    previews: () => Object.fromEntries([...S.previews].map(([k, v]) => [k, v.state])),
    options: () => ({ ...S.opts, type: S.type }), openFile,
  };
}

function sanitizeOptions(o) {
  const out = {};
  if (PHOTO_METHODS.includes(o.method)) out.method = o.method;
  if (o.scale === "auto" || [1, 0.5, 0.25].includes(o.scale)) out.scale = o.scale;
  for (const [key, lo, hi] of [["tolerance", 0.05, 20], ["shape_tolerance", 0.05, 10]]) {
    if (typeof o[key] === "number" && o[key] >= lo && o[key] <= hi) out[key] = o[key];
  }
  if (o.threshold === null || (typeof o.threshold === "number" && o.threshold > 0 && o.threshold < 1)) out.threshold = o.threshold;
  for (const key of ["refine", "faint", "named", "quality"]) if (typeof o[key] === "boolean") out[key] = o[key];
  if (["auto", 1, 2].includes(o.upscale)) out.upscale = o.upscale;
  return out;
}

let saveTimer = 0;
function saveOptions() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api("POST", "api/settings", { options: S.opts }).catch(() => {}), 500);
}

function saveSession() {
  if (S.mode !== "app") return;
  try {
    sessionStorage.setItem("line2func.session", JSON.stringify({
      screen: S.screen, imageId: S.image?.image_id || null, type: S.type, jobId: S.result?.snap?.job_id || null,
    }));
  } catch { /* storage may be blocked */ }
}

async function restoreSession() {
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem("line2func.session") || "null"); } catch { /* ignore */ }
  if (!saved?.imageId) return false;
  try {
    S.image = await api("GET", `api/images/${saved.imageId}`);
  } catch {
    return false;
  }
  S.type = saved.type === "photo" ? "photo" : saved.type === "lineart" ? "lineart" : S.image.suggested;
  if (saved.screen === "result" && saved.jobId) {
    try {
      const snap = await api("GET", `api/jobs/${saved.jobId}`);
      if (snap.state === "done") { loadResult(snap); return true; }
    } catch { /* fall back to the settings */ }
  }
  enterConfigure();
  return true;
}

// ---------- server events and lifetime ----------
let events = null;
function connectEvents() {
  events = new EventSource("api/events");
  events.addEventListener("hello", (e) => { for (const snap of JSON.parse(e.data).jobs || []) onJob(snap); });
  events.addEventListener("job", (e) => onJob(JSON.parse(e.data)));
  events.addEventListener("bye", () => { events.close(); fatal(); });
  events.onerror = () => { if (!S.gone) setTimeout(checkServer, 1500); };
}

async function checkServer() {
  if (S.gone) return;
  try {
    const response = await fetch("api/info", { cache: "no-store" });
    if (response.ok) return;
  } catch { /* gone */ }
  fatal();
}

function fatal() {
  if (S.gone) return;
  S.gone = true;
  if (events) events.close();
  $("#running").hidden = true;
  $("#fatal").hidden = false;
  setTimeout(() => window.close(), 300); // closes an app window (one history entry); a normal tab stays
}

async function quit() {
  try { await api("POST", "api/shutdown"); } catch { /* already gone */ }
  fatal();
}

// ---------- import: drag and drop, file picker, paste ----------
function setupImport() {
  const input = $("#file-input");
  $("#choose").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const file = input.files[0];
    input.value = "";
    if (file) openFile(file);
  });
}

function hasFiles(e) {
  return !!e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
}

function onDragOver(e) {
  e.preventDefault();
  if (internalDrag || !hasFiles(e)) {
    if (e.dataTransfer) e.dataTransfer.dropEffect = "none";
    return;
  }
  e.dataTransfer.dropEffect = "copy";
  $("#drop-overlay").hidden = false;
  clearTimeout(dragTimer);
  dragTimer = setTimeout(hideDrop, 150); // dragover repeats while the file is over the window
}

function onDragLeave(e) {
  if (e.relatedTarget === null) hideDrop();
}

function hideDrop() {
  clearTimeout(dragTimer);
  $("#drop-overlay").hidden = true;
}

function onDrop(e) {
  e.preventDefault();
  hideDrop();
  if (internalDrag || !e.dataTransfer) return;
  const dt = e.dataTransfer;
  const items = Array.from(dt.items || []).filter((item) => item.kind === "file");
  if (items.some((item) => item.webkitGetAsEntry?.()?.isDirectory)) { toast(t("error.folder"), "error"); return; }
  const files = Array.from(dt.files || []);
  if (!files.length) {
    if (Array.from(dt.types || []).some((type) => type === "text/uri-list" || type === "text/html")) toast(t("error.noFile"), "error");
    return;
  }
  if (files.length > 1) toast(t("error.multiple", { n: files.length }));
  openFile(files[0]);
}

function onPaste(e) {
  if (isFormField(e.target)) return;
  const files = Array.from(e.clipboardData?.files || []).filter((file) => file.type.startsWith("image/"));
  if (!files.length) return;
  e.preventDefault();
  openFile(files[0]);
}

let uploads = 0;
async function openFile(file) {
  if (S.mode !== "app") { toast(t("static.dropHint")); return; }
  if (S.gone || !file) return;
  if (file.size > S.info.limits.max_bytes) { toast(errorText({ code: "too_large" }), "error"); return; }
  // the current image and result stay until the new file has been read: a wrong file loses nothing
  const mine = ++uploads;
  const name = file.name || "image.png";
  const busy = t("import.uploading", { name });
  toast(busy, "", true);
  try {
    const info = await api("POST", "api/images", file, {
      "X-Filename": encodeURIComponent(name), "Content-Type": file.type || "application/octet-stream",
    });
    if (mine !== uploads) return;
    if ($("#toast").textContent === busy) $("#toast").hidden = true;
    cancelTrace(true);
    S.ticket++;
    S.previews.clear(); S.previewKey = ""; S.result = null; S.copyArmed = false;
    viewer.unload();
    $("#img-lineart").removeAttribute("src");
    $("#img-original").removeAttribute("src");
    shownPreview = "";
    S.image = info;
    S.type = info.suggested;
    enterConfigure();
  } catch (err) {
    if (mine === uploads) toast(errorText(err), "error");
  }
}

function renderFileName() {
  const el = $("#file-name");
  const img = S.image;
  el.hidden = !(S.mode === "app" && img && S.screen !== "import");
  el.textContent = img ? `${img.name} · ${img.width} × ${img.height}` : "";
}

// ---------- configure: type, line-art preview, options ----------
const cmp = { s: 1, tx: 0, ty: 0, drag: null };

function setupConfigure() {
  const form = $("#cfg-form");
  form.addEventListener("submit", (e) => { e.preventDefault(); startTrace(); });
  $("#back").addEventListener("click", () => { S.image = null; S.previews.clear(); show("import"); $("#choose").focus(); });
  for (const radio of form.querySelectorAll("input[name=type]")) {
    radio.addEventListener("change", () => { S.type = radio.value; layoutCompare(); renderTypeNote(); renderQualityNote(); requestPreview(); saveSession(); });
  }
  for (const radio of form.querySelectorAll("input[name=method]")) {
    radio.addEventListener("change", () => { S.opts.method = radio.value; renderMethods(); requestPreview(); saveOptions(); });
  }
  $("#scale").addEventListener("change", (e) => {
    S.opts.scale = e.target.value === "auto" ? "auto" : Number(e.target.value);
    renderQualityNote(); requestPreview(); saveOptions();
  });
  const number = (id, key, lo, hi) => $(id).addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    S.opts[key] = Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : DEFAULTS[key];
    e.target.value = S.opts[key];
    saveOptions();
  });
  number("#opt-tolerance", "tolerance", 0.05, 20);
  number("#opt-shape", "shape_tolerance", 0.05, 10);
  const check = (id, key) => $(id).addEventListener("change", (e) => { S.opts[key] = e.target.checked; saveOptions(); });
  check("#opt-refine", "refine");
  check("#opt-faint", "faint");
  check("#opt-named", "named");
  check("#opt-quality", "quality");
  $("#opt-upscale").addEventListener("change", (e) => { S.opts.upscale = e.target.value === "auto" ? "auto" : Number(e.target.value); saveOptions(); });
  $("#opt-threshold-auto").addEventListener("change", (e) => {
    S.opts.threshold = e.target.checked ? null : Number($("#opt-threshold").value);
    syncThreshold(); saveOptions();
  });
  $("#opt-threshold").addEventListener("input", (e) => { S.opts.threshold = Number(e.target.value); syncThreshold(); saveOptions(); });

  // the two panes share one zoom and pan, so the line art lines up with the original
  for (const vp of document.querySelectorAll("#compare .viewport")) {
    vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = vp.getBoundingClientRect();
      zoomCompare(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
    }, { passive: false });
    vp.addEventListener("pointerdown", (e) => {
      vp.setPointerCapture(e.pointerId);
      cmp.drag = { x: e.clientX, y: e.clientY, tx: cmp.tx, ty: cmp.ty };
      vp.classList.add("dragging");
    });
    vp.addEventListener("pointermove", (e) => {
      if (!cmp.drag) return;
      cmp.tx = cmp.drag.tx + e.clientX - cmp.drag.x; cmp.ty = cmp.drag.ty + e.clientY - cmp.drag.y;
      applyCompare();
    });
    const end = () => { cmp.drag = null; vp.classList.remove("dragging"); };
    vp.addEventListener("pointerup", end);
    vp.addEventListener("pointercancel", end);
    vp.addEventListener("dblclick", fitCompare);
  }
  $("#zoom-in").addEventListener("click", () => zoomCenter(1.25));
  $("#zoom-out").addEventListener("click", () => zoomCenter(0.8));
  $("#zoom-fit").addEventListener("click", fitCompare);
  for (const img of [$("#img-original"), $("#img-lineart")]) img.addEventListener("load", applyCompare);
  new ResizeObserver(() => { if (S.screen === "configure") layoutCompare(); }).observe($("#compare"));
  $("#cancel").addEventListener("click", () => cancelTrace(false));
}

function enterConfigure() {
  const img = S.image;
  if (!img) { show("import"); return; }
  show("configure");
  const original = $("#img-original");
  const src = `api/images/${img.image_id}/preview`;
  if (original.getAttribute("src") !== src) original.src = src;
  syncForm();
  layoutCompare();
  renderPreview();
  requestPreview();
}

function syncForm() {
  const form = $("#cfg-form");
  for (const radio of form.querySelectorAll("input[name=type]")) radio.checked = radio.value === S.type;
  renderScaleOptions();
  renderMethods();
  renderTypeNote();
  $("#opt-tolerance").value = S.opts.tolerance;
  $("#opt-shape").value = S.opts.shape_tolerance;
  $("#opt-refine").checked = S.opts.refine;
  $("#opt-faint").checked = S.opts.faint;
  $("#opt-named").checked = S.opts.named;
  $("#opt-quality").checked = S.opts.quality;
  $("#opt-upscale").value = String(S.opts.upscale);
  syncThreshold();
  renderQualityNote();
}

function syncThreshold() {
  const auto = S.opts.threshold === null;
  $("#opt-threshold-auto").checked = auto;
  $("#opt-threshold").disabled = auto;
  if (!auto) $("#opt-threshold").value = S.opts.threshold;
  $("#opt-threshold-value").textContent = auto ? "" : Number(S.opts.threshold).toFixed(2);
}

function currentScale() {
  return S.opts.scale === "auto" ? S.image.auto_scale : S.opts.scale;
}

function renderScaleOptions() {
  const img = S.image, select = $("#scale");
  select.innerHTML = "";
  for (const choice of ["auto", 1, 0.5, 0.25]) {
    const scale = choice === "auto" ? img.auto_scale : choice;
    const w = String(Math.max(1, Math.round(img.width * scale))), h = String(Math.max(1, Math.round(img.height * scale)));
    const label = choice === "auto" ? t("cfg.scale.auto", { w, h }) : t("cfg.scale.pct", { pct: choice * 100, w, h });
    const option = new Option(label, String(choice));
    option.disabled = choice !== "auto" && choice > img.max_scale + 1e-6;
    select.add(option);
  }
  const wanted = String(S.opts.scale);
  const option = [...select.options].find((o) => o.value === wanted && !o.disabled);
  select.value = option ? wanted : "auto";
  if (!option) S.opts.scale = "auto";
}

function renderMethods() {
  const methods = S.info.methods;
  if (!methods[S.opts.method]?.available) S.opts.method = PHOTO_METHODS.find((m) => methods[m]?.available) || "canny";
  let reason = null;
  for (const radio of document.querySelectorAll("input[name=method]")) {
    const status = methods[radio.value] || { available: false };
    radio.disabled = !status.available;
    radio.checked = radio.value === S.opts.method;
    const label = radio.closest("label");
    label.title = status.available ? t("methodHint." + radio.value) : t("reason." + status.reason, { cmd: S.info.weights_command });
    if (!status.available) reason = status.reason;
  }
  const note = $("#method-note");
  note.innerHTML = "";
  note.classList.toggle("warn-text", false);
  note.append(t("methodHint." + S.opts.method));
  if (reason) {
    const extra = document.createElement("span");
    extra.className = "warn-text";
    const [before, after] = t("reason." + reason, { cmd: "\0" }).split("\0");
    extra.append(document.createElement("br"), before);
    if (after !== undefined) {
      const code = document.createElement("code");
      code.textContent = S.info.weights_command;
      extra.append(code, after);
    }
    note.append(extra);
  }
  $("#method-set").hidden = S.type !== "photo";
}

function renderTypeNote() {
  if (!S.image) return;
  $("#type-auto").textContent = t("cfg.autoDetected", { type: t("cfg.typeName." + S.image.suggested) });
  $("#original-size").textContent = t("cfg.size", { w: String(S.image.width), h: String(S.image.height) });
  $("#method-set").hidden = S.type !== "photo";
}

function renderQualityNote() {
  if (!S.image) return;
  const mp = S.info.limits.quality_max_pixels / 1e6, scale = currentScale();
  const tooBig = S.image.width * S.image.height * scale * scale > S.info.limits.quality_max_pixels;
  $("#opt-quality").disabled = tooBig;
  $("#quality-hint").textContent = tooBig ? t("cfg.qualityTooBig", { mp }) : t("cfg.qualityHint", { mp });
}

function layoutCompare() {
  const photo = S.type === "photo";
  const compare = $("#compare");
  $("#pane-lineart").hidden = !photo;
  compare.classList.toggle("single", !photo);
  compare.classList.remove("stacked");
  if (photo && S.image) {
    // side by side or one above the other, whichever shows the image larger
    const r = compare.getBoundingClientRect(), w = S.image.width, h = S.image.height;
    const side = Math.min(r.width / 2 / w, r.height / h), stacked = Math.min(r.width / w, r.height / 2 / h);
    compare.classList.toggle("stacked", stacked > side * 1.15);
  }
  fitCompare();
}

function fitCompare() {
  const vp = $("#pane-original .viewport"), img = S.image;
  if (!img || !vp.clientWidth || !vp.clientHeight) return;
  cmp.s = Math.min(vp.clientWidth / img.width, vp.clientHeight / img.height) * 0.98;
  cmp.tx = (vp.clientWidth - img.width * cmp.s) / 2;
  cmp.ty = (vp.clientHeight - img.height * cmp.s) / 2;
  applyCompare();
}

function zoomCompare(px, py, factor) {
  const s = Math.min(64, Math.max(0.02, cmp.s * factor));
  const ix = (px - cmp.tx) / cmp.s, iy = (py - cmp.ty) / cmp.s;
  cmp.s = s; cmp.tx = px - ix * s; cmp.ty = py - iy * s;
  applyCompare();
}

function zoomCenter(factor) {
  const vp = $("#pane-original .viewport");
  zoomCompare(vp.clientWidth / 2, vp.clientHeight / 2, factor);
}

function applyCompare() {
  const img = S.image;
  if (!img) return;
  for (const el of [$("#img-original"), $("#img-lineart")]) {
    el.style.width = img.width + "px";
    el.style.height = img.height + "px";
    el.style.transform = `translate(${cmp.tx}px, ${cmp.ty}px) scale(${cmp.s})`;
    el.classList.toggle("pixelated", (cmp.s * img.width) / (el.naturalWidth || img.width) > 2);
  }
}

// ---------- line-art previews (photos) ----------
function previewKey() {
  return `${S.opts.method}|${currentScale()}`;
}

async function requestPreview() {
  // (a preview must not replace a queued conversion: the server keeps one waiting job)
  if (S.type !== "photo" || !S.image || S.gone || S.trace) { renderPreview(); return; }
  const key = previewKey();
  S.previewKey = key;
  const known = S.previews.get(key);
  if (known && !["error", "cancelled"].includes(known.state)) { renderPreview(); return; }
  const entry = { state: "queued", jobId: null, url: null, error: null, stage: null };
  S.previews.set(key, entry);
  renderPreview();
  try {
    const snap = await api("POST", "api/jobs", { image_id: S.image.image_id, kind: "lineart", method: S.opts.method, scale: S.opts.scale });
    entry.jobId = snap.job_id;
    onJob(S.jobs.get(snap.job_id) || snap);
  } catch (err) {
    entry.state = "error";
    entry.error = err;
    renderPreview();
  }
}

let shownPreview = "";
function renderPreview() {
  const img = $("#img-lineart"), spinner = $("#lineart-spinner"), msg = $("#lineart-msg"), state = $("#lineart-state");
  const entry = S.type === "photo" ? S.previews.get(S.previewKey) : null;
  if (!entry) { spinner.hidden = true; msg.hidden = true; state.textContent = ""; return; }
  msg.hidden = entry.state !== "error";
  if (entry.state === "error") {
    spinner.hidden = true;
    img.classList.add("stale");
    msg.textContent = errorText(entry.error);
    state.textContent = "";
    return;
  }
  if (entry.state === "done" && entry.url) {
    state.textContent = "";
    if (shownPreview === entry.url) { spinner.hidden = true; img.classList.remove("stale"); return; }
    const url = entry.url, next = new Image();
    next.src = url;
    next.decode().then(() => {
      if (S.previews.get(S.previewKey)?.url !== url) return; // the user moved on meanwhile
      img.src = url;
      shownPreview = url;
      img.classList.remove("stale");
      spinner.hidden = true;
    }).catch(() => { spinner.hidden = true; });
    return;
  }
  spinner.hidden = false;
  img.classList.toggle("stale", !!img.getAttribute("src"));
  state.textContent = entry.stage === "load_model" ? t("cfg.previewLoading")
    : entry.state === "running" ? t("cfg.previewRunning") : t("cfg.previewWaiting");
}

function onJob(snap) {
  if (!snap?.job_id) return;
  S.jobs.set(snap.job_id, snap);
  for (const [key, entry] of S.previews) {
    if (entry.jobId !== snap.job_id) continue;
    entry.state = snap.state;
    entry.stage = snap.stage;
    if (snap.state === "done") entry.url = `api/jobs/${snap.job_id}/data/lineart.png`;
    if (snap.state === "error") entry.error = apiError(snap.error?.code, snap.error);
    if (key === S.previewKey) renderPreview();
    // replaced while queued (by a newer job): ask again, unless a conversion took its place
    if (snap.state === "cancelled" && key === S.previewKey && S.screen === "configure" && !S.trace) requestPreview();
  }
  if (S.trace && S.trace.jobId === snap.job_id) onTraceUpdate(snap);
}

// ---------- converting ----------
function startTrace() {
  if (!S.image || S.gone || S.trace) return;
  const mine = ++S.ticket;
  const method = S.type === "photo" ? S.opts.method : "none";
  const params = {
    image_id: S.image.image_id, kind: "trace", method, scale: S.opts.scale, tolerance: S.opts.tolerance,
    threshold: S.opts.threshold, refine: S.opts.refine, named: S.opts.named, shape_tolerance: S.opts.shape_tolerance,
    upscale: S.opts.upscale, faint: S.opts.faint, quality: S.opts.quality && !$("#opt-quality").disabled,
  };
  saveOptions();
  S.trace = { jobId: null, snap: null, params, started: performance.now(), group: null };
  $("#running").hidden = false;
  renderRunning();
  S.trace.timer = setInterval(renderElapsed, 200);
  $("#cancel").focus();
  api("POST", "api/jobs", params).then((snap) => {
    if (mine !== S.ticket || !S.trace) { api("POST", `api/jobs/${snap.job_id}/cancel`).catch(() => {}); return; }
    S.trace.jobId = snap.job_id;
    onTraceUpdate(S.jobs.get(snap.job_id) || snap);
  }, (err) => {
    if (mine !== S.ticket) return;
    stopRunning();
    toast(errorText(err), "error");
  });
}

function stopRunning() {
  if (S.trace?.timer) clearInterval(S.trace.timer);
  S.trace = null;
  $("#running").hidden = true;
}

function cancelTrace(silent) {
  const trace = S.trace;
  if (!trace) return;
  S.ticket++;
  stopRunning();
  if (trace.jobId) api("POST", `api/jobs/${trace.jobId}/cancel`).catch(() => {});
  if (!silent) { toast(t("run.cancelled")); $("#convert").focus(); }
}

function onTraceUpdate(snap) {
  const trace = S.trace;
  trace.snap = snap;
  if (snap.state === "done") { stopRunning(); loadResult(snap); return; }
  if (snap.state === "error") { stopRunning(); toast(errorText(apiError(snap.error?.code, snap.error)), "error"); return; }
  if (snap.state === "cancelled") { stopRunning(); return; }
  renderRunning();
}

function stepsFor(params) {
  const steps = ["prepare", "trace", "finish"];
  if (params.quality) steps.push("quality");
  return steps;
}

function renderRunning() {
  const trace = S.trace;
  if (!trace) return;
  const snap = trace.snap, stage = snap?.stage || null, steps = stepsFor(trace.params);
  const current = snap?.state === "running" ? STEP_OF[stage] || "prepare" : null;
  const list = $("#steps");
  list.innerHTML = "";
  let passed = true;
  for (const step of steps) {
    const li = document.createElement("li");
    li.textContent = t("step." + step);
    if (step === current) { li.className = "active"; passed = false; } else if (passed && current) li.className = "done";
    list.append(li);
  }
  $("#run-stage").textContent = !snap || snap.state === "queued" ? t("run.queued") : stage ? t("stage." + stage, {}, stage) : "";
  // progress: the share of the typical work before the current stage
  const order = Object.keys(WEIGHTS).filter((s) => (s !== "quality" || trace.params.quality) && (s !== "refine" || trace.params.refine));
  const total = order.reduce((sum, s) => sum + WEIGHTS[s], 0);
  const index = stage ? order.indexOf(stage) : -1;
  const done = index > 0 ? order.slice(0, index).reduce((sum, s) => sum + WEIGHTS[s], 0) : 0;
  $("#run-bar").style.width = `${Math.max(3, Math.round((100 * done) / total))}%`;
  if (current && current !== trace.group) { trace.group = current; $("#run-live").textContent = t("step." + current); }
  renderElapsed();
}

function renderElapsed() {
  if (!S.trace) return;
  $("#elapsed").textContent = t("run.elapsed", { s: ((performance.now() - S.trace.started) / 1000).toFixed(1) });
}

// ---------- result ----------
const DOWNLOAD_NAMES = { "curves.json": "{stem}.json", "out.svg": "{stem}.svg", "desmos.txt": "{stem}-desmos.txt",
                         "equations.tex": "{stem}.tex", zip: "{stem}-line2func.zip" };

function setupResult() {
  for (const button of document.querySelectorAll("[data-file]")) button.addEventListener("click", () => download(button.dataset.file));
  $("#copy-all").addEventListener("click", copyAll);
  $("#adjust").addEventListener("click", () => enterConfigure());
  $("#new-image").addEventListener("click", () => {
    S.image = null; S.previews.clear(); S.result = null; viewer.unload(); show("import"); $("#choose").focus();
  });
}

async function loadResult(snap) {
  const mine = ++S.ticket;
  const base = `api/jobs/${snap.job_id}/data/`;
  show("result");
  viewer.setLoading();
  renderBanner();
  try {
    const response = await fetch(base + "curves.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const doc = await response.json();
    if (mine !== S.ticket) return;
    const files = new Set(snap.files);
    viewer.load(doc, {
      original: `api/images/${snap.image_id}/preview`,
      lineart: files.has("lineart.png") ? base + "lineart.png" : null,
      quality: files.has("quality.png") ? base + "quality.png" : null,
      qualityJson: files.has("quality.json") ? base + "quality.json" : null,
    });
    S.result = { base, doc, snap, name: S.image?.name || doc.meta?.source || "line2func", desmos: null };
    S.copyArmed = false;
    prefetchDesmos(S.result);
    renderBanner();
    saveSession();
  } catch (err) {
    if (mine === S.ticket) viewer.fail(err.message);
  }
}

function prefetchDesmos(result) {
  // fetched ahead, so "Copy all" can write to the clipboard right inside the click
  fetch(result.base + "desmos.txt").then((r) => (r.ok ? r.text() : null)).then((text) => { result.desmos = text; }).catch(() => {});
}

function stemOf(name) {
  const dot = name.lastIndexOf(".");
  return (dot > 0 ? name.slice(0, dot) : name).trim() || "line2func";
}

async function download(file) {
  const result = S.result;
  if (!result) return;
  const name = DOWNLOAD_NAMES[file].replace("{stem}", stemOf(result.name));
  const url = file === "zip" ? result.base.replace(/data\/$/, "zip") : result.base + file;
  if (S.mode === "app" && window.showSaveFilePicker) {
    let handle = null;
    try {
      handle = await window.showSaveFilePicker({ suggestedName: name }); // must be the first await of the click
    } catch (err) {
      if (err.name === "AbortError") return;
    }
    if (handle) {
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const writable = await handle.createWritable();
        await writable.write(await response.blob());
        await writable.close();
        toast(t("result.saved", { name: handle.name }));
      } catch (err) {
        toast(t("error.save", { detail: err.message }), "error");
      }
      return;
    }
  }
  const a = document.createElement("a");
  a.href = url + "?download";
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
}

function copyAll() {
  const result = S.result;
  if (!result || result.desmos === null) return;
  const n = result.doc.curves.length;
  if (n > DESMOS_LIMIT && !S.copyArmed) { S.copyArmed = true; toast(t("result.confirmCopy", { n, limit: DESMOS_LIMIT })); return; }
  S.copyArmed = false;
  navigator.clipboard.writeText(result.desmos).then(
    () => toast(t("result.copiedAll", { n })),
    () => toast(t("viewer.copyFailed"), "error"));
}

function renderBanner() {
  const banner = $("#banner");
  const summary = S.screen === "result" ? S.result?.snap?.summary : null;
  const warnings = summary?.warnings || [];
  banner.hidden = !warnings.length;
  banner.innerHTML = "";
  for (const warning of warnings) {
    const row = document.createElement("div");
    row.className = "banner-row";
    const vars = { n: summary.curves, limit: DESMOS_LIMIT, mp: (S.info?.limits?.quality_max_pixels || 4e6) / 1e6 };
    row.append(t("warning." + warning, vars, warning));
    const action = (label, fn) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.addEventListener("click", fn);
      row.append(button);
    };
    if (warning === "no_lines" && S.type === "lineart" && S.image) {
      action(t("warning.no_lines.action"), () => { S.type = "photo"; enterConfigure(); });
    }
    if (warning === "over_desmos_limit" && S.image) {
      action(t("warning.over_desmos_limit.action"), () => {
        S.opts.tolerance = Math.min(20, Math.round(S.opts.tolerance * 2 * 100) / 100);
        enterConfigure();
        $("#advanced").open = true;
        $("#opt-tolerance").focus();
      });
    }
    banner.append(row);
  }
}

// ---------- keyboard ----------
function onKey(e) {
  if (e.isComposing || e.defaultPrevented) return;
  const mod = e.ctrlKey || e.metaKey;
  if (mod && !e.altKey && !e.shiftKey && (e.key === "o" || e.key === "O")) {
    e.preventDefault(); // never let the browser open a file in this window
    if (S.mode === "app" && !S.gone) $("#file-input").click();
    return;
  }
  if (mod || e.altKey || isFormField(e.target)) return;
  if (S.trace && e.key === "Escape") { e.preventDefault(); cancelTrace(false); return; }
  if (S.screen === "result") {
    if (viewer.handleKey(e)) e.preventDefault();
  } else if (S.screen === "configure" && !S.trace) {
    if (e.key === "+" || e.key === "=") zoomCenter(1.25);
    else if (e.key === "-") zoomCenter(0.8);
    else if (e.key === "0" || e.key === "f" || e.key === "F") fitCompare();
    else return;
    e.preventDefault();
  }
}
