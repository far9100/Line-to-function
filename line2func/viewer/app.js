// line2func app: one page, the curve viewer. Without an image it shows a drop zone; a dropped (chosen or
// pasted) line drawing is shown in place, and Convert traces it into functions or parametric equations like
// the command line does (up to 5,000 curves, with a quality check). /api/info says who does the work:
// - "app": `python -m line2func` serves the page and traces on this computer;
// - "web": the online page (static files, line2func/website.py) traces in the browser (engine.js, Pyodide);
// - "static" (or no answer): `python -m line2func.serve out/` shows a saved result, the plain result viewer.
import { t, setLang, getLang, detectLang, onLangChange } from "./i18n.js";
import { createViewer, DESMOS_LIMIT } from "./viewer.js";
import { createEngine } from "./engine.js";

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
  spacer: $("#spacer"), detail: $("#detail"), stats: $("#stats"), fit: $("#fit"),
  originalOnly: $("#original-only"), viewMode: $("#view-mode"), bgAlpha: $("#bg-alpha"),
  lineColor: $("#line-color"), colorReroll: $("#color-reroll"), lineWidth: $("#line-width"),
  convertBar: $("#convert-bar"), listToggleLabel: $("#list-toggle-label"),
  onLineColor: (choice) => {
    S.lineColor = choice.mode; S.colorSeed = choice.seed; S.lineWidth = choice.width;
    saveOptions();
  },
});

const FORMS = ["function", "parametric"]; // how the lines are written (line2func.export)
const LINE_COLORS = ["bw", "palette", "random"]; // = line2func.viewer.viewer LINE_COLOR_MODES
const LINE_COLOR = "palette"; // the colors the viewer has always drawn; "measured" is the SVG's own
const LINE_WIDTHS = ["measured", "uniform"]; // = line2func.export.LINE_WIDTH_MODES
const LINE_WIDTH = "measured"; // what the SVG has always written: each stroke as thick as its ink
const DENOISE = 50; // = line2func.pipeline.DENOISE, the noise filters' default strength (tests check it)
const FAINT = 50; // = line2func.pipeline.FAINT_SENSITIVITY, the faint-line sensitivity's default (tests check it)
// the stages in the order they run (the online engine's start, pipeline.trace, app.run_job) -> the step shown,
// and typical cost
const STEP_OF = { load_engine: "prepare", resize: "prepare", load_model: "prepare", lineart: "prepare", upscale: "prepare",
                  vectorize: "trace", refine: "trace", measure: "finish", outline: "finish", residual: "finish",
                  fill: "finish", count: "finish", optimize: "finish", shapes: "finish", export: "finish", quality: "quality" };
const WEIGHTS = { load_engine: 12, resize: 1, load_model: 6, lineart: 6, upscale: 3, vectorize: 45, refine: 20, measure: 8,
                  outline: 5, residual: 8, fill: 3, count: 10, optimize: 10, shapes: 6, export: 3, quality: 14 };
const NEVER = ["load_model", "optimize"]; // stages the page's conversions do not run

const S = {
  mode: "boot", info: null, gone: false, engine: null,
  view: "empty", // empty (the drop zone), preview (an image to convert) or result
  image: null, form: "function", denoise: DENOISE, denoiseOn: true, faint: FAINT, jobs: new Map(),
  lineColor: LINE_COLOR, colorSeed: 0, // the chosen line color; colorSeed only matters for "random"
  lineWidth: LINE_WIDTH, // the chosen line thickness (LINE_WIDTHS)
  trace: null, result: null, ticket: 0, copyArmed: false,
};
const converts = () => S.mode === "app" || S.mode === "web"; // the page can open and convert images

// ---------- small helpers ----------
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
  return !!el?.closest?.("input, select, textarea, [contenteditable='true'], [contenteditable='']");
}

// ---------- boot ----------
boot();

async function boot() {
  let info = null;
  try {
    const response = await fetch("api/info", { cache: "no-store" });
    if (response.ok) info = await response.json();
  } catch { /* the plain viewer may not answer at all */ }
  if (info?.mode === "app") startApp(info);
  else if (info?.mode === "web") startWeb(info);
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
  if (!converts()) return;
  $("#import-limits").textContent = t("import.limits", { mb: Math.round(S.info.limits.max_bytes / 1048576) });
  renderFileName();
  renderConvert();
  renderEngine();
  if (S.trace) renderRunning();
  renderBanner();
}

// ---------- static mode: the plain result viewer ----------
async function startStatic(info) {
  S.mode = "static";
  S.view = "result";
  document.documentElement.dataset.mode = "static";
  initLang(readLocal("line2func.lang"));
  setupResult();
  viewer.setLoading();
  // the server lists the optional files it has; an older server doesn't, so try them all
  const has = (name) => !Array.isArray(info?.files) || info.files.includes(name);
  const source = (name) => (has(name) ? "data/" + name : null);
  try {
    const response = await fetch("data/curves.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const doc = await response.json();
    viewer.load(doc, { original: source("source.png"), quality: source("quality.png"), qualityJson: source("quality.json") });
    S.result = { url: (file) => "data/" + file, doc, name: doc.meta?.source || "line2func", desmos: null, snap: null };
    prefetchDesmos(S.result);
  } catch (err) {
    viewer.fail(err.message);
  }
}

// ---------- app mode: python -m line2func traces on this computer ----------
async function startApp(info) {
  S.mode = "app";
  S.info = info;
  document.documentElement.dataset.mode = "app";
  document.documentElement.dataset.view = "empty"; // before restoreSession() awaits: no flash of result chrome
  viewer.setDesmosWarning(false); // the result banner says it
  useOptions(info.settings?.options);
  initLang(info.settings?.lang);
  setupApp();
  setupResult();
  $("#quit").addEventListener("click", quit);
  connectEvents();
  if (!(await restoreSession())) showEmpty();
  window.__l2f = testHooks();
}

// ---------- web mode: the online page traces in the browser ----------
function startWeb(info) {
  S.mode = "web";
  S.info = info;
  document.documentElement.dataset.mode = "web";
  document.documentElement.dataset.view = "empty";
  viewer.setDesmosWarning(false); // the result banner says it
  useOptions(readLocal("line2func.options", true));
  S.engine = createEngine(info.engine, { onJob, onStatus: renderEngine });
  initLang(readLocal("line2func.lang"));
  setupApp();
  setupResult();
  showEmpty();
  // the engine (about 25 MB the first time) starts at once, unless the browser asks to save data: then with
  // the first image
  if (!navigator.connection?.saveData) S.engine.start();
  window.__l2f = { ...testHooks(), engine: () => S.engine.status() };
}

function readLocal(key, json = false) {
  try {
    const text = localStorage.getItem(key);
    return json ? JSON.parse(text || "null") : text;
  } catch {
    return null; // storage may be blocked, or hold something else
  }
}

function useOptions(saved) {
  if (!saved || typeof saved !== "object") return;
  if (FORMS.includes(saved.form)) S.form = saved.form;
  if (typeof saved.denoise === "number" && saved.denoise >= 0 && saved.denoise <= 100) S.denoise = saved.denoise;
  if (typeof saved.denoise_on === "boolean") S.denoiseOn = saved.denoise_on;
  if (typeof saved.faint_sensitivity === "number" && saved.faint_sensitivity >= 0 && saved.faint_sensitivity <= 100) {
    S.faint = saved.faint_sensitivity;
  }
  if (LINE_COLORS.includes(saved.line_color)) S.lineColor = saved.line_color;
  if (LINE_WIDTHS.includes(saved.line_width)) S.lineWidth = saved.line_width;
  if (Number.isInteger(saved.color_seed) && saved.color_seed >= 0 && saved.color_seed <= 0xffffffff) {
    S.colorSeed = saved.color_seed;
  }
  viewer.setLineColor(S.lineColor, S.colorSeed);
  viewer.setLineWidth(S.lineWidth);
}

function testHooks() {
  return {
    state: () => S.view, mode: () => S.mode, debug: () => viewer.debug(), info: () => S.info,
    image: () => S.image, result: () => S.result && { job: S.result.snap, name: S.result.name },
    desmos: () => S.result?.desmos ?? null, form: () => S.form, openFile,
  };
}

function renderEngine() {
  const el = $("#engine-status");
  if (S.mode !== "web") return;
  const status = S.engine.status();
  el.classList.toggle("error", status.state === "failed");
  if (status.state === "failed") el.textContent = errorText(apiError(status.code, { detail: status.detail }));
  else el.textContent = t(status.state === "ready" ? "engine.ready" : status.state === "idle" ? "engine.idle" : "engine.loading");
}

function setupApp() {
  const input = $("#file-input");
  $("#choose").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const file = input.files[0];
    input.value = "";
    if (file) openFile(file);
  });
  for (const radio of document.querySelectorAll("input[name=form]")) {
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      S.form = radio.value;
      saveOptions();
    });
  }
  $("#denoise-on").addEventListener("change", (e) => { S.denoiseOn = e.target.checked; renderDenoise(); saveOptions(); });
  $("#denoise").addEventListener("input", (e) => { S.denoise = Number(e.target.value); renderDenoise(); });
  $("#denoise").addEventListener("change", saveOptions);
  $("#faint").addEventListener("input", (e) => { S.faint = Number(e.target.value); renderFaint(); });
  $("#faint").addEventListener("change", saveOptions);
  $("#convert").addEventListener("click", startTrace);
  $("#clear").addEventListener("click", clearImage);
  $("#cancel").addEventListener("click", () => cancelTrace(false));
}

function saveOptions() {
  const options = { form: S.form, denoise: S.denoise, denoise_on: S.denoiseOn, faint_sensitivity: S.faint,
                    line_color: S.lineColor, color_seed: S.colorSeed, line_width: S.lineWidth };
  if (S.mode === "app") api("POST", "api/settings", { options }).catch(() => {});
  else try { localStorage.setItem("line2func.options", JSON.stringify(options)); } catch { /* storage may be blocked */ }
}

function saveSession() {
  if (S.mode !== "app") return;
  try {
    sessionStorage.setItem("line2func.session", JSON.stringify({
      view: S.view, imageId: S.image?.image_id || null, jobId: S.result?.snap?.job_id || null,
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
  if (saved.view === "result" && saved.jobId) {
    try {
      const snap = await api("GET", `api/jobs/${saved.jobId}`);
      if (snap.state === "done") { loadResult(snap); return true; }
    } catch { /* show the image again */ }
  }
  showPreview();
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

// ---------- opening an image: drag and drop, file picker, paste ----------
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
  if (!converts()) { toast(t("static.dropHint")); return; }
  if (S.gone || !file) return;
  if (file.size > S.info.limits.max_bytes) { toast(errorText({ code: "too_large" }), "error"); return; }
  // the current image and result stay until the new file has been read: a wrong file loses nothing (online,
  // a conversion still running stops first: the engine does one thing at a time)
  if (S.mode === "web") cancelTrace(true);
  const mine = ++uploads;
  const name = file.name || "image.png";
  const busy = t("import.uploading", { name });
  toast(busy, "", true);
  try {
    const info = S.mode === "web" ? await S.engine.open(file, name) : await api("POST", "api/images", file, {
      "X-Filename": encodeURIComponent(name), "Content-Type": file.type || "application/octet-stream",
    });
    if (mine !== uploads) return;
    if ($("#toast").textContent === busy) $("#toast").hidden = true;
    cancelTrace(true);
    S.image = info;
    showPreview();
  } catch (err) {
    if (mine === uploads) toast(errorText(err), "error");
  }
}

// On a phone the equation list starts collapsed to its header row, so the drawing gets the screen;
// the button is display:none on a desktop, where the list is always open.
function toggleList() {
  const aside = $("#list-toggle").closest("aside");
  const open = !aside.hasAttribute("data-open");
  aside.toggleAttribute("data-open", open);
  $("#list-toggle").setAttribute("aria-expanded", String(open));
  if (open) viewer.refreshList(); // it was display:none, so it has no rows yet
}

// ---------- the page's three states: the drop zone, an image to convert, a result ----------
function setView(view) {
  S.view = view;
  document.documentElement.dataset.view = view; // the layout's three states, for index.html's media queries
  $("#empty").hidden = view !== "empty";
  $("#convert-bar").hidden = view !== "preview";
  $("#clear").hidden = view === "empty";
  for (const button of document.querySelectorAll("[data-file], #copy-all")) button.disabled = view !== "result";
  $("#fit").disabled = view === "empty";
  renderFileName();
  renderBanner();
  saveSession();
}

function showEmpty() {
  S.image = null; S.result = null; S.copyArmed = false;
  viewer.unload();
  setView("empty");
}

function showPreview() {
  S.ticket++;
  S.result = null; S.copyArmed = false;
  setView("preview");
  renderConvert();
  viewer.preview(previewURL(S.image.image_id), S.image.width, S.image.height);
  $("#convert").focus();
}

function previewURL(imageId) {
  return S.mode === "web" ? S.engine.previewURL(imageId) : `api/images/${imageId}/preview`;
}

function clearImage() {
  cancelTrace(true);
  S.ticket++;
  showEmpty();
  $("#choose").focus();
}

function renderFileName() {
  const el = $("#file-name");
  const img = S.image;
  el.hidden = !(converts() && img);
  el.textContent = img ? `${img.name} · ${img.width} × ${img.height}` : "";
}

function renderDenoise() {
  $("#denoise-on").checked = S.denoiseOn;
  const slider = $("#denoise");
  slider.value = String(S.denoise);
  slider.disabled = !S.denoiseOn;
  $("#denoise-value").textContent = S.denoiseOn ? t("convert.strength", { n: String(S.denoise) }) : t("convert.off");
}

function renderFaint() {
  $("#faint").value = String(S.faint);
  $("#faint-value").textContent = S.faint > 0 ? t("convert.faintLevel", { n: String(S.faint) }) : t("convert.off");
}

function renderConvert() {
  for (const radio of document.querySelectorAll("input[name=form]")) radio.checked = radio.value === S.form;
  renderDenoise();
  renderFaint();
  let note = t(S.mode === "web" ? "convert.noteWeb" : "convert.note", { n: new Intl.NumberFormat().format(S.info.desmos_limit) });
  if (S.image?.suggested === "photo") note += " " + t("convert.photo");
  $("#convert-note").textContent = note;
}

// ---------- converting ----------
function startTrace() {
  if (!S.image || S.gone || S.trace) return;
  const mine = ++S.ticket;
  const img = S.image;
  // as the command line makes it: up to desmos_limit curves, with the quality check (the server skips that,
  // with a warning, for images that are too large)
  const params = { image_id: img.image_id, kind: "trace", method: "none", scale: "auto", form: S.form,
                   curves: S.info.desmos_limit, quality: true, denoise: S.denoiseOn ? S.denoise : 0,
                   faint_sensitivity: S.faint };
  const checked = img.width * img.height * img.auto_scale ** 2 <= S.info.limits.quality_max_pixels;
  S.trace = { jobId: null, snap: null, params, checked, started: performance.now(), group: null, engine: false };
  $("#running").hidden = false;
  renderRunning();
  S.trace.timer = setInterval(renderElapsed, 200);
  $("#cancel").focus();
  const submitted = S.mode === "web" ? S.engine.trace(params) : api("POST", "api/jobs", params);
  submitted.then((snap) => {
    if (mine !== S.ticket || !S.trace) { cancelJob(snap.job_id); return; }
    S.trace.jobId = snap.job_id;
    onTraceUpdate(S.jobs.get(snap.job_id) || snap);
  }, (err) => {
    if (mine !== S.ticket) return;
    stopRunning();
    toast(errorText(err), "error");
  });
}

function cancelJob(jobId) {
  if (S.mode === "web") S.engine.cancel(jobId);
  else api("POST", `api/jobs/${jobId}/cancel`).catch(() => {});
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
  if (trace.jobId) cancelJob(trace.jobId);
  if (!silent) { toast(t("run.cancelled")); $("#convert").focus(); }
}

function onJob(snap) {
  if (!snap?.job_id) return;
  S.jobs.set(snap.job_id, snap);
  if (S.trace && S.trace.jobId === snap.job_id) onTraceUpdate(snap);
}

function onTraceUpdate(snap) {
  const trace = S.trace;
  trace.snap = snap;
  if (snap.stage === "load_engine") trace.engine = true; // this conversion waits for the online engine to start
  if (snap.state === "done") { stopRunning(); loadResult(snap); return; }
  if (snap.state === "error") { stopRunning(); toast(errorText(apiError(snap.error?.code, snap.error)), "error"); return; }
  if (snap.state === "cancelled") { stopRunning(); return; }
  renderRunning();
}

function renderRunning() {
  const trace = S.trace;
  if (!trace) return;
  const snap = trace.snap, stage = snap?.stage || null;
  const steps = ["prepare", "trace", "finish", ...(trace.checked ? ["quality"] : [])];
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
  const waiting = !snap || snap.state === "queued"; // online, nothing waits behind another job
  $("#run-stage").textContent = waiting ? (S.mode === "web" ? "" : t("run.queued")) : stage ? t("stage." + stage, {}, stage) : "";
  // progress: the share of the typical work before the current stage
  const order = Object.keys(WEIGHTS).filter((s) => !NEVER.includes(s) && (s !== "quality" || trace.checked)
                                                && (s !== "load_engine" || trace.engine));
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
                         "desmos.js": "{stem}-desmos.js", "equations.tex": "{stem}.tex",
                         zip: "{stem}-line2func.zip" };
const STYLED_FILES = ["out.svg", "desmos.js"]; // = line2func.jobs.RESTYLED: written again in the page's line style

function setupResult() {
  $("#list-toggle").addEventListener("click", toggleList);
  for (const button of document.querySelectorAll("[data-file]")) button.addEventListener("click", () => download(button.dataset.file));
  $("#copy-all").addEventListener("click", copyAll);
  viewer.setLineColor(S.lineColor, S.colorSeed); // the selects start on whatever the viewer is drawing
  viewer.setLineWidth(S.lineWidth);
}

// The styled files the local server sends are re-styled to match the page, through the query it takes. The
// online engine's files are blob: URLs made when the trace finished, so no query reaches it; download() asks
// the engine itself for a styled file instead.
function colored(url, file) {
  if (!STYLED_FILES.includes(file) || S.mode === "web") return url;
  return `${url}?color=${encodeURIComponent(S.lineColor)}&seed=${S.colorSeed >>> 0}`
       + `&width=${encodeURIComponent(S.lineWidth)}`;
}

// A result's files by name ("zip": all of them): from the local server, or blob: URLs of the online engine.
function resultURL(snap) {
  if (S.mode === "web") return (file) => S.engine.fileURL(snap.job_id, file);
  const base = `api/jobs/${snap.job_id}/`;
  return (file) => base + (file === "zip" ? "zip" : "data/" + file);
}

async function loadResult(snap) {
  const mine = ++S.ticket;
  const url = resultURL(snap);
  S.result = null;
  setView("result");
  viewer.setLoading();
  try {
    const response = await fetch(url("curves.json"));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const doc = await response.json();
    if (mine !== S.ticket) return;
    const files = new Set(snap.files);
    viewer.load(doc, {
      original: previewURL(snap.image_id),
      quality: files.has("quality.png") ? url("quality.png") : null,
      qualityJson: files.has("quality.json") ? url("quality.json") : null,
    });
    S.result = { url, doc, snap, name: S.image?.name || doc.meta?.source || "line2func", desmos: null };
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
  fetch(result.url("desmos.txt")).then((r) => (r.ok ? r.text() : null)).then((text) => { result.desmos = text; }).catch(() => {});
}

function stemOf(name) {
  const dot = name.lastIndexOf(".");
  return (dot > 0 ? name.slice(0, dot) : name).trim() || "line2func";
}

async function download(file) {
  const result = S.result;
  if (!result) return;
  const name = DOWNLOAD_NAMES[file].replace("{stem}", stemOf(result.name));
  let url = colored(result.url(file), file);
  if (STYLED_FILES.includes(file) && S.mode === "web" && result.snap) {
    // online: the engine writes the file again in the style the page shows (no server to ask, nothing re-traced)
    try {
      url = (await S.engine.styledFile(result.snap.job_id, file,
                                       { color: S.lineColor, seed: S.colorSeed, width: S.lineWidth })) || url;
    } catch (err) {
      toast(errorText(err), "error"); // the unstyled file is still there to download
    }
  }
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
  // a blob: URL takes no query; a server URL may already carry the line color
  a.href = S.mode === "web" ? url : url + (url.includes("?") ? "&" : "?") + "download";
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
}

function copyAll() {
  const result = S.result;
  if (!result || result.desmos === null) return;
  const n = result.desmos.split("\n").filter((line) => line.trim()).length; // equations: curves, or functions
  if (n > DESMOS_LIMIT && !S.copyArmed) { S.copyArmed = true; toast(t("result.confirmCopy", { n, limit: DESMOS_LIMIT })); return; }
  S.copyArmed = false;
  navigator.clipboard.writeText(result.desmos).then(
    () => toast(t("result.copiedAll", { n })),
    () => toast(t("viewer.copyFailed"), "error"));
}

function renderBanner() {
  const banner = $("#banner");
  const summary = S.view === "result" ? S.result?.snap?.summary : null;
  const warnings = summary?.warnings || [];
  banner.hidden = !warnings.length;
  banner.innerHTML = "";
  for (const warning of warnings) {
    const row = document.createElement("div");
    row.className = "banner-row";
    const vars = { n: summary.equations ?? summary.curves, limit: DESMOS_LIMIT,
                   mp: (S.info?.limits?.quality_max_pixels || 4e6) / 1e6 };
    row.append(t("warning." + warning, vars, warning));
    banner.append(row);
  }
}

// ---------- keyboard ----------
function onKey(e) {
  if (e.isComposing || e.defaultPrevented) return;
  const mod = e.ctrlKey || e.metaKey;
  if (mod && !e.altKey && !e.shiftKey && (e.key === "o" || e.key === "O")) {
    e.preventDefault(); // never let the browser open a file in this window
    if (converts() && !S.gone) $("#file-input").click();
    return;
  }
  if (mod || e.altKey || isFormField(e.target)) return;
  if (S.trace && e.key === "Escape") { e.preventDefault(); cancelTrace(false); return; }
  if (S.view !== "empty" && !S.trace && viewer.handleKey(e)) e.preventDefault();
}
