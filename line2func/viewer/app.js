// line2func app: one page, the curve viewer. Without an image it shows a drop zone; a dropped (chosen or
// pasted) line drawing is shown in place, and Convert traces it into functions or parametric equations like
// the command line does (up to 5,000 curves, with a quality check). Served by `python -m line2func`. When
// `python -m line2func.serve out/` serves it, the same page is the plain result viewer ("static" mode:
// /api/info answers {mode: "static"} or nothing).
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
  spacer: $("#spacer"), detail: $("#detail"), stats: $("#stats"), fit: $("#fit"),
  originalOnly: $("#original-only"), viewMode: $("#view-mode"), bgAlpha: $("#bg-alpha"),
});

const FORMS = ["function", "parametric"]; // how the lines are written (line2func.export)
const DENOISE = 50; // = line2func.pipeline.DENOISE, the noise filters' default strength (tests check it)
// the server's stages in the order they run (pipeline.trace, app.run_job) -> the step shown, and typical cost
const STEP_OF = { resize: "prepare", load_model: "prepare", lineart: "prepare", upscale: "prepare",
                  vectorize: "trace", refine: "trace", measure: "finish", outline: "finish", residual: "finish",
                  fill: "finish", count: "finish", optimize: "finish", shapes: "finish", export: "finish", quality: "quality" };
const WEIGHTS = { resize: 1, load_model: 6, lineart: 6, upscale: 3, vectorize: 45, refine: 20, measure: 8,
                  outline: 5, residual: 8, fill: 3, count: 10, optimize: 10, shapes: 6, export: 3, quality: 14 };
const NEVER = ["load_model", "optimize"]; // stages the page's conversions do not run

const S = {
  mode: "boot", info: null, gone: false,
  view: "empty", // empty (the drop zone), preview (an image to convert) or result
  image: null, form: "function", denoise: DENOISE, denoiseOn: true, jobs: new Map(),
  trace: null, result: null, ticket: 0, copyArmed: false,
};

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
  renderConvert();
  if (S.trace) renderRunning();
  renderBanner();
}

// ---------- static mode: the plain result viewer ----------
async function startStatic(info) {
  S.mode = "static";
  S.view = "result";
  document.documentElement.dataset.mode = "static";
  let saved = null;
  try { saved = localStorage.getItem("line2func.lang"); } catch { /* storage may be blocked */ }
  initLang(saved);
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
  viewer.setDesmosWarning(false); // the result banner says it
  const saved = info.settings?.options || {};
  if (FORMS.includes(saved.form)) S.form = saved.form;
  if (typeof saved.denoise === "number" && saved.denoise >= 0 && saved.denoise <= 100) S.denoise = saved.denoise;
  if (typeof saved.denoise_on === "boolean") S.denoiseOn = saved.denoise_on;
  initLang(info.settings?.lang);
  setupApp();
  setupResult();
  $("#quit").addEventListener("click", quit);
  connectEvents();
  if (!(await restoreSession())) showEmpty();
  window.__l2f = {
    state: () => S.view, mode: () => S.mode, debug: () => viewer.debug(), info: () => S.info,
    image: () => S.image, result: () => S.result && { job: S.result.snap, name: S.result.name },
    form: () => S.form, openFile,
  };
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
  $("#convert").addEventListener("click", startTrace);
  $("#clear").addEventListener("click", clearImage);
  $("#cancel").addEventListener("click", () => cancelTrace(false));
}

function saveOptions() {
  api("POST", "api/settings", { options: { form: S.form, denoise: S.denoise, denoise_on: S.denoiseOn } }).catch(() => {});
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
    S.image = info;
    showPreview();
  } catch (err) {
    if (mine === uploads) toast(errorText(err), "error");
  }
}

// ---------- the page's three states: the drop zone, an image to convert, a result ----------
function setView(view) {
  S.view = view;
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
  // fitted above the convert bar (it floats 16 px above the bottom)
  viewer.preview(`api/images/${S.image.image_id}/preview`, S.image.width, S.image.height, $("#convert-bar").offsetHeight + 24);
  $("#convert").focus();
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
  el.hidden = !(S.mode === "app" && img);
  el.textContent = img ? `${img.name} · ${img.width} × ${img.height}` : "";
}

function renderDenoise() {
  $("#denoise-on").checked = S.denoiseOn;
  const slider = $("#denoise");
  slider.value = String(S.denoise);
  slider.disabled = !S.denoiseOn;
  $("#denoise-value").textContent = S.denoiseOn ? t("convert.strength", { n: String(S.denoise) }) : t("convert.off");
}

function renderConvert() {
  for (const radio of document.querySelectorAll("input[name=form]")) radio.checked = radio.value === S.form;
  renderDenoise();
  let note = t("convert.note", { n: new Intl.NumberFormat().format(S.info.desmos_limit) });
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
                   curves: S.info.desmos_limit, quality: true, denoise: S.denoiseOn ? S.denoise : 0 };
  const checked = img.width * img.height * img.auto_scale ** 2 <= S.info.limits.quality_max_pixels;
  S.trace = { jobId: null, snap: null, params, checked, started: performance.now(), group: null };
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

function onJob(snap) {
  if (!snap?.job_id) return;
  S.jobs.set(snap.job_id, snap);
  if (S.trace && S.trace.jobId === snap.job_id) onTraceUpdate(snap);
}

function onTraceUpdate(snap) {
  const trace = S.trace;
  trace.snap = snap;
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
  $("#run-stage").textContent = !snap || snap.state === "queued" ? t("run.queued") : stage ? t("stage." + stage, {}, stage) : "";
  // progress: the share of the typical work before the current stage
  const order = Object.keys(WEIGHTS).filter((s) => !NEVER.includes(s) && (s !== "quality" || trace.checked));
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
}

async function loadResult(snap) {
  const mine = ++S.ticket;
  const base = `api/jobs/${snap.job_id}/data/`;
  S.result = null;
  setView("result");
  viewer.setLoading();
  try {
    const response = await fetch(base + "curves.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const doc = await response.json();
    if (mine !== S.ticket) return;
    const files = new Set(snap.files);
    viewer.load(doc, {
      original: `api/images/${snap.image_id}/preview`,
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
    if (S.mode === "app" && !S.gone) $("#file-input").click();
    return;
  }
  if (mod || e.altKey || isFormField(e.target)) return;
  if (S.trace && e.key === "Escape") { e.preventDefault(); cancelTrace(false); return; }
  if (S.view !== "empty" && !S.trace && viewer.handleKey(e)) e.preventDefault();
}
