// The online page's engine: line2func's Python running in the browser (worker.js, Pyodide). It gives app.js
// what the local server gives it: image info for an opened file, and job snapshots ({job_id, state, stage,
// summary, files, ...}, as line2func/app.py sends them) while a trace runs, so the page shows progress,
// results and downloads the same way. Files are blob: URLs.
//
// One request runs at a time. Python cannot be interrupted, so Cancel stops the worker and starts a new one;
// so does a crash, and a worker whose memory has grown large (WebAssembly memory never shrinks). Every request
// carries the image's bytes, so a new worker loses nothing.
export const PROTOCOL = 2; // = worker.js PROTOCOL
const RECYCLE_BYTES = 1 << 30; // after a trace, a worker with more memory than this is replaced
const KEEP_RESULTS = 2; // results whose files stay available
const TYPES = { "curves.json": "application/json", "out.svg": "image/svg+xml", "desmos.txt": "text/plain;charset=utf-8",
                "desmos.js": "text/javascript;charset=utf-8", "equations.tex": "text/plain;charset=utf-8",
                "quality.json": "application/json", "quality.png": "image/png",
                "lineart.png": "image/png", zip: "application/zip" };

// config: api/info's "engine" ({pyodide, package, packages, build}). onJob(snapshot) is called on every change
// of a job, onStatus(status) on every change of the engine. Errors are like app.js's: an Error with .code
// (the page shows t("error." + code)) and .info ({detail}).
export function createEngine(config, { onJob = () => {}, onStatus = () => {}, Worker = globalThis.Worker } = {}) {
  let worker = null, state = "idle", step = null, failure = null, version = null; // idle, loading, ready, failed
  let current = null; // the request the worker is working on
  const queue = []; // requests waiting for it
  const images = new Map(); // key -> {data, name, preview}: the image opened last
  const results = new Map(); // job id -> {file name: blob URL}
  const styled = new Map(); // job id -> {"color|seed|width": blob URL}: out.svg written again in that style
  const jobs = new Map(); // job id -> job
  let seq = 0;

  function error(code, info = {}) {
    const err = new Error(code);
    err.code = code;
    err.info = info;
    return err;
  }

  function status() {
    return { state, step, version, code: failure?.code ?? null, detail: failure?.detail ?? null };
  }

  // ---------- the worker ----------
  function start() {
    if (worker) return;
    state = "loading"; step = null; failure = null;
    worker = new Worker(new URL(`./worker.js?v=${encodeURIComponent(config.build || "")}`, import.meta.url), { type: "module" });
    worker.onmessage = receive;
    worker.onerror = (e) => {
      e.preventDefault?.();
      if (state === "ready") crashed(e.message || "worker error");
      else fail("engine_failed", e.message || "the engine's script could not be loaded");
    };
    worker.postMessage({ type: "init", pyodide: config.pyodide, packages: config.packages,
                         package: new URL(config.package, globalThis.document?.baseURI ?? import.meta.url).href });
    onStatus(status());
  }

  function restart() {
    if (worker) worker.terminate();
    worker = null;
    start();
  }

  function fail(code, detail) {
    if (worker) worker.terminate();
    worker = null;
    state = "failed";
    failure = { code, detail };
    onStatus(status());
    const pending = [current, ...queue.splice(0)].filter(Boolean);
    current = null;
    for (const r of pending) r.reject(error(code, { detail }));
  }

  function crashed(detail) {
    const r = current;
    current = null;
    restart();
    if (r) r.reject(error("engine_crashed", { detail }));
  }

  function receive({ data: m }) {
    if (m.type === "status") { step = m.step; onStatus(status()); return; }
    if (m.type === "failed") { fail("engine_failed", m.detail); return; }
    if (m.type === "ready") {
      if (m.protocol !== PROTOCOL) { fail("bad_token", "old engine files"); return; } // the page is out of date
      state = "ready"; version = m.version;
      onStatus(status());
      pump();
      return;
    }
    if (!current || m.id !== current.id) return; // for a request that was cancelled
    if (m.type === "stage") { current.onStage?.(m.stage); return; }
    if (m.type === "crashed") { crashed(m.detail); return; }
    if (m.type === "answer") {
      const r = current;
      current = null;
      if (r.message.type === "trace" && !(m.heap <= RECYCLE_BYTES)) restart(); // its memory back, while idle
      r.resolve(m);
      pump();
    }
  }

  function request(message, job = null, onStage = null) {
    return new Promise((resolve, reject) => {
      queue.push({ message, job, onStage, resolve, reject });
      if (state === "idle" || state === "failed") restart(); // not started yet, or try again
      else pump();
    });
  }

  function pump() {
    if (current || state !== "ready" || !queue.length) return;
    current = queue.shift();
    current.id = ++seq;
    worker.postMessage({ ...current.message, id: current.id });
  }

  // ---------- images ----------
  // Read an image file (a File or Blob); resolves with its info, as the server's POST /api/images answers.
  async function open(file, name) {
    const data = new Uint8Array(await file.arrayBuffer());
    const key = `image-${++seq}`;
    const m = await request({ type: "open", key, name, data });
    const answer = JSON.parse(m.answer);
    if (answer.error) throw error(answer.error.code, answer.error);
    for (const old of images.values()) URL.revokeObjectURL(old.preview);
    images.clear();
    images.set(key, { data, name, preview: URL.createObjectURL(new Blob([m.preview], { type: answer.preview_type })) });
    return answer.image;
  }

  function previewURL(imageId) {
    return images.get(imageId)?.preview || "";
  }

  // ---------- jobs ----------
  function snapshot(job) {
    const end = job.finished ?? performance.now();
    return { job_id: job.id, kind: "trace", image_id: job.imageId, state: job.state, stage: job.stage,
             elapsed: Math.round(end - job.started) / 1000, error: job.error, summary: job.summary,
             files: job.files, params: job.params };
  }

  function finish(job, state, fields = {}) {
    if (job.state !== "running") return;
    Object.assign(job, fields, { state, finished: performance.now() });
    if (state === "done") job.stage = null;
    onJob(snapshot(job));
  }

  // Start a trace job (params as app.js sends them to the server); resolves with its first snapshot, and
  // onJob reports the rest. It is "running" from the start (stage "load_engine" until the engine is ready).
  function trace(params) {
    const image = images.get(params.image_id);
    if (!image) return Promise.reject(error("unknown_image"));
    const job = { id: `job-${++seq}`, imageId: params.image_id, state: "running",
                  stage: state === "ready" ? null : "load_engine", error: null, summary: null, files: [], params,
                  started: performance.now(), finished: null };
    jobs.set(job.id, job);
    const onStage = (stage) => {
      if (job.state !== "running") return;
      job.stage = stage;
      onJob(snapshot(job));
    };
    const message = { type: "trace", key: params.image_id, name: image.name, data: image.data, params: JSON.stringify(params) };
    request(message, job, onStage).then((m) => {
      const answer = JSON.parse(m.answer);
      if (answer.error) {
        finish(job, "error", { error: answer.error });
        if (answer.error.code === "out_of_memory") restart();
        return;
      }
      const urls = {};
      for (const [name, bytes] of Object.entries(m.files)) urls[name] = URL.createObjectURL(new Blob([bytes], { type: TYPES[name] }));
      if (m.zip) urls.zip = URL.createObjectURL(new Blob([m.zip], { type: TYPES.zip }));
      results.set(job.id, urls);
      for (const id of [...results.keys()].slice(0, -KEEP_RESULTS)) forget(id);
      finish(job, "done", { summary: answer.summary, params: answer.params, files: Object.keys(m.files).sort() });
    }, (err) => {
      if (err.code !== "cancelled") finish(job, "error", { error: { code: err.code, detail: err.info?.detail ?? null } });
    });
    return Promise.resolve(snapshot(job));
  }

  function cancel(jobId) {
    const job = jobs.get(jobId);
    if (!job || job.state !== "running") return;
    if (current?.job === job) {
      const r = current;
      current = null;
      restart(); // Python cannot be interrupted: stop the worker, start a new one
      r.reject(error("cancelled"));
    } else {
      const waiting = queue.findIndex((r) => r.job === job);
      if (waiting >= 0) queue.splice(waiting, 1)[0].reject(error("cancelled"));
    }
    finish(job, "cancelled");
  }

  function fileURL(jobId, name) {
    return results.get(jobId)?.[name] || "";
  }

  // out.svg or desmos.js written again with the line style the page is showing, as the local server's
  // ?color=&seed=&width= does. The curves come from the result's own curves.json, so nothing is traced
  // again. Resolves with a blob: URL (kept until the result is forgotten), or "" if it cannot be made.
  async function styledFile(jobId, name, { color = "measured", seed = 0, width = "measured" } = {}) {
    const key = `${name}|${color}|${seed >>> 0}|${width}`;
    const have = styled.get(jobId)?.[key];
    if (have) return have;
    const source = results.get(jobId)?.["curves.json"];
    if (!source) return "";
    const curves = new Uint8Array(await (await fetch(source)).arrayBuffer());
    const m = await request({ type: "svg", curves, name, color, seed: seed >>> 0, width });
    const answer = JSON.parse(m.answer);
    if (answer.error) throw error(answer.error.code, answer.error);
    const url = URL.createObjectURL(new Blob([m.svg], { type: TYPES[name] }));
    if (!styled.has(jobId)) styled.set(jobId, {});
    styled.get(jobId)[key] = url;
    return url;
  }

  function forget(jobId) {
    for (const url of Object.values(results.get(jobId) || {})) URL.revokeObjectURL(url);
    for (const url of Object.values(styled.get(jobId) || {})) URL.revokeObjectURL(url);
    results.delete(jobId);
    styled.delete(jobId);
    jobs.delete(jobId);
  }

  return { start, status, open, trace, cancel, previewURL, fileURL, styledFile };
}
