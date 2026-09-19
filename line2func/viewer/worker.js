// The online page's tracing engine: a module Web Worker that runs line2func's Python with Pyodide
// (line2func/web.py). engine.js starts it and is the only one who talks to it:
//   page -> worker  {type: "init", pyodide, package, packages}   Pyodide's folder URL, line2func's zip, packages
//                   {type: "open", id, key, name, data}          read an image file (data: its bytes)
//                   {type: "trace", id, key, name, data, params} trace it (params: the job's JSON)
//   worker -> page  {type: "status", step} while starting; {type: "ready", protocol, version, heap} or
//                   {type: "failed", detail}; {type: "stage", id, stage} during a trace;
//                   {type: "answer", id, answer, preview | files + zip, heap}, or {type: "crashed", id, detail}
// It keeps no state of its own: every request brings the image, so a worker that is stopped (Cancel), that
// crashed or that is replaced (its memory only grows) loses nothing.
const PROTOCOL = 1; // = engine.js PROTOCOL
let py = null, web = null;

self.onmessage = ({ data: m }) => {
  if (m.type === "init") init(m);
  else if (m.type === "open") call(m.id, () => web.open_image(m.data, m.name, m.key));
  else if (m.type === "trace") {
    const progress = (stage) => self.postMessage({ type: "stage", id: m.id, stage }); // called from Python
    call(m.id, () => web.trace(m.data, m.name, m.params, progress, m.key));
  }
};

async function init({ pyodide, package: pkg, packages }) {
  try {
    const zip = fetch(pkg).then((r) => {
      if (!r.ok) throw new Error(`${pkg}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    zip.catch(() => {}); // reported where it is awaited
    self.postMessage({ type: "status", step: "runtime" });
    const { loadPyodide } = await import(pyodide + "pyodide.mjs");
    py = await loadPyodide({ indexURL: pyodide });
    self.postMessage({ type: "status", step: "packages" });
    await py.loadPackage(packages, { messageCallback: () => {} }); // checked against Pyodide's lock file
    self.postMessage({ type: "status", step: "line2func" });
    py.unpackArchive(await zip, "zip", { extractDir: py.runPython("import site; site.getsitepackages()[0]") });
    py.runPython("import importlib; importlib.invalidate_caches(); import line2func.web, line2func.quality");
    web = py.pyimport("line2func.web");
    self.postMessage({ type: "ready", protocol: PROTOCOL, version: web.VERSION, heap: heap() });
  } catch (err) {
    self.postMessage({ type: "failed", detail: describe(err) });
  }
}

function call(id, run) {
  let answer;
  try {
    answer = run(); // line2func.web answers Python's own errors; what is thrown here is Pyodide failing
  } catch (err) {
    self.postMessage({ type: "crashed", id, detail: describe(err) });
    return;
  }
  try {
    const out = answer.toJs({ dict_converter: Object.fromEntries }); // bytes -> Uint8Array copies
    const buffers = new Set([out.preview, out.zip, ...Object.values(out.files || {})].filter(Boolean).map((a) => a.buffer));
    self.postMessage({ type: "answer", id, ...out, heap: heap() }, [...buffers]);
  } catch (err) {
    self.postMessage({ type: "crashed", id, detail: describe(err) });
  } finally {
    answer.destroy();
  }
}

function heap() {
  try { return py._module.HEAPU8.byteLength; } catch { return null; } // WebAssembly memory (not public API)
}

function describe(err) {
  return String(err?.message ?? err).slice(0, 500);
}
