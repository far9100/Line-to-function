// line2func's online engine (line2func.web) in Pyodide under Node.js, set up like the page's worker
// (line2func/viewer/worker.js). For tests/test_pyodide.py:
//   node tests/pyodide/run.mjs <line2func zip> <image> [job params as JSON]
// prints {pyodide, webp, stages, answer, files: {name: sha256[:16]}, styled, standalone, heap} as JSON.
import { loadPyodide } from "pyodide";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

const PAGE = { kind: "trace", method: "none", scale: "auto", form: "function", curves: 5000, quality: true, denoise: 50,
               faint_sensitivity: 50 };
const [zipPath, imagePath, params = JSON.stringify(PAGE)] = process.argv.slice(2);
const py = await loadPyodide();
await py.loadPackage(["numpy", "scipy", "pillow"], { messageCallback: () => {} });
const site = py.runPython("import site; site.getsitepackages()[0]");
py.unpackArchive(new Uint8Array(await readFile(zipPath)), "zip", { extractDir: site });
py.runPython("import importlib; importlib.invalidate_caches(); import line2func.web");
const web = py.pyimport("line2func.web");
const stages = [];
const answer = web.trace(new Uint8Array(await readFile(imagePath)), path.basename(imagePath), params, (s) => stages.push(s), "k");
const out = answer.toJs({ dict_converter: Object.fromEntries });
answer.destroy();
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex").slice(0, 16);

// what the page asks for when a download must match the line style it is showing (viewer/engine.js styledSVG)
function restyle(color, seed, width) {
  const call = web.export_svg(out.files["curves.json"], color, seed, width);
  const got = call.toJs({ dict_converter: Object.fromEntries });
  call.destroy();
  const answer = JSON.parse(got.answer);
  if (!got.svg) return { answer };
  const text = new TextDecoder().decode(got.svg);
  const uniq = (re) => [...new Set(text.match(re) || [])].length;
  return { answer, widths: uniq(/stroke-width="[^"]+"/g), colors: uniq(/ stroke="[^"]+"/g), sha: sha(got.svg) };
}
console.log(JSON.stringify({
  pyodide: py.version,
  webp: py.runPython("from PIL import features; features.check('webp')"),
  stages,
  answer: JSON.parse(out.answer),
  files: Object.fromEntries(Object.entries(out.files).map(([name, bytes]) => [name, sha(bytes)])),
  styled: { bwUniform: restyle("bw", 0, "uniform"), randomMeasured: restyle("random", 7, "measured"),
            bad: restyle("nope", 0, "uniform") },
  standalone: out.zip ? out.zip.buffer.byteLength === out.zip.byteLength : null, // a copy: can be transferred
  heap: py._module.HEAPU8.byteLength,
}));
