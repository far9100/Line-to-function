// Drives viewer/engine.js with a fake worker and prints what happened as JSON (tests/test_viewer_engine.py).
//   node tests/engine_check.mjs <file URL of engine.js>
const { createEngine, PROTOCOL } = await import(process.argv[2]);

class FakeWorker {
  static created = [];
  static behaviour = { protocol: PROTOCOL, trace: "answer", heap: 100 };

  constructor(url, options) {
    this.url = String(url);
    this.options = options;
    this.terminated = false;
    this.received = [];
    FakeWorker.created.push(this);
  }

  postMessage(m) {
    this.received.push(m.type);
    setTimeout(() => this.handle(m));
  }

  terminate() {
    this.terminated = true;
  }

  reply(m) {
    if (!this.terminated) this.onmessage?.({ data: m });
  }

  handle(m) {
    const b = FakeWorker.behaviour;
    if (m.type === "init") {
      this.init = m;
      this.reply({ type: "status", step: "runtime" });
      this.reply({ type: "ready", protocol: b.protocol, version: "test", heap: 1 });
    } else if (m.type === "open") {
      const image = { image_id: m.key, name: m.name, width: 10, height: 8, auto_scale: 1 };
      this.reply({ type: "answer", id: m.id, answer: JSON.stringify({ image, preview_type: "image/png" }),
                   preview: new Uint8Array([1, 2]), heap: 1 });
    } else if (m.type === "trace" && b.trace === "crash") {
      this.reply({ type: "crashed", id: m.id, detail: "boom" });
    } else if (m.type === "trace" && b.trace === "answer") {
      this.reply({ type: "stage", id: m.id, stage: "resize" });
      this.reply({ type: "stage", id: m.id, stage: "vectorize" });
      this.reply({ type: "answer", id: m.id, heap: b.heap, zip: new Uint8Array([80, 75]),
                   answer: JSON.stringify({ summary: { curves: 2 }, params: { method: "none", scale: 1 } }),
                   files: { "curves.json": new TextEncoder().encode('{"curves": []}'), "desmos.txt": new Uint8Array([120]) } });
    } else if (m.type === "svg") {
      const svg = `<svg data-style="${m.name}|${m.color}|${m.seed}|${m.width}"/>`;
      this.reply({ type: "answer", id: m.id, heap: 1, answer: JSON.stringify({ ok: true }),
                   svg: new TextEncoder().encode(svg) });
    } // "hang": never answers
  }
}

const settle = () => new Promise((r) => setTimeout(r, 20));
const config = { pyodide: "https://cdn.example/pyodide/", package: "py/line2func.zip", packages: ["numpy"], build: "b1" };
const out = {};

// a trace from start to finish; the second one waits for a new worker (the first one used too much memory)
{
  const snaps = [], statuses = [];
  let engine = null, image = null, second = null;
  const onJob = (s) => {
    snaps.push(s);
    if (s.state === "done" && !second) { // at once: the worker that used too much memory is being replaced
      FakeWorker.behaviour.heap = 100;
      second = engine.trace({ image_id: image.image_id, kind: "trace", method: "none" });
    }
  };
  engine = createEngine(config, { onJob, onStatus: (s) => statuses.push(s.state), Worker: FakeWorker });
  engine.start();
  await settle();
  image = await engine.open(new Blob([new Uint8Array([9, 9, 9])]), "a.png");
  FakeWorker.behaviour.heap = 2 ** 31;
  const first = await engine.trace({ image_id: image.image_id, kind: "trace", method: "none" });
  await settle();
  second = await second;
  await settle();
  const done = snaps.filter((s) => s.state === "done");
  out.normal = {
    worker: FakeWorker.created[0].url, init: FakeWorker.created[0].init, statuses, image,
    preview: engine.previewURL(image.image_id).startsWith("blob:"),
    first: { state: first.state, stage: first.stage }, second: { state: second.state, stage: second.stage },
    stages: snaps.filter((s) => s.job_id === first.job_id).map((s) => s.stage),
    done: done.map((s) => ({ files: s.files, summary: s.summary, params: s.params })),
    urls: ["curves.json", "desmos.txt", "zip", "quality.png"].map((f) => engine.fileURL(first.job_id, f).startsWith("blob:")),
    workers: FakeWorker.created.length, terminated: FakeWorker.created[0].terminated,
    curves: await (await fetch(engine.fileURL(first.job_id, "curves.json"))).text(),
  };
}

// cancel: the worker is stopped and a new one started
{
  FakeWorker.created = [];
  FakeWorker.behaviour.trace = "hang";
  const snaps = [];
  const engine = createEngine(config, { onJob: (s) => snaps.push(s), Worker: FakeWorker });
  const image = await engine.open(new Blob([new Uint8Array([1])]), "b.png"); // starts the engine by itself
  const job = await engine.trace({ image_id: image.image_id });
  await settle();
  engine.cancel(job.job_id);
  await settle();
  out.cancel = { states: snaps.map((s) => s.state), workers: FakeWorker.created.length,
                 terminated: FakeWorker.created[0].terminated, status: engine.status().state };
}

// a crash: the job fails with engine_crashed, and a new worker is started
{
  FakeWorker.created = [];
  FakeWorker.behaviour.trace = "crash";
  const snaps = [];
  const engine = createEngine(config, { onJob: (s) => snaps.push(s), Worker: FakeWorker });
  const image = await engine.open(new Blob([new Uint8Array([1])]), "c.png");
  await engine.trace({ image_id: image.image_id });
  await settle();
  out.crash = { last: snaps.at(-1), workers: FakeWorker.created.length, unknown: await engine.trace({ image_id: "nope" }).catch((e) => e.code) };
}

// styledFile: out.svg / desmos.js written again in the page's line style, cached per file and style
// and dropped with the result
{
  FakeWorker.created = [];
  FakeWorker.behaviour.trace = "answer";
  FakeWorker.behaviour.heap = 100;
  const engine = createEngine(config, { Worker: FakeWorker });
  const image = await engine.open(new Blob([new Uint8Array([1])]), "e.png");
  const job = await engine.trace({ image_id: image.image_id });
  await settle();
  const worker = FakeWorker.created[FakeWorker.created.length - 1];
  const style = { color: "bw", seed: 0, width: "uniform" };
  const first = await engine.styledFile(job.job_id, "out.svg", style);
  const again = await engine.styledFile(job.job_id, "out.svg", style); // the same style must not ask twice
  const other = await engine.styledFile(job.job_id, "out.svg", { color: "random", seed: 7, width: "measured" });
  // the same style of another file is a different answer, so the cache must not hand back the SVG
  const js = await engine.styledFile(job.job_id, "desmos.js", style);
  out.styled = {
    isBlob: first.startsWith("blob:"), cached: first === again, differs: first !== other,
    asked: worker.received.filter((t) => t === "svg").length,
    body: await (await fetch(first)).text(),
    otherBody: await (await fetch(other)).text(),
    jsBody: await (await fetch(js)).text(),
    unknownJob: await engine.styledFile("nope", "out.svg", style),
  };
}

// old engine files: the engine fails with "bad_token" (the page is out of date)
{
  FakeWorker.created = [];
  FakeWorker.behaviour.protocol = PROTOCOL + 1;
  const engine = createEngine(config, { Worker: FakeWorker });
  const failed = await engine.open(new Blob([new Uint8Array([1])]), "d.png").catch((e) => e.code);
  out.protocol = { failed, status: engine.status() };
}

console.log(JSON.stringify(out));
process.exit(0);
