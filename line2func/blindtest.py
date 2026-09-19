"""Blind comparison of the two engines on real drawings (the third enabling condition, README Evaluation).

    python -m line2func.eval --ckpt runs/m3/best.pt --fullset data/full_v1   # builds the kit
    # raters open runs/m3/blindtest/index.html, vote, and click "Export votes"
    python -m line2func.blindtest score runs/m3/blindtest votes_alice.json votes_bob.json

For every drawing the kit shows the original and the two engines' traced
curves side by side, as "A" and "B" in a random order that is stored only in
``key.json``, which raters must not open. Raters pick A, B or "tie". Scoring
unblinds the votes: the model passes when it is rated better or tied in at
least 60% of the votes.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

from line2func import baseline, lineart
from line2func.render import render_overlay

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PASS_SHARE = 0.60


def _png_data_uri(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_kit(folder: str | Path, out: str | Path, model_vectorize, seed: int = 0) -> int:
    """Trace every drawing in ``folder`` with both engines and write the kit to ``out``."""
    folder, out = Path(folder), Path(out)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES) if folder.is_dir() else []
    if not images:
        raise FileNotFoundError(f"no drawings ({', '.join(IMAGE_TYPES)}) in {folder}")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items, key = [], {}
    for i, path in enumerate(images):
        rgb = lineart.load_rgb(path)
        ink = lineart.extract(rgb, "none")
        traced = {"baseline": baseline.vectorize(ink), "model": model_vectorize(ink)}
        # the curves alone on white, so raters judge the tracing, not the photo
        blank = np.full_like(rgb, 255)
        views = {k: _png_data_uri(render_overlay(blank, v, line_width=1.2, fade=0.0, colors=[(0, 0, 0)]))
                 for k, v in traced.items()}
        order = ["baseline", "model"]
        rng.shuffle(order)
        item_id = f"d{i:03d}"
        key[item_id] = {"A": order[0], "B": order[1], "file": path.name}
        items.append({"id": item_id, "original": _png_data_uri(rgb), "A": views[order[0]], "B": views[order[1]]})
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8", newline="\n")
    html = _PAGE.replace("__ITEMS__", json.dumps(items))
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    return len(items)


def score(kit: str | Path, vote_files: list[str | Path]) -> dict:
    """Unblind votes: share of votes where the model was rated better or tied."""
    key = json.loads((Path(kit) / "key.json").read_text(encoding="utf-8"))
    counts = {"model": 0, "baseline": 0, "tie": 0}
    for vf in vote_files:
        votes = json.loads(Path(vf).read_text(encoding="utf-8"))["votes"]
        for item_id, choice in votes.items():
            if item_id not in key or choice not in ("A", "B", "tie"):
                continue
            counts["tie" if choice == "tie" else key[item_id][choice]] += 1
    total = sum(counts.values())
    share = (counts["model"] + counts["tie"]) / total if total else float("nan")
    return {"votes": total, **counts, "better_or_tied": share, "pass": bool(total) and share >= PASS_SHARE}


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>line2func blind test</title>
<style>
 :root { --bg:#f5f5f3; --panel:#fff; --text:#1d1d1f; --muted:#6b6b70; --border:#e2e2de; --accent:#d9480f; }
 @media (prefers-color-scheme: dark) { :root { --bg:#151517; --panel:#1e1e21; --text:#ececee; --muted:#9a9aa2;
   --border:#2f2f34; --accent:#ff7a3d; } }
 body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui, sans-serif; }
 header { padding:12px 16px; background:var(--panel); border-bottom:1px solid var(--border); display:flex;
   gap:16px; align-items:center; flex-wrap:wrap; }
 main { padding:16px; max-width:1400px; margin:0 auto; }
 .grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
 figure { margin:0; background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px; }
 figcaption { font-weight:600; margin-bottom:6px; }
 img { width:100%; height:auto; image-rendering:pixelated; background:#fff; border-radius:4px; }
 .votes { display:flex; gap:8px; margin:16px 0; justify-content:center; }
 button { font:inherit; padding:8px 18px; border-radius:6px; border:1px solid var(--border);
   background:var(--panel); color:var(--text); cursor:pointer; }
 button.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
 .muted { color:var(--muted); }
 @media (max-width: 760px) { .grid { grid-template-columns:1fr; } }
</style></head><body>
<header><strong>Blind test</strong><span class="muted" id="progress"></span>
 <span style="flex:1"></span><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
 <button id="export">Export votes</button></header>
<main>
 <p class="muted">Which tracing reproduces the drawing better? Judge completeness, accuracy and clean strokes.
 Keys: 1 = A, 2 = B, 3 = tie, arrows = navigate.</p>
 <div class="grid">
  <figure><figcaption>Original</figcaption><img id="orig" alt="original drawing"></figure>
  <figure><figcaption>A</figcaption><img id="a" alt="tracing A"></figure>
  <figure><figcaption>B</figcaption><img id="b" alt="tracing B"></figure>
 </div>
 <div class="votes"><button data-v="A">A is better</button><button data-v="tie">Tie</button>
  <button data-v="B">B is better</button></div>
</main>
<script>
const ITEMS = __ITEMS__;
const STORE = "line2func-blindtest";
let votes = {}; try { votes = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
let i = 0;
function save() { try { localStorage.setItem(STORE, JSON.stringify(votes)); } catch (e) {} }
function show() {
  const it = ITEMS[i];
  document.getElementById("orig").src = it.original;
  document.getElementById("a").src = it.A;
  document.getElementById("b").src = it.B;
  document.querySelectorAll("[data-v]").forEach(b => b.classList.toggle("on", votes[it.id] === b.dataset.v));
  document.getElementById("progress").textContent =
    `drawing ${i + 1} / ${ITEMS.length} · ${Object.keys(votes).length} voted`;
}
function vote(v) { votes[ITEMS[i].id] = v; save(); if (i < ITEMS.length - 1) i++; show(); }
document.querySelectorAll("[data-v]").forEach(b => b.onclick = () => vote(b.dataset.v));
document.getElementById("prev").onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById("next").onclick = () => { i = Math.min(ITEMS.length - 1, i + 1); show(); };
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify({ votes }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "votes.json"; a.click();
};
addEventListener("keydown", e => {
  if (e.key === "1") vote("A"); else if (e.key === "2") vote("B"); else if (e.key === "3") vote("tie");
  else if (e.key === "ArrowRight") document.getElementById("next").click();
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
});
show();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.blindtest", description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="build a kit (same as eval --fullset)")
    m.add_argument("folder", type=Path)
    m.add_argument("--ckpt", type=Path, required=True)
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--device", default=None)
    s = sub.add_parser("score", help="unblind and score exported votes")
    s.add_argument("kit", type=Path)
    s.add_argument("votes", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "make":
        from line2func.model.infer import load_vectorizer

        out = args.out or args.ckpt.parent / "blindtest"
        n = make_kit(args.folder, out, load_vectorizer(str(args.ckpt), args.device))
        print(f"blind test kit: {n} drawings -> {out / 'index.html'} (do not show raters key.json)")
        return 0
    r = score(args.kit, args.votes)
    print(f"{r['votes']} votes: model better {r['model']}, baseline better {r['baseline']}, tie {r['tie']}")
    print(f"model better or tied: {r['better_or_tied']:.1%} (need >= {PASS_SHARE:.0%}) -> "
          f"{'PASS' if r['pass'] else 'FAIL'}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
