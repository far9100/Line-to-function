# line2func: the full manual

The [README](../README.md) shows how to install and use line2func; this page has everything else.<br>
[README](../README.md) 說明怎麼安裝和使用 line2func；其他內容都在這一頁。

[English](#english) · [繁體中文](#繁體中文)

---

## English

### Contents

1. [Install](#1-install)
2. [Trace your first drawing](#2-trace-your-first-drawing)
3. [What you get](#3-what-you-get)
4. [The viewer](#4-the-viewer)
5. [Using the equations in Desmos](#5-using-the-equations-in-desmos)
6. [Photos and color images](#6-photos-and-color-images)
7. [All `demo` options](#7-all-demo-options)
8. [Tips for good results](#8-tips-for-good-results)
9. [The neural engine (optional)](#9-the-neural-engine-optional)
10. [Evaluation and the blind test](#10-evaluation-and-the-blind-test)
11. [Using line2func from Python](#11-using-line2func-from-python)
12. [Project layout and tests](#12-project-layout-and-tests)
13. [Status and roadmap](#13-status-and-roadmap)

### 1. Install

Requires **Python 3.10 or newer**.

#### Basic install (CPU only, baseline engine)

```bash
git clone https://github.com/far9100/Line-to-function.git line2func
cd line2func
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
```

This installs `numpy`, `pillow` and `scipy`, which is all you need to trace line
art, export, and view the results.

#### Optional: PyTorch (pretrained photo line art, neural engine, training)

Install a PyTorch build that matches your GPU **first**, then the extras:

```bash
# NVIDIA RTX 50-series (Blackwell) needs a CUDA 12.8+ build, e.g. CUDA 13.0:
pip install torch --index-url https://download.pytorch.org/whl/cu130
# (CPU only: pip install torch)

pip install -e ".[train,dev]"     # adds pyyaml (configs) and pytest
```

Verified setup: torch 2.14.0+cu130, Python 3.14, RTX 5070 (driver 596.21),
Windows 11.

### 2. Trace your first drawing

#### In the browser (drag and drop)

```bash
python -m line2func          # or just: line2func   (after pip install -e .)
```

The page opens in a new tab of your default browser. Then:

1. **Drop a line drawing** onto the page. You can also click **Choose a
   file…** or paste with Ctrl+V. PNG, JPEG, WebP, BMP, TIFF and GIF are
   accepted, up to 64 MB. Photos from phones are turned upright.
2. The drawing appears in place. Choose how to write the lines, as
   **functions** `y = f(x)`, `x = g(y)` (section 5) or as **parametric
   equations** `x(t), y(t)`, and how strongly noise is removed (**Remove
   noise**, strength 0-100, 50 by default: lower keeps more detail, higher cleans
   noisy scans; section 8) and how light a line may be (**Faint lines**, 0-100,
   50 by default: higher keeps lighter strands of hair and background), and
   click **Convert**. It is traced like `demo`
   traces it: up to 5,000 curves, with the quality check (images larger than
   2048 px are shrunk first). A progress panel shows each step, and **Cancel**
   stops at the next step.
3. The result opens in the same page, the viewer (section 4). From there you
   can download SVG, JSON, Desmos, LaTeX or a ZIP of everything, or
   **Copy all for Desmos**.
4. **Clear image** at the top removes the image, so the next one can be
   dropped. You can also drop another image at any time.

The web page traces every image as line art. For photos, use `demo` with
`--lineart` (section 6); the command line also has all the other options
(section 7).

The **中文 / EN** switch in the top right changes the language, and the choice is
remembered. Ctrl+C in the terminal, or the **Quit** button on the page, ends the
program.

| Option | Meaning |
|---|---|
| `--browser {auto,chrome,edge,default,none}` | `default` (default): a tab in the default browser; `chrome`, `edge`: an app window without tabs or an address bar, which ends the program when closed; `auto`: Chrome's app window, else Edge's, else a tab; `none`: only print the address |
| `--port N` | port (default: any free port; a busy port falls back to a free one) |
| `--keep-running` | with an app window: keep serving after it is closed |

The server binds only to `127.0.0.1` and accepts requests only for this machine.
Uploaded images and results stay in memory and are gone when the program ends.

#### From the command line

```bash
python -m line2func.demo drawing.png --out out/
python -m line2func.serve out/
```

The first command traces `drawing.png` with the baseline engine and prints a
summary, for example (a 256×256 synthetic test drawing):

```
sample_lineart.png: 256x256, 136 curves in 10 strokes, 0.25 s (3.88 s/MP); 132 recognized as lines or arcs [faint strokes on, learned decisions, all curves of the fine tracing, fewer than the 5000 allowed]
  out\curves.json
  out\out.svg
  out\desmos.txt
  out\equations.tex
  out\overlay.png
```

The second command opens the viewer in your browser (Ctrl+C stops it).

Good inputs are clean line art: sketches, ink drawings, diagrams, comic line
art, scanned pencil or pen drawings. Dark lines on light paper work best, but
light lines on a dark background are detected and inverted automatically.

### 3. What you get

| File | Contents |
|---|---|
| `curves.json` | Every curve: 4 control points, stroke id, confidence, measured width and color, its recognized shape (line/arc) if any, and with `--form function` its functions |
| `out.svg` | Vector version; each stroke in its measured width and color. Opens in browsers, Inkscape, Illustrator |
| `desmos.txt` | One Desmos expression per line: parametric, named (`--form named`) or functions (`--form function`), section 5 |
| `equations.tex` | The same equations as a LaTeX `align*` block |
| `overlay.png` | The curves drawn over the original, one color per stroke, for a quick check |
| `source.png` | A copy of the input (shown behind the curves in the viewer) |

**Curves and strokes.** A *stroke* is one drawn line. Long or strongly bent
strokes are split into several cubic *curves* that join end to end and share
the same `stroke` id. Sharp corners are kept as corners.

**Coordinates.** `curves.json` and `out.svg` use image pixels (origin at the
top-left, y down). `desmos.txt` and `equations.tex` flip to math orientation
(y up), so the picture is not upside down in Desmos.

**Confidence** is the fraction of a curve that lies on ink. It is below 1 where
the tracer bridged a gap.

**Filled areas.** Solid black areas, large ones and heavy strokes such as
thick eyelashes, are traced by their outline only (tagged `fill_outline`; the
outline of a thick or strongly tapered stroke is tagged `outline`). The SVG
fills them. Desmos cannot fill pasted curves, so rings inside each area (tagged `fill`, 1.5 px apart) make it look
solid there; the SVG leaves the rings out.

<details>
<summary><code>curves.json</code> format</summary>

```json
{
  "format": "line2func.curves",
  "version": 1,
  "image": {"width": 640, "height": 480},
  "coordinates": "image-pixels-y-down",
  "meta": {"engine": "baseline", "line_width": 2.1, "lineart": "none", "refined": true},
  "curves": [
    {"id": 0, "stroke": 0,
     "ctrl": [[10.0, 90.0], [30.0, 0.0], [70.0, 0.0], [90.0, 90.0]],
     "confidence": 1.0, "tags": [],
     "width": 2.08, "color": "#1b1b1b",
     "shape": {"type": "line", "p0": [...], "p1": [...], "desmos": "y=..."}}
  ]
}
```

`width`, `color` and `shape` are optional (written only when known).
</details>

### 4. The viewer

The web page (section 2) shows every result in this viewer. To open a saved output
folder in it:

```bash
python -m line2func.serve out/                 # opens http://localhost:8000/
python -m line2func.serve out/ --port 0        # any free port
python -m line2func.serve out/ --no-browser    # just print the URL
```

| Action | How |
|---|---|
| Zoom | mouse wheel, pinch |
| Pan | drag |
| Fit to window | **Fit** button, double-click, or `F` |
| Inspect a curve | hover it (tooltip with `x(t)`, `y(t)`); click to select |
| Browse all equations | scroll the list on the right; click a row to jump to the curve |
| Next / previous curve | `↓` / `↑`; `Esc` deselects |
| Copy an equation | select a curve, then **Copy for Desmos** (parametric), **Copy named** (for lines/arcs) or **Copy functions** (results made with `--form function`) |
| Display | **Lines only**, **Lines + original**, or **Lines + missed detail** (after a quality check: the curves in gray, missed lines red, missed faint ink orange, curves without ink blue), with the background's opacity slider |
| The original alone | **Original**: shows only the original image |
| Download | **SVG**, **JSON**, **Desmos**, **LaTeX** buttons (and **ZIP** with `python -m line2func`) |
| All equations at once | **Copy all for Desmos**, then paste into the first expression box |
| Language | **中文 / EN** switch in the top right |

The viewer is built for large results (10,000+ curves): it draws with Canvas2D
from a few cached paths, finds the curve under the pointer with a spatial
index, and only renders the visible rows of the equation list. It binds only
to `127.0.0.1` and serves only the output files (the ones above, plus the
quality check's when present).

### 5. Using the equations in Desmos

1. Open `desmos.txt`, select all, copy.
2. Click the first expression box in [Desmos](https://www.desmos.com/calculator) and paste.
   Multi-line text is usually split into one expression per line automatically.

A parametric line looks like this (Desmos's default domain 0 ≤ t ≤ 1 is exactly right):

```
\left(-40.00t^{3}+60.00t^{2}+60.00t+10.00,\ 20.00t^{3}-270.00t^{2}+250.00t+10.00\right)
```

With `--form named` (or `--named`), straight lines and circular arcs are written in closed form:

```
y=0.5000x+5.00\left\{10.00\le x\le 90.00\right\}
\left(x-50.00\right)^{2}+\left(y-50.00\right)^{2}=40.00^{2}\left\{0.7071x-0.7071y-28.28\ge 0\right\}
```

The arc keeps only the side of its chord that holds it, which is exact for any
arc. All other curves stay parametric.

Numbers never use scientific notation, because Desmos would read `1e-5` as
`1·e − 5`. Rounding moves any point by at most about 0.02 px.

**Up to 5,000 curves by default.** `demo` makes at most 5,000 curves
(`--curves N` for another count); a drawing that needs fewer keeps all of its
finely traced curves. Above 5,000 line2func warns you. If Desmos gets slow on
your computer, ask for fewer, e.g. `--curves 2000`.

#### Functions instead of parametric curves: `--form function`

With `--form function`, every curve is written as explicit functions. The arch
from the top of this page becomes three: rising steeply (`x = g(y)`), flat over
the top (`y = f(x)`) and falling (`x = g(y)`):

```
x=17.92+0.3813\left(y-36.29\right)+0.00561\left(y-36.29\right)^{2}+0.0000977\left(y-36.29\right)^{3}\left\{10.00\le y\le 62.58\right\}
y=70.05-0.0171\left(x-49.49\right)-0.03067\left(x-49.49\right)^{2}\left\{33.59\le x\le 65.40\right\}
x=81.52-0.4101\left(y-36.01\right)-0.00564\left(y-36.01\right)^{2}-0.0000929\left(y-36.01\right)^{3}\left\{10.00\le y\le 62.03\right\}
```

- Each curve is cut where its slope is +1 or −1. Where it is flatter it becomes
  `y = f(x)`, where it is steeper `x = g(y)`. In between the curve never turns
  back, so the function exists. Where it does turn back (a cusp), a new piece
  starts.
- Each piece is a polynomial of degree 1 to 3, centered on the piece:
  `y = a + b(x−c) + d(x−c)² + e(x−c)³`. Straight pieces are written
  `y = mx + c`. Every piece passes through both of its ends, so neighbouring
  pieces and curves join exactly.
- Every function stays within `--function-tolerance` (default 0.25 px) of its
  curve, checked on the printed, rounded numbers. A piece that does not fit is
  halved. Circular arcs are approximated like any other curve.

There are more functions than curves. On the four test drawings (section 8),
at up to 5,000 curves:

| | Curves | Functions | Per curve | Straight pieces |
|---|---|---|---|---|
| Drawing 1 | 3,541 | 5,291 | 1.49 | 73% |
| Drawing 2 | 5,000 | 7,249 | 1.45 | 72% |
| Drawing 3 | 5,000 | 8,145 | 1.63 | 65% |
| Drawing 4 | 5,000 | 6,969 | 1.39 | 72% |

Making the functions takes under a second. Above 5,000 functions `demo` warns
and suggests a smaller `--curves`. Fewer, longer curves each give a few more
functions, and the suggestion allows for that: on the test drawings it gave
4,555 to 4,993 functions. In the viewer, **Copy functions** copies the
functions of the selected curve.

### 6. Photos and color images

Photos need a line-extraction step first (`--lineart`):

| Method | Needs | Result |
|---|---|---|
| `informative` | PyTorch + weights | Pretrained line-art network: drawing-like lines. **Best for photos** |
| `informative-coarse` | PyTorch + weights | Same network, bolder and simpler lines |
| `canny` | nothing | Edge detection; thick lines get an edge on each side |
| `xdog` | nothing | Stylized edges; works on high-contrast images |
| `none` (default) | nothing | The input already is line art |

The pretrained weights are never bundled. Download them once; they are checked
against a pinned SHA-256 and stored in `~/.cache/line2func` (override with
`LINE2FUNC_HOME`):

```bash
python -m line2func.weights list            # what is available, licenses, status
python -m line2func.weights fetch informative   # or: fetch all
python -m line2func.weights verify          # re-check hashes

python -m line2func.demo photo.jpg --lineart informative --out out/
```

Photos usually give many short curves. `demo` merges them down to 5,000
(section 8). The web page traces every image as line art, so photos go through
`demo`.

### 7. All `demo` options

```
python -m line2func.demo IMAGE [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | `out` | Output folder |
| `--lineart {none,canny,xdog,informative,informative-coarse}` | `none` | Line extraction (section 6) |
| `--vectorizer {baseline,model}` | `baseline` | Tracing engine |
| `--ckpt FILE` | – | Checkpoint for `--vectorizer model` (section 9) |
| `--curves N` | `5000` | Make N curves: trace finely, then merge the neighbouring pieces whose merge changes the drawing least (section 8). A drawing that gives fewer keeps all of them |
| `--tolerance PX` | off | Trace to this max curve-fitting error instead of a number of curves. Larger gives fewer, smoother curves |
| `--decisions {learned,rules,PATH}` | `learned` | Who decides where strokes continue, where breaks are joined, which junctions are one crossing and where corners are: the learned scorer, the angle rules, or other learned weights (section 8) |
| `--threshold 0..1` | automatic | Ink threshold. Automatic: Otsu's, but for line art at most 0.25 so light strokes stay whole (kept at Otsu's when the paper itself would be traced). Lower it if faint lines are missed, raise it if paper texture is traced |
| `--form {parametric,named,function}` | `parametric` | How `desmos.txt` / `equations.tex` write each curve: parametric, named (lines and arcs; the other curves parametric) or as functions `y = f(x)` / `x = g(y)` (section 5) |
| `--named` | off | The same as `--form named` |
| `--function-tolerance PX` | `0.25` | With `--form function`: the largest distance between a function and its curve |
| `--shape-tolerance PX` | `0.5` | How close a curve must be to a line/circle to count as one |
| `--no-refine` | refine on | Skip snapping curves to the ink centerline (refinement roughly halves the distance to the true lines) |
| `--upscale {auto,1,2,3,4}` | `auto` | Trace at N× resolution, then map the curves back. `auto` uses 2× when lines are thinner than ~1.75 px. The tolerance stays in original pixels |
| `--no-faint` | faint on | Do not add faint strokes below the threshold, nor very faint lines (the step is conservative: it adds nothing on noisy or shaded paper) |
| `--denoise 0..100` | `50` | How strongly specks and short faint pieces are dropped as noise: 0 keeps them all (most detail; on a noisy scan the noise is traced too), 100 is twice as strict (section 8) |
| `--faint-sensitivity 0..100` | `50` | How light a line may be and still be traced: higher keeps lighter strands of hair and background (at the top some pencil texture, as short dashes), 0 traces only ink above the threshold (section 8) |
| `--quality` | off | Judge the result against the image: `quality.json`, `quality.png` and a summary (section 8) |
| `--no-residual` | second pass on | Skip the second pass that traces the ink the first pass left uncovered (section 8) |
| `--no-outline` | outlines on | Keep solid areas (heavy eyelashes) and thick or wedge-shaped strokes (brush strokes) as centerlines instead of filled outlines |
| `--no-fill` | rings on | Leave filled areas hollow in Desmos: no rings inside them (the SVG fills them anyway) |
| `--optimize` | off | Refine every curve by render-and-compare. Needs PyTorch; for a 760×818 drawing about 3 s on an RTX 5070, 13 s on the CPU (section 8) |

### 8. Tips for good results

The measurements in this section come from four real test drawings (anime line
art found online, not included in the repository) and from synthetic drawings
with known answers.

- **Resolution:** lines about 1.5–5 px wide trace best. Thin-lined drawings are
  traced at 2× automatically (`--upscale auto`); scale very large scans down
  yourself. On two real drawings with ~1.7 px lines, auto upscaling plus faint
  strokes raised the share of lines kept from 95–97% to 98–99% and cut the
  missed ink roughly in half, at 1.2–1.4× more curves.
- **Light and faint strokes:** for line art the automatic ink threshold is at
  most 0.25, so light gray strokes (such as loose strands of hair) are traced
  whole instead of breaking into dashes. Fainter lines are added when they
  clearly stand out from their local background, and the faintest, down to the
  paper's own noise, when they are clearly lines (below). For very faint pencil
  work on clean paper, try `--threshold 0.15` or lower (more detail, more
  curves).
- **Engine choice for real drawings:** use the baseline. On a real anime line
  drawing the neural engine missed more ink and produced more fragments,
  although it wins on synthetic tests.
- **Width range:** if the thickest line is more than ~4.5× the thinnest,
  quality can drop. Much thicker areas are treated as fills.
- **Noisy scans / JPEG:** the baseline removes small specks (on clean paper it
  keeps those that continue a line, below). If dust is still traced, clean the
  scan or raise `--threshold`.
- **Too many or too few curves:** `demo` makes up to 5,000; `--curves N` gives
  another count (below). Fewer curves this way keep more of the drawing than a
  larger `--tolerance`.
- **Thin lines only:** in a drawing made *only* of lines thinner than ~2 px,
  measured widths may read up to 0.5 px too wide.

#### Closing the gap: second pass, outlines, render-and-compare

Three steps target detail that a single tracing pass loses:

- **Second pass** (on by default, `--no-residual` to skip): after the first
  pass, the ink that no curve covers yet is traced again on its own. Pieces that
  are too short, mostly off the ink, or that repeat an existing curve are
  dropped.
- **Outlines** (on by default, `--no-outline` to skip): solid areas, such as a
  heavy eyelash, and strokes much thicker than the drawing's lines or strongly
  tapered (a brush tip) become closed outlines. The SVG fills them, and rings
  inside make them look solid in Desmos (below).
- **Render-and-compare** (`--optimize`, needs PyTorch): all curves are drawn
  with a differentiable renderer and moved by gradient descent until the
  drawing matches the image. Strokes stay joined and filled outlines are kept.

When they were added, the three steps together raised the share of lines kept
on two real drawings from 98.3% and 99.0% to 99.7% and 99.5%, and cut the
missed ink from 1.8% to 0.6% and from 6.3% to 5.2%. What none of them recovers
is ink that is too soft to be a line (blurred shading, the soft ring of an
iris).

#### Light and very faint strokes

**Light strokes.** Otsu's automatic ink threshold sits between the paper and
the dark lines, around 0.33–0.35 on the test drawings, so light gray strokes
just under it broke into dashes. For line art the automatic threshold is at
most 0.25. The typical line width, solid areas and the faint-stroke test are
still judged at Otsu's threshold, so only the light strokes change. If what the
lower threshold adds looks like paper (wide patches of shading, or more than
twice the ink already there), Otsu's threshold is kept. On two drawings with
light strokes, the share of faint strokes kept rose from 98.3% to 99.3% and
from 94.4% to 96.9%, and the missed ink roughly halved.

**Very faint lines.** How faint may a line be and still be traced? Measured as
local contrast (how much darker than the paper around it) on the test drawings:

| | Local contrast |
|---|---|
| Blank paper: grain, JPEG noise | at most 0.031–0.038 |
| Butterflies drawn very lightly in one drawing's background | 0.043–0.084, median 0.060 |
| Pencil texture on a hat in the same drawing | top tenth 0.059, top hundredth 0.135 |

Anything fainter than the paper's own noise must be ignored. Between that and
the faint-stroke level (0.07–0.10), contrast alone cannot tell the butterflies
from the texture, but shape can: a line is long, thin and hardly branches;
texture is short, dense and criss-crossed. So, for line art, such lines are
traced when they are at least 15 line widths long (pieces of one line that
fades here and there counted together), thin, and have at most one real
junction per 17 line widths. The noise level is measured on each drawing's
blank paper, so grainier paper raises it; without blank paper nothing is
added. With up to 5,000 curves:

| | Drawing 1 | Drawing 2 | Drawing 3 | Drawing 4 (butterflies) |
|---|---|---|---|---|
| Butterfly outlines traced | – | – | – | 36% → **98%** |
| Missed ink | 0.35% → 0.35% | 5.0% → **4.4%** | 1.01% → 0.96% | 4.8% → 4.7% |
| Curve length off the ink (counting faint ink) | 0.02% → 0.02% | 0.01% → 0.24% | 0.12% → 0.18% | 0.07% → 0.12% |
| Time | 8.0 → 8.4 s | 16.3 → 18.4 s | 20.3 → 21.5 s | 17.5 → 19.1 s |

The quality tool counts a curve on such a line as on ink ("counting faint ink");
its stricter headline, against ink over half the threshold, rises instead
(drawing 4: 1.9% → 5.2%). In the SVG these lines are as thin and light as in
the drawing.

**Lines broken into dots and dashes.** The tracer drops specks and short faint
pieces as noise. But a light line that fades in and out, such as a loose strand
of hair or a fold, breaks into just such dots and dashes, and it was dropped
with the noise. So in line art, pieces within 2 line widths of each other are
judged together: a broken line stays, lone specks still go. With up to 5,000
curves:

| | Faint strokes kept | Missed ink |
|---|---|---|
| Drawing 1 | 99.26% → 99.60% | 0.35% → 0.24% |
| Drawing 2 | 91.78% → **95.45%** | 4.44% → **2.80%** |
| Drawing 3 | 96.93% → 98.77% | 0.96% → 0.46% |
| Drawing 4 | 91.10% → **97.09%** | 4.74% → **1.75%** |

The share of curve length off the ink (counting faint ink) did not rise, so the
new curves lie on real lines; tracing takes 1-4 s longer, and lines that fade
in and out may be traced as dashes. On noisy paper, chains of noise specks would
pass as lines: on 100 synthetic noisy scans the share of curve length on true
lines fell from 0.93 to 0.87 (0.10 on the worst scan). So this is only done
when the paper is clean (its pixel noise, measured on the blank paper, below
0.005); on those scans the result is then unchanged.

#### How much noise to remove: `--denoise`

`--denoise` (in the web page: **Remove noise** and its slider) sets how strictly
specks and short faint pieces are dropped: 50 is the default described above, 0
keeps them all, and 100 doubles the limits and no longer joins the pieces of
broken lines. Unticking **Remove noise** is 0. On drawing 4 and on 100 synthetic
noisy scans (noise, specks, JPEG):

| Strength | Drawing 4: faint strokes kept | Drawing 4: missed ink | Noisy scans: curve length on true lines (worst scan) |
|---|---|---|---|
| 0 | 98.58% | 1.12% | 0.849 (0.105) |
| 25 | 97.95% | 1.40% | 0.923 (0.737) |
| 50 | 97.09% | 1.75% | 0.929 (0.747) |
| 75 | 94.54% | 2.90% | 0.932 (0.750) |
| 100 | 83.12% | 8.68% | 0.934 (0.752) |

On clean line art a lower strength keeps more detail and invents nothing (the
curve length off the ink stays at 0.10-0.13%). On a noisy scan, 0 traces the
noise too (55% more curves); from 25 up the result hardly changes, and a higher
strength only gives up a little of the lines (recall 0.991 at 50, 0.988 at 100).

#### How light a line may be: `--faint-sensitivity`

Light strands of hair and background are often lighter than the ink threshold.
The tracer still finds them by their contrast over the paper around them
(faint strokes) and, fainter still, by their shape (very faint lines: long,
thin, hardly branching). Hair crosses itself a lot and runs next to darker
strands, so many light strands failed those tests; lowering the ink threshold
instead merges neighbouring strands into blobs (on the drawing below the line
width grew from 1.7 to 2.4 px at a threshold of 0.08, and the eyes became
zigzag outlines).

`--faint-sensitivity` (in the web page: **Faint lines**) loosens those tests
together: 50 is as tuned, 100 needs less than half the contrast (0.12 instead
of 0.3 of the threshold), looks closer to dark lines, and lets very faint lines
be shorter (6 line widths instead of 15), a little wider and branch more (one
junction per 5 line widths instead of 17). Below 50 faint strokes need more
contrast; 0 traces only ink above the threshold. Tolerance 1.0:

| Drawing | Missed ink at 50 | at 100 | Curve length on ink (with faint) at 100 | Line width 50 / 100 |
|---|---|---|---|---|
| huaban-6611354694 | 2.61% | 1.01% | 0.986 | 1.70 / 1.77 px |
| huaban-6611349997 | 1.79% | 0.56% | 0.991 | 1.54 / 1.58 px |
| af26b7b7 | 2.99% | 1.64% | 0.993 | 1.71 / 1.75 px |

At the top, pencil texture comes in as short dashes, and on noisy scans the
noise is traced too (as with `--denoise 0`): on 25 synthetic noisy scans the
share of curve length on true lines fell from 0.929 at 50 to 0.902 at 75 and
0.826 at 100 (0.165 on the worst scan). Clean synthetic scans are unchanged.

#### Heavy eyelashes and other solid areas

Thick black strokes such as heavy eyelashes used to thin down to a tangle of
short centerlines, and Desmos drew them as thin lines. Ink that is at least
2.5 line widths thick over some length, and as dark in its middle as the
drawing's darkest ink, is traced as one filled area. Rings inside the area, 1.5
px apart, make it solid in Desmos too (`--no-fill` leaves it hollow). Two lines
drawn so close that their ink merges do not count: their ink is lighter in
between, so they stay lines.

Drawing 1 at tolerance 1.0:

| | Before | Now |
|---|---|---|
| Dark eyelash ink that Desmos draws (every curve 2.5 px wide, whole drawing on screen) | 89.6% | **97.0%** |
| Lines kept / d_M | 99.65% / 0.527 px | 99.66% / 0.531 px |
| PSNR / SSIM | 21.9 dB / 0.922 | **22.1 dB / 0.923** |
| Curves | 1,078 | 1,111 (35 of them rings) |

On a light sketch whose close double lines merge in many places, only 2 small
areas qualify. Zoomed in far in Desmos, the rings show as rings; with the whole
drawing on screen they merge into a solid area.

#### The number of curves: up to 5,000 by default, `--curves N`

`demo` makes up to 5,000 curves, and `--curves N` any other count. The drawing
is traced finely (0.35 px, or 0.25 px when that gives fewer than N curves;
0.25 px right away when the drawing's lines are too short for N curves at 0.35
px, about one curve per 10 px of line). Then the two neighbouring pieces of a
stroke whose single-curve replacement changes the drawing least are merged,
again and again, until N remain. Intricate parts keep their pieces, smooth
parts merge first, and strokes stay connected. If the drawing gives fewer than
N curves even when traced finely, all of them are kept (with `--curves`, `demo`
says so).

At the same count this is closer to the drawing than a plain tolerance, and
for small counts it keeps more of the lines (drawing 1, quality tool):

| Curves | Plain `--tolerance` | `--curves N` |
|---|---|---|
| 849 | 99.2% of lines, PSNR 19.7 dB | **99.7%**, **20.6 dB** |
| 1,111 | 99.7%, 22.1 dB | **99.8%**, **22.3 dB** |
| 1,751 | 99.8%, 23.7 dB | 99.8%, **23.9 dB** |
| 2,000 | – | 99.8%, 24.1 dB |
| 3,541 (all it gives: the 5,000 default) | – | 99.8%, 24.4 dB |

More curves mostly make the lines follow the drawing more closely; the share of
lines kept hardly changes. Time and PSNR / SSIM on the four test drawings:

| | 2,000 curves | 5,000 curves (the default) |
|---|---|---|
| Drawing 1 | 9.8 s, 23.8 dB / 0.939 | all 3,541 it gives, 8.0 s, 24.4 dB / 0.942 |
| Drawing 2 | 13.9 s, 21.9 dB / 0.578 | 16.4 s, 22.1 dB / 0.587 |
| Drawing 3 | 17.3 s, 18.1 dB / 0.795 | 20.4 s, **19.7 dB / 0.837** |
| Drawing 4 | 14.6 s, 21.4 dB / 0.610 | 17.6 s, 21.5 dB / 0.618 |

Drawing 3 gains most: at 2,000 its merges moved curves by up to 1.1 px, at
5,000 by 0.16 px. A tolerance of 1.0 takes 5–6 s. If Desmos gets slow on your
computer, `--curves 2000` or `--curves 1000` (99.7% of drawing 1's lines, PSNR
21.6 dB) are smaller.

#### How strokes are joined: `--decisions`

The baseline engine makes all the geometry: the skeleton, the stroke graph and
the curves. At four places it has to choose, and those choices decide which
pieces become one stroke:

- where a line continues through a junction;
- which breaks are joined;
- which two nearby junctions are really one shallow crossing;
- where a stroke has a sharp corner.

By default a small learned scorer makes these choices; `--decisions rules`
uses fixed angle rules instead. The scorer sees more than the angles: line
widths, ink darkness, faint ink inside a break, and whether a link would cross
another line.

- It was trained on 20,000 synthetic tracings whose right answers are known,
  in 17 s on an RTX 5070.
- It runs on numpy, so PyTorch is not needed.
- Its weights (150 KB) ship with line2func.
- For inputs unlike its training data, the rules decide.

Measured on synthetic test drawings with known strokes:

| | Rules | Learned |
|---|---|---|
| Lines continuing through crossings (hard set) | 0.762 | **0.826** |
| Breaks joined (hard set / thin pen lines) | 0.774 / 0.086 | **0.898 / 0.975** |
| Wrong joins per 100 strokes (hard set) | 2.43 | **1.58** |
| Corner precision (hard set) | 0.525 | **0.675** |

On two real test drawings it made 4–5% fewer curves, kept the same share of the
lines and invented no more. It joined breaks in faint lines well. At dense
junctions it was only slightly better than the rules, and it misses a few real
sharp tips. Tracing takes about 10–15% longer than with the rules.

#### Judging the result: `--quality`

Real drawings have no "correct answer", so line2func judges the curves against
the image itself:

```bash
python -m line2func.demo drawing.png --quality --out out/
python -m line2func.quality out/          # or later, on an existing output folder
```

```
detail kept (centerline within 2 px of a curve): lines 96.6%, with faint strokes 74.1%; ink by darkness: ...
invented lines: 0.2% of curve length is > 2 px from ink; farthest curve point from any ink 8.9 px
accuracy: d_M 0.553 px, ink->curve p95 1.621 px (max 12.38)
missed ink: 11.9% of all ink; by cause: below_threshold 84.8%, speck 6.3%, untraced 8.8%
1147 curves / 562 strokes, 4.162 per 100 px of ink
```

| Number | Meaning | If it is bad |
|---|---|---|
| **detail kept: lines** | Share of the drawing's lines (ink centerline, specks removed) within 2 px of a curve | `--upscale 2` if auto did not (dense detail) |
| **with faint strokes** | Same, counting strokes down to half the threshold | Lower `--threshold` (more detail, more curves) |
| **invented lines** / farthest point | Curve length with no ink under it (also shown counting faint ink, including very faint lines that stand out from the paper). A stray-curve flag is raised when a curve runs over 10 px from ink | Report it: that is a bug |
| **d_M, p95, max** | Mean / 95th-percentile / worst distance between ink and curves (px) | Keep refinement on; upscale thin lines |
| **missed ink by cause** | `below_threshold` = too faint to trace; `speck` = removed as noise; `untraced` = a visible line the tracer did not follow (dense detail, very short branches) | Follow the matching fix above |
| **curves / strokes** | Compactness; more fidelity always costs more curves | A smaller `--curves N` for Desmos |

`quality.png` marks every problem on the image: **red** = missed line,
**orange** = missed faint ink, **blue** = curve without ink. In the viewer,
choose the display **Lines + missed detail**.

These numbers were validated on synthetic drawings with known answers. The
"lines" recall is within about 0.01 of the true recall on average (Spearman
0.93 over 500 tracings), the distance d_M tracks the true error (Spearman
0.97), and the stray-curve flag caught 98% of injected stray lines with no
false alarms. IoU is reported too but should not be trusted on lines thinner
than ~2 px: a 1 px shift can halve it.

### 9. The neural engine (optional)

The neural engine is **experimental**. On synthetic test drawings it passes
every enabling condition checked there (section 10): compared with the baseline
engine using the angle rules, it continues strokes through crossings better
(0.92 vs 0.76) and closes more gaps (0.95 vs 0.78). The baseline's default
learned decisions narrow that gap (0.83 and 0.90). On a real anime drawing it
missed more ink and produced more fragments than the baseline, and the final
condition, a blind test on real drawings, has not been run. No pretrained
weights are published, so you train your own.

#### Train

```bash
python -m line2func.train --config configs/overfit_multi.yaml   # 5-minute self-check, must print PASSED
python -m line2func.train --config configs/multi_curve.yaml --device cuda
```

- The full run is 150k steps: **about 4.5 h on an RTX 5070 + Ryzen 7 9700X**.
  Data is generated on the fly, so no dataset download is needed.
- Output goes to `runs/m3/`: `best.pt`, `last.pt` and `log.csv`.
- **Time cap:** each run stops cleanly after 36 h (`--max-hours` to change) and
  saves `last.pt`. Continue with:
  ```bash
  python -m line2func.train --config configs/multi_curve.yaml --device cuda --resume
  ```
  A resumed run continues the same data stream and learning-rate schedule. On
  CPU the final weights are bit-identical to an uninterrupted run (tested); on
  GPU, cuDNN's nondeterminism causes tiny numerical differences.
- `last.pt` is also saved every 30 minutes and on Ctrl+C.
- Windows: DataLoader workers use `spawn`; the defaults (7 workers) already
  account for that.

#### Use it

```bash
python -m line2func.demo drawing.png --vectorizer model --ckpt runs/m3/best.pt --out out/
```

The model reads the image in overlapping 64×64 tiles and stitches the curves
into strokes. After that come the same refinement, measurement and export steps
as the baseline.

#### Other configs

| Config | Purpose |
|---|---|
| `configs/overfit.yaml` | Self-check for the single-curve model |
| `configs/single_curve.yaml` | Single-curve model (a stepping stone; ~0.8 h) |
| `configs/overfit_multi.yaml` | Self-check for the multi-curve model |
| `configs/multi_curve.yaml` | The model used by `--vectorizer model` |

Preview the synthetic training data:

```bash
python -m line2func.synth preview --kind hard --out preview.png
python -m line2func.synth preview --kind hard --patches --out patches.png
```

### 10. Evaluation and the blind test

The engines are scored on synthetic drawings with ground truth, in a "clean"
set and a "hard" set (noise, broken lines, dense crossings, width variation).
Besides whether the curves lie on the lines, the scores check that the strokes
are right:

| Metric | Meaning |
|---|---|
| F_GT@2 | How well the curves match the true lines within 2 px |
| Crossing continuity | Whether a line is still one stroke after passing through a crossing |
| Gap closure | Whether small breaks in a line are bridged |
| Fragments per stroke | How many pieces one stroke is split into; ideally 1 |
| Curve count ratio | Output curves ÷ true curves; closer to 1 is more compact |

- **Baseline targets:** F_GT@2 ≥ 0.97 on the clean set, at most 5 s of CPU per
  megapixel.
- **Conditions for enabling the neural engine by default:**
  - on the hard set, crossing continuity and gap closure each at least 0.05
    higher than the baseline, and F_GT@2 no lower;
  - on the clean set, F_GT@2 at most 0.01 below the baseline;
  - in a blind comparison on real line drawings, at least 60% rated better or
    tied (below).

```bash
# a fixed synthetic validation set (100 clean + 100 hard scenes, identical on every machine)
python -m line2func.synth valset --out data/val_v1

# baseline scores: F_GT@2, crossing continuity, gap closure, fragments/stroke, curve ratio, s/MP
python -m line2func.eval --valset data/val_v1

# both engines on whole drawings + the enabling conditions (PASS/FAIL)
python -m line2func.eval --ckpt runs/m3/best.pt --valset data/val_v1

# a model on patches vs the baseline
python -m line2func.eval --ckpt runs/m3/best.pt
```

Add `--json results.json` to save the numbers, and `--limit N` for a quick run.

#### Decision scorer: evaluate and retrain

```bash
python -m line2func.eval --valset data/val_v1 --decisions learned        # learned vs rules + gates
python -m line2func.synth valset --version 2                             # data/val_v2: T-junctions, hatching, thin lines
python -m line2func.eval --valset data/val_v2 --decisions learned --upscale auto
python -m line2func.labels oracle --valset data/val_v1                   # what perfect decisions would gain
python -m line2func.labels disagree drawing.png --decisions learned      # where learned and rules differ, to review
python -m line2func.decisions_data --out data/decisions_v1               # ~25 min with 7 worker processes (20,000 tracings)
python -m line2func.train_decisions --data data/decisions_v1 --out runs/decisions/v1   # needs PyTorch; ~20 s on a GPU
```

#### Blind test on real drawings

This is the last condition for enabling the neural engine: human raters must
judge it better or tied on at least 60% of real drawings.

1. Put real line drawings (PNG/JPEG) in `data/full_v1/`.
2. Build the kit:
   ```bash
   python -m line2func.eval --ckpt runs/m3/best.pt --fullset data/full_v1
   ```
   This writes `runs/m3/blindtest/index.html`, an offline page. For each
   drawing it shows the original and the two tracings as **A** and **B** in
   random order. The page never names the engines, and the answer key is in
   `key.json`, which raters must not open.
3. Each rater opens the page and votes (keys `1` = A, `2` = B, `3` = tie). Then
   they click **Export votes** and send you their `votes.json`. Use at least
   two raters.
4. Score:
   ```bash
   python -m line2func.blindtest score runs/m3/blindtest votes_alice.json votes_bob.json
   ```

### 11. Using line2func from Python

The whole flow in one call, as `demo` and the web page run it (the second pass,
outlines and the rings that fill them in Desmos are on by default; both ask for
up to 5,000 curves; `optimize=True` needs PyTorch):

```python
from line2func import functions, lineart, pipeline
from line2func.export import write_outputs

rgb = lineart.load_rgb("drawing.png")
curves, ink = pipeline.trace(rgb, upscale="auto", curve_count=5000)   # up to 5,000 curves, as demo
curves, ink = pipeline.trace(rgb, upscale="auto", fit_tolerance=1.0)  # a fitting tolerance instead
curves, ink = pipeline.trace(rgb, upscale="auto", optimize=True)
curves, ink = pipeline.trace(rgb, upscale="auto", decisions="rules")  # the angle rules decide

for c in curves:
    print(c.stroke, c.ctrl.tolist(), c.width, c.color, c.shape and c.shape["type"])
write_outputs(curves, "out", source_image=rgb, named=True)  # lines and arcs as named equations
report = functions.attach(curves)  # every curve as y = f(x) / x = g(y) pieces (c.functions), within 0.25 px
write_outputs(curves, "out_functions", source_image=rgb, form="function")
```

The building blocks, step by step (the baseline engine on its own uses the
angle rules and Otsu's threshold, without the pipeline's extra steps):

```python
from line2func import attributes, baseline, lineart, shapes
from line2func.export import to_desmos

rgb = lineart.load_rgb("drawing.png")
ink = lineart.extract(rgb, "none")        # float map, 1 = line
curves = baseline.vectorize(ink)          # CurveSet
attributes.refine(curves, ink)            # snap to the ink centerline
attributes.measure(curves, ink, rgb)      # width and color per curve
shapes.recognize(curves)                  # mark lines and arcs
print(to_desmos(curves, named=True))
```

With the neural engine:

```python
from line2func.model.infer import load_vectorizer
run = load_vectorizer("runs/m3/best.pt")   # uses the GPU if available
curves = run(ink)
```

Other useful modules: `line2func.geometry` (evaluate, split, flatten, arc
length, bounding box, closest point), `line2func.render` (anti-aliased
rasterizer), `line2func.fit.fit_polyline` (Schneider fitting),
`line2func.metrics` (F-score, chamfer, structure scores).

### 12. Project layout and tests

```
line2func/
  app.py  browser.py  __main__.py          # the web page's server (python -m line2func), browser launcher
  viewer/                                  # web page: index.html, app.js, viewer.js, i18n.js, i18n.json
  demo.py  serve.py  pipeline.py           # commands, and the tracing flow they share
  lineart.py  lineart_model.py  weights.py # line extraction, pretrained model, downloads
  baseline.py  fit.py                      # baseline engine, Schneider fitting
  attributes.py  shapes.py  export.py      # refinement, width/color, lines/arcs, exports
  functions.py                             # curves as functions y = f(x) / x = g(y) (--form function)
  residual.py  outline.py  fill.py         # second pass, thick strokes as outlines, rings for Desmos
  budget.py  optimize.py  quality.py       # an exact number of curves, render-and-compare, quality check
  decisions.py  decision_features.py      # the tracer's decisions as scores; candidate features
  decision_model.py  data/                # the learned scorer (numpy) and its bundled weights
  labels.py  decisions_data.py  train_decisions.py   # ground-truth labels and oracle, data, training
  geometry.py  curves.py  render.py        # geometry core, data model, rasterizer
  synth.py  metrics.py  eval.py  blindtest.py        # synthetic data, metrics, evaluation, blind test
  train.py  model/                         # networks, data, losses, tiled inference
configs/        # training configs
docs/           # details.md (this manual), third_party.md (licenses of third-party code and weights)
tests/          # pytest suite
```

```bash
python -m pytest            # about 370 tests; model tests skip without PyTorch, viewer JS checks without Node.js
```

Generated folders (`out/`, `runs/`, `data/`, `.venv/`) are git-ignored.

### 13. Status and roadmap

This is version 1.0. The baseline engine is the one to use; the neural engine is experimental.

- [x] Geometry core and renderer
- [x] Baseline engine; SVG, Desmos and LaTeX export; viewer; `demo` and `serve`
- [x] Synthetic data generator; single- and multi-curve neural models; whole-image inference
- [x] Pretrained line art for photos (Informative Drawings, MIT; downloaded with a SHA-256 check)
- [x] Named equations for lines and arcs, curve refinement, line width and color
- [x] Web page (`python -m line2func`) in English and Traditional Chinese
- [x] Second pass over uncovered ink, thick strokes as filled outlines, render-and-compare (`--optimize`)
- [x] Learned decisions (the default; `--decisions rules` for the angle rules)
- [x] Solid areas such as heavy eyelashes, with rings for Desmos; up to 5,000 curves by default; light and very faint lines
- [x] Function mode: every curve as pieces of `y = f(x)` / `x = g(y)` (`--form function`)
- [x] One-screen web page with three display modes; lines broken into dots and dashes kept; a noise filter slider (`--denoise`)
- [x] Faint-line sensitivity (`--faint-sensitivity`)
- [ ] Blind test on real drawings (tooling ready; needs human raters)
- [ ] Optional: retrain the neural engine on these line styles and fine-tune it on real drawings

---

## 繁體中文

### 目錄

1. [安裝](#1-安裝)
2. [描第一張圖](#2-描第一張圖)
3. [輸出內容](#3-輸出內容)
4. [檢視器](#4-檢視器)
5. [在 Desmos 中使用算式](#5-在-desmos-中使用算式)
6. [照片與彩色圖片](#6-照片與彩色圖片)
7. [`demo` 的所有選項](#7-demo-的所有選項)
8. [取得好結果的訣竅](#8-取得好結果的訣竅)
9. [神經網路引擎（選用）](#9-神經網路引擎選用)
10. [評估與盲測](#10-評估與盲測)
11. [在 Python 中使用](#11-在-python-中使用)
12. [專案結構與測試](#12-專案結構與測試)
13. [現況與路線圖](#13-現況與路線圖)

### 1. 安裝

需要 **Python 3.10 以上**。

#### 基本安裝（只用 CPU、傳統引擎）

```bash
git clone https://github.com/far9100/Line-to-function.git line2func
cd line2func
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
```

這會安裝 `numpy`、`pillow` 和 `scipy`，描線稿、匯出和檢視結果只需要這些。

#### 選用：PyTorch（照片用的預訓練線稿模型、神經網路引擎、訓練）

**先**安裝符合你 GPU 的 PyTorch，再安裝額外套件：

```bash
# NVIDIA RTX 50 系列（Blackwell）需要 CUDA 12.8 以上的版本，例如 CUDA 13.0：
pip install torch --index-url https://download.pytorch.org/whl/cu130
# （只用 CPU：pip install torch）

pip install -e ".[train,dev]"     # 加上 pyyaml（設定檔）與 pytest
```

已驗證的環境：torch 2.14.0+cu130、Python 3.14、RTX 5070（驅動程式 596.21）、Windows 11。

### 2. 描第一張圖

#### 用瀏覽器（拖放）

```bash
python -m line2func          # 或直接打：line2func（執行過 pip install -e . 之後）
```

會在預設瀏覽器開一個新分頁。接著：

1. **把線稿拖進頁面**。也可以按〔選擇檔案…〕，或按 Ctrl+V 貼上。支援 PNG、JPEG、WebP、BMP、TIFF、GIF，最大 64 MB。手機拍的照片會自動轉正。
2. 圖片會出現在原處。選擇線段的寫法：**函數** `y = f(x)`、`x = g(y)`（見第 5 節）或**參數方程式** `x(t)`、`y(t)`，以及去雜訊的強度（〔去雜訊〕，0–100，預設 50：調低保留更多細節，有雜訊的掃描圖可以調高；見第 8 節）與〔淡線〕靈敏度（0–100，預設 50：調高保留更淡的頭髮和背景線），再按〔確認 ▶〕。描線方式和 `demo` 相同：最多 5,000 條曲線，並做品質檢查（超過 2048 px 的圖會先縮小）。處理時會顯示目前的步驟，按〔取消〕會在下一個步驟停下。
3. 結果會在同一頁的檢視器中開啟（見第 4 節）。可以下載 SVG、JSON、Desmos、LaTeX，或把全部打包成 ZIP，也可以〔全部複製到 Desmos〕。
4. 上方的〔清除圖片〕會清掉目前的圖，接著就能拖入下一張；隨時直接拖入新圖片也可以。

網頁版會把每張圖都當成線稿描線。照片請用 `demo` 加上 `--lineart`（見第 6 節）；其他選項也都在指令列（見第 7 節）。

右上角的〔中文｜EN〕可以切換語言，選擇會被記住。在終端機按 Ctrl+C，或按頁面上的〔結束〕，就會結束程式。

| 選項 | 意義 |
|---|---|
| `--browser {auto,chrome,edge,default,none}` | `default`（預設）：預設瀏覽器的分頁；`chrome`、`edge`：沒有分頁和網址列的 App 視窗，關閉視窗就結束程式；`auto`：Chrome 的 App 視窗，沒有就 Edge，再沒有就用分頁；`none`：只印出網址 |
| `--port N` | 連接埠（預設：任一可用的連接埠；被佔用時自動改用別的） |
| `--keep-running` | 用 App 視窗時，關閉視窗後仍繼續執行 |

伺服器只監聽 `127.0.0.1`，也只接受指向本機的請求。上傳的圖片與結果只放在記憶體裡，程式結束就會消失。

#### 用指令

```bash
python -m line2func.demo drawing.png --out out/
python -m line2func.serve out/
```

第一行用傳統引擎描 `drawing.png`，並印出摘要，例如（一張 256×256 的合成測試圖）：

```
sample_lineart.png: 256x256, 136 curves in 10 strokes, 0.25 s (3.88 s/MP); 132 recognized as lines or arcs [faint strokes on, learned decisions, all curves of the fine tracing, fewer than the 5000 allowed]
  out\curves.json
  out\out.svg
  out\desmos.txt
  out\equations.tex
  out\overlay.png
```

第二行會在瀏覽器開啟檢視器（按 Ctrl+C 停止）。

最適合的輸入是乾淨的線稿：素描、墨線稿、示意圖、漫畫線稿，以及掃描的鉛筆或鋼筆稿。淺色紙上的深色線條效果最好；深色背景上的淺色線條會被自動偵測並反轉。

### 3. 輸出內容

| 檔案 | 內容 |
|---|---|
| `curves.json` | 每條曲線的 4 個控制點、所屬筆畫、信心值、量測到的線寬與顏色、辨識出的形狀（直線／圓弧，若有），以及 `--form function` 時的函數 |
| `out.svg` | 向量圖，每一筆畫使用量測到的線寬與顏色；可用瀏覽器、Inkscape、Illustrator 開啟 |
| `desmos.txt` | 每行一個 Desmos 算式：參數式、具名式（`--form named`）或函數（`--form function`），見第 5 節 |
| `equations.tex` | 同樣的算式，寫成 LaTeX `align*` 區塊 |
| `overlay.png` | 曲線疊在原圖上，每一筆畫一種顏色，方便快速檢查 |
| `source.png` | 輸入圖的副本（在檢視器中顯示在曲線後方） |

**曲線與筆畫。**「筆畫」是畫出來的一條線。較長或彎曲較大的筆畫會拆成幾段首尾相接的三次「曲線」，它們共用同一個 `stroke` 編號。銳利的轉角會保留為轉角。

**座標。** `curves.json` 和 `out.svg` 使用圖片像素座標（原點在左上角，y 軸朝下）。`desmos.txt` 和 `equations.tex` 會翻轉成數學方向（y 軸朝上），所以在 Desmos 裡圖不會上下顛倒。

**信心值**是曲線落在墨跡上的比例。描線器補過缺口的地方，信心值會低於 1。

**填滿的區域。** 塗黑的區域（大片的，以及像很粗的睫毛這種粗重筆畫）只描外框，標上 `fill_outline`；粗筆畫或兩端粗細差很多的筆畫，其外框標上 `outline`。SVG 會把它們填滿。Desmos 無法填滿貼上的曲線，所以每個區域內會加上一圈圈的曲線（標上 `fill`，間隔 1.5 px），讓它在 Desmos 裡看起來也是實心的；SVG 不含這些圈。

<details>
<summary><code>curves.json</code> 格式</summary>

```json
{
  "format": "line2func.curves",
  "version": 1,
  "image": {"width": 640, "height": 480},
  "coordinates": "image-pixels-y-down",
  "meta": {"engine": "baseline", "line_width": 2.1, "lineart": "none", "refined": true},
  "curves": [
    {"id": 0, "stroke": 0,
     "ctrl": [[10.0, 90.0], [30.0, 0.0], [70.0, 0.0], [90.0, 90.0]],
     "confidence": 1.0, "tags": [],
     "width": 2.08, "color": "#1b1b1b",
     "shape": {"type": "line", "p0": [...], "p1": [...], "desmos": "y=..."}}
  ]
}
```

`width`、`color`、`shape` 是選用欄位，只有在有資料時才會寫出。
</details>

### 4. 檢視器

網頁版（第 2 節）的每個結果都在這個檢視器裡顯示。要用它開啟已存好的輸出資料夾：

```bash
python -m line2func.serve out/                 # 開啟 http://localhost:8000/
python -m line2func.serve out/ --port 0        # 使用任一可用的連接埠
python -m line2func.serve out/ --no-browser    # 只印出網址，不開瀏覽器
```

| 動作 | 操作方式 |
|---|---|
| 縮放 | 滑鼠滾輪、雙指縮放 |
| 平移 | 拖曳 |
| 回到全圖 | 〔符合視窗〕按鈕、雙擊，或按 `F` |
| 查看曲線 | 滑過曲線（提示框顯示 `x(t)`、`y(t)`）；點擊選取 |
| 瀏覽所有算式 | 捲動右側清單；點一列就會跳到該曲線 |
| 上一條／下一條 | `↑`／`↓`；`Esc` 取消選取 |
| 複製算式 | 選取曲線後按〔複製到 Desmos〕（參數式），直線與圓弧可按〔複製具名算式〕，`--form function` 的結果可按〔複製函數〕 |
| 顯示方式 | 〔只顯示線段〕〔線段＋原圖〕或〔線段＋遺漏線段〕（做過品質檢查後：曲線改為灰色，漏掉的線紅色、漏掉的淡線橘色、沒有墨跡的曲線藍色），以及背景的透明度滑桿 |
| 只看原圖 | 〔原圖〕：只顯示原圖 |
| 下載 | 〔SVG〕〔JSON〕〔Desmos〕〔LaTeX〕按鈕（用 `python -m line2func` 時還有〔ZIP〕） |
| 一次複製全部算式 | 〔全部複製到 Desmos〕，再貼到 Desmos 的第一個算式欄 |
| 語言 | 右上角的〔中文｜EN〕 |

檢視器是為大量曲線（一萬條以上）設計的：用 Canvas2D 從少數快取的路徑繪圖，以空間索引找出游標下的曲線，算式清單也只繪製看得見的那幾列。它只監聽 `127.0.0.1`，也只提供輸出檔（上面列出的，加上品質檢查的檔案，如果有的話）。

### 5. 在 Desmos 中使用算式

1. 打開 `desmos.txt`，全選並複製。
2. 點 [Desmos](https://www.desmos.com/calculator) 的第一個算式欄並貼上。多行內容通常會自動拆成一行一個算式。

參數式長這樣（Desmos 預設的定義域 0 ≤ t ≤ 1 正好適用）：

```
\left(-40.00t^{3}+60.00t^{2}+60.00t+10.00,\ 20.00t^{3}-270.00t^{2}+250.00t+10.00\right)
```

加上 `--form named`（或 `--named`）時，直線與圓弧會寫成具名算式：

```
y=0.5000x+5.00\left\{10.00\le x\le 90.00\right\}
\left(x-50.00\right)^{2}+\left(y-50.00\right)^{2}=40.00^{2}\left\{0.7071x-0.7071y-28.28\ge 0\right\}
```

圓弧只保留弦的其中一側（圓弧所在的那一側），這對任何圓弧都是精確的。其他曲線維持參數式。

數字一律不用科學記號，因為 Desmos 會把 `1e-5` 讀成 `1·e − 5`。四捨五入讓任何一點的位置最多偏移約 0.02 px。

**預設最多 5,000 條曲線。** `demo` 最多產生 5,000 條曲線（用 `--curves N` 指定其他數量）；需要的曲線比這少的圖，會保留細緻描線的全部曲線。超過 5,000 條時 line2func 會提出警告。如果在你的電腦上 Desmos 變慢，可以指定少一點，例如 `--curves 2000`。

#### 用函數代替參數式：`--form function`

加上 `--form function` 時，每條曲線都會寫成顯函數。本頁開頭那道拱形會變成三段：陡升的一段（`x = g(y)`）、頂端平緩的一段（`y = f(x)`）和陡降的一段（`x = g(y)`）：

```
x=17.92+0.3813\left(y-36.29\right)+0.00561\left(y-36.29\right)^{2}+0.0000977\left(y-36.29\right)^{3}\left\{10.00\le y\le 62.58\right\}
y=70.05-0.0171\left(x-49.49\right)-0.03067\left(x-49.49\right)^{2}\left\{33.59\le x\le 65.40\right\}
x=81.52-0.4101\left(y-36.01\right)-0.00564\left(y-36.01\right)^{2}-0.0000929\left(y-36.01\right)^{3}\left\{10.00\le y\le 62.03\right\}
```

- 每條曲線在斜率為 +1 或 −1 的地方切開。較平的部分寫成 `y = f(x)`，較陡的部分寫成 `x = g(y)`。切點之間曲線不會往回走，所以函數一定存在；曲線往回折的地方（尖點）會從新的一段開始。
- 每一段都是 1 到 3 次的多項式，以該段的中心展開：`y = a + b(x−c) + d(x−c)² + e(x−c)³`；直的部分寫成 `y = mx + c`。每一段都通過自己的兩個端點，所以相鄰的段落與曲線會精確相接。
- 每個函數與曲線的距離都在 `--function-tolerance`（預設 0.25 px）以內，而且是用印出來、四捨五入後的數字檢查；不合格的段落會二分。圓弧和其他曲線一樣用多項式近似。

函數會比曲線多。四張測試圖（見第 8 節）在最多 5,000 條曲線時：

| | 曲線 | 函數 | 每條曲線 | 直的段落 |
|---|---|---|---|---|
| 圖 1 | 3,541 | 5,291 | 1.49 | 73% |
| 圖 2 | 5,000 | 7,249 | 1.45 | 72% |
| 圖 3 | 5,000 | 8,145 | 1.63 | 65% |
| 圖 4 | 5,000 | 6,969 | 1.39 | 72% |

產生函數不到一秒。超過 5,000 條函數時 `demo` 會警告，並建議較小的 `--curves`。曲線越少、越長，每條切出的函數就會多一些，建議值已經考慮到這點：在測試圖上得到 4,555 到 4,993 條函數。在檢視器中，〔複製函數〕會複製選取曲線的所有函數。

### 6. 照片與彩色圖片

照片需要先抽出線稿（`--lineart`）：

| 方法 | 需要 | 結果 |
|---|---|---|
| `informative` | PyTorch 與權重 | 預訓練線稿網路，產生像手繪的線條。**最適合照片** |
| `informative-coarse` | PyTorch 與權重 | 同一個網路，線條較粗、較簡潔 |
| `canny` | 無 | 邊緣偵測；粗線的兩側會各描出一條邊 |
| `xdog` | 無 | 風格化的邊緣；適合高對比的圖 |
| `none`（預設） | 無 | 輸入本身已經是線稿 |

預訓練權重不隨專案附帶。下載一次即可；下載後會用固定的 SHA-256 驗證，存放在 `~/.cache/line2func`（可用環境變數 `LINE2FUNC_HOME` 更改）：

```bash
python -m line2func.weights list            # 列出可用的權重、授權與下載狀態
python -m line2func.weights fetch informative   # 或：fetch all
python -m line2func.weights verify          # 重新檢查雜湊值

python -m line2func.demo photo.jpg --lineart informative --out out/
```

照片通常會得到很多短曲線。`demo` 會把它們合併到 5,000 條（見第 8 節）。網頁版會把每張圖都當成線稿描線，所以照片請用 `demo`。

### 7. `demo` 的所有選項

```
python -m line2func.demo IMAGE [options]
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `--out DIR` | `out` | 輸出資料夾 |
| `--lineart {none,canny,xdog,informative,informative-coarse}` | `none` | 抽線稿方法（見第 6 節） |
| `--vectorizer {baseline,model}` | `baseline` | 描線引擎 |
| `--ckpt FILE` | – | `--vectorizer model` 使用的 checkpoint（見第 9 節） |
| `--curves N` | `5000` | 產生 N 條曲線：先細緻地描線，再合併「合併後對圖影響最小」的相鄰片段（見第 8 節）。曲線本來就比 N 少的圖會全部保留 |
| `--tolerance PX` | 關閉 | 改用曲線擬合的最大誤差來描線，而不是指定曲線數量。數值越大，曲線越少、越平滑 |
| `--decisions {learned,rules,PATH}` | `learned` | 由誰決定線條在交叉處怎麼接、哪些斷口要接起來、哪些相鄰的交叉點其實是同一個淺角交叉、哪裡是轉角：學習式評分器、角度規則，或其他學習權重（見第 8 節） |
| `--threshold 0..1` | 自動 | 墨跡門檻。自動：採用 Otsu 門檻，但線稿最高只到 0.25，讓淺色的筆畫保持完整（如果連紙面都會被描出來，就維持 Otsu 門檻）。淡的線被漏掉時調低，紙張紋理被描出來時調高 |
| `--form {parametric,named,function}` | `parametric` | `desmos.txt`／`equations.tex` 怎麼寫每條曲線：參數式、具名式（直線與圓弧；其他曲線仍是參數式），或函數 `y = f(x)`／`x = g(y)`（見第 5 節） |
| `--named` | 關閉 | 等於 `--form named` |
| `--function-tolerance PX` | `0.25` | `--form function` 時，函數與曲線之間允許的最大距離 |
| `--shape-tolerance PX` | `0.5` | 曲線要多接近直線或圓才算是直線或圓弧 |
| `--no-refine` | 精修開啟 | 跳過把曲線貼齊墨跡中心線的步驟（精修約可讓與真實線條的距離減半） |
| `--upscale {auto,1,2,3,4}` | `auto` | 以 N 倍解析度描線，再把曲線換算回原尺寸。`auto` 在線條細於約 1.75 px 時放大 2 倍；容許誤差仍以原圖像素計 |
| `--no-faint` | 淡線開啟 | 不加入門檻以下的淡線，也不加入極淡的線（這個步驟很保守：在有雜訊或陰影的紙上不會加入任何東西） |
| `--denoise 0..100` | `50` | 小雜點和短的淡線片段當成雜訊丟掉的強度：0 全部保留（細節最多；有雜訊的掃描圖連雜訊也會描出來），100 是兩倍嚴格（見第 8 節） |
| `--faint-sensitivity 0..100` | `50` | 多淡的線也算線條：調高會保留更淡的頭髮和背景線（最高時鉛筆紋理也會變成短線），0 只描門檻以上的墨跡（見第 8 節） |
| `--quality` | 關閉 | 拿結果和原圖比對：輸出 `quality.json`、`quality.png` 與摘要（見第 8 節） |
| `--no-residual` | 第二遍開啟 | 跳過第二遍描線（第二遍只描第一遍沒蓋到的墨跡，見第 8 節） |
| `--no-outline` | 外框開啟 | 實心區域（粗重的睫毛）以及粗筆畫、楔形筆畫（筆刷）維持中心線，不改成填滿的外框 |
| `--no-fill` | 圈線開啟 | 在 Desmos 裡讓填滿的區域保持空心：不在裡面加圈線（SVG 本來就會填滿） |
| `--optimize` | 關閉 | 用「渲染後比對」微調每一條曲線。需要 PyTorch；760×818 的圖在 RTX 5070 上約 3 秒，CPU 約 13 秒（見第 8 節） |

### 8. 取得好結果的訣竅

本節的量測數據來自四張真實測試圖（網路上找到的動漫線稿，不包含在儲存庫裡），以及有標準答案的合成圖。

- **解析度：** 線寬約 1.5–5 px 時效果最好。細線的圖會自動以 2 倍解析度描線（`--upscale auto`）；很大的掃描圖請自行縮小。在兩張線寬約 1.7 px 的真實線稿上，自動放大加上淡線，讓線條保留率從 95–97% 提高到 98–99%，漏掉的墨跡大約減半，曲線數為原本的 1.2–1.4 倍。
- **淺色與很淡的線：** 線稿的自動墨跡門檻最高只到 0.25，所以淺灰色的筆畫（例如飄散的髮絲）會完整描出，而不是斷成一段一段的虛線。比這更淡的線，只要明顯比周圍背景深也會被加入；最淡的、一直到紙張本身的雜訊為止，只要明顯是線也會被加入（見下文）。在乾淨紙面上很淡的鉛筆稿，可以試試 `--threshold 0.15` 或更低（細節更多，曲線也更多）。
- **真實線稿請用傳統引擎：** 在一張真實的動漫線稿上，神經網路引擎漏掉較多墨跡、碎片也較多，雖然它在合成測試中勝過傳統引擎。
- **線寬差距：** 最粗的線如果超過最細的約 4.5 倍，品質可能下降；粗很多的區域會被當成填色區域。
- **掃描雜訊、JPEG：** 傳統引擎會去除小雜點（紙面乾淨時，接續某條線的小點會保留，見後面）。如果灰塵還是被描出來，先清理掃描圖或調高 `--threshold`。
- **曲線太多或太少：** `demo` 最多產生 5,000 條；用 `--curves N` 指定其他數量（見下文）。用這個方法減少曲線，比調高 `--tolerance` 保留更多細節。
- **全是細線的圖：** 如果整張圖只有細於約 2 px 的線，量到的線寬可能偏寬最多 0.5 px。

#### 補回細節：第二遍、外框、渲染後比對

有三個步驟專門處理單次描線會遺失的細節：

- **第二遍描線**（預設開啟，`--no-residual` 關閉）：第一遍描完後，把還沒有任何曲線蓋到的墨跡單獨再描一次。太短的、大部分不在墨跡上的、或只是重複描既有曲線的片段都會被丟掉。
- **外框**（預設開啟，`--no-outline` 關閉）：實心區域（例如粗重的睫毛），以及比圖中一般線條粗很多、或兩端粗細差很多的筆畫（例如筆刷的尖端），會改用封閉的外框表示。SVG 會把它們填滿，裡面的圈線則讓它們在 Desmos 裡看起來也是實心的（見下文）。
- **渲染後比對**（`--optimize`，需要 PyTorch）：用可微分的渲染器把所有曲線畫出來，再用梯度下降移動曲線，直到畫出來的圖和原圖一致。同一筆畫的各段曲線會保持相連，填滿的外框則維持不動。

加入這三個步驟時，它們一起讓兩張真實線稿的線條保留率從 98.3% 和 99.0% 提高到 99.7% 和 99.5%，漏掉的墨跡從 1.8% 降到 0.6%、從 6.3% 降到 5.2%。這三個步驟都救不回來的，是淡到不像線的墨跡（模糊的陰影、虹膜柔和的光環）。

#### 淺色與極淡的線

**淺色的筆畫。** Otsu 自動門檻落在紙和深色線條之間，在測試圖上約 0.33–0.35，所以剛好比它淺一點的灰色筆畫會斷成虛線。線稿的自動門檻最高只到 0.25。一般線寬、實心區域和淡線偵測仍然以 Otsu 門檻判斷，所以只有淺色筆畫會改變。如果調低門檻後多出來的墨跡看起來像紙張（大片陰影，或比原有墨跡多兩倍以上），就維持 Otsu 門檻。在兩張有淺色筆畫的圖上，淡線的保留率從 98.3% 提高到 99.3%、從 94.4% 提高到 96.9%，漏掉的墨跡大約減半。

**極淡的線。** 多淡的線還應該描？以「局部對比度」（比周圍紙面深多少）在測試圖上量測：

| | 局部對比度 |
|---|---|
| 空白紙面：紙紋、JPEG 雜訊 | 最高 0.031–0.038 |
| 一張圖背景裡畫得很淡的蝴蝶 | 0.043–0.084，中位數 0.060 |
| 同一張圖帽子上的鉛筆紋理 | 最高 10% 約 0.059，最高 1% 約 0.135 |

比紙張本身的雜訊還淡的一定要忽略。介於雜訊和淡線門檻（0.07–0.10）之間時，光看深淺分不出蝴蝶和紋理，但形狀可以：線條又長、又細、幾乎不分岔；紋理則短小、密集、互相交錯。所以線稿裡這種線，只有在長度至少 15 個線寬（同一條線中途變淡斷開的片段合起來算）、夠細、而且每 17 個線寬最多一個真正的分岔點時才描。雜訊的標準從每張圖的空白紙面量出，紙越粗糙標準越高；沒有空白紙面就不加。最多 5,000 條曲線時：

| | 圖 1 | 圖 2 | 圖 3 | 圖 4（蝴蝶） |
|---|---|---|---|---|
| 蝴蝶外框被描到 | – | – | – | 36% → **98%** |
| 漏掉的墨跡 | 0.35% → 0.35% | 5.0% → **4.4%** | 1.01% → 0.96% | 4.8% → 4.7% |
| 不在墨跡上的曲線長度（算上淡墨） | 0.02% → 0.02% | 0.01% → 0.24% | 0.12% → 0.18% | 0.07% → 0.12% |
| 時間 | 8.0 → 8.4 秒 | 16.3 → 18.4 秒 | 20.3 → 21.5 秒 | 17.5 → 19.1 秒 |

品質工具會把畫在這種線上的曲線算成在墨跡上（「算上淡墨」）；它較嚴格的主要數字只把門檻一半以上的墨當墨跡，所以反而會上升（圖 4：1.9% → 5.2%）。在 SVG 裡這些線和原圖一樣又細又淡。

**斷成點與虛線的線。** 描線器會把小雜點和短的淡線片段當成雜訊丟掉。但忽隱忽現的淺色線條（例如飄散的髮絲、衣服的皺褶）正好會斷成這種點和短虛線，結果跟雜訊一起被丟掉了。所以線稿裡，彼此相距 2 個線寬以內的片段會合起來判斷：斷掉的線會保留，孤立的小雜點照樣丟掉。最多 5,000 條曲線時：

| | 淡線保留率 | 漏掉的墨跡 |
|---|---|---|
| 圖 1 | 99.26% → 99.60% | 0.35% → 0.24% |
| 圖 2 | 91.78% → **95.45%** | 4.44% → **2.80%** |
| 圖 3 | 96.93% → 98.77% | 0.96% → 0.46% |
| 圖 4 | 91.10% → **97.09%** | 4.74% → **1.75%** |

不在墨跡上的曲線長度（算上淡墨）沒有增加，表示新增的曲線都畫在真的線上；描線時間多 1–4 秒，忽隱忽現的線可能會描成虛線。在有雜訊的紙上，一串雜點也會被當成線：在 100 張有雜訊的合成掃描圖上，曲線落在真實線條上的比例從 0.93 降到 0.87（最差的一張只剩 0.10）。所以只有紙面乾淨時（在空白紙面量到的像素雜訊低於 0.005）才這樣做；這些掃描圖的結果因此不受影響。

#### 去雜訊的強度：`--denoise`

`--denoise`（網頁版是〔去雜訊〕與它的拖動條）決定小雜點和短的淡線片段丟得多嚴格：50 是上面說明的預設值，0 全部保留，100 則把各項門檻加倍，而且不再把斷線的片段合起來判斷。取消勾選〔去雜訊〕等於 0。在圖 4 和 100 張有雜訊的合成掃描圖（雜訊、雜點、JPEG）上：

| 強度 | 圖 4：淡線保留率 | 圖 4：漏掉的墨跡 | 雜訊掃描圖：曲線落在真實線條上的比例（最差一張） |
|---|---|---|---|
| 0 | 98.58% | 1.12% | 0.849（0.105） |
| 25 | 97.95% | 1.40% | 0.923（0.737） |
| 50 | 97.09% | 1.75% | 0.929（0.747） |
| 75 | 94.54% | 2.90% | 0.932（0.750） |
| 100 | 83.12% | 8.68% | 0.934（0.752） |

乾淨的線稿上，強度越低保留越多細節，也不會多畫錯的線（不在墨跡上的曲線長度維持在 0.10–0.13%）。有雜訊的掃描圖設成 0 會連雜訊一起描（曲線多 55%）；25 以上結果幾乎不變，強度越高只會少掉一點點線條（召回率 50 時 0.991，100 時 0.988）。

#### 多淡的線也算線條：`--faint-sensitivity`

頭髮和背景的淺色線條常常比墨跡門檻還淡。描線器仍然會用它們和周圍紙面的對比找出來（淡線），更淡的則看形狀（極淡的線：長、細、很少分岔）。頭髮常常互相交叉，又緊貼著較深的髮絲，所以許多淺色髮絲過不了這些檢查；若改成調低墨跡門檻，相鄰的髮絲會黏成一團（下面這張圖在門檻 0.08 時線寬從 1.7 變成 2.4 px，眼睛也變成鋸齒狀的外框）。

`--faint-sensitivity`（網頁版是〔淡線〕）一起放寬這些檢查：50 是調校好的預設，100 只需要不到一半的對比（門檻的 0.12 而不是 0.3）、更靠近深色線也會找，極淡的線可以更短（6 個線寬而不是 15）、稍寬、分岔更多（每 5 個線寬一個交叉點，而不是 17）。低於 50 時淡線需要更高的對比；0 只描門檻以上的墨跡。容許誤差 1.0：

| 圖 | 50 時漏掉的墨跡 | 100 時 | 100 時曲線落在墨跡上的比例（含淡墨） | 線寬 50／100 |
|---|---|---|---|---|
| huaban-6611354694 | 2.61% | 1.01% | 0.986 | 1.70／1.77 px |
| huaban-6611349997 | 1.79% | 0.56% | 0.991 | 1.54／1.58 px |
| af26b7b7 | 2.99% | 1.64% | 0.993 | 1.71／1.75 px |

調到最高時，鉛筆紋理會變成短線被描出來；有雜訊的掃描圖連雜訊也會描（和 `--denoise 0` 一樣）：在 25 張有雜訊的合成掃描圖上，曲線落在真實線條上的比例從 50 時的 0.929 降到 75 時的 0.902、100 時的 0.826（最差一張 0.165）。乾淨的合成掃描圖不受影響。

#### 粗重的睫毛與其他實心區域

像粗重睫毛這種很粗的黑色筆畫，以前會被細化成一團短短的中心線，在 Desmos 裡也只畫成細線。至少 2.5 個線寬粗、有一定長度，而且中間和圖中最黑的墨一樣黑的墨跡，會被描成一個填滿的區域。區域內每隔 1.5 px 加一圈曲線，讓它在 Desmos 裡也是實心的（`--no-fill` 可保持空心）。兩條靠得太近、墨跡黏在一起的線不算：它們中間的墨比較淡，所以仍然是線。

圖 1，容差 1.0：

| | 之前 | 現在 |
|---|---|---|
| Desmos 畫到的深色睫毛墨跡（每條曲線 2.5 px 寬、整張圖顯示在畫面上） | 89.6% | **97.0%** |
| 線條保留率／d_M | 99.65%／0.527 px | 99.66%／0.531 px |
| PSNR／SSIM | 21.9 dB／0.922 | **22.1 dB／0.923** |
| 曲線數 | 1,078 | 1,111（其中 35 條是圈線） |

在一張很多地方雙線黏在一起的淡色草稿上，只有 2 個小區域符合條件。在 Desmos 裡放得很大時看得出一圈圈的線；整張圖顯示在畫面上時，它們會連成實心的一塊。

#### 曲線數量：預設最多 5,000 條，`--curves N`

`demo` 預設最多產生 5,000 條曲線，`--curves N` 可指定其他數量。做法是先細緻地描線（容許誤差 0.35 px；如果這樣得到的曲線不到 N 條，就改用 0.25 px；如果圖上的線太短，在 0.35 px 下明顯湊不到 N 條，約每 10 px 線長一條，就直接用 0.25 px），再反覆把同一筆畫中「換成一條曲線後對圖影響最小」的兩段相鄰片段合併，直到剩下 N 條。精細的地方會保留較多片段，平滑的地方先合併，筆畫也維持相連。如果細緻描線後的曲線還不到 N 條，就全部保留（有指定 `--curves` 時，`demo` 會提示）。

在相同曲線數下，這個方法比單純調整容許誤差更貼近原圖；曲線數少的時候，保留的線條也更多（圖 1，品質工具量測）：

| 曲線數 | 單純調 `--tolerance` | `--curves N` |
|---|---|---|
| 849 | 線條保留 99.2%，PSNR 19.7 dB | **99.7%**，**20.6 dB** |
| 1,111 | 99.7%，22.1 dB | **99.8%**，**22.3 dB** |
| 1,751 | 99.8%，23.7 dB | 99.8%，**23.9 dB** |
| 2,000 | – | 99.8%，24.1 dB |
| 3,541（能描出的全部：5,000 條預設） | – | 99.8%，24.4 dB |

曲線變多主要是讓線條更貼合原圖，線條保留率幾乎不變。四張測試圖的時間與 PSNR／SSIM：

| | 2,000 條 | 5,000 條（預設） |
|---|---|---|
| 圖 1 | 9.8 秒，23.8 dB／0.939 | 全部 3,541 條，8.0 秒，24.4 dB／0.942 |
| 圖 2 | 13.9 秒，21.9 dB／0.578 | 16.4 秒，22.1 dB／0.587 |
| 圖 3 | 17.3 秒，18.1 dB／0.795 | 20.4 秒，**19.7 dB／0.837** |
| 圖 4 | 14.6 秒，21.4 dB／0.610 | 17.6 秒，21.5 dB／0.618 |

圖 3 進步最多：2,000 條時合併讓曲線最多偏移 1.1 px，5,000 條時只有 0.16 px。容差 1.0 大約 5–6 秒。如果在你的電腦上 Desmos 變慢，可以用 `--curves 2000` 或 `--curves 1000`（仍保留圖 1 的 99.7% 線條，PSNR 21.6 dB）。

#### 筆畫怎麼接起來：`--decisions`

傳統引擎負責產生所有幾何：骨架、筆畫圖和曲線。但有四個地方必須做選擇，而這些選擇決定了哪些片段會成為同一筆：

- 線條在交叉處往哪裡延續；
- 哪些斷口要接起來；
- 兩個相鄰的交叉點是不是其實只是一個淺角交叉；
- 筆畫在哪裡有尖角。

預設由一個小型的學習式評分器做這些選擇；`--decisions rules` 則改用固定的角度規則。評分器看到的不只是角度，還有線寬、墨色深淺、斷口裡的淡墨，以及一條連線會不會穿過別的線。

- 它用 2 萬次有標準答案的合成描線訓練，在 RTX 5070 上只花 17 秒。
- 推論只用 numpy，不需要 PyTorch。
- 權重只有 150 KB，隨 line2func 一起附帶。
- 遇到和訓練資料差很多的輸入時，仍由規則決定。

在有已知筆畫的合成測試圖上的量測結果：

| | 規則 | 學習式 |
|---|---|---|
| 線條穿過交叉處不斷（困難組） | 0.762 | **0.826** |
| 斷口被接起來（困難組／細筆線） | 0.774 / 0.086 | **0.898 / 0.975** |
| 每 100 筆的錯誤接合（困難組） | 2.43 | **1.58** |
| 轉角精確率（困難組） | 0.525 | **0.675** |

在兩張真實測試圖上，曲線數少了 4–5%，線條保留率不變，也沒有多出不在墨上的曲線。淡線的斷口接得很好；在密集的交叉處只比規則略好，而且會漏掉少數真正的尖端。描線時間比用規則時約多 10–15%。

#### 評判結果：`--quality`

真實線稿沒有「標準答案」，所以 line2func 直接拿曲線和原圖比對：

```bash
python -m line2func.demo drawing.png --quality --out out/
python -m line2func.quality out/          # 也可以之後對既有的輸出資料夾執行
```

```
detail kept (centerline within 2 px of a curve): lines 96.6%, with faint strokes 74.1%; ink by darkness: ...
invented lines: 0.2% of curve length is > 2 px from ink; farthest curve point from any ink 8.9 px
accuracy: d_M 0.553 px, ink->curve p95 1.621 px (max 12.38)
missed ink: 11.9% of all ink; by cause: below_threshold 84.8%, speck 6.3%, untraced 8.8%
1147 curves / 562 strokes, 4.162 per 100 px of ink
```

| 數字 | 意義 | 不理想時怎麼辦 |
|---|---|---|
| **detail kept: lines**（細節保留率） | 圖中線條（墨跡中心線，去除雜點）有多少比例在曲線 2 px 內 | 如果沒有自動放大，手動加 `--upscale 2`（細節密集時） |
| **with faint strokes** | 同上，但連門檻一半以上的淡線也算進去 | 調低 `--threshold`（細節更多，曲線也更多） |
| **invented lines**／最遠點 | 底下沒有墨跡的曲線長度（也會顯示連淡墨一起算的數字，包括比紙面明顯深的極淡線）；曲線離墨跡超過 10 px 時會發出亂線警告 | 請回報，這是 bug |
| **d_M、p95、max** | 墨跡與曲線之間的平均、第 95 百分位、最大距離（px） | 保持精修開啟；細線先放大 |
| **missed ink by cause**（遺漏原因） | `below_threshold` = 太淡而沒描；`speck` = 當成雜點移除；`untraced` = 看得到但沒描到的線（細節太密、分支太短） | 依上面對應的方法處理 |
| **curves／strokes** | 精簡程度；想更忠實就一定要更多曲線 | 要用 Desmos 時指定較小的 `--curves N` |

`quality.png` 會把問題標在圖上：**紅色** = 漏掉的線、**橘色** = 漏掉的淡墨、**藍色** = 沒有墨跡的曲線。在檢視器中，顯示方式選〔線段＋遺漏線段〕即可查看。

這些數字已在有標準答案的合成圖上驗證過。「lines」保留率與真實值平均只差約 0.01（500 組結果的 Spearman 相關係數 0.93），距離 d_M 與真實誤差高度一致（Spearman 0.97），亂線檢查抓到 98% 刻意加入的亂線，而且沒有誤報。報告中也有 IoU，但線寬細於約 2 px 時不可信：只偏移 1 px 就可能讓它減半。

### 9. 神經網路引擎（選用）

神經網路引擎目前是**實驗功能**。在合成測試圖上，它通過了能在合成圖上檢查的所有啟用條件（見第 10 節）：和使用角度規則的傳統引擎相比，它穿過交叉點時更能保持筆畫連續（0.92 對 0.76），也補起更多缺口（0.95 對 0.78）。傳統引擎預設的學習式決策縮小了這個差距（0.83 與 0.90）。在一張真實的動漫線稿上，它漏掉的墨跡和碎片都比傳統引擎多；最後一項條件，也就是真實線稿的盲測，尚未進行。目前沒有公開的預訓練權重，需要自行訓練。

#### 訓練

```bash
python -m line2func.train --config configs/overfit_multi.yaml   # 約 5 分鐘的自我檢查，必須印出 PASSED
python -m line2func.train --config configs/multi_curve.yaml --device cuda
```

- 完整訓練是 15 萬步，**在 RTX 5070 + Ryzen 7 9700X 上約 4.5 小時**。資料是即時產生的，不需要下載資料集。
- 輸出在 `runs/m3/`：`best.pt`、`last.pt`、`log.csv`。
- **時間上限：** 每次訓練最多 36 小時（可用 `--max-hours` 更改），到時會存 `last.pt` 並正常停止。接續訓練：
  ```bash
  python -m line2func.train --config configs/multi_curve.yaml --device cuda --resume
  ```
  接續時會沿用同一份資料流與學習率排程。在 CPU 上，最後的權重與不中斷訓練逐位元相同（有測試驗證）；在 GPU 上，cuDNN 的非確定性會造成極小的數值差異。
- 另外每 30 分鐘和按 Ctrl+C 時也會存 `last.pt`。
- Windows：DataLoader 的 worker 以 `spawn` 方式啟動；預設值（7 個 worker）已經考慮到這點。

#### 使用

```bash
python -m line2func.demo drawing.png --vectorizer model --ckpt runs/m3/best.pt --out out/
```

模型以互相重疊的 64×64 圖塊讀取整張圖，再把曲線接成筆畫。之後的精修、量測與匯出步驟和傳統引擎相同。

#### 其他設定檔

| 設定檔 | 用途 |
|---|---|
| `configs/overfit.yaml` | 單曲線模型的自我檢查 |
| `configs/single_curve.yaml` | 單曲線模型（過渡用的模型；約 0.8 小時） |
| `configs/overfit_multi.yaml` | 多曲線模型的自我檢查 |
| `configs/multi_curve.yaml` | `--vectorizer model` 使用的模型 |

預覽合成的訓練資料：

```bash
python -m line2func.synth preview --kind hard --out preview.png
python -m line2func.synth preview --kind hard --patches --out patches.png
```

### 10. 評估與盲測

兩種引擎都用有標準答案的合成圖評分，分成「乾淨組」和「困難組」（雜訊、斷線、密集交叉、線寬變化）。除了曲線有沒有落在線上，也檢查筆畫是否正確：

| 指標 | 意義 |
|---|---|
| F_GT@2 | 曲線與真實線條在 2 px 內的吻合程度 |
| 交叉接續 | 線條穿過交叉點後是否仍是同一筆 |
| 斷線補齊 | 線條上的小斷口是否被接起來 |
| 每筆畫碎片數 | 一筆被拆成幾段，理想是 1 |
| 曲線數比 | 輸出曲線數 ÷ 真實曲線數，越接近 1 越精簡 |

- **傳統引擎的目標：** 乾淨組 F_GT@2 ≥ 0.97，每百萬像素最多 5 秒 CPU。
- **神經網路引擎成為預設的條件：**
  - 困難組的交叉接續與斷線補齊都比傳統引擎高至少 0.05，而且 F_GT@2 不低於傳統引擎；
  - 乾淨組的 F_GT@2 最多比傳統引擎低 0.01；
  - 在真實線稿的盲測中，至少 60% 被判定更好或打平（見下文）。

```bash
# 固定的合成驗證集（乾淨、困難各 100 張，在每台電腦上都完全相同）
python -m line2func.synth valset --out data/val_v1

# 傳統引擎的分數：F_GT@2、交叉接續、斷線補齊、每筆畫碎片數、曲線數比、每百萬像素秒數
python -m line2func.eval --valset data/val_v1

# 兩種引擎描整張圖，並檢查啟用條件（PASS／FAIL）
python -m line2func.eval --ckpt runs/m3/best.pt --valset data/val_v1

# 模型與傳統引擎在圖塊上的比較
python -m line2func.eval --ckpt runs/m3/best.pt
```

加上 `--json results.json` 可以儲存數據，`--limit N` 可以快速跑少量場景。

#### 決策評分器：評估與重新訓練

```bash
python -m line2func.eval --valset data/val_v1 --decisions learned        # 學習式對規則，並檢查關卡
python -m line2func.synth valset --version 2                             # data/val_v2：T 字、排線、細線
python -m line2func.eval --valset data/val_v2 --decisions learned --upscale auto
python -m line2func.labels oracle --valset data/val_v1                   # 完美決策的上限
python -m line2func.labels disagree drawing.png --decisions learned      # 學習式與規則判斷不同的地方，供人工審查
python -m line2func.decisions_data --out data/decisions_v1               # 7 個行程約 25 分鐘（2 萬次描線）
python -m line2func.train_decisions --data data/decisions_v1 --out runs/decisions/v1   # 需要 PyTorch；GPU 約 20 秒
```

#### 真實線稿的盲測

這是啟用神經網路引擎的最後一項條件：在真實線稿上，評分者要有至少 60% 判定它更好或打平。

1. 把真實線稿（PNG／JPEG）放進 `data/full_v1/`。
2. 產生盲測網頁：
   ```bash
   python -m line2func.eval --ckpt runs/m3/best.pt --fullset data/full_v1
   ```
   這會產生 `runs/m3/blindtest/index.html`，一個離線網頁。每張圖會顯示原圖和兩種描線結果，以 **A**、**B** 隨機排列。網頁不會透露引擎名稱，答案存在 `key.json`，評分者不可以打開。
3. 每位評分者打開網頁投票（按鍵 `1` = A、`2` = B、`3` = 打平），再按 **Export votes**，把匯出的 `votes.json` 交給你。至少請兩位評分者。
4. 計分：
   ```bash
   python -m line2func.blindtest score runs/m3/blindtest votes_alice.json votes_bob.json
   ```

### 11. 在 Python 中使用

一次跑完整個流程，和 `demo` 與網頁版的做法相同（第二遍、外框和讓外框在 Desmos 裡填滿的圈線預設開啟；兩者都指定最多 5,000 條曲線；`optimize=True` 需要 PyTorch）：

```python
from line2func import functions, lineart, pipeline
from line2func.export import write_outputs

rgb = lineart.load_rgb("drawing.png")
curves, ink = pipeline.trace(rgb, upscale="auto", curve_count=5000)   # 最多 5,000 條曲線，和 demo 一樣
curves, ink = pipeline.trace(rgb, upscale="auto", fit_tolerance=1.0)  # 改用容差
curves, ink = pipeline.trace(rgb, upscale="auto", optimize=True)
curves, ink = pipeline.trace(rgb, upscale="auto", decisions="rules")  # 改由角度規則決定

for c in curves:
    print(c.stroke, c.ctrl.tolist(), c.width, c.color, c.shape and c.shape["type"])
write_outputs(curves, "out", source_image=rgb, named=True)  # 直線與圓弧寫成具名算式
report = functions.attach(curves)  # 每條曲線切成 y = f(x)／x = g(y)（c.functions），誤差 0.25 px 以內
write_outputs(curves, "out_functions", source_image=rgb, form="function")
```

也可以一步一步使用各個元件（單獨使用傳統引擎時採用角度規則與 Otsu 門檻，不含流程裡額外的步驟）：

```python
from line2func import attributes, baseline, lineart, shapes
from line2func.export import to_desmos

rgb = lineart.load_rgb("drawing.png")
ink = lineart.extract(rgb, "none")        # 浮點數墨跡圖，1 = 線條
curves = baseline.vectorize(ink)          # CurveSet
attributes.refine(curves, ink)            # 貼齊墨跡中心線
attributes.measure(curves, ink, rgb)      # 量測每條曲線的線寬與顏色
shapes.recognize(curves)                  # 標出直線與圓弧
print(to_desmos(curves, named=True))
```

使用神經網路引擎：

```python
from line2func.model.infer import load_vectorizer
run = load_vectorizer("runs/m3/best.pt")   # 有 GPU 時會自動使用
curves = run(ink)
```

其他可用的模組：`line2func.geometry`（求值、分割、轉折線、弧長、外框、最近點）、`line2func.render`（反鋸齒渲染器）、`line2func.fit.fit_polyline`（Schneider 擬合）、`line2func.metrics`（F 分數、chamfer 距離、結構指標）。

### 12. 專案結構與測試

```
line2func/
  app.py  browser.py  __main__.py          # 網頁版的伺服器（python -m line2func）、瀏覽器啟動器
  viewer/                                  # 網頁：index.html、app.js、viewer.js、i18n.js、i18n.json
  demo.py  serve.py  pipeline.py           # 指令，以及它們共用的描線流程
  lineart.py  lineart_model.py  weights.py # 抽線稿、預訓練模型、權重下載
  baseline.py  fit.py                      # 傳統引擎、Schneider 擬合
  attributes.py  shapes.py  export.py      # 精修、線寬顏色、直線圓弧、匯出
  functions.py                             # 把曲線寫成函數 y = f(x)／x = g(y)（--form function）
  residual.py  outline.py  fill.py         # 第二遍描線、粗筆畫外框、Desmos 用的圈線
  budget.py  optimize.py  quality.py       # 指定曲線數量、渲染後比對、品質檢查
  decisions.py  decision_features.py      # 描線決策的評分介面、候選特徵
  decision_model.py  data/                # 學習式評分器（numpy）與附帶的權重
  labels.py  decisions_data.py  train_decisions.py   # 標準答案標籤與神諭、訓練資料、訓練
  geometry.py  curves.py  render.py        # 幾何核心、資料結構、渲染器
  synth.py  metrics.py  eval.py  blindtest.py        # 合成資料、指標、評估、盲測
  train.py  model/                         # 網路、資料、損失函數、分塊推論
configs/        # 訓練設定檔
docs/           # details.md（本說明）、third_party.md（第三方程式碼與權重的授權）
tests/          # pytest 測試
```

```bash
python -m pytest            # 約 370 個測試；沒有 PyTorch 時略過模型測試，沒有 Node.js 時略過檢視器 JS 檢查
```

產生的資料夾（`out/`、`runs/`、`data/`、`.venv/`）已列在 `.gitignore`。

### 13. 現況與路線圖

這是 1.0 版。建議使用傳統引擎；神經網路引擎是實驗功能。

- [x] 幾何核心與渲染器
- [x] 傳統引擎；SVG、Desmos、LaTeX 匯出；檢視器；`demo` 與 `serve`
- [x] 合成資料產生器；單曲線與多曲線神經網路模型；整張圖推論
- [x] 照片用的預訓練線稿模型（Informative Drawings，MIT；下載時檢查 SHA-256）
- [x] 直線與圓弧的具名算式、曲線精修、線寬與顏色
- [x] 網頁版（`python -m line2func`），繁體中文與英文介面
- [x] 第二遍描線、粗筆畫改為填滿的外框、渲染後比對（`--optimize`）
- [x] 學習式決策（預設；`--decisions rules` 改用角度規則）
- [x] 粗重睫毛等實心區域，並加上 Desmos 用的圈線；預設最多 5,000 條曲線；淺色與極淡的線
- [x] 函數模式：每條曲線切成 `y = f(x)`／`x = g(y)` 的顯函數（`--form function`）
- [x] 單一畫面的網頁版，三種顯示方式；斷成點和虛線的線會保留；去雜訊強度拖動條（`--denoise`）
- [x] 淡線靈敏度（`--faint-sensitivity`）
- [ ] 真實線稿的盲測（工具已完成，需要人工評分者）
- [ ] 選用：以這些線條風格重新訓練神經網路引擎，並在真實線稿上微調
