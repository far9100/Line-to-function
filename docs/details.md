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
9. [Evaluation](#9-evaluation)
10. [Using line2func from Python](#10-using-line2func-from-python)
11. [Project layout and tests](#11-project-layout-and-tests)
12. [Status and roadmap](#12-status-and-roadmap)

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

#### Optional: PyTorch (pretrained photo line art, `--optimize`)

Install a PyTorch build that matches your GPU **first**, then the extras:

```bash
# NVIDIA RTX 50-series (Blackwell) needs a CUDA 12.8+ build, e.g. CUDA 13.0:
pip install torch --index-url https://download.pytorch.org/whl/cu130
# (CPU only: pip install torch)

pip install -e ".[torch,dev]"     # adds pytest
```

Verified setup: torch 2.14.0+cu130, Python 3.14, RTX 5070 (driver 596.21),
Windows 11.

### 2. Trace your first drawing

#### Online, without installing

<https://far9100.github.io/Line-to-function/> is the web page below, with
line2func running in your browser through [Pyodide](https://pyodide.org)
(Python compiled to WebAssembly). Nothing is installed and the image never
leaves your computer. Pyodide with numpy, SciPy and Pillow, about 25 MB, comes
from the jsDelivr CDN on the first visit; the browser keeps it for later.

- It traces like `python -m line2func` (up to 5,000 curves, with the quality
  check), within tighter limits for a browser tab: files up to 32 MB, images
  kept at up to 2048 px, traced at up to 2.5 megapixels (larger ones are
  shrunk first) and quality-checked at up to 2.0 (above that the check is
  skipped, and the page says so).
- It takes 1.5 to 2 times as long as installed. Five real drawings of about
  0.6 megapixels took 17–56 s each (Edge or Node.js on a Ryzen 7 9700X,
  including the quality check), against 11–30 s installed, and used up to
  about 460 MB of memory. A photograph at the full 2.5 megapixels took about
  11 s, line extraction included, and 487 MB. Slower computers and phones take
  longer, and a phone may run out of memory on a large picture.
- WebAssembly rounds some calculations slightly differently, so a curve or two
  can come out differently from the installed version; the share of lines kept
  is the same.
- **Cancel** stops at once. There is no **Quit**: close the tab.
- It needs Chrome, Edge or Firefox 112 or newer, or Safari 16.4 or newer. The
  settings and the language are kept in the browser.

Photos work here too (see below). The pretrained line-art network and the
other options (sections 6 and 7) need the installed version.

#### In the browser (drag and drop)

```bash
python -m line2func          # or just: line2func   (after pip install -e .)
```

The page opens in a new tab of your default browser. Then:

1. **Drop an image** onto the page. You can also click **Choose a
   file…** or paste with Ctrl+V. PNG, JPEG, WebP, BMP, TIFF and GIF are
   accepted, up to 64 MB. Photos from phones are turned upright.
2. The drawing appears in place. Choose how to write the lines, as
   **functions** `y = f(x)`, `x = g(y)` (section 5) or as **parametric
   equations** `x(t), y(t)`, and how strongly noise is removed (**Remove
   noise**, strength 0-100: lower keeps more detail, higher cleans noisy scans;
   section 8) and how light a line may be (**Faint lines**, 0-100: higher keeps
   lighter strands of hair and background), and click **Convert**. The page
   starts on parametric equations and filters nothing - **Remove noise** off
   with its slider at 0, **Faint lines** at 100 - so a first run shows every
   line it found and the sliders are there to take some away; `demo` keeps its
   own defaults of 50 for both. It is traced like `demo`
   traces it: up to 5,000 curves, with the quality check (images larger than
   2048 px are shrunk first). A progress panel shows each step, and **Cancel**
   stops at the next step.
3. The result opens in the same page, the viewer (section 4). From there you
   can download SVG, JSON, Desmos, LaTeX or a ZIP of everything, or
   **Copy all for Desmos**.
4. **Clear image** at the top removes the image, so the next one can be
   dropped. You can also drop another image at any time.

The web page takes photos as well as drawings. A photo is recognized as it is
opened and the lines in it are found first (`flow`, section 6); the page shows
you that line art, with **Line-art detail** to adjust it and a pair of chips to
compare it with the original, and traces it only when you convert. **Shade dark
areas**, if you tick it, also draws how dark the picture is, as hatching
(section 8). The
pretrained network and the other options are on the command line (sections 6
and 7).

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
sample_lineart.png: 256x256, 136 curves in 10 strokes, 0.25 s (3.88 s/MP); 132 recognized as lines or arcs [faint strokes on, all curves of the fine tracing, fewer than the 5000 allowed]
  out\curves.json
  out\out.svg
  out\desmos.txt
  out\desmos.js
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
| `desmos.js` | The same expressions, each carrying its measured line width and color, for the Desmos API, section 5 |
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

**Filled areas.** Areas of flat ink, large ones and heavy strokes such as thick
eyelashes, are traced by their outline only (tagged `fill_outline`; the outline
of a thick or strongly tapered stroke is tagged `outline`). Each one carries how
dark it is, as `tone` (0 paper, 1 black). **Nothing is ever filled.** An area is
drawn with curves tagged `fill` across it, spaced by one formula for every area
(section 8): rings where it is deeper than one spacing, 45 degree hatching where
it is not. Desmos cannot fill a pasted expression, so this is the only way tone
reaches it - and the page and the SVG draw the same curves, so all three outputs
show the same drawing. Its outline is stroked in the color measured inside it,
and closed, so the curves inside land in a shape.

Its boundary is traced with Moore-neighbour contours
(:func:`line2func.boundary.contours`), not by thinning the edge and walking it as
a skeleton. A boundary has to come back closed: the walk splits a ring wherever
two rings touch, and an open chain is then closed by a straight chord when it is
filled, inventing the area between the two. On `lineArt (5).jpg` that invented
129,399 px against a true filled area of 87,337 - it nearly doubled it - and on
`lineArt (11).jpg` it painted over the blank upper half of a plum-blossom pupil.
With contours every loop closes, and the invented area falls to 2,050 px.

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
| Display | **Lines only**, **Lines + original** (where it starts, the original at 50% opacity), or **Lines + missed detail** (after a quality check: the curves in gray, missed lines red, missed faint ink orange, curves without ink blue, the map at 90%), with the background's opacity slider |
| The original alone | **Original**: shows only the original image |
| Line color | **Black & white** (where it starts), **Colors** (eight hues by stroke) or **Random colors**, with **Re-roll** for a fresh set of random hues. A downloaded SVG uses the same colors and thickness; the other files and the ZIP are unchanged. Over **Lines + missed detail** the curves stay gray so its marks still stand out |
| Line width | **Measured width** (the default: every stroke as thick as the ink it was traced from) or **Uniform width** (one thickness, the result's own `line_width`, for every line). The canvas draws the measured widths too, so it shows what the SVG will contain; a thin stroke is never drawn narrower than 0.75 screen pixels, so it stays visible when zoomed out |
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

#### Colors and line widths: `desmos.js`

A pasted expression is drawn by Desmos as a line of one fixed width and one
color, so `desmos.txt` can only say how dark an area is by how densely it is
drawn (section 8, *Shadows, heavy eyelashes and other filled areas*). The
Desmos API takes a `color` and a `lineWidth` per expression, and `desmos.js` is
the same expressions written with them - the width and color measured along
each stroke, and for the curves inside a filled area, the area's own gray.

Its math is byte for byte what `desmos.txt` holds; only the styling is added.
It is a JavaScript file that ends in a `setExpressions` call, so:

- in a page that embeds the [Desmos API](https://www.desmos.com/api), run it or
  call `calculator.setExpressions(LINE2FUNC)` yourself;
- with a calculator open, pasting the whole file into the browser console does
  the same, where the page exposes a calculator to it. Desmos does not document
  that, so it may or may not work for you - the API route always does.

The curves inside an area are drawn as wide as they are *spaced*, so they meet
instead of leaving paper between them and the area comes out a solid patch of
its measured gray. The spacing already says the tone once; drawing them at
their own spacing takes it back out, and the color says it instead. Measured
against the two JPEG drawings used through this page, as the local average
tone over a 12 px window, against the shaded part of the original:

| | mean abs. tone error | more than 0.05 too dark |
|---|---|---|
| lineArt (11), `desmos.txt` | 0.176 | 79.6% |
| lineArt (11), `desmos.js` | **0.052** | **3.2%** |
| lineArt (9), `desmos.txt` | 0.164 | 77.4% |
| lineArt (9), `desmos.js` | **0.041** | **6.3%** |

Two caveats. `lineWidth` is in screen pixels while the widths are measured in
image pixels, so they read as measured with the drawing at its own size on
screen, and thicken or thin as you zoom. And what is left of the error is now
mostly on the light side: a stroke measured thinner than 0.5 px is drawn at
0.5 px so it does not disappear, but a faint JPEG line drawn thin and pale is
fainter than the original.

The page's own line color and width choices apply to this file too, so what you
download is what the page is showing.

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

Photos need a line-extraction step first (`--lineart`). Both web pages do it by
themselves for anything `lineart.suggest_mode` calls a photo, and show you the
result before tracing it; on the command line it is a flag.

| Method | Needs | Result |
|---|---|---|
| `flow` | nothing | Coherent line drawing: long, connected lines. **The default for photos, and the best one that needs nothing downloaded** |
| `informative` | PyTorch + weights | Pretrained line-art network: drawing-like lines. **Best of all, where PyTorch is installed** |
| `informative-coarse` | PyTorch + weights | Same network, bolder and simpler lines |
| `canny` | nothing | Edge detection; thick lines get an edge on each side |
| `xdog` | nothing | Stylized edges; works on high-contrast images |
| `none` (default) | nothing | The input already is line art |

**Why `flow`, and why it is the one in the browser.** A tracer does not want
pretty edges, it wants *strokes*: it binarizes, thins to a skeleton, walks that
skeleton into a graph and fits curves to the runs it finds. Every disconnected
fragment is either thrown away by the speck filter or costs a curve of its own,
which is why the other image-to-Desmos tools say photos give them thousands of
equations. `flow` is a difference of Gaussians taken **across** the edge tangent
flow and then smoothed **along** it (Kang, Lee & Chui, NPAR 2007): a speck of
noise has no line to agree with it and is averaged away, while a real line is
reinforced from both ends. Measured on a photo-like picture (a drawing blurred,
unevenly lit and given noise), with the ink thinned and its connected pieces
counted:

| Method | seconds | skeleton px | pieces | pieces per 1,000 px | line width |
|---|---|---|---|---|---|
| `flow` | 0.85 | 17,163 | 558 | **32.5** | 2.40 |
| `xdog` | 0.02 | 10,462 | 1,983 | 189.5 | 1.78 |
| `canny` | 0.06 | 49,495 | 755 | 15.3 | 1.12 |

and with the noise raised from 0.02 to 0.06, which is an ordinary photograph:

| Method | skeleton px | pieces | pieces per 1,000 px | change |
|---|---|---|---|---|
| `flow` | 18,169 | 633 | **34.8** | +7% |
| `xdog` | 19,393 | 8,248 | 425.3 | +124% |
| `canny` | 48,502 | 806 | 16.6 | +9% |

That is a controlled experiment on one picture. On **32 real photographs**
(section 9), paired photograph by photograph with a 95% bootstrap interval:

| flow minus | pieces per 1,000 px | skeleton px | line width | extraction |
|---|---|---|---|---|
| `xdog` | **−132.7** [−174.6, −96.8] | **+3,362** [+1,368, +7,845] | **+0.55 px** [+0.47, +0.67] | +2.70 s |
| `canny` | +34.4 [+26.6, +46.7] | **−23,270** [−37,860, −11,170] | +1.13 px [+1.01, +1.22] | +2.58 s |

Every interval misses zero. Against `xdog`, `flow` hands the tracer lines that
are a third as broken up (median 57 pieces per 1,000 skeleton pixels against
182) and it finds *more* line while doing it, not less. Against `canny` it has
more pieces per 1,000 pixels and that is the wrong way to read the column:
`canny` has **1.8x the skeleton** for the same photographs, because every line
gets an edge on each side, so it has more pixels to divide by and twice as much
to trace.

The last column of that comparison decides something else. Of the 32
photographs, the lines came out thinner than `pipeline.AUTO_UPSCALE_BELOW`
(1.75 px) in **30 of them for `xdog` and 32 of 32 for `canny`, against 1 of 32
for `flow`**. Before the rule that closes this section, that asked for the 2x
upscale on nearly
every photograph anyone would open - four times the tracing work, to interpolate
a decision the extractor had already made.

**What this does not buy is a smaller result.** Traced the way `demo` traces -
`--curves 5000`, `auto` upscaling - two photographs from each group of the
corpus came out like this:

| Method | median curves | median seconds | reached the budget |
|---|---|---|---|
| `flow` | 5,000 | 23 s | 6 of 10 |
| `xdog` | 4,348 | 9 s | 5 of 10 |
| `canny` | 5,000 | 16 s | 6 of 10 |

A photograph fills the budget whichever method found its lines, and `flow` is
the slowest of the three. What differs is **what those 5,000 curves are**:
`flow` spends them on long strokes (57 pieces per 1,000 skeleton pixels) and
`xdog` on dust (182). On one portrait `xdog`'s whole extraction was 8,650
skeleton pixels in about 3,800 pieces - an average of 2.3 pixels each, which
the speck filter is right to throw away. Being quick about finding nothing is
not an advantage; it is why `flow` is the default and `xdog` is still offered
for a slow machine or a very large picture.

Two properties of `flow` matter as much as the coherence. A flat area gives
*exactly* zero ink, and so does a smooth gradient - a difference of Gaussians is
zero on anything linear, and what is left has the wrong sign to be ink - so a
sky or a cheek contributes nothing to the curve budget. And its lines are wide
enough to trace as they are: a median of 2.2 px on the 32 photographs, and
under `pipeline.AUTO_UPSCALE_BELOW` on one of them.

**What `flow` reads, since 2.0.0.** The tables above were measured with the
extractor as 2.0.0 shipped it, which read the picture in gray and held every
picture to one bar. It has changed in three ways since, each for a fault that
showed on a painted illustration - nine figures on a night sky, where the faces
came out blank and the sky full of specks:

- **Colour.** Two colours of the same brightness - hair against sky, cloth on
  cloth - are no edge at all in gray. At each pixel the extractor now finds the
  direction in RGB the picture changes along and reads the ridge along that, so
  a red|blue boundary is as much of a step as black|white. A gray picture is
  read as its one channel and pays nothing for this.
- **One bar everywhere.** Kang's formulation leaves `(1 - tau) x gray` as the
  bar, which is five times higher on a light face than in a dark sky; and
  drawing the lines found in black between rounds reinforced a line on a light
  ground four times as hard as the same line on a dark one. The bar is now a
  plain `eps`, and lines are drawn a fixed step darker than what is there.
- **The picture's own noise sets how far the bar rises.** The noise is read
  off the picture (0.010, 0.030 and 0.058 for 0.01, 0.03 and 0.06 put in) and a
  ridge must clear four standard deviations of what that noise becomes after
  the two passes. A clean illustration is read at the low bar its faint edges
  need, a grainy photograph at one its grain cannot reach.

On the illustration, at the default detail: **109,293 skeleton pixels in 19.9
pieces per 1,000, against 57,103 in 37.5** - nearly twice the line, half as
broken. On drawings given camera blur, uneven light and noise, where the true
lines are known:

| noise | lines found | false line | pieces per 1,000 px |
|---|---|---|---|
| 0.01 | 99.3% (was 97.5%) | 0.5% (was 4.1%) | 1.2 (was 4.6) |
| 0.03 | 99.3% (was 97.5%) | 0.6% (was 3.6%) | 1.2 (was 5.8) |
| 0.06 | 99.4% (was 97.2%) | 2.4% (was 8.0%) | 10.2 (was 24.7) |

On eight photographs at 2.5 MP it gave fewer pieces per 1,000 pixels on every
one that has lines in it (2.6 to 40.3, against 3.9 to 58.1), more line on each,
and lines 2.3 to 2.8 px wide. **What it costs is time**: 4.1 s a photograph
against 3.0 s. **What has not been done** is to repeat the 32-photograph
comparison above with it; those figures stand for the earlier extractor.

The pretrained weights are never bundled. Download them once; they are checkedThe pretrained weights are never bundled. Download them once; they are checked
against a pinned SHA-256 and stored in `~/.cache/line2func` (override with
`LINE2FUNC_HOME`):

```bash
python -m line2func.weights list            # what is available, licenses, status
python -m line2func.weights fetch informative   # or: fetch all
python -m line2func.weights verify          # re-check hashes

python -m line2func.demo photo.jpg --lineart informative --out out/
```

Photos usually give many short curves. `demo` merges them down to 5,000
(section 8).

**An extractor's ink map is not enlarged.** `--upscale auto` doubles the
resolution of a *drawing* whose own lines are thinner than one pixel can carry;
an extractor's output is already the lines it decided on, at the picture's own
resolution, so enlarging it only interpolates that decision at four times the
work. `auto` therefore stays at 1x for every `--lineart` method other than
`none`; `--upscale 2` still does what it says.

### 7. All `demo` options

```
python -m line2func.demo IMAGE [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | `out` | Output folder |
| `--lineart {none,canny,xdog,flow,informative,informative-coarse}` | `none` | Line extraction (section 6) |
| `--lineart-detail 0..100` | `50` | With `--lineart flow`: how much of the picture becomes lines (section 8) |
| `--shade` | off | For photos and paintings: also draw how dark the picture is, as hatching (section 8) |
| `--curves N` | `5000` | Make N curves: trace finely, then merge the neighbouring pieces whose merge changes the drawing least (section 8). A drawing that gives fewer keeps all of them |
| `--tolerance PX` | off | Trace to this max curve-fitting error instead of a number of curves. Larger gives fewer, smoother curves |
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
| `--no-fill` | on | Leave filled areas hollow: only their outline, no rings or hatching inside |
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
  tapered (a brush tip) become closed outlines, with rings or hatching inside
  them at their own tone (below). Nothing is filled, in any output.
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
broken lines. Unticking **Remove noise** is 0, which is where the page starts.
On drawing 4 and on 100 synthetic
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
together: 50 is as tuned and the page starts at 100, which needs less than
half the contrast (0.12 instead
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

#### How much of a photo becomes a line: `--lineart-detail`

With `--lineart flow`, one slider from 0 to 100 decides how weak a ridge still
counts as a line. It moves three settings together, because they are three
views of that one decision: the width of the kernel taken across the flow
(`sigma_e`), how far the answer is smoothed along it (`sigma_m`), and the bar a
ridge has to clear (`eps`, on gray in 0..1). Measured on a 2.07 MP illustration
(1920 x 1080, nine figures on a night sky):

| detail | `sigma_e` | `sigma_m` | `eps` | extract | ink | skeleton px | pieces/1,000 px | line width |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.80 | 3.80 | 0.0300 | 4.42 s | 6.8% | 35,918 | 26.4 | 3.42 px |
| 25 | 1.40 | 3.20 | 0.0134 | 3.78 s | 14.0% | 78,680 | 17.8 | 3.19 px |
| **50** | **1.00** | **2.60** | **0.0060** | **3.22 s** | **15.9%** | **109,293** | **19.9** | **2.62 px** |
| 75 | 0.90 | 2.40 | 0.0035 | 2.81 s | 16.6% | 124,720 | 21.1 | 2.39 px |
| 100 | 0.80 | 2.20 | 0.0020 | 2.66 s | 17.4% | 138,444 | 22.8 | 2.26 px |

Three things in that table are worth knowing before turning the slider.

**More detail never finds less line.** That is the promise the control makes,
and it is what the endpoints were chosen for. `eps` is moved *in proportion*
rather than linearly, because it spans more than an order of magnitude across
the slider; interpolated straight, the fine half would barely move it while the
two sigmas fell quickly.

**The slowest setting is 0, not 100.** Higher detail means narrower kernels and
so fewer samples per pixel: extraction falls from 4.42 s to 2.66 s across the
range. What goes up is the tracing, because there is more line to trace.

**The top end is where the curve budget goes.** At 100 there is a quarter more
line than at the tuned setting and it is a little more broken (22.8 pieces per
1,000 skeleton pixels against 19.9), because what is coming in is texture
rather than strokes. This picture reaches the 5,000-curve limit at 50 already.

**`eps` is the bar on a clean picture; a noisy one raises it by itself.** See
*What `flow` reads* in section 6.

#### How dark a photo is: `--shade`

Line art says where a picture's edges are. A photo or a painting is mostly
*tone* - a night sky, a black coat, dark hair - and none of that is an edge.
`--shade` draws it as well: straight 45 degree hatching, closer together the
darker the picture is under it. On the web page it is the **Shade dark areas**
box, shown for a photo and left for you to tick.

- **Spacing is the tone.** A tone asks for a spacing of `closest x dark / tone`
  - half as dark, twice as far apart - where `dark` is the tone of the
  picture's darkest 1% (never taken as lighter than 0.5, so a pale picture is
  not hatched as if its darkest gray were black). Parallel straight lines cannot
  change their spacing gradually, so it is rounded to one of three: `closest`,
  twice that and four times that. Anything lighter stays paper. The wider sets
  are subsets of the closer ones, so a line that runs from a shadow into a
  mid-tone simply carries on.
- **It is paid for out of `--curves`.** The hatching may take up to 30% of the
  count, and `closest` is the tightest of 2.5, 3, 4, 5, 6, 8, 10, 12 and 16 px
  that fits; the lines are merged down to what is left. Without `--curves` it
  has 2,000 curves of its own.
- **Each hatch line is one straight segment**, tagged `shade`, 1 px wide, and
  carrying the tone it stands for as its colour; the SVG and `desmos.js` draw
  it in that gray, so it reads lighter than the outlines. The page draws in the
  line colour you chose, and `desmos.txt` pasted into Desmos is one colour, so
  there the tone is in the spacing alone. As a function each one is
  `y = x + c` over a range.
- **The quality check leaves it out.** It judges how the ink was traced, and
  hatching traces no ink.

On the 2.07 MP illustration of the table above, with `--curves 5000`:

| share of the count | hatching | closest spacing | lines | lines kept | largest merge error |
|---|---|---|---|---|---|
| none | - | - | 5,000 | 99.3% | - |
| **30%** | **1,445 curves** | **6 px** | **3,555** | **97.5%** | **2.9 px** |
| 40% | 1,904 curves | 5 px | 3,096 | 91.2% | 6.2 px |

That is why the share is 30%: at 40% the outlines are merged far enough to show
it. A picture with large dark areas costs its outlines something either way.

#### Shadows, heavy eyelashes and other filled areas

Thick black strokes such as heavy eyelashes used to thin down to a tangle of
short centerlines, and Desmos drew them as thin lines. Ink that is at least
2.5 line widths thick over some length, and *flat* in its middle, is traced as
one filled area instead. Two lines drawn so close that their ink merges do not
count: their ink is lighter in between, so they stay lines. Flatness is what
separates a wash of tone from a cluster of strokes, and it cannot be loosened -
at 0.7 x the darkest ink nearby instead of 0.8, the areas on `lineArt (11)` of
the JPEG set go from 1.1% of the page to 3.5% as stroke clusters start to pass.

**How far the area reaches.** The thick part is found as the disks that fit
inside the ink, and those disks are then grown back by their own radius, so on
its own the area takes in every piece of ink within that radius - however thin
it is. A hair strand crossing a heavy eyelash is annexed, and the paper between
the two ends up inside the area's outline. The area therefore only grows along
ink that stays at least half the seed radius deep (`solid_depth`), which follows
the lash down its own taper and stops at a strand a third as thick. On a chibi
line drawing whose bangs cross both eyes, that takes the traced lash from 1.25
and 1.19 times the drawing's own black down to 1.01 and 0.95, and the share of
each area that is ink under 4 px wide - the annexed hair - from 25% to about
12%. Where the ink threshold was lowered below Otsu's, the area is judged on the
strong ink and the fainter rim around it is added back afterwards; that rim is
one pixel (`solid_rim`), because two reached past the soft edge and into the
next stroke (the same two lashes: 1.25 and 1.19 times the black, against 1.17
and 1.10 at one pixel).

**Every filled area is measured for how dark it is**, and is drawn at that tone.
Nothing fills it: Desmos cannot fill a pasted expression and draws every line at
one darkness, so the tone becomes line density there, and the SVG and the page
draw the same curves Desmos gets, so all three show the same drawing. A shadow
gets no outline drawn around it either - its edge is where the tone fades out,
not a line anyone drew.

An area's outline has ink on one side only, so no width is measured along it,
and the page, the SVG and `desmos.js` used to fall back to the drawing's own
line width - which is measured at the ink threshold, soft edges included, and so
is wider than most of the drawing's strokes (3.43 px against a measured median
of 2.36 px on the chibi drawing above). Stroked on the area's edge, half of that
lands outside it. The outline is its edge, not a line anyone drew, so it is
stroked at 1 px, just wide enough to close the area, and the curves inside carry
the tone - the width a thick stroke's own outline has been written at all along
(`line2func.outline`). On the two lashes that took the drawn area from 1.60 and
1.57 times the drawing's black to 1.26 and 1.24; with the reach above as well,
to 1.02 and 0.96, and the drawn half width from 8.25 px and 7.28 px to 5.39 px
and 5.00 px, where the ink's own are 4.12 px and 3.61 px.

Over the 81 real drawings, paired: **lines kept and missed ink do not move at all** (median
difference 0, interval [0, 0]), and `precision`, `d_M`, PSNR, SSIM and the curve count do not
reach significance. The cost is 0.4% more curve length per unit of skeleton - a smaller area
hands its rim back to the line tracer - and 0.1 s on a 7.3 s median.

One formula sets the spacing for every area, with no step anywhere in it:

```
spacing = clip(0.88 x 2.5 px x dark / tone, 1.5 px, 12 px)
```

A curve inks 2.5 px of every `spacing`, so the share of the area Desmos inks is
the share of the drawing's dark ink that its own ink is. The 0.88 is measured,
not chosen: curves tiled at exactly their own width cover 92% of an area rather
than 100%, because rings curve, and without that overlap the darkest areas come
out with paper showing through them.

*What* is drawn at that spacing is chosen by the area's shape, not by its tone:
**rings** (contour lines of the distance to its edge) where the area is deeper
than one spacing, and **45 degree hatching** where it is not. A ring at depth
*k* x spacing only exists where the area is deeper than that, so rings alone
leave a shadow along a jaw or a finger with nothing in it - 23 of the 59 areas
on `lineArt (9)`, 12% of the shaded pixels, got no ring at all. A hatch line
crosses an area however thin it is. An area is drawn all the way in, as many
rings as it is deep, which costs 66 curves on `lineArt (11)` and takes its solid
areas from 95.1% covered to 99.8%.

An area used to have to be as dark as the drawing's own dark ink (the 90th
percentile) to count as filled at all. That bar is relative, so on a light
pencil drawing with no true black it collapses: `lineArt (9)` measures an ink
p90 of 0.576, which put the bar at 0.46 and promoted every mid-gray shadow to
solid ink - and then every shadow in the drawing was drawn equally black, which
is what this replaced.

| | Curves | PSNR | SSIM |
|---|---|---|---|
| `lineArt (9).jpg` (soft gray shading) | 2,066 -> 2,062 | 22.52 -> **22.88 dB** | 0.781 -> 0.776 |
| `lineArt (11).jpg` (dense pencil, black eyes) | 3,585 -> **3,111** | 19.00 -> **19.71 dB** | 0.670 -> 0.668 |
| `lineArt (7).jpg` (little real shading) | 1,253 -> 1,347 | 17.96 -> 17.92 dB | 0.582 -> 0.578 |
| `lineArt (5).jpg` (real black) | 8,072 -> 8,039 | 8.32 -> **8.82 dB** | 0.631 -> 0.629 |
| `lineArt (3).jpg` (pure line art) | 1,432 -> 1,432 | 22.10 -> 22.10 dB | identical |

Filled areas' measured tone on `lineArt (9)` runs 0.30 / 0.40 / 0.60
(p10/p50/p90) where every one of them used to be drawn solid black. Pure line
art is untouched, curve for curve, and a drawing that does have black keeps its
solid areas (`lineArt (5)`, ink p90 1.00: its areas measure 0.81-0.91).

Over the 66 drawings of `data/real_v1`, paired: **recall and missed ink do not
move at all** (median difference 0, interval [0, 0]), `precision` and `d_M` do
not reach significance, and the cost is +18 curves (0.7% of the median drawing)
and +0.13 s. The eyelash benchmark drawing improves on every measure.

Drawing 1 at tolerance 1.0:

| | Before | Now |
|---|---|---|
| Dark eyelash ink that Desmos draws (every curve 2.5 px wide, whole drawing on screen) | 89.6% | **97.0%** |
| Lines kept / d_M | 99.65% / 0.527 px | 99.66% / 0.531 px |
| PSNR / SSIM | 21.9 dB / 0.922 | **22.1 dB / 0.923** |
| Curves | 1,078 | 1,111 (35 of them rings) |

A lash that dark still gets rings, so those numbers stand. On a light sketch
whose close double lines merge in many places, only 2 small areas qualify.
Zoomed in far in Desmos, the rings show as rings; with the whole drawing on
screen they merge into a solid area.

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

#### Why there is no learned scorer

Where strokes continue through a junction, which breaks are one line, which
close junction pairs are one shallow crossing and where a stroke turns sharply
enough to split are decided by the angle rules (`line2func/decisions.py`).

There used to be a learned scorer here as well, on by default: five small MLPs
run with numpy, trained on generated scenes. It was removed in 1.2. It was
better, and the difference could not be seen.

Paired over the 66 drawings of `data/real_v1`, bootstrapped, learned against
the rules it replaced:

| | median | 95% interval |
|---|---|---|
| Curve length on ink (precision) | +0.0044 | [+0.0040, +0.0058] |
| PSNR | +0.175 dB | [+0.13, +0.20] |
| SSIM | +0.0030 | [+0.0026, +0.0032] |
| **Seconds per drawing** | **+1.76** | **[+1.49, +1.945]** |

Replicated on 15 held-out JPEG drawings at +0.0029 precision and +0.11 dB.
Missed ink, faint-line coverage, drawn length and `d_M` all had intervals
containing zero: it did not find more of the drawing, it placed what it found
slightly better.

**Slightly enough not to see.** Rendering the same drawing both ways and
comparing pixel by pixel, 89-93% of the difference is the same stroke drawn
within 2 px of itself. What is left is 7-25 places per drawing where one engine
put ink and the other did not, the largest of them 49 px - a 20 px stretch of
line, inside dense hair texture. Around 7-9% of all decisions flipped, several
hundred per drawing, to move precision by 0.4 of a percentage point.

Against that: 24% of the package, a fifth of the tracing time, and 52% of the
browser download (the zip went 298,548 -> 142,807 bytes when it went).

**Widening the rules instead does not work, and that is measured too.** The
obvious cheap substitute is to let the rules consider more candidates: two
wider variants (`--decisions r2` and `r2-gaps`, removed along with the scorer)
did that, and on synthetic scenes they looked
strong (gap closure 0.854 against the rules' 0.774, two thirds of the way to
the scorer's 0.898). On real drawings they were **worse than plain rules**:
precision -0.0014 [-0.0022, -0.0007] over the 66 and -0.0009 over the 15. They
drew 5.1% more curve length, bridging real breaks and imaginary ones together,
and the imaginary ones cost more than the real ones gained. So removing the
scorer means the rules, not a wider version of them.

The same caution applies to the synthetic numbers the scorer won on: the
project measured that they do not transfer. Gap closure 0.774 -> 0.898 and
corner precision 0.525 -> 0.624 are real on generated pages and did not reach
the output.

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

### 9. Evaluation

Preview the synthetic data the metrics below are measured on:

```bash
python -m line2func.synth preview --kind hard --out preview.png
python -m line2func.synth preview --kind hard --patches --out patches.png
```

The tracer is scored on synthetic drawings with ground truth, in a "clean"
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

```bash
# a fixed synthetic validation set (100 clean + 100 hard scenes, identical on every machine)
python -m line2func.synth valset --out data/val_v1

# the scores: F_GT@2, crossing continuity, gap closure, fragments/stroke, curve ratio, s/MP
python -m line2func.eval --valset data/val_v1
```

Add `--json results.json` to save the numbers, and `--limit N` for a quick run.

Real drawings have no ground truth, so `--realset FOLDER` judges them against
their own ink instead (`line2func.quality`), and `--compare` puts two such runs
side by side as the median per-drawing difference with a bootstrap interval:

```bash
python -m line2func.eval --realset path/to/drawings --json runs/a.json
python -m line2func.eval --realset path/to/drawings --set local_trim=false --json runs/b.json
python -m line2func.eval --compare runs/b.json runs/a.json
```

`--lineart METHOD` (with `--lineart-detail N`) runs the set through a line
extractor first, which is how a folder of **photographs** is measured:

```bash
python -m line2func.eval --realset path/to/photos --lineart flow --json runs/photo_flow.json
python -m line2func.eval --realset path/to/photos --lineart xdog --json runs/photo_xdog.json
python -m line2func.eval --compare runs/photo_xdog.json runs/photo_flow.json
```

**Most of the columns cannot be compared between two extractors, and the report
says so when one is used.** The judge scores the curves against the ink map that
arm's own extractor produced - the threshold comes from that same ink - so
`kept`, `precision`, `missed_ink`, `d_M`, `psnr_db` and `ssim` all answer "did
we draw our own extraction faithfully". An extractor that finds almost nothing
scores beautifully on every one of them. Three columns do survive the
comparison, because they do not divide by what the arm chose to find:

| Measure | What it says |
|---|---|
| `curves` | Whether the budget is spent (section 8) |
| `seconds` | What it cost |
| `pieces_per_kpx` | Connected pieces per 1,000 skeleton pixels of the ink: how broken up the lines handed to the tracer are. This is the one that separates a method that draws strokes from one that shatters (section 6) |

The 66 drawings and the 15 held-out JPEGs that the measurements in this manual
were taken on are third-party line art. They are kept outside the repository
and are not published with it, so `data/real_v1` below names where they were,
not a folder you will find in a checkout. `--realset` takes any folder.

The photograph measurements in section 6 were taken on a separate set of **32
CC0 photographs** from Wikimedia Commons, every file's licence checked against
the API rather than assumed from its category, scaled to the 2.5 megapixels the
online page traces, and spread deliberately over the cases the photo path has to
survive: 8 portraits (smooth tone must give no lines at all), 9 of foliage and
grass (the budget-flooding case), 5 of architecture (hard edges, where the older
methods should compete), 5 of streets at night (grain, where a plain difference
of Gaussians shatters) and 5 plain objects. They are third-party pictures and
are kept outside the repository too, with a `sources.json` recording each file's
page on Commons. The line drawings stay the **regression** set for that work:
the line-art path must not move, and over all 81 of them it did not - every
measure zero, with a zero-width interval.

### 10. Using line2func from Python

The whole flow in one call, as `demo` and the web page run it (the second pass,
outlines and the curves that fill them in Desmos are on by default; both ask for
up to 5,000 curves; `optimize=True` needs PyTorch):

```python
from line2func import functions, lineart, pipeline
from line2func.export import write_outputs

rgb = lineart.load_rgb("drawing.png")
curves, ink = pipeline.trace(rgb, upscale="auto", curve_count=5000)   # up to 5,000 curves, as demo
curves, ink = pipeline.trace(rgb, upscale="auto", fit_tolerance=1.0)  # a fitting tolerance instead
curves, ink = pipeline.trace(rgb, upscale="auto", optimize=True)
curves, ink = pipeline.trace(rgb, upscale="auto")

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

Other useful modules: `line2func.geometry` (evaluate, split, flatten, arc
length, bounding box, closest point), `line2func.render` (anti-aliased
rasterizer), `line2func.fit.fit_polyline` (Schneider fitting),
`line2func.metrics` (F-score, chamfer, structure scores).

### 11. Project layout and tests

```
line2func/
  app.py  browser.py  __main__.py          # the web page's server (python -m line2func), browser launcher
  jobs.py                                  # the web page's jobs: reading images, tracing (server and online)
  web.py  website.py                       # the online page's engine (Pyodide), building the online page
  viewer/                                  # web page: index.html, app.js, viewer.js, i18n.js, i18n.json;
                                           #   engine.js, worker.js: tracing in the browser (online)
  demo.py  serve.py  pipeline.py           # commands, and the tracing flow they share
  lineart.py  lineart_model.py  weights.py # line extraction, pretrained model, downloads
  baseline.py  fit.py                      # baseline engine, Schneider fitting
  boundary.py                              # the outline of a filled area as closed loops
  attributes.py  shapes.py  export.py      # refinement, width/color, lines/arcs, exports
  functions.py                             # curves as functions y = f(x) / x = g(y) (--form function)
  residual.py  outline.py  fill.py         # second pass, thick strokes as outlines, filling for Desmos
  budget.py  optimize.py  quality.py       # an exact number of curves, render-and-compare, quality check
  decisions.py                            # the tracer's decisions as scores (the angle rules)
  geometry.py  curves.py  render.py        # geometry core, data model, rasterizer
  synth.py  metrics.py  eval.py            # synthetic data, metrics, evaluation
docs/           # details.md (this manual), third_party.md (licenses of third-party code and weights)
tests/          # pytest suite; tests/pyodide/: Pyodide for the WebAssembly test
.github/workflows/pages.yml   # tests, builds and publishes the online page
```

```bash
python -m pytest            # about 430 tests; PyTorch tests skip without PyTorch, viewer JS checks without Node.js
npm ci --prefix tests/pyodide && LINE2FUNC_PYODIDE=1 python -m pytest tests/test_pyodide.py   # in WebAssembly
```

The online page (section 2) is built by

```bash
python -m line2func.website --out _site            # the page, api/info, line2func as a ZIP
python -m line2func.website --out _site --serve    # and try it at http://127.0.0.1:8000/
```

and published by `.github/workflows/pages.yml` on every push to `main`, once
the tests pass (in the repository settings, Pages must use "GitHub Actions").
Pyodide comes from jsDelivr; `--pyodide-url` points the page to another copy
of Pyodide 314.0.7.

Generated folders (`out/`, `runs/`, `data/`, `_site/`, `.venv/`) are git-ignored.

### 12. Status and roadmap

This is version 2.0.

- [x] Geometry core and renderer
- [x] Baseline engine; SVG, Desmos and LaTeX export; viewer; `demo` and `serve`
- [x] Synthetic data generator with ground truth, and the evaluation harness
- [x] Pretrained line art for photos (Informative Drawings, MIT; downloaded with a SHA-256 check)
- [x] Named equations for lines and arcs, curve refinement, line width and color
- [x] Web page (`python -m line2func`) in English and Traditional Chinese
- [x] Second pass over uncovered ink, thick strokes as filled outlines, render-and-compare (`--optimize`)
- [x] Filled areas such as heavy eyelashes and shadows, each measured for how dark it is and drawn at that tone (one spacing formula for every area, drawn as rings or as hatching by its depth, so Desmos shows the difference); up to 5,000 curves by default; light and very faint lines
- [x] Function mode: every curve as pieces of `y = f(x)` / `x = g(y)` (`--form function`)
- [x] One-screen web page with three display modes; lines broken into dots and dashes kept; a noise filter slider (`--denoise`)
- [x] Faint-line sensitivity (`--faint-sensitivity`)
- [x] Online page: the web page with line2func running in the browser (Pyodide), published with GitHub Pages
- [x] Any image, not only line art: photographs are recognized as they are opened and the lines in them are found first, with a coherent line drawing (`--lineart flow`) that needs nothing downloaded and so runs in the browser too. Both pages show that line art and let it be adjusted (**Line-art detail**, `--lineart-detail`) before anything is traced

Next:

- [ ] Line art from **regions** rather than edges: quantize the colors, merge the small regions and trace the boundaries. It would give closed loops instead of strokes, and a measured tone per region, which is what the filled-area machinery already draws from - so a photograph could come out shaded rather than outlined. It needs its own curve budget policy, because a loop per region plus rings inside each one is the one way to flood the 5,000, and its own measurement

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
9. [評估](#9-評估)
10. [在 Python 中使用](#10-在-python-中使用)
11. [專案結構與測試](#11-專案結構與測試)
12. [現況與路線圖](#12-現況與路線圖)

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

#### 選用：PyTorch（照片用的預訓練線稿模型、`--optimize`）

**先**安裝符合你 GPU 的 PyTorch，再安裝額外套件：

```bash
# NVIDIA RTX 50 系列（Blackwell）需要 CUDA 12.8 以上的版本，例如 CUDA 13.0：
pip install torch --index-url https://download.pytorch.org/whl/cu130
# （只用 CPU：pip install torch）

pip install -e ".[torch,dev]"     # 加上 pytest
```

已驗證的環境：torch 2.14.0+cu130、Python 3.14、RTX 5070（驅動程式 596.21）、Windows 11。

### 2. 描第一張圖

#### 線上使用（免安裝）

<https://far9100.github.io/Line-to-function/> 就是下面介紹的網頁版，只是 line2func 透過 [Pyodide](https://pyodide.org)（編譯成 WebAssembly 的 Python）在你的瀏覽器裡執行。不用安裝任何東西，圖片也不會離開你的電腦。Pyodide 連同 numpy、SciPy、Pillow 約 25 MB，第一次使用時從 jsDelivr CDN 下載，之後由瀏覽器保存。

- 描線方式和 `python -m line2func` 相同（最多 5,000 條曲線，並做品質檢查），但為了瀏覽器分頁而限制較嚴：檔案最大 32 MB，圖片最多保留 2048 px，描線最多 2.5 百萬像素（更大的圖會先縮小），品質檢查最多 2.0 百萬像素（超過就略過檢查，頁面會說明）。
- 所需時間是安裝版的 1.5 到 2 倍。五張約 0.6 百萬像素的真實線稿每張要 17–56 秒（Ryzen 7 9700X 上的 Edge 或 Node.js，含品質檢查），安裝版是 11–30 秒；記憶體最多用了約 460 MB。一張 2.5 百萬像素的照片，連同抽線稿約 11 秒、487 MB。較慢的電腦與手機會更久，手機遇到大圖也可能記憶體不足。
- WebAssembly 的部分計算捨入方式略有不同，所以偶爾會有一兩條曲線和安裝版的結果不同；保留線條的比例則相同。
- 〔取消〕會立刻停止。沒有〔結束〕按鈕：關閉分頁即可。
- 需要 Chrome、Edge 或 Firefox 112 以上，或 Safari 16.4 以上。設定和語言會保存在瀏覽器裡。

照片在這裡也能用（見下文）。預訓練的線稿網路和其他選項（第 6、7 節）需要安裝版。

#### 用瀏覽器（拖放）

```bash
python -m line2func          # 或直接打：line2func（執行過 pip install -e . 之後）
```

會在預設瀏覽器開一個新分頁。接著：

1. **把圖片拖進頁面**。也可以按〔選擇檔案…〕，或按 Ctrl+V 貼上。支援 PNG、JPEG、WebP、BMP、TIFF、GIF，最大 64 MB。手機拍的照片會自動轉正。
2. 圖片會出現在原處。選擇線段的寫法：**函數** `y = f(x)`、`x = g(y)`（見第 5 節）或**參數方程式** `x(t)`、`y(t)`，以及去雜訊的強度（〔去雜訊〕，0–100：調低保留更多細節，有雜訊的掃描圖可以調高；見第 8 節）與〔淡線〕靈敏度（0–100：調高保留更淡的頭髮和背景線），再按〔確認 ▶〕。頁面一開始選的是參數方程式，而且什麼都不過濾——〔去雜訊〕關閉、拖動條在 0，〔淡線〕100——所以第一次描線會畫出找到的每一條線，拖動條是用來減的；`demo` 自己的預設值兩者都是 50。描線方式和 `demo` 相同：最多 5,000 條曲線，並做品質檢查（超過 2048 px 的圖會先縮小）。處理時會顯示目前的步驟，按〔取消〕會在下一個步驟停下。
3. 結果會在同一頁的檢視器中開啟（見第 4 節）。可以下載 SVG、JSON、Desmos、LaTeX，或把全部打包成 ZIP，也可以〔全部複製到 Desmos〕。
4. 上方的〔清除圖片〕會清掉目前的圖，接著就能拖入下一張；隨時直接拖入新圖片也可以。

網頁版線稿和照片都收。照片在開啟時就會被認出來，並先找出裡面的線條（`flow`，見第 6 節）；頁面會把那份線稿顯示出來，可以用〔線稿細節〕調整，也可以用兩個按鈕和原圖來回對照，按下確認才會描線。勾選〔深色處畫上明暗〕會把畫面的深淺也用排線畫出來（見第 8 節）。預訓練網路與其他選項仍在指令列（見第 6、7 節）。

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
sample_lineart.png: 256x256, 136 curves in 10 strokes, 0.25 s (3.88 s/MP); 132 recognized as lines or arcs [faint strokes on, all curves of the fine tracing, fewer than the 5000 allowed]
  out\curves.json
  out\out.svg
  out\desmos.txt
  out\desmos.js
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
| `desmos.js` | 同樣的算式，每一條都帶著量到的線寬與顏色，給 Desmos API 用，見第 5 節 |
| `equations.tex` | 同樣的算式，寫成 LaTeX `align*` 區塊 |
| `overlay.png` | 曲線疊在原圖上，每一筆畫一種顏色，方便快速檢查 |
| `source.png` | 輸入圖的副本（在檢視器中顯示在曲線後方） |

**曲線與筆畫。**「筆畫」是畫出來的一條線。較長或彎曲較大的筆畫會拆成幾段首尾相接的三次「曲線」，它們共用同一個 `stroke` 編號。銳利的轉角會保留為轉角。

**座標。** `curves.json` 和 `out.svg` 使用圖片像素座標（原點在左上角，y 軸朝下）。`desmos.txt` 和 `equations.tex` 會翻轉成數學方向（y 軸朝上），所以在 Desmos 裡圖不會上下顛倒。

**信心值**是曲線落在墨跡上的比例。描線器補過缺口的地方，信心值會低於 1。

**填滿的區域。** 墨色平坦的區域（大片的，以及像很粗的睫毛這種粗重筆畫）只描外框，標上 `fill_outline`；粗筆畫或兩端粗細差很多的筆畫，其外框標上 `outline`。每個區域都帶著自己的濃淡 `tone`（0 是紙白，1 是全黑）。**完全不填色。** 區域是用標上 `fill` 的曲線畫出來的，所有區域共用同一條間距算式（見第 8 節）：深度大於一個間距的地方畫圈線，不夠深的地方畫 45 度排線。Desmos 無法填滿貼上的算式，所以這是濃淡唯一能傳達過去的方式 —— 而網頁與 SVG 畫的是同一批曲線，所以三種輸出看到的是同一張圖。區域的外框用在內部量到的顏色描邊，並且封閉，裡面的曲線才會落在一個形狀內。

區域的邊界是用 Moore 鄰域輪廓追蹤取得的（:func:`line2func.boundary.contours`），不是把邊緣細化後當骨架走。邊界必須是封閉的：兩個環相接的地方，骨架走訪會把環切開，而開放的鏈在填色時會被一條直線弦接起來，憑空造出兩者之間的面積。在 `lineArt (5).jpg` 上這憑空多出 129,399 px（真實填色區域是 87,337，幾乎翻倍），在 `lineArt (11).jpg` 上則把梅花狀瞳孔空白的上半部塗掉了。改用輪廓追蹤後每個環都封閉，憑空多出的面積降到 2,050 px。

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
| 顯示方式 | 〔只顯示線段〕、〔線段＋原圖〕（一開始就是這個，原圖透明度 50%）或〔線段＋遺漏線段〕（做過品質檢查後：曲線改為灰色，漏掉的線紅色、漏掉的淡線橘色、沒有墨跡的曲線藍色，該圖透明度 90%），以及背景的透明度滑桿 |
| 只看原圖 | 〔原圖〕：只顯示原圖 |
| 線條顏色 | 〔黑白〕（一開始就是這個）、〔彩色〕（依筆畫分成八種色相）或〔隨機顏色〕，〔重新隨機〕可換一組隨機色相。下載的 SVG 會套用同樣的顏色與粗細，其他檔案和 ZIP 不受影響。在〔線段＋遺漏線段〕下曲線一律維持灰色，這樣標記才看得出來 |
| 線條粗細 | 〔實際粗細〕（一開始就是這個：每一筆畫和它描到的墨跡一樣粗）或〔統一線寬〕（全部用結果自己的 `line_width`）。畫布也用實際粗細繪製，所以看到的就是 SVG 會有的內容；細的筆畫在畫面上不會細於 0.75 px，縮小時仍然看得見 |
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

#### 顏色與線寬：`desmos.js`

貼進算式列的算式，Desmos 一律用同一種線寬、同一種顏色畫，所以 `desmos.txt` 只能靠「畫得多密」來表達一塊區域有多深（見第 8 節〈陰影、粗睫毛與其他色塊〉）。Desmos API 則可以逐條算式指定 `color` 和 `lineWidth`，`desmos.js` 就是把同樣的算式配上這兩項寫出來 —— 每一筆畫量到的線寬與顏色，而色塊內部的曲線，用的是那塊區域自己的灰階。

裡面的數學和 `desmos.txt` 逐位元相同，只是多了樣式。它是一個 JavaScript 檔，結尾是一次 `setExpressions` 呼叫，所以：

- 在嵌入 [Desmos API](https://www.desmos.com/api) 的網頁裡，執行它，或自己呼叫 `calculator.setExpressions(LINE2FUNC)`；
- 開著計算機時，把整個檔案貼進瀏覽器主控台也是同一件事 —— 前提是那個頁面有把計算機物件露出來。Desmos 沒有把這件事寫進文件，所以不保證在你那邊可用；走 API 那條路則一定可以。

色塊內部的曲線，線寬會等於它們的**間距**，所以彼此相接、中間不留白紙，整塊區域就成為一片它量到的灰。間距本身已經表達過一次深淺了；把線寬設成間距等於把這層表達抵銷掉，改由顏色來說。用本頁一直在用的兩張 JPEG 線稿量測（以 12 px 視窗的區域平均色調，對照原圖有陰影的部分）：

| | 平均色調誤差 | 比原圖深超過 0.05 的比例 |
|---|---|---|
| lineArt (11)，`desmos.txt` | 0.176 | 79.6% |
| lineArt (11)，`desmos.js` | **0.052** | **3.2%** |
| lineArt (9)，`desmos.txt` | 0.164 | 77.4% |
| lineArt (9)，`desmos.js` | **0.041** | **6.3%** |

兩點要注意。`lineWidth` 的單位是螢幕像素，而線寬是在圖片像素上量的，所以只有當圖以原尺寸顯示時兩者才吻合，縮放時線會跟著變粗或變細。另外，剩下的誤差現在多半偏淡：量到比 0.5 px 更細的筆畫會用 0.5 px 畫，免得整條消失，但一條又細又淡的 JPEG 線這樣畫出來，仍比原圖淡一些。

頁面上選的線條顏色與線寬也會套用到這個檔案，所以下載到的就是頁面正在顯示的樣子。

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

照片需要先抽出線稿（`--lineart`）。兩個網頁版都會自己對 `lineart.suggest_mode` 判定為照片的圖做這件事，並在描線前先顯示結果；指令列則是一個旗標。

| 方法 | 需要 | 結果 |
|---|---|---|
| `flow` | 無 | 連貫線稿：線條長而相連。**照片的預設值，也是不需下載任何東西之中最好的** |
| `informative` | PyTorch 與權重 | 預訓練線稿網路，產生像手繪的線條。**裝了 PyTorch 的話，這個最好** |
| `informative-coarse` | PyTorch 與權重 | 同一個網路，線條較粗、較簡潔 |
| `canny` | 無 | 邊緣偵測；粗線的兩側會各描出一條邊 |
| `xdog` | 無 | 風格化的邊緣；適合高對比的圖 |
| `none`（預設） | 無 | 輸入本身已經是線稿 |

**為什麼是 `flow`，以及為什麼瀏覽器裡用它。** 描線器要的不是漂亮的邊緣，而是**筆畫**：它會二值化、細化成骨架、把骨架走成圖，再對找到的每一段擬合曲線。每一個斷開的碎片不是被去雜訊丟掉，就是各自吃掉一條曲線——這正是其他 image-to-Desmos 工具都說照片會產生上千條算式的原因。`flow` 是**跨著**邊緣切向流場做高斯差，再**沿著**流場平滑（Kang, Lee & Chui, NPAR 2007）：一個雜訊點沒有線可以附和它，就被平均掉了；真正的線則從兩端被強化。在一張類照片的圖上實測（把線稿加上模糊、不均勻打光與雜訊），把 ink 細化後數連通片：

| 方法 | 秒 | 骨架像素 | 片數 | 每千像素片數 | 線寬 |
|---|---|---|---|---|---|
| `flow` | 0.85 | 17,163 | 558 | **32.5** | 2.40 |
| `xdog` | 0.02 | 10,462 | 1,983 | 189.5 | 1.78 |
| `canny` | 0.06 | 49,495 | 755 | 15.3 | 1.12 |

把雜訊從 0.02 提高到 0.06，也就是一般照片的程度：

| 方法 | 骨架像素 | 片數 | 每千像素片數 | 變化 |
|---|---|---|---|---|
| `flow` | 18,169 | 633 | **34.8** | +7% |
| `xdog` | 19,393 | 8,248 | 425.3 | +124% |
| `canny` | 48,502 | 806 | 16.6 | +9% |

那是單一張圖的控制實驗。在 **32 張真實照片**上（見第 9 節），逐張配對、附 95% bootstrap 區間：

| flow 減去 | 每千像素片數 | 骨架像素 | 線寬 | 抽取時間 |
|---|---|---|---|---|
| `xdog` | **−132.7** [−174.6, −96.8] | **+3,362** [+1,368, +7,845] | **+0.55 px** [+0.47, +0.67] | +2.70 s |
| `canny` | +34.4 [+26.6, +46.7] | **−23,270** [−37,860, −11,170] | +1.13 px [+1.01, +1.22] | +2.58 s |

每一個區間都不包含 0。對上 `xdog`，`flow` 交給描線器的線破碎程度只有三分之一（每千個骨架像素 57 片，對 182 片），而且同時找到**更多**線，不是更少。對上 `canny`，`flow` 的每千像素片數比較高，但那一欄不能這樣讀：同樣這些照片，`canny` 的**骨架長度是 1.8 倍**，因為每條線兩側都畫了一條邊，所以分母比較大，而要描的東西是兩倍。

那份比較的最後一欄還決定了另一件事。這 32 張照片裡，線寬低於 `pipeline.AUTO_UPSCALE_BELOW`（1.75 px）的，`xdog` 有 **30 張、`canny` 是 32 張全中，而 `flow` 只有 1 張**。在本節最後那條規則之前，這等於幾乎每一張會被打開的照片都要求 2 倍放大——四倍的描線工作量，只為了把抽取器早就做好的決定內插一次。

**這並不會讓結果變小。** 照 `demo` 的方式描（`--curves 5000`、`auto` 放大），從語料每一組各取兩張照片：

| 方法 | 曲線數中位數 | 秒數中位數 | 撞到預算 |
|---|---|---|---|
| `flow` | 5,000 | 23 s | 10 張中 6 張 |
| `xdog` | 4,348 | 9 s | 10 張中 5 張 |
| `canny` | 5,000 | 16 s | 10 張中 6 張 |

不管線是哪個方法找出來的，照片都會把預算填滿，而且 `flow` 是三者中最慢的。差別在於**那 5,000 條曲線是什麼**：`flow` 把它們花在長筆畫上（每千個骨架像素 57 片），`xdog` 花在粉塵上（182 片）。在其中一張人像上，`xdog` 整張抽出來只有 8,650 個骨架像素、分成約 3,800 片——平均每片 2.3 個像素，去雜訊把它們丟掉是對的。「很快地找到幾乎什麼都沒有」不是優點；這就是 `flow` 成為預設、而 `xdog` 仍然保留給慢機器或超大圖的原因。

`flow` 還有兩個和連貫性同樣重要的性質。純色區域給出的 ink **恰好是 0**，平滑漸層也一樣——高斯差對任何線性的東西都是 0，剩下的部分符號也不對，成不了 ink——所以天空或臉頰完全不會吃掉曲線預算。而且它的線夠寬，可以照原樣描：在那 32 張照片上中位數是 2.2 px，低於 `pipeline.AUTO_UPSCALE_BELOW` 的只有 1 張。

**`flow` 讀的是什麼（2.0.0 之後）。** 上面的表是用 2.0.0 發布時的抽取器量的，當時它用灰階讀圖，而且每張圖都用同一個門檻。之後它改了三件事，每一件都是為了一張手繪插畫上看得到的毛病——夜空前的九個人物，臉是空白的，天空卻滿是碎點：

- **顏色。** 兩個亮度相同的顏色——頭髮對天空、布料疊布料——在灰階裡根本不是邊。現在抽取器在每個像素找出圖片在 RGB 裡變化的方向，沿著那個方向讀脊，所以紅|藍的交界和黑|白一樣是一個台階。灰階圖片只當成一個通道來讀，不必付顏色的額外成本。
- **到處都是同一個門檻。** Kang 的寫法留下 `(1 - tau) x gray` 當門檻，亮的臉上比暗的天空高五倍；而兩輪之間把找到的線用黑色畫上去，讓亮底上的線被強化的力道是暗底上同一條線的四倍。現在門檻是單純的 `eps`，線則是畫得比原處深一個固定的量。
- **圖片自己的雜訊決定門檻抬多高。** 雜訊是從圖片上讀出來的（放進 0.01、0.03、0.06，讀到 0.010、0.030、0.058），一條脊必須跨過「那個雜訊經過兩次濾波後」的四個標準差。乾淨的插畫用它的淡邊需要的低門檻來讀，有顆粒的照片用顆粒搆不到的門檻來讀。

在那張插畫上、預設細節：**109,293 個骨架像素、每千像素 19.9 片，原本是 57,103 個、37.5 片**——線將近兩倍，破碎程度減半。在加上相機模糊、不均勻光線與雜訊、而且知道真實線條在哪裡的合成圖上：

| 雜訊 | 找到的線 | 假線 | 每千像素片數 |
|---|---|---|---|
| 0.01 | 99.3%（原 97.5%） | 0.5%（原 4.1%） | 1.2（原 4.6） |
| 0.03 | 99.3%（原 97.5%） | 0.6%（原 3.6%） | 1.2（原 5.8） |
| 0.06 | 99.4%（原 97.2%） | 2.4%（原 8.0%） | 10.2（原 24.7） |

在八張 2.5 MP 的照片上，凡是有線的每一張，每千像素片數都更少（2.6 到 40.3，原本 3.9 到 58.1），線都更多，線寬 2.3 到 2.8 px。**代價是時間**：每張照片 4.1 秒，原本 3.0 秒。**還沒做的**是用它重跑上面那 32 張照片的比較；那些數字代表的是先前的抽取器。

預訓練權重不隨專案附帶。下載一次即可；預訓練權重不隨專案附帶。下載一次即可；下載後會用固定的 SHA-256 驗證，存放在 `~/.cache/line2func`（可用環境變數 `LINE2FUNC_HOME` 更改）：

```bash
python -m line2func.weights list            # 列出可用的權重、授權與下載狀態
python -m line2func.weights fetch informative   # 或：fetch all
python -m line2func.weights verify          # 重新檢查雜湊值

python -m line2func.demo photo.jpg --lineart informative --out out/
```

照片通常會得到很多短曲線。`demo` 會把它們合併到 5,000 條（見第 8 節）。

**抽出來的 ink map 不會被放大。** `--upscale auto` 會把**線稿**中細到一個像素裝不下的線放大兩倍；但抽取器的輸出本來就是它決定好的線，而且已經是原圖的解析度，放大只是把那個決定內插一遍，工作量卻變成四倍。所以除了 `none` 以外的每一個 `--lineart` 方法，`auto` 都維持 1 倍；明確指定 `--upscale 2` 仍然照做。

### 7. `demo` 的所有選項

```
python -m line2func.demo IMAGE [options]
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `--out DIR` | `out` | 輸出資料夾 |
| `--lineart {none,canny,xdog,flow,informative,informative-coarse}` | `none` | 抽線稿方法（見第 6 節） |
| `--lineart-detail 0..100` | `50` | 搭配 `--lineart flow`：畫面裡有多少東西會變成線條（見第 8 節） |
| `--shade` | 關閉 | 給照片與繪畫用：把畫面的深淺也用排線畫出來（見第 8 節） |
| `--curves N` | `5000` | 產生 N 條曲線：先細緻地描線，再合併「合併後對圖影響最小」的相鄰片段（見第 8 節）。曲線本來就比 N 少的圖會全部保留 |
| `--tolerance PX` | 關閉 | 改用曲線擬合的最大誤差來描線，而不是指定曲線數量。數值越大，曲線越少、越平滑 |
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
| `--no-fill` | 開啟 | 讓填滿的區域保持空心：只留外框，裡面不加圈線或排線 |
| `--optimize` | 關閉 | 用「渲染後比對」微調每一條曲線。需要 PyTorch；760×818 的圖在 RTX 5070 上約 3 秒，CPU 約 13 秒（見第 8 節） |

### 8. 取得好結果的訣竅

本節的量測數據來自四張真實測試圖（網路上找到的動漫線稿，不包含在儲存庫裡），以及有標準答案的合成圖。

- **解析度：** 線寬約 1.5–5 px 時效果最好。細線的圖會自動以 2 倍解析度描線（`--upscale auto`）；很大的掃描圖請自行縮小。在兩張線寬約 1.7 px 的真實線稿上，自動放大加上淡線，讓線條保留率從 95–97% 提高到 98–99%，漏掉的墨跡大約減半，曲線數為原本的 1.2–1.4 倍。
- **淺色與很淡的線：** 線稿的自動墨跡門檻最高只到 0.25，所以淺灰色的筆畫（例如飄散的髮絲）會完整描出，而不是斷成一段一段的虛線。比這更淡的線，只要明顯比周圍背景深也會被加入；最淡的、一直到紙張本身的雜訊為止，只要明顯是線也會被加入（見下文）。在乾淨紙面上很淡的鉛筆稿，可以試試 `--threshold 0.15` 或更低（細節更多，曲線也更多）。
- **線寬差距：** 最粗的線如果超過最細的約 4.5 倍，品質可能下降；粗很多的區域會被當成填色區域。
- **掃描雜訊、JPEG：** 傳統引擎會去除小雜點（紙面乾淨時，接續某條線的小點會保留，見後面）。如果灰塵還是被描出來，先清理掃描圖或調高 `--threshold`。
- **曲線太多或太少：** `demo` 最多產生 5,000 條；用 `--curves N` 指定其他數量（見下文）。用這個方法減少曲線，比調高 `--tolerance` 保留更多細節。
- **全是細線的圖：** 如果整張圖只有細於約 2 px 的線，量到的線寬可能偏寬最多 0.5 px。

#### 補回細節：第二遍、外框、渲染後比對

有三個步驟專門處理單次描線會遺失的細節：

- **第二遍描線**（預設開啟，`--no-residual` 關閉）：第一遍描完後，把還沒有任何曲線蓋到的墨跡單獨再描一次。太短的、大部分不在墨跡上的、或只是重複描既有曲線的片段都會被丟掉。
- **外框**（預設開啟，`--no-outline` 關閉）：實心區域（例如粗重的睫毛），以及比圖中一般線條粗很多、或兩端粗細差很多的筆畫（例如筆刷的尖端），會改用封閉的外框表示，裡面用圈線或排線畫出它自己的濃淡（見下文）。任何一種輸出都不填色。
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

`--denoise`（網頁版是〔去雜訊〕與它的拖動條）決定小雜點和短的淡線片段丟得多嚴格：50 是上面說明的預設值，0 全部保留，100 則把各項門檻加倍，而且不再把斷線的片段合起來判斷。取消勾選〔去雜訊〕等於 0，網頁版一開始就是取消勾選的。在圖 4 和 100 張有雜訊的合成掃描圖（雜訊、雜點、JPEG）上：

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

`--faint-sensitivity`（網頁版是〔淡線〕，一開始是 100）一起放寬這些檢查：50 是調校好的預設，100 只需要不到一半的對比（門檻的 0.12 而不是 0.3）、更靠近深色線也會找，極淡的線可以更短（6 個線寬而不是 15）、稍寬、分岔更多（每 5 個線寬一個交叉點，而不是 17）。低於 50 時淡線需要更高的對比；0 只描門檻以上的墨跡。容許誤差 1.0：

| 圖 | 50 時漏掉的墨跡 | 100 時 | 100 時曲線落在墨跡上的比例（含淡墨） | 線寬 50／100 |
|---|---|---|---|---|
| huaban-6611354694 | 2.61% | 1.01% | 0.986 | 1.70／1.77 px |
| huaban-6611349997 | 1.79% | 0.56% | 0.991 | 1.54／1.58 px |
| af26b7b7 | 2.99% | 1.64% | 0.993 | 1.71／1.75 px |

調到最高時，鉛筆紋理會變成短線被描出來；有雜訊的掃描圖連雜訊也會描（和 `--denoise 0` 一樣）：在 25 張有雜訊的合成掃描圖上，曲線落在真實線條上的比例從 50 時的 0.929 降到 75 時的 0.902、100 時的 0.826（最差一張 0.165）。乾淨的合成掃描圖不受影響。

#### 照片裡有多少東西會變成線條：`--lineart-detail`

搭配 `--lineart flow` 時，一支 0 到 100 的滑桿決定「多弱的脊還算一條線」。它會同時移動三個設定，因為那是同一個決定的三個面向：跨著流場取的核有多寬（`sigma_e`）、答案沿著流場平滑多遠（`sigma_m`），以及一條脊必須跨過的門檻（`eps`，以 0 到 1 的灰階計）。在一張 2.07 MP 的插畫上實測（1920 x 1080，夜空前的九個人物）：

| 細節 | `sigma_e` | `sigma_m` | `eps` | 抽取 | ink | 骨架像素 | 每千像素片數 | 線寬 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.80 | 3.80 | 0.0300 | 4.42 s | 6.8% | 35,918 | 26.4 | 3.42 px |
| 25 | 1.40 | 3.20 | 0.0134 | 3.78 s | 14.0% | 78,680 | 17.8 | 3.19 px |
| **50** | **1.00** | **2.60** | **0.0060** | **3.22 s** | **15.9%** | **109,293** | **19.9** | **2.62 px** |
| 75 | 0.90 | 2.40 | 0.0035 | 2.81 s | 16.6% | 124,720 | 21.1 | 2.39 px |
| 100 | 0.80 | 2.20 | 0.0020 | 2.66 s | 17.4% | 138,444 | 22.8 | 2.26 px |

這張表裡有三件事，在動滑桿之前值得知道。

**調高細節不會讓線變少。** 這是這個控制項給的承諾，端點也是為了它才這樣選。`eps` 是**按比例**移動而不是線性移動的，因為它在整支滑桿上橫跨超過一個數量級；直接線性內插的話，細的那一半幾乎不動它、兩個 sigma 卻掉得很快。

**最慢的設定是 0，不是 100。** 細節調高代表核更窄、每個像素取的樣本更少：抽取時間從 4.42 秒降到 2.66 秒。變多的是描線，因為有更多線要描。

**曲線預算是花在高端的。** 細節 100 的線比調好的設定多四分之一，也稍微更破碎（每千個骨架像素 22.8 片，對 19.9 片），因為進來的是紋理而不是筆畫。這張圖在 50 就已經撞到 5,000 條上限。

**`eps` 是乾淨圖片上的門檻；有雜訊的圖片會自己把它抬高。** 見第 6 節的〈`flow` 讀的是什麼〉。

#### 照片有多深：`--shade`

線稿說的是一張圖的邊在哪裡。照片或繪畫大部分是**深淺**——夜空、黑外套、深色頭髮——而那些都不是邊。`--shade` 把它也畫出來：45 度的直線排線，底下的畫面越深就排得越密。在網頁上是〔深色處畫上明暗〕這個勾選框，開啟的是照片時才出現，要不要勾由你決定（預設不勾）。

- **間距就是深淺。** 一個色調要的間距是 `最密間距 x dark / 色調`——深度減半，間距加倍——其中 `dark` 是畫面最深的 1% 的色調（最低當作 0.5，所以淡色的圖不會被當成「最深的灰就是黑」來畫）。平行直線無法逐漸改變間距，所以取整到三種：最密、兩倍、四倍。再淡的就留白。較疏的那組線是較密那組的子集，所以從陰影畫進中間調的線會直接延續下去。
- **它算在 `--curves` 裡。** 排線最多用掉總數的 30%，最密間距從 2.5、3、4、5、6、8、10、12、16 px 裡挑放得下的最密者；輪廓線則合併到剩下的數量。沒有指定 `--curves` 時，排線自己有 2,000 條的額度。
- **每條排線是一段直線**，標上 `shade`、寬 1 px，並把它代表的色調當成自己的顏色；SVG 與 `desmos.js` 會用那個灰階來畫，所以看起來比輪廓淡。頁面用的是你選的線條顏色，而 `desmos.txt` 貼進 Desmos 時全部是同一個顏色，這兩處的深淺只靠間距表現。寫成函數時每一條都是某個範圍內的 `y = x + c`。
- **品質檢查不算它。** 品質檢查評的是墨跡描得如何，而排線沒有描任何墨跡。

在上表那張 2.07 MP 的插畫上，`--curves 5000`：

| 佔總數的比例 | 排線 | 最密間距 | 輪廓線 | 線條保留 | 最大合併誤差 |
|---|---|---|---|---|---|
| 不畫 | - | - | 5,000 | 99.3% | - |
| **30%** | **1,445 條** | **6 px** | **3,555** | **97.5%** | **2.9 px** |
| 40% | 1,904 條 | 5 px | 3,096 | 91.2% | 6.2 px |

所以比例定在 30%：到 40% 時輪廓被合併到看得出來。深色面積大的圖，不管怎麼分，輪廓都要付出一些代價。

#### 陰影、粗重的睫毛與其他填滿的區域

像粗重睫毛這種很粗的黑色筆畫，以前會被細化成一團短短的中心線，在 Desmos 裡也只畫成細線。至少 2.5 個線寬粗、有一定長度，而且中間夠**平坦**的墨跡，會被描成一個填滿的區域。兩條靠得太近、墨跡黏在一起的線不算：它們中間的墨比較淡，所以仍然是線。平坦度正是分辨「一片灰階塗抹」和「一堆擠在一起的線條」的關鍵，而且不能放寬：門檻從附近最黑墨色的 0.8 倍改成 0.7 倍時，JPEG 測試集裡 `lineArt (11)` 的區域面積就從整頁的 1.1% 衝到 3.5%，開始把密集線條也算進去。

**區域會長到多大。** 粗的部分是用「塞得進墨裡的圓盤」找出來的，接著再以同樣的半徑往外長回去；所以單靠這一步，半徑內的每一塊墨都會被併進來——不管它自己有多細。橫過粗重睫毛的一撮頭髮就會被吃掉，連帶兩者之間的紙白也落進區域的外框裡。因此區域只會沿著「自己也至少有半個種子半徑深」的墨生長（`solid_depth`）：這樣會順著睫毛自己的錐形一路跟下去，但在只有三分之一粗的髮絲前停住。在一張瀏海蓋過雙眼的 Q 版線稿上，描出來的睫毛從原圖黑色的 1.25 倍與 1.19 倍降到 1.01 倍與 0.95 倍，而區域裡「寬度不到 4 px 的墨」（也就是被吃進來的頭髮）從 25% 降到約 12%。若墨門檻被調到 Otsu 之下，區域是用強墨判定的，之後再把周圍較淡的軟邊補回去；這個軟邊是 1 px（`solid_rim`），因為補 2 px 會越過軟邊、伸進旁邊的筆畫（同樣那兩道睫毛：補 2 px 是黑色的 1.25 倍與 1.19 倍，補 1 px 是 1.17 倍與 1.10 倍）。

**每個填滿的區域都會量測自己有多深**，並照那個濃淡畫出來。**沒有任何地方會填色**：Desmos 沒辦法填滿貼上的算式，而且每條線都一樣深，所以在那裡濃淡是用線的密度表示；而 SVG 與網頁畫的就是 Desmos 拿到的同一批曲線，所以三種輸出看到的是同一張圖。陰影也不另外描邊——它的邊界是濃淡淡出去的地方，不是誰畫的線。

區域的外框只有一側有墨，所以量不出自己的線寬；網頁、SVG 與 `desmos.js` 以前都退回用全圖的線寬——那是在墨門檻上量的、含軟邊，所以比圖中大多數筆畫還粗（上面那張 Q 版線稿是 3.43 px，實際量到的線寬中位數只有 2.36 px）。以區域邊界為中心描這麼粗的邊，等於有一半畫在區域外面。外框是區域的邊界、不是誰畫的線，所以只用 1 px 描邊，剛好把區域封起來，裡面的濃淡交給區域內的曲線——粗筆畫自己的外框一直都是用這個線寬寫出來的（`line2func.outline`）。同樣那兩道睫毛，畫出來的面積因此從原圖黑色的 1.60 倍與 1.57 倍降到 1.26 倍與 1.24 倍；再加上前面的生長規則，降到 1.02 倍與 0.96 倍，畫出來的半寬也從 8.25 px 與 7.28 px 降到 5.39 px 與 5.00 px（原圖的墨本身是 4.12 px 與 3.61 px）。

在 81 張實測線稿上做配對比較：**保留的線與漏掉的墨完全沒有變動**（中位差 0，區間 [0, 0]），`precision`、`d_M`、PSNR、SSIM 與曲線數都未達顯著。代價是每單位骨架長度的曲線長度多 0.4%（區域縮小後，邊緣還給了線描），以及中位 7.3 秒多 0.1 秒。

所有區域共用同一條間距算式，中間沒有任何跳階：

```
間距 = clip(0.88 × 2.5 px × 暗墨 ÷ 濃淡, 1.5 px, 12 px)
```

一條曲線每隔一個「間距」就畫上 2.5 px 的墨，所以 Desmos 畫到的面積比例，正好等於這塊區域的墨色占全圖暗墨的比例。其中 0.88 是量出來的，不是選的：曲線以自身線寬並排時只覆蓋區域的 92% 而非 100%（因為圈線是彎的），少了這個重疊，最深的區域會透出紙白。

**畫什麼**由區域的形狀決定，而不是由濃淡決定：深度大於一個間距的地方畫**圈線**（到邊緣距離的等高線），不夠深的地方畫 **45 度排線**。第 *k* 圈只存在於「深度大於 k × 間距」的地方，所以光靠圈線，下顎或手指邊上只有幾個像素深的陰影會完全空白——`lineArt (9)` 的 59 個區域就有 23 個（占陰影像素的 12%）畫不出任何一圈。排線則不管區域多薄都能穿過去。區域會一路畫到最裡面，有多深就畫多少圈，在 `lineArt (11)` 上只花 66 條曲線，就讓實心區域的覆蓋率從 95.1% 提高到 99.8%。

以前區域必須和圖中自己的暗墨（第 90 百分位）一樣黑才算得上填滿區域。這個門檻是相對的，所以在沒有純黑的淺鉛筆稿上就會塌掉：`lineArt (9)` 量到的墨色 p90 是 0.576，門檻只剩 0.46，於是每一塊中灰陰影都被升級成實心墨塊——然後整張圖的陰影就都一樣黑了，這正是這次改掉的東西。

| | 曲線數 | PSNR | SSIM |
|---|---|---|---|
| `lineArt (9).jpg`（柔和灰階陰影） | 2,066 → 2,062 | 22.52 → **22.88 dB** | 0.781 → 0.776 |
| `lineArt (11).jpg`（濃密鉛筆、黑眼睛） | 3,585 → **3,111** | 19.00 → **19.71 dB** | 0.670 → 0.668 |
| `lineArt (7).jpg`（幾乎沒有真陰影） | 1,253 → 1,347 | 17.96 → 17.92 dB | 0.582 → 0.578 |
| `lineArt (5).jpg`（有真黑） | 8,072 → 8,039 | 8.32 → **8.82 dB** | 0.631 → 0.629 |
| `lineArt (3).jpg`（純線稿） | 1,432 → 1,432 | 22.10 → 22.10 dB | 完全相同 |

`lineArt (9)` 的填滿區域量到的濃淡是 0.30／0.40／0.60（p10／p50／p90），而這些以前全部都畫成實心黑。純線稿一條不差；本來就有純黑的圖也保住它的實心區域（`lineArt (5)` 墨色 p90 為 1.00，區域量到 0.81–0.91）。

在 `data/real_v1` 的 66 張上做配對比較：**召回率與漏掉的墨完全沒有變動**（中位差 0，區間 [0, 0]），`precision` 與 `d_M` 未達顯著，代價是曲線 +18（中位圖的 0.7%）與 +0.13 秒。睫毛基準圖每一項都變好。

圖 1，容差 1.0：

| | 之前 | 現在 |
|---|---|---|
| Desmos 畫到的深色睫毛墨跡（每條曲線 2.5 px 寬、整張圖顯示在畫面上） | 89.6% | **97.0%** |
| 線條保留率／d_M | 99.65%／0.527 px | 99.66%／0.531 px |
| PSNR／SSIM | 21.9 dB／0.922 | **22.1 dB／0.923** |
| 曲線數 | 1,078 | 1,111（其中 35 條是圈線） |

那麼深的睫毛仍然會拿到圈線，所以這些數字依然成立。在一張很多地方雙線黏在一起的淡色草稿上，只有 2 個小區域符合條件。在 Desmos 裡放得很大時看得出一圈圈的線；整張圖顯示在畫面上時，它們會連成實心的一塊。

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

#### 為何沒有學習式評分器

筆畫在接點怎麼延續、哪些斷口其實是同一條線、哪些相鄰的交叉點是同一個淺角交叉、哪裡轉得夠急要切開 —— 這些都由角度規則決定（`line2func/decisions.py`）。

這裡曾經還有一個預設開啟的學習式評分器：五個用 numpy 跑的小型 MLP，在生成場景上訓練。它在 1.2 被移除。它確實比較好，而且那個差別看不出來。

在 `data/real_v1` 的 66 張上做配對比較（bootstrap），學習式對上它取代的規則：

| | 中位數 | 95% 區間 |
|---|---|---|
| 曲線長度落在墨上（precision） | +0.0044 | [+0.0040, +0.0058] |
| PSNR | +0.175 dB | [+0.13, +0.20] |
| SSIM | +0.0030 | [+0.0026, +0.0032] |
| **每張圖的秒數** | **+1.76** | **[+1.49, +1.945]** |

在另外 15 張 held-out 的 JPEG 上複現為 precision +0.0029、PSNR +0.11 dB。漏掉的墨、淡線覆蓋、畫出的總長度與 `d_M` 區間全部含 0：它並沒有多找到圖上的東西，只是把找到的東西擺得稍微準一點。

**準到看不出來。** 把同一張圖用兩種方式畫出來逐像素比對，89–93% 的差異是同一條線在 2 px 之內的位移。剩下的是每張圖 7–25 處「一邊有墨、另一邊沒有」，最大的一處 49 px —— 約 20 px 長的線段，夾在密集的髮絲紋理裡。大約 7–9% 的決策翻轉了，每張圖數百個，換來 0.4 個百分點的 precision。

代價則是：包裡 24% 的程式碼、五分之一的描圖時間，以及 52% 的瀏覽器下載（移除後 zip 從 298,548 降到 142,807 bytes）。

**把規則放寬來代替也行不通，這一樣量過。** 最顯然的便宜替代方案是讓規則多考慮一些候選：兩個放寬版本（`--decisions r2` 和 `r2-gaps`，已隨評分器一起移除）就是這麼做的，在合成場景上看起來很強（斷線補齊 0.854 對規則的 0.774，已經走了三分之二到評分器的 0.898）。但在真實線稿上它們**比純規則還差**：66 張上 precision −0.0014 [−0.0022, −0.0007]，15 張上 −0.0009。它們多畫了 5.1% 的曲線長度，把真正的斷點和不存在的斷點一起接起來，而後者的代價大於前者的收穫。所以移除評分器等於回到規則，而不是回到放寬版的規則。

評分器贏的那些合成數字也適用同樣的警告：專案量過它們不會轉移。斷線補齊 0.774 → 0.898、轉角 precision 0.525 → 0.624 在生成的頁面上是真的，但沒有到達輸出。

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

### 9. 評估

預覽下面各項指標所使用的合成資料：

```bash
python -m line2func.synth preview --kind hard --out preview.png
python -m line2func.synth preview --kind hard --patches --out patches.png
```

描線器用有標準答案的合成圖評分，分成「乾淨組」和「困難組」（雜訊、斷線、密集交叉、線寬變化）。除了曲線有沒有落在線上，也檢查筆畫是否正確：

| 指標 | 意義 |
|---|---|
| F_GT@2 | 曲線與真實線條在 2 px 內的吻合程度 |
| 交叉接續 | 線條穿過交叉點後是否仍是同一筆 |
| 斷線補齊 | 線條上的小斷口是否被接起來 |
| 每筆畫碎片數 | 一筆被拆成幾段，理想是 1 |
| 曲線數比 | 輸出曲線數 ÷ 真實曲線數，越接近 1 越精簡 |

- **傳統引擎的目標：** 乾淨組 F_GT@2 ≥ 0.97，每百萬像素最多 5 秒 CPU。

```bash
# 固定的合成驗證集（乾淨、困難各 100 張，在每台電腦上都完全相同）
python -m line2func.synth valset --out data/val_v1

# 分數：F_GT@2、交叉接續、斷線補齊、每筆畫碎片數、曲線數比、每百萬像素秒數
python -m line2func.eval --valset data/val_v1
```

加上 `--json results.json` 可以儲存數據，`--limit N` 可以快速跑少量場景。

真實線稿沒有標準答案，所以 `--realset 資料夾` 改用線稿自己的墨來評判（`line2func.quality`），
`--compare` 則把兩次結果並排比較，報告每張圖差異的中位數與 bootstrap 區間：

```bash
python -m line2func.eval --realset path/to/drawings --json runs/a.json
python -m line2func.eval --realset path/to/drawings --set local_trim=false --json runs/b.json
python -m line2func.eval --compare runs/b.json runs/a.json
```

`--lineart 方法`（可搭配 `--lineart-detail N`）會先讓整個資料夾經過抽線稿，一資料夾的**照片**就是這樣測量的：

```bash
python -m line2func.eval --realset path/to/photos --lineart flow --json runs/photo_flow.json
python -m line2func.eval --realset path/to/photos --lineart xdog --json runs/photo_xdog.json
python -m line2func.eval --compare runs/photo_xdog.json runs/photo_flow.json
```

**大部分的欄位不能在兩個抽取器之間比較，用了抽取器時報告也會這樣提醒。** 評判是拿曲線去對該組自己的抽取器產生的 ink map——門檻也來自同一份 ink——所以 `kept`、`precision`、`missed_ink`、`d_M`、`psnr_db`、`ssim` 回答的都是「我們有沒有忠實畫出自己抽到的東西」。一個幾乎什麼都沒抽到的抽取器，在這每一項上都會很漂亮。有三欄經得起比較，因為它們不是除以該組自己選擇要找的東西：

| 指標 | 意義 |
|---|---|
| `curves` | 預算有沒有用完（見第 8 節） |
| `seconds` | 花了多少時間 |
| `pieces_per_kpx` | ink 每 1,000 個骨架像素有幾個相連的片段：交給描線器的線有多破碎。分得出「畫出筆畫」和「碎成粉塵」的就是這一欄（見第 6 節） |

本手冊中的測量所用的 66 張線稿與 15 張 held-out JPEG 都是第三方線稿作品，保存在儲存庫之外，
也不隨專案發布。因此下文提到的 `data/real_v1` 指的是它們當初的位置，而不是 checkout 裡找得到的
資料夾；`--realset` 可以指向任何資料夾。

第 6 節的照片測量用的是另一組 **32 張 CC0 照片**，來自 Wikimedia Commons，每個檔案的授權都是向 API 查過而不是從分類推定的，縮到線上版描線的 2.5 百萬像素，並刻意分散在照片這條路必須撐過的幾種情況：8 張人像（平滑的色調必須完全不產生線）、9 張枝葉與草地（會灌爆預算的情況）、5 張建築（硬邊，舊方法應該有競爭力的地方）、5 張夜間街景（顆粒，單純的高斯差會碎掉的地方）、5 張單純的物件。它們是第三方的圖片，同樣保存在儲存庫之外，並有一份 `sources.json` 記錄每個檔案在 Commons 上的頁面。線稿仍然是這項工作的**回歸**測試集：線稿這條路不能動，而在全部 81 張上它也確實沒動——每一項指標都是 0，區間寬度為零。

### 10. 在 Python 中使用

一次跑完整個流程，和 `demo` 與網頁版的做法相同（第二遍、外框和讓外框在 Desmos 裡填滿的曲線預設開啟；兩者都指定最多 5,000 條曲線；`optimize=True` 需要 PyTorch）：

```python
from line2func import functions, lineart, pipeline
from line2func.export import write_outputs

rgb = lineart.load_rgb("drawing.png")
curves, ink = pipeline.trace(rgb, upscale="auto", curve_count=5000)   # 最多 5,000 條曲線，和 demo 一樣
curves, ink = pipeline.trace(rgb, upscale="auto", fit_tolerance=1.0)  # 改用容差
curves, ink = pipeline.trace(rgb, upscale="auto", optimize=True)
curves, ink = pipeline.trace(rgb, upscale="auto")

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

其他可用的模組：`line2func.geometry`（求值、分割、轉折線、弧長、外框、最近點）、`line2func.render`（反鋸齒渲染器）、`line2func.fit.fit_polyline`（Schneider 擬合）、`line2func.metrics`（F 分數、chamfer 距離、結構指標）。

### 11. 專案結構與測試

```
line2func/
  app.py  browser.py  __main__.py          # 網頁版的伺服器（python -m line2func）、瀏覽器啟動器
  jobs.py                                  # 網頁版的工作：讀取圖片、描線（伺服器和線上版共用）
  web.py  website.py                       # 線上版的引擎（Pyodide）、建置線上版網頁
  viewer/                                  # 網頁：index.html、app.js、viewer.js、i18n.js、i18n.json；
                                           #   engine.js、worker.js：在瀏覽器裡描線（線上版）
  demo.py  serve.py  pipeline.py           # 指令，以及它們共用的描線流程
  lineart.py  lineart_model.py  weights.py # 抽線稿、預訓練模型、權重下載
  baseline.py  fit.py                      # 傳統引擎、Schneider 擬合
  boundary.py                              # 把色塊的外框描成封閉的環
  attributes.py  shapes.py  export.py      # 精修、線寬顏色、直線圓弧、匯出
  functions.py                             # 把曲線寫成函數 y = f(x)／x = g(y)（--form function）
  residual.py  outline.py  fill.py         # 第二遍描線、粗筆畫外框、Desmos 用的填色曲線
  budget.py  optimize.py  quality.py       # 指定曲線數量、渲染後比對、品質檢查
  decisions.py                            # 描線決策的評分（角度規則）
  geometry.py  curves.py  render.py        # 幾何核心、資料結構、渲染器
  synth.py  metrics.py  eval.py            # 合成資料、指標、評估
docs/           # details.md（本說明）、third_party.md（第三方程式碼與權重的授權）
tests/          # pytest 測試；tests/pyodide/：WebAssembly 測試用的 Pyodide
.github/workflows/pages.yml   # 測試、建置並發布線上版網頁
```

```bash
python -m pytest            # 約 430 個測試；沒有 PyTorch 會略過需要它的測試，沒有 Node.js 會略過檢視器的 JS 檢查
npm ci --prefix tests/pyodide && LINE2FUNC_PYODIDE=1 python -m pytest tests/test_pyodide.py   # 在 WebAssembly 中
```

線上版網頁（第 2 節）用以下指令建置：

```bash
python -m line2func.website --out _site            # 網頁、api/info、打包成 ZIP 的 line2func
python -m line2func.website --out _site --serve    # 並在 http://127.0.0.1:8000/ 試用
```

每次 push 到 `main` 且測試通過後，由 `.github/workflows/pages.yml` 發布（repo 設定裡的 Pages 要選「GitHub Actions」）。Pyodide 從 jsDelivr 載入；`--pyodide-url` 可以改用另一份 Pyodide 314.0.7。

產生的資料夾（`out/`、`runs/`、`data/`、`_site/`、`.venv/`）已列在 `.gitignore`。

### 12. 現況與路線圖

這是 2.0 版。

- [x] 幾何核心與渲染器
- [x] 傳統引擎；SVG、Desmos、LaTeX 匯出；檢視器；`demo` 與 `serve`
- [x] 有標準答案的合成資料產生器，以及評估工具
- [x] 照片用的預訓練線稿模型（Informative Drawings，MIT；下載時檢查 SHA-256）
- [x] 直線與圓弧的具名算式、曲線精修、線寬與顏色
- [x] 網頁版（`python -m line2func`），繁體中文與英文介面
- [x] 第二遍描線、粗筆畫改為填滿的外框、渲染後比對（`--optimize`）
- [x] 粗重睫毛、陰影等填滿的區域，每一塊都量測自己有多深並照那個濃淡畫出來（所有區域共用一條間距算式，依深度畫成圈線或排線，Desmos 裡也分得出深淺）；預設最多 5,000 條曲線；淺色與極淡的線
- [x] 函數模式：每條曲線切成 `y = f(x)`／`x = g(y)` 的顯函數（`--form function`）
- [x] 單一畫面的網頁版，三種顯示方式；斷成點和虛線的線會保留；去雜訊強度拖動條（`--denoise`）
- [x] 淡線靈敏度（`--faint-sensitivity`）
- [x] 線上版網頁：line2func 在瀏覽器裡執行（Pyodide），以 GitHub Pages 發布
- [x] 任何圖片都可以，不只線稿：照片在開啟時就會被認出來，並先找出裡面的線條——用的是連貫線稿（`--lineart flow`），不需下載任何東西，所以瀏覽器裡也能跑。兩個頁面都會把那份線稿顯示出來、可以調整（〔線稿細節〕、`--lineart-detail`），確認後才描線

接下來：

- [ ] 從**區域**而不是邊緣抽線稿：把顏色量化、合併小區域，再描出區域邊界。這樣得到的是封閉迴圈而不是筆畫，而且每個區域都量得到自己的色調——那正是填滿區域那套機制已經在用的東西——所以照片可以畫成有明暗的，而不只是輪廓。它需要自己的曲線預算策略（一個區域一個迴圈、裡面再加環，是唯一會灌爆 5,000 條的做法），也需要自己的量測
