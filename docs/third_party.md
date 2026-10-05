# Third-party code, models and data

line2func uses only resources whose license allows commercial use. Third-party
pretrained weights are **never bundled**: `python -m line2func.weights fetch NAME` downloads
them into `$LINE2FUNC_HOME` (default `~/.cache/line2func`) and rejects any file
whose SHA-256 differs from the value pinned in `line2func/weights.py`.

Nothing here is bundled: line2func ships no weights of its own and no trained
model of any kind. The two files below are downloaded on request.

## Measurement corpora

Neither is in the repository, and neither is published with it.

| Set | What | Licence |
|---|---|---|
| 66 + 15 line drawings | The tracer's own measurements throughout this manual | Third-party line art, kept locally, never redistributed |
| 32 photographs | The photo-path measurements in section 6 of `docs/details.md` | **CC0**, from Wikimedia Commons (the Unsplash import). Each file's licence was read from the API and checked, not assumed from its category; a `sources.json` beside them records every title, page URL and licence string |

## Methods implemented from their papers

No code is copied; these are the published methods the implementations follow,
and the papers are cited for the ideas rather than for any source.

| Where | Method |
|---|---|
| `lineart.flow`, `lineart.flow_field`, `lineart.flow_dog` | Coherent line drawing: Henry Kang, Seungyong Lee and Charles K. Chui, *Coherent Line Drawing*, Proc. NPAR 2007. The flow is built as the minor eigenvector of a smoothed structure tensor, after Jan Eric Kyprianidis and Jürgen Döllner, rather than by Kang's iterative smoothing; the two are the same field, and the note in `flow_field` says why one Gaussian replaces the rounds. |
| `lineart.flow_field` (the colour axis) | The structure tensor summed over the colour channels: Silvano Di Zenzo, *A note on the gradient of a multi-image*, Computer Vision, Graphics, and Image Processing 33, 1986. |
| `lineart.noise_level` | John Immerkaer, *Fast noise variance estimation*, Computer Vision and Image Understanding 64(2), 1996, with the median in place of the mean so that edges do not count as noise. |
| `lineart.xdog` | eXtended difference-of-Gaussians: Holger Winnemöller, Jan Eric Kyprianidis and Sven C. Olsen, *XDoG*, Computers & Graphics 36(6), 2012. Only the edge term is kept; the docstring says why. |
| `lineart.canny` | John Canny, *A Computational Approach to Edge Detection*, PAMI 1986. |
| `baseline.thin` | The Guo-Hall and Lam-Lee-Suen thinning rules, written from the published lookup tables. |

## Pretrained weights

| Name | File | Source | License | SHA-256 |
|---|---|---|---|---|
| `informative` | `sk_model.pth` (17.2 MB) | Informative Drawings, fine lines; mirrored at `huggingface.co/lllyasviel/Annotators` | MIT, Copyright (c) 2022 Caroline Chan | `c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6` |
| `informative-coarse` | `sk_model2.pth` (17.2 MB) | Informative Drawings, coarse lines; same mirror | MIT, Copyright (c) 2022 Caroline Chan | `30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0` |

The Hugging Face repository declares "license: other" because it bundles
models under several licenses. These two files are the Informative Drawings
generators; their license is the MIT license of the original project, which is
also shipped next to them in ControlNet (`annotator/lineart/LICENSE`).

## Loaded by the online page

The online page (`python -m line2func.website`) bundles none of these. The
browser loads them from the jsDelivr CDN at run time, pinned to Pyodide
314.0.7; Pyodide checks every package against the SHA-256 in its lock file.

| What | License |
|---|---|
| [Pyodide](https://pyodide.org) 314.0.7: CPython 3.14 compiled to WebAssembly | MPL-2.0 (Pyodide), PSF License (CPython) |
| NumPy 2.4.6 | BSD-3-Clause |
| SciPy 1.18.0 | BSD-3-Clause |
| Pillow 12.2.0 | MIT-CMU (HPND) |

## Ported code

| Where | From | License |
|---|---|---|
| `line2func/lineart_model.py` (`Generator`) | [carolineec/informative-drawings](https://github.com/carolineec/informative-drawings) `model.py`, with ControlNet's lineart configuration (3 residual blocks) | MIT, Copyright (c) 2022 Caroline Chan |
| `line2func/baseline.py` (`thin` lookup tables) | Reimplemented from the thinning algorithm in Lam, Lee & Suen (1992), as used by `skimage.morphology.thin` | Algorithm description; no code copied |

## Algorithms (no code used)

- Curve fitting: Schneider, "An Algorithm for Automatically Fitting Digitized Curves", *Graphics Gems*, 1990.
- `line2func/boundary.py` (`contours`): Moore-neighbour contour following, the classical
  eight-neighbour outline walk. No particular source was followed.
- The total-length metric (`metrics.stroke_length_scores`) follows the stroke-length error reported in
  *Deep Sketch Vectorization* (SIGGRAPH 2024); the resolution-relative distance unit and the F sweep
  (`metrics.distance_base`, `metrics.f_sweep`) follow the Rough Sketch Cleanup Benchmark's conventions.
- XDoG: Winnemöller, Kyprianidis & Olsen, 2012.

## MIT license text (Informative Drawings)

```
MIT License

Copyright (c) 2022 Caroline Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
