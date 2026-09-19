# Third-party code, models and data

line2func uses only resources whose license allows commercial use. Third-party
pretrained weights are **never bundled**: `python -m line2func.weights fetch NAME` downloads
them into `$LINE2FUNC_HOME` (default `~/.cache/line2func`) and rejects any file
whose SHA-256 differs from the value pinned in `line2func/weights.py`.

The one bundled weights file, `line2func/data/decisions_v1.npz` (the learned
decision scorer, 150 KB), was trained by this project with
`line2func.train_decisions`. Its training data is line2func's own synthetic
generator (`line2func.synth`), so it contains no third-party data or code.

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
- Model design inspired by *Deep Vectorization of Technical Drawings* (ECCV 2020, arXiv:2003.05471) and DETR (Carion et al., ECCV 2020).
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
