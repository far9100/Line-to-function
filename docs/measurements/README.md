# Kept measurements

Evaluation runs that **cannot be produced again**, because the code that made
one side of them has been removed. Everything else in this project can be
re-measured from the current tree, so nothing else belongs here.

| File | What it is |
|---|---|
| `decisions_rules.json` | `eval --realset data/real_v1` with the angle rules |
| `decisions_learned.json` | the same run with the learned decision scorer |

These two are the raw data behind the learned-against-rules table in
`docs/details.md`, section 7 ("Why there is no learned scorer"). The scorer was
removed in 8e43ed4 together with its weights, its training code and the
`--decisions` flag, so the `learned` side can never be measured again: this file
is the only surviving record of it. `data/real_v1` is gitignored as well, so
even the rules side cannot be reproduced from a clean checkout.

Compare them with:

    python -m line2func.eval --compare docs/measurements/decisions_rules.json docs/measurements/decisions_learned.json

Each file holds the per-drawing scores under `images` and the paired bootstrap
under `median`. The headline: precision +0.0044, PSNR +0.175 dB, SSIM +0.0030,
and +1.76 seconds per drawing - measurable, and not visible.
