"""Download and verify third-party pretrained weights.

    python -m line2func.weights list
    python -m line2func.weights fetch informative        # or: fetch all
    python -m line2func.weights verify

Weights are never bundled with the project. Every file in :data:`WEIGHTS` has a
fixed URL and SHA-256; a download is written to a temporary file, checked, and
only then moved into the cache (``$LINE2FUNC_HOME``, default
``~/.cache/line2func``). A file whose hash does not match is rejected. Only
resources whose license allows commercial use are listed (see
docs/third_party.md).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

WEIGHTS: dict[str, dict] = {
    "informative": {
        "file": "sk_model.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
        "sha256": "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
        "about": "Informative Drawings line-art generator, fine lines (Chan et al., CVPR 2022)",
        "license": "MIT, Copyright (c) 2022 Caroline Chan",
    },
    "informative-coarse": {
        "file": "sk_model2.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth",
        "sha256": "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
        "about": "Informative Drawings line-art generator, coarse lines (Chan et al., CVPR 2022)",
        "license": "MIT, Copyright (c) 2022 Caroline Chan",
    },
}


class WeightsError(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("LINE2FUNC_HOME") or Path.home() / ".cache" / "line2func")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_for(name: str) -> Path:
    if name not in WEIGHTS:
        raise WeightsError(f"unknown weights {name!r}; available: {', '.join(WEIGHTS)}")
    return cache_dir() / WEIGHTS[name]["file"]


def fetch(name: str, quiet: bool = False) -> Path:
    """Download ``name`` if needed and return its verified path."""
    spec = WEIGHTS.get(name)
    if spec is None:
        raise WeightsError(f"unknown weights {name!r}; available: {', '.join(WEIGHTS)}")
    dest = path_for(name)
    if dest.is_file() and _sha256(dest) == spec["sha256"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if not quiet:
        print(f"downloading {name} ({spec['license']}) from {spec['url']}", file=sys.stderr)
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "line2func"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if not quiet and total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    if not quiet:
        print(file=sys.stderr)
    digest = _sha256(tmp)
    if digest != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise WeightsError(f"{name}: SHA-256 mismatch (got {digest}, expected {spec['sha256']}); file rejected")
    os.replace(tmp, dest)
    return dest


def require(name: str) -> Path:
    """Verified path of already-downloaded weights, or a clear error telling how to fetch them."""
    dest = path_for(name)
    if not dest.is_file():
        raise WeightsError(f"weights {name!r} are not downloaded; run: python -m line2func.weights fetch {name}")
    if _sha256(dest) != WEIGHTS[name]["sha256"]:
        raise WeightsError(f"{dest} fails its SHA-256 check; re-download with: python -m line2func.weights fetch {name}")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.weights", description="Manage pretrained weights.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show available weights and their licenses")
    f = sub.add_parser("fetch", help="download and verify weights")
    f.add_argument("names", nargs="+", help="weights names, or 'all'")
    sub.add_parser("verify", help="check the hashes of downloaded weights")
    args = p.parse_args(argv)
    if args.cmd == "list":
        for name, spec in WEIGHTS.items():
            state = "downloaded" if path_for(name).is_file() else "not downloaded"
            print(f"{name:<20} {state:<15} {spec['about']}\n{'':<20} license: {spec['license']}")
        print(f"cache: {cache_dir()}")
        return 0
    if args.cmd == "fetch":
        names = list(WEIGHTS) if args.names == ["all"] else args.names
        try:
            for name in names:
                print(f"{name}: {fetch(name)}")
        except (WeightsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    bad = 0
    for name in WEIGHTS:
        dest = path_for(name)
        if not dest.is_file():
            print(f"{name:<20} not downloaded")
            continue
        ok = _sha256(dest) == WEIGHTS[name]["sha256"]
        bad += not ok
        print(f"{name:<20} {'OK' if ok else 'HASH MISMATCH'}  {dest}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
