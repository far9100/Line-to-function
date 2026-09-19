"""The viewer's two languages: complete, consistent, and every text on the page goes through them."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from line2func import pipeline

VIEWER = Path(__file__).resolve().parents[1] / "line2func" / "viewer"
STRINGS = json.loads((VIEWER / "i18n.json").read_text(encoding="utf-8"))
HTML = (VIEWER / "index.html").read_text(encoding="utf-8")
SCRIPTS = {p.name: p.read_text(encoding="utf-8") for p in VIEWER.glob("*.js")}
# keys built at run time from codes the server sends (stage, step, error, ...)
DYNAMIC = ("stage.", "step.", "error.", "warning.", "tag.", "shape.")
CJK = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")


def _placeholders(text) -> set[str]:
    if isinstance(text, dict):
        return set().union(*(_placeholders(v) for v in text.values()))
    return set(re.findall(r"\{(\w+)\}", text))


def _used_keys() -> set[str]:
    """Keys in data-i18n attributes and t("...") calls; "prefix." entries come from t("prefix." + code)."""
    keys = set(re.findall(r'data-i18n(?:-title|-aria-label)?="([^"]+)"', HTML))
    for source in SCRIPTS.values():
        keys |= set(re.findall(r'\bt\(\s*"([^"]+)"', source))
        # keys chosen in an expression, e.g. t(mode === "static" ? "static.title" : "app.title")
        keys |= {k for k in re.findall(r'"([\w-]+(?:\.[\w-]+)+)"', source) if k in STRINGS["en"]}
    return keys


def test_both_languages_have_the_same_keys_and_placeholders():
    en, zh = STRINGS["en"], STRINGS["zh-TW"]
    assert set(en) == set(zh)
    for key in en:
        assert _placeholders(en[key]) == _placeholders(zh[key]), key
        for text in (en[key], zh[key]):
            if isinstance(text, dict):
                assert "other" in text, key  # plural forms always have the general case


def test_every_key_the_page_uses_exists():
    used = _used_keys()
    prefixes = sorted(k for k in used if k.endswith("."))
    assert all(p in DYNAMIC for p in prefixes), prefixes
    assert sorted(k for k in used if not k.endswith(".") and k not in STRINGS["en"]) == []


def test_every_key_is_used_or_built_from_a_code():
    used = _used_keys()
    assert sorted(k for k in STRINGS["en"] if k not in used and not k.startswith(DYNAMIC)) == []


def test_codes_from_the_server_are_translated():
    en = STRINGS["en"]
    for stage in (*pipeline.STAGES, "resize", "load_model", "export", "quality", "encode"):
        assert f"stage.{stage}" in en, stage
    app_source = (VIEWER.parent / "app.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'ApiError\(HTTPStatus\.\w+, "(\w+)"', app_source)) | {"out_of_memory", "internal"}
    generic = {"bad_json", "bad_request", "forbidden", "length_required", "not_found", "not_ready"}  # -> error.generic
    assert sorted(c for c in codes - generic if f"error.{c}" not in en) == []
    for warning in ("no_lines", "over_desmos_limit", "quality_skipped"):
        assert f"warning.{warning}" in en


def test_chinese_only_lives_in_the_translations():
    for name, source in [("index.html", HTML), *SCRIPTS.items()]:
        stray = [ch for ch in CJK.findall(source.replace(">中文<", "><"))]  # the language button names itself
        assert not stray, (name, "".join(stray)[:40])


class _VisibleText(HTMLParser):
    """Text nodes of the page body that are not covered by a data-i18n attribute."""

    SKIP = {"script", "style", "title"}
    OK = {"line2func", "中文", "EN", "SVG", "JSON", "Desmos", "LaTeX", "ZIP", "−", "+", "2×"}

    def __init__(self):
        super().__init__()
        self.stack: list[tuple[str, bool]] = []
        self.untranslated: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("input", "img", "br", "meta", "link"):
            return
        self.stack.append((tag, any(name == "data-i18n" for name, _ in attrs)))

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()[0] == tag:
                break

    def handle_data(self, data):
        text = data.strip()
        if not text or any(tag in self.SKIP for tag, _ in self.stack) or text in self.OK:
            return
        if not (self.stack and self.stack[-1][1]):
            self.untranslated.append(text)


def test_all_visible_page_text_is_translated():
    parser = _VisibleText()
    parser.feed(HTML)
    assert parser.untranslated == []
