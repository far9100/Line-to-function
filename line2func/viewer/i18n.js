// UI text in English and Traditional Chinese (i18n.json).
//
// Static text: elements carry data-i18n (text), data-i18n-title (tooltip) or
// data-i18n-aria-label; applyI18n() fills them in. Dynamic text: t(key, vars).
// A translation may be {one, other} for English plurals, chosen by vars.n.
// Numbers passed as vars are formatted for the language; numbers inside
// equations are never localized (they are built elsewhere).
import STRINGS from "./i18n.json" with { type: "json" };

export const LANGS = ["en", "zh-TW"];
let lang = "en";
const listeners = new Set();
const pluralRules = new Map();

// The language to start with: the saved choice, else the browser's (any Chinese -> zh-TW).
export function detectLang(saved) {
  if (LANGS.includes(saved)) return saved;
  const prefs = (typeof navigator !== "undefined" && (navigator.languages || [navigator.language])) || [];
  for (const pref of prefs) {
    const low = String(pref || "").toLowerCase();
    if (low.startsWith("zh")) return "zh-TW";
    if (low.startsWith("en")) return "en";
  }
  return "en";
}

export function getLang() {
  return lang;
}

export function setLang(next) {
  lang = LANGS.includes(next) ? next : "en";
  document.documentElement.lang = lang;
  applyI18n(document);
  for (const listener of listeners) listener(lang);
}

export function onLangChange(listener) {
  listeners.add(listener);
}

function format(value) {
  return typeof value === "number" ? value.toLocaleString(lang) : String(value);
}

// Translate `key`, filling {name} placeholders from `vars`. Unknown keys return `fallback`
// (or the key itself), so a missing translation never breaks the page.
export function t(key, vars = {}, fallback) {
  let text = STRINGS[lang]?.[key] ?? STRINGS.en[key];
  if (text === undefined) return fallback !== undefined ? fallback : key;
  if (typeof text === "object") {
    let rules = pluralRules.get(lang);
    if (!rules) pluralRules.set(lang, (rules = new Intl.PluralRules(lang)));
    text = text[rules.select(Number(vars.n ?? 0))] ?? text.other;
  }
  return text.replace(/\{(\w+)\}/g, (match, name) => (name in vars ? format(vars[name]) : match));
}

export function applyI18n(root) {
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of root.querySelectorAll("[data-i18n-aria-label]")) el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel));
}
