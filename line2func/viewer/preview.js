// What the page shows about a photo's line art, worked out apart from the page so it can be tested without
// one. `on` is whether the open image goes through an extractor at all; `preview` is app.js's S.preview
// ({url, fresh, pending, error, ...}) or null. `url` is the picture that is up, and it stays up while the next
// one is made, so it says nothing about being current: `fresh` does, and only a fresh picture may be traced -
// Convert sends the settings as they are now, which a stale picture was not made with.
export function lineartState(on, preview) {
  if (!on) return { state: "off", shown: false, ready: false, convertDisabled: false, status: "" };
  const failed = !!preview?.error, pending = !failed && !!preview?.pending;
  const ready = !failed && !pending && !!preview?.url && !!preview.fresh;
  return {
    state: failed ? "failed" : pending ? "running" : ready ? "ready" : "off",
    shown: !!preview?.url, // there is a picture to compare with the original, current or not
    ready,
    convertDisabled: !ready, // a photo is looked at before it is traced
    status: failed ? "error" : pending ? "stage" : ready ? "note" : "",
  };
}
