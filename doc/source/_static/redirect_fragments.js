// Redirect pages with redundant anchors
// E.g. `site/foo.object#foo.Object` gets redirected to `site/foo.object`
(function () {
  if (!location.hash) return;

  const frag = location.hash.slice(1);
  const path = location.pathname;

  // Normalize: lowercase and strip punctuation
  function norm(s) {
    return s.toLowerCase().replace(/[._-]/g, "");
  }

  const tail = path.substring(path.lastIndexOf("/") + 1);

  if (norm(tail) === norm(frag)) {
    // Replace URL without reloading the full page twice
    history.replaceState({}, "", path);
  }
})();
