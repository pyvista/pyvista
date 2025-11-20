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

  // Strip trailing .html for comparison and redirect target
  const cleanPath = path.replace(/\.html$/, "");
  const tail = cleanPath.substring(cleanPath.lastIndexOf("/") + 1);

  if (norm(tail) === norm(frag)) {
    // Replace URL without reloading the page
    history.replaceState({}, "", cleanPath);
  }
})();
