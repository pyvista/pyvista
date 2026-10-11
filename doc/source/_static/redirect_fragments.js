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

// Point an object anchor at the section that documents it
// E.g. `typing#pyvista.typing.VectorLikeInt` goes to `typing#pyvista-typing-vectorlikeint`
(function () {
  function toSection() {
    const frag = decodeURIComponent(location.hash.slice(1));
    if (!frag) return;
    const target = document.getElementById(frag);
    const section = target && target.closest("section");
    if (!section) return;
    if (section.id !== frag.toLowerCase().replace(/[._]/g, "-")) return;
    history.replaceState(null, "", "#" + section.id);
    section.scrollIntoView();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", toSection);
  } else {
    toSection();
  }
  window.addEventListener("hashchange", toSection);
})();
