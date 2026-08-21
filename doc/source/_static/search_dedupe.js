// Collapse duplicate search results.
//
// Sphinx searches several indices independently (objects, titles, full text) and
// concatenates the hits, so an API page shows up twice. Searching `add_mesh`
// lists both
//
//   pyvista.Plotter.add_mesh (Python method, in Plotter.add_mesh)
//       -> pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh
//   Plotter.add_mesh
//       -> pyvista.Plotter.add_mesh.html
//
// The de-duplication in Sphinx's own `searchtools.js` keys on the anchor and the
// description, so it never merges the two. Every autosummary stub page hits this
// because the page documents a single object whose name is also the page title.
//
// Drop the whole-page hit whenever the same page is already listed through an
// anchor, keeping the object hit: it links straight to the definition and says
// what kind of object it is.
(function () {
  if (typeof Search === "undefined" || !Search._performSearch) return;

  // Kinds from `SearchResultKind` in searchtools.js. Object and index hits
  // point at a specific definition; text and title hits are the page itself.
  const ANCHOR_KINDS = new Set(["object", "index"]);
  const PAGE_KINDS = new Set(["text", "title"]);

  // Each result is [docname, title, anchor, descr, score, filename, kind].
  const DOCNAME = 0;
  const ANCHOR = 2;
  const KIND = 6;

  const performSearch = Search._performSearch;

  Search._performSearch = function (...args) {
    const results = performSearch.apply(this, args);

    const anchored = new Set();
    for (const result of results) {
      if (result[ANCHOR] && ANCHOR_KINDS.has(result[KIND]))
        anchored.add(result[DOCNAME]);
    }
    if (!anchored.size) return results;

    return results.filter(
      (result) =>
        !(
          PAGE_KINDS.has(result[KIND]) &&
          !result[ANCHOR] &&
          anchored.has(result[DOCNAME])
        ),
    );
  };
})();
