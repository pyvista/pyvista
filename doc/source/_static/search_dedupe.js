// Collapse duplicate search results: Sphinx searches objects, titles, and
// full text independently, so an autosummary stub page shows up both as an
// object hit (`pyvista.Plotter.add_mesh (Python method, in Plotter.add_mesh)`)
// and as a page hit (`Plotter.add_mesh`). Drop the page hit, then turn the
// object hit into a page-style hit (no anchor, no descriptor) so it links to
// the page top and gets a docstring summary (see `search_summaries.js`).
(function () {
  if (typeof Search === "undefined" || !Search._performSearch) return;

  // Kinds from `SearchResultKind` in searchtools.js. Object and index hits
  // point at a specific definition; text and title hits are the page itself.
  const ANCHOR_KINDS = new Set(["object", "index"]);
  const PAGE_KINDS = new Set(["text", "title"]);

  // Each result is [docname, title, anchor, descr, score, filename, kind].
  const DOCNAME = 0;
  const TITLE = 1;
  const ANCHOR = 2;
  const DESCR = 3;
  const KIND = 6;

  // An object hit whose full name is its page's basename is the page.
  const isOwnPage = (result) =>
    result[KIND] === "object" &&
    (result[DOCNAME] === result[TITLE] ||
      result[DOCNAME].endsWith("/" + result[TITLE]));

  const performSearch = Search._performSearch;

  Search._performSearch = function (...args) {
    const results = performSearch.apply(this, args);

    const anchored = new Set();
    for (const result of results) {
      if (result[ANCHOR] && ANCHOR_KINDS.has(result[KIND]))
        anchored.add(result[DOCNAME]);
    }
    if (!anchored.size) return results;

    const deduped = results.filter(
      (result) =>
        !(
          PAGE_KINDS.has(result[KIND]) &&
          !result[ANCHOR] &&
          anchored.has(result[DOCNAME])
        ),
    );

    for (const result of deduped) {
      if (isOwnPage(result)) {
        result[ANCHOR] = "";
        result[DESCR] = null;
      }
    }
    return deduped;
  };
})();
