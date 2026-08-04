// Force the header download buttons to save rather than render.
//
// sphinx-book-theme renders them with its generic link macro, so the browser
// displays .py and .ipynb inline as text. Sphinx's own download links set the
// HTML5 `download` attribute; do the same for these.
(function () {
  const SELECTOR =
    "a.btn-download-source-button, a.btn-download-notebook-button";

  function markDownloads() {
    for (const link of document.querySelectorAll(SELECTOR)) {
      link.setAttribute("download", "");
      // Redundant once the link downloads, and it flashes a blank tab
      link.removeAttribute("target");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", markDownloads);
  } else {
    markDownloads();
  }
})();
