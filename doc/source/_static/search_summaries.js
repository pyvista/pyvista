// Show the target page's (or section's) lead paragraph as each search
// result's snippet, instead of upstream's raw match-context text, which on
// example pages usually lands mid-code-block. Page-level snippets come from
// the build-time `searchsummaries.json` index (one request instead of one
// page fetch per result); section-anchored results and pages missing from
// the index fall back to fetching the page, upstream style.
(function () {
  if (
    typeof Search === "undefined" ||
    typeof DOCUMENTATION_OPTIONS === "undefined"
  )
    return;

  // Upstream `_displayItem` fetches every result page for its snippet; turn
  // that off and render the snippets from the observer below instead.
  DOCUMENTATION_OPTIONS.SHOW_SEARCH_SUMMARY = false;

  // Markup that never belongs in a snippet.
  const NOISE = [
    "pre",
    ".highlight",
    "table",
    ".admonition",
    ".rubric",
    ".topic",
    ".sidebar",
    ".deprecated",
    ".versionadded",
    ".versionchanged",
    "a[download]",
    ".headerlink",
    "script",
    "style",
    "nav",
    "footer",
    ".prev-next-area",
  ].join(", ");

  // Same length budget as the upstream snippet.
  const MAX_LENGTH = 240;

  const contentRoot = document.documentElement.dataset.content_root ?? "";

  const summaryIndex = fetch(contentRoot + "searchsummaries.json")
    .then((response) => (response.ok ? response.json() : null))
    .catch(() => null);

  const searchTerms = (() => {
    try {
      const query = new URLSearchParams(window.location.search).get("q") ?? "";
      return Search._parseQuery(query)[1];
    } catch {
      return new Set();
    }
  })();

  const clip = (text) => {
    if (text.length <= MAX_LENGTH) return text;
    const cut = text.lastIndexOf(" ", MAX_LENGTH);
    return text.slice(0, cut > 0 ? cut : MAX_LENGTH) + "...";
  };

  const leadParagraph = (scope) => {
    for (const p of scope.querySelectorAll("p")) {
      const text = p.textContent.replace(/\s+/g, " ").trim();
      if (/[a-zA-Z]/.test(text)) return text;
    }
    return "";
  };

  // Upstream's behavior: the text around the last query-term match.
  const matchContext = (text) => {
    const textLower = text.toLowerCase();
    const start = [...searchTerms]
      .map((term) => textLower.indexOf(term.toLowerCase()))
      .filter((i) => i > -1)
      .slice(-1)[0];
    const from = Math.max((start ?? 0) - 120, 0);
    const head = from === 0 ? "" : "...";
    const tail = from + MAX_LENGTH < text.length ? "..." : "";
    return head + text.substr(from, MAX_LENGTH).trim() + tail;
  };

  const summarizePage = (htmlText, anchor) => {
    const page = new DOMParser().parseFromString(htmlText, "text/html");
    page.querySelectorAll(NOISE).forEach((el) => el.remove());
    let scope = null;
    if (anchor) {
      try {
        scope = page.querySelector(`[role="main"] ${anchor}`);
      } catch {
        scope = null; // anchors are not sanitized for use in selectors
      }
    }
    scope =
      scope ||
      page.querySelector('[role="main"] article') ||
      page.querySelector('[role="main"]') ||
      page.body;
    const lead = clip(leadParagraph(scope));
    if (lead) return lead;
    const full = scope.textContent.replace(/\s+/g, " ").trim();
    return full ? matchContext(full) : "";
  };

  const pageFetches = new Map();
  const fetchPage = (url) => {
    if (!pageFetches.has(url))
      pageFetches.set(
        url,
        fetch(url)
          .then((response) => (response.ok ? response.text() : ""))
          .catch(() => ""),
      );
    return pageFetches.get(url);
  };

  const attach = (listItem, text) => {
    if (!text) return;
    const summary = document.createElement("p");
    summary.classList.add("context");
    summary.textContent = text;
    listItem.appendChild(summary);
  };

  const annotate = async (listItem) => {
    // Results with a descriptor (`(Python class, in ...)`) get no snippet,
    // matching upstream.
    const link = listItem.querySelector("a");
    if (!link || listItem.querySelector(":scope > span, p.context")) return;
    const [path, anchor] = (link.getAttribute("href") ?? "").split("#");
    if (!path) return;

    if (!anchor) {
      const suffix = DOCUMENTATION_OPTIONS.LINK_SUFFIX;
      const docname = decodeURIComponent(
        path.endsWith(suffix) ? path.slice(0, -suffix.length) : path,
      );
      const index = await summaryIndex;
      if (index && index[docname]) {
        attach(listItem, clip(index[docname]));
        return;
      }
    }
    const htmlText = await fetchPage(contentRoot + path);
    attach(listItem, summarizePage(htmlText, anchor ? "#" + anchor : ""));
  };

  // This script loads in <head>, before #search-results exists.
  const start = () => {
    const results = document.getElementById("search-results");
    if (!results) return;
    new MutationObserver((mutations) => {
      for (const mutation of mutations)
        for (const node of mutation.addedNodes)
          if (node.nodeName === "LI") annotate(node);
    }).observe(results, { childList: true, subtree: true });
    results.querySelectorAll("ul.search li").forEach(annotate);
  };
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", start);
  else start();
})();
