// Show the target page's (or section's) lead paragraph as each search
// result's snippet, instead of upstream's raw match-context text, which on
// example pages usually lands mid-code-block.
(function () {
  if (typeof Search === "undefined" || !Search.makeSearchSummary) return;

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
  ].join(", ");

  // Same length budget as the upstream snippet.
  const MAX_LENGTH = 240;

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
  const matchContext = (text, keywords) => {
    const textLower = text.toLowerCase();
    const start = [...keywords]
      .map((k) => textLower.indexOf(k.toLowerCase()))
      .filter((i) => i > -1)
      .slice(-1)[0];
    const from = Math.max((start ?? 0) - 120, 0);
    const head = from === 0 ? "" : "...";
    const tail = from + MAX_LENGTH < text.length ? "..." : "";
    return head + text.substr(from, MAX_LENGTH).trim() + tail;
  };

  Search.makeSearchSummary = (htmlText, keywords, anchor) => {
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
    scope = scope || page.querySelector('[role="main"]') || page.body;

    let text = clip(leadParagraph(scope));
    if (!text) {
      const fullText = scope.textContent.replace(/\s+/g, " ").trim();
      if (fullText === "") return null;
      text = matchContext(fullText, keywords);
    }

    const summary = document.createElement("p");
    summary.classList.add("context");
    summary.textContent = text;
    return summary;
  };
})();
