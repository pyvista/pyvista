// Turn the URL behind this 404 into suggestions. The fragment never reaches
// the server, so the page reads the URL itself: it splits the path and
// fragment into search terms, most specific first, then looks them up in the
// site's search index for pages and objects with exactly that name ("Did you
// mean") and for the best title and object hits ("Related pages"). Plain
// search links are rendered first and stay as the fallback.
(function () {
  // The page is served for any missing path, so URLs are built from the site
  // root rather than from the page-relative `data-content_root`.
  const script = document.currentScript;
  const root = script
    ? new URL(script.src, location.href).pathname.replace(/_static\/[^/]*$/, "")
    : "/";

  // Path pieces that describe the site's layout rather than its content,
  // compared without the leading underscore Sphinx gives its own directories.
  const NOISE = new Set([
    "pyvista",
    "api",
    "autosummary",
    "index",
    "version",
    "stable",
    "dev",
    "latest",
    "html",
    "htm",
    "static",
    "images",
    "downloads",
    "modules",
    "sources",
    "py",
    "ipynb",
    "rst",
    "txt",
    "png",
    "jpg",
    "jpeg",
    "svg",
    "gif",
    "zip",
  ]);
  const VERSION_RE = /^v?\d+(\.\d+)*$/;
  const MAX_TERMS = 5; // search links shown
  const MAX_QUERIES = 4; // terms run through the index
  const MAX_EXACT = 3;
  const MAX_RELATED = 5;

  const decode = (text) => {
    try {
      return decodeURIComponent(text);
    } catch {
      return text;
    }
  };

  // Words of one or two letters are mostly stopwords (`is`, `of`, `to`).
  const isNoise = (word) =>
    (word.length < 3 && !/\p{N}/u.test(word)) ||
    NOISE.has(word.toLowerCase().replace(/^_+/, "")) ||
    VERSION_RE.test(word);

  // The requested page relative to the site root, without the `.html` suffix
  // and without a `version/<x.y>/` prefix, and the ranked search terms it and
  // the fragment yield. A path segment or the fragment goes in whole
  // (`DataSetFilters.voxelize`, `what mesh`), then identifier by identifier
  // and word by word; `_`-joined and CamelCase parts go last, since `add_mesh`
  // and `DataSetFilters` are better queries than `add`, `mesh` or `Data`.
  // Only the last segment and the fragment can name an object outright, and
  // the fragment is the most specific name the URL carries, so exact matches
  // are looked up in that order; directories never name an object.
  const parseUrl = () => {
    const pathname = decode(location.pathname);
    const relative = pathname.startsWith(root)
      ? pathname.slice(root.length)
      : pathname.slice(1);
    const versioned = relative.match(/^version\/([^/]+)\/(.*)$/);
    const page = (versioned ? versioned[2] : relative).replace(/\.html?$/i, "");
    const segments = page.split("/").filter(Boolean);
    const fragment = decode(location.hash.slice(1));

    const terms = [];
    const extras = [];
    const whole = new Set();
    const seen = new Set();
    const add = (list, term) => {
      const key = term.toLowerCase();
      if (seen.has(key) || isNoise(key)) return;
      seen.add(key);
      list.push(term);
    };
    const addPiece = (text, { names = false } = {}) => {
      const identifier = text.replace(/^pyvista\./, "");
      const identifiers = identifier
        .split(/[^\p{L}\p{N}_]+/u)
        .filter((word) => !isNoise(word));
      const words = identifiers
        .flatMap((word) => word.split("_"))
        .filter((word) => !isNoise(word));
      if (!words.length) return;
      const dotted = identifier.split(".").filter((part) => !isNoise(part));
      const piece = dotted.length > 1 ? dotted.join(".") : words.join(" ");
      if (names) {
        whole.add(piece.toLowerCase());
        if (identifiers.length === 1) whole.add(identifiers[0].toLowerCase());
      }
      add(terms, piece);
      identifiers.forEach((word) => add(terms, word));
      words.forEach((word) => add(terms, word));
      words.forEach((word) => {
        const parts = word
          .replace(/([\p{Ll}\p{N}])(\p{Lu})/gu, "$1 $2")
          .split(" ");
        if (parts.length > 1) parts.forEach((part) => add(extras, part));
      });
    };
    if (segments.length)
      addPiece(segments[segments.length - 1], { names: true });
    const fromFragment = terms.length;
    if (fragment) addPiece(fragment, { names: true });
    const fragmentTerms = terms.slice(fromFragment);
    segments
      .slice(0, -1)
      .reverse()
      .forEach((segment) => addPiece(segment));
    terms.push(...extras);
    return {
      page,
      versioned,
      terms,
      whole,
      lookup: [...fragmentTerms, ...terms],
    };
  };

  // Titles in the search index are HTML, e.g. `&lt;no title&gt;`, and object
  // names carry the package prefix that the page titles leave out.
  const label = (name) => {
    const textarea = document.createElement("textarea");
    textarea.innerHTML = name;
    return textarea.value.replace(/^pyvista\./, "");
  };
  const link = (href, text) => {
    const a = document.createElement("a");
    a.href = href;
    a.textContent = label(text);
    return a;
  };
  const paragraph = (...content) => {
    const p = document.createElement("p");
    p.append(...content);
    return p;
  };
  const section = (heading, items) => {
    const el = document.createElement("section");
    const h2 = document.createElement("h2");
    h2.textContent = heading;
    const ul = document.createElement("ul");
    for (const [href, text] of items) {
      const li = document.createElement("li");
      li.append(link(href, text));
      ul.append(li);
    }
    el.append(h2, ul);
    return el;
  };
  const searchUrl = (term) =>
    root + "search.html?q=" + encodeURIComponent(term);
  const pageUrl = (docname, anchor) =>
    root +
    docname +
    DOCUMENTATION_OPTIONS.LINK_SUFFIX +
    (anchor ? "#" + anchor : "");

  const loadScript = (src) =>
    new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src;
      el.onload = resolve;
      el.onerror = reject;
      document.head.append(el);
    });

  // What the search page loads on its own; only the search page loads them.
  const loadIndex = async () => {
    if (typeof Search === "undefined")
      await loadScript(root + "_static/searchtools.js");
    if (typeof Stemmer === "undefined")
      await loadScript(root + "_static/language_data.js");
    if (!Search.hasIndex()) await loadScript(root + "searchindex.js");
  };

  // Pages whose path or file name is the requested one, then the objects named
  // by the most specific term that names any: by full name, or by last dotted
  // part when the term was a whole segment and few objects share the name.
  const exactMatches = (index, { page, lookup, whole }, shown) => {
    const hits = [];
    const listed = new Set();
    const push = (docIndex, anchor, text) => {
      const docname = index.docnames[docIndex];
      if (listed.has(docname + "#" + anchor)) return;
      listed.add(docname + "#" + anchor);
      shown.add(docname);
      hits.push([pageUrl(docname, anchor), text]);
    };

    const wanted = page.toLowerCase();
    const basename = wanted.slice(wanted.lastIndexOf("/") + 1);
    index.docnames.forEach((docname, i) => {
      const lower = docname.toLowerCase();
      if (
        lower === wanted ||
        (!isNoise(basename) &&
          (lower === basename || lower.endsWith("/" + basename)))
      )
        push(i, "", index.titles[i]);
    });

    // Each entry is [docIndex, objtypeIndex, priority, anchor, name].
    const byName = new Map();
    const byLastName = new Map();
    const remember = (map, key, value) => {
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(value);
    };
    for (const [prefix, entries] of Object.entries(index.objects)) {
      for (const [docIndex, objtype, , anchor, name] of entries) {
        const fullname = prefix ? prefix + "." + name : name;
        const docname = index.docnames[docIndex];
        const ownPage =
          docname === fullname || docname.endsWith("/" + fullname);
        const object = {
          docIndex,
          text: fullname,
          anchor: ownPage
            ? ""
            : anchor === ""
              ? fullname
              : anchor === "-"
                ? index.objnames[objtype][1] + "-" + fullname
                : anchor,
        };
        remember(byName, label(fullname).toLowerCase(), object);
        remember(byLastName, name.toLowerCase(), object);
      }
    }
    for (const term of lookup) {
      const key = term.toLowerCase();
      const named = byName.get(key) ?? [];
      const lastNamed = (
        whole.has(key) ? (byLastName.get(key) ?? []) : []
      ).filter((object) => !named.includes(object));
      const matches =
        named.length + lastNamed.length <= MAX_EXACT
          ? [...named, ...lastNamed]
          : named;
      for (const { docIndex, anchor, text } of matches)
        push(docIndex, anchor, text);
      if (matches.length) break;
    }
    return hits.slice(0, MAX_EXACT);
  };

  // `Search._parseQuery` stores the query for sphinx_highlight.js to mark on
  // the next page, which would put a "Hide Search Matches" bar on it.
  const parseQuery = (query) => {
    const key = "sphinx_highlight_terms";
    const stored = localStorage.getItem(key);
    try {
      return Search._parseQuery(query);
    } finally {
      if (stored === null) localStorage.removeItem(key);
      else localStorage.setItem(key, stored);
    }
  };

  // The search page's own ranking for the most specific term that finds
  // anything. Body-text matches score `Scorer.term`; anything above it matched
  // a title or an object name, which is the bar for calling a page related.
  const relatedPages = ({ terms }, shown) => {
    const hits = [];
    for (const query of terms.slice(0, MAX_QUERIES)) {
      const [, searchTerms, excludedTerms, highlightTerms, objectTerms] =
        parseQuery(query);
      const results = Search._performSearch(
        query,
        searchTerms,
        excludedTerms,
        highlightTerms,
        objectTerms,
      );
      // Sorted for `pop()`, so the best result comes last.
      for (const [docname, title, anchor, , score] of results.reverse()) {
        if (score <= Scorer.term || shown.has(docname)) continue;
        shown.add(docname);
        hits.push([pageUrl(docname, anchor.replace(/^#/, "")), title]);
        if (hits.length === MAX_RELATED) return hits;
      }
      if (hits.length) break;
    }
    return hits;
  };

  const suggest = (container, request) => {
    const shown = new Set();
    const before = container.firstChild;
    const exact = exactMatches(Search._index, request, shown);
    if (exact.length)
      container.insertBefore(section("Did you mean", exact), before);
    const related = relatedPages(request, shown);
    if (related.length)
      container.insertBefore(section("Related pages", related), before);
  };

  const start = () => {
    const container = document.getElementById("notfound");
    if (!container) return;
    const request = parseUrl();
    const { page, versioned, terms } = request;

    if (terms.length) {
      const links = terms
        .slice(0, MAX_TERMS)
        .map((term) => link(searchUrl(term), term));
      container.append(
        paragraph(
          "Maybe search for: ",
          ...links.flatMap((el, i) => (i ? [", ", el] : [el])),
          ".",
        ),
      );
    }
    if (versioned)
      container.append(
        paragraph(
          `This link is for version ${versioned[1]}. Try the `,
          link(
            root + versioned[2] + location.hash,
            "same page in the current version",
          ),
          ".",
        ),
      );

    if (terms.length || page)
      loadIndex()
        .then(() => suggest(container, request))
        .catch(() => {});
  };

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", start);
  else start();
})();
