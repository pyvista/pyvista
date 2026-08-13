// Client-side multi-select facet filter + name search for the dataset gallery.
// Builds a checkbox dropdown per facet from each card's `:class-card:`
// classes (e.g. "dtype-polydata"); a card matches when it has at least one
// checked value per facet with a selection (OR within a facet, AND across
// facets), and its search text contains the search query.
(function () {
  "use strict";

  function loadManifest() {
    // {labels: {"dtype-polydata": "PolyData", ...}, order: {"size": [...]}}
    const el = document.getElementById("facet-manifest");
    if (!el) return { labels: {}, order: {} };
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      return { labels: {}, order: {} };
    }
  }

  function collectFacetValues(cards, facet, order) {
    const present = new Set();
    cards.forEach((card) => {
      card.classList.forEach((cls) => {
        if (cls.indexOf(facet + "-") === 0) {
          present.add(cls.slice(facet.length + 1));
        }
      });
    });
    const explicitOrder = order[facet];
    const values = explicitOrder
      ? explicitOrder.filter((v) => present.has(v))
      : Array.from(present).sort();
    // Pin "N/A" first so it doesn't get lost among the real values.
    const naIndex = values.indexOf("na");
    if (naIndex > 0) {
      values.splice(naIndex, 1);
      values.unshift("na");
    }
    return values;
  }

  function buildPanel(dropdown, facet, values, labels, onChange) {
    const panel = dropdown.querySelector(".facet-panel");
    values.forEach((value) => {
      const id = "facet-" + facet + "-" + value;
      const label = document.createElement("label");
      label.className = "facet-option";
      label.setAttribute("for", id);

      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = id;
      input.value = value;
      input.addEventListener("change", onChange);

      const text = document.createElement("span");
      text.textContent = labels[facet + "-" + value] || value;

      label.appendChild(input);
      label.appendChild(text);
      panel.appendChild(label);
    });
  }

  function closeAllPanels(dropdowns) {
    dropdowns.forEach((dropdown) => {
      dropdown.querySelector(".facet-panel").hidden = true;
      dropdown
        .querySelector(".facet-toggle")
        .setAttribute("aria-expanded", "false");
    });
  }

  function getSelected(dropdown) {
    return Array.from(dropdown.querySelectorAll("input:checked")).map(
      (i) => i.value,
    );
  }

  function updateToggleCount(dropdown) {
    const n = getSelected(dropdown).length;
    dropdown.querySelector(".facet-toggle-count").textContent = n
      ? "(" + n + ")"
      : "";
  }

  function cardSearchText(card) {
    const el = card.querySelector(".gallery-search-text");
    return el ? el.textContent : "";
  }

  function applyFilters(cards, dropdowns, searchInput) {
    const active = {};
    dropdowns.forEach((dropdown) => {
      const selected = getSelected(dropdown);
      if (selected.length) active[dropdown.dataset.facet] = selected;
      updateToggleCount(dropdown);
    });

    const query = searchInput ? searchInput.value.trim().toLowerCase() : "";

    let visibleCount = 0;
    cards.forEach((card) => {
      const matchesFacets = Object.keys(active).every((facet) =>
        active[facet].some((value) =>
          card.classList.contains(facet + "-" + value),
        ),
      );
      const matchesSearch =
        !query || cardSearchText(card).indexOf(query) !== -1;
      const matches = matchesFacets && matchesSearch;
      card.classList.toggle("gallery-hidden", !matches);
      if (matches) visibleCount += 1;
    });

    const countEl = document.getElementById("filter-count");
    if (countEl) {
      countEl.textContent =
        visibleCount + " of " + cards.length + " datasets shown";
    }

    syncUrl(active, query);
  }

  function syncUrl(active, query) {
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    Object.keys(active).forEach((facet) => {
      params.set(facet, active[facet].join(","));
    });
    const queryString = params.toString();
    const newUrl =
      window.location.pathname +
      (queryString ? "?" + queryString : "") +
      window.location.hash;
    window.history.replaceState(null, "", newUrl);
  }

  function restoreFromUrl(dropdowns, searchInput) {
    const params = new URLSearchParams(window.location.search);
    dropdowns.forEach((dropdown) => {
      const raw = params.get(dropdown.dataset.facet);
      if (!raw) return;
      raw.split(",").forEach((value) => {
        const input = dropdown.querySelector(
          'input[value="' + CSS.escape(value) + '"]',
        );
        if (input) input.checked = true;
      });
    });
    const q = params.get("q");
    if (q && searchInput) searchInput.value = q;
  }

  function init() {
    const cards = Array.from(document.querySelectorAll(".gallery-card"));
    const bar = document.getElementById("gallery-filter-bar");
    if (!cards.length || !bar) return;

    const manifest = loadManifest();
    const dropdowns = Array.from(bar.querySelectorAll(".facet-dropdown"));
    const searchInput = document.getElementById("gallery-search");
    const onChange = () => applyFilters(cards, dropdowns, searchInput);

    dropdowns.forEach((dropdown) => {
      const facet = dropdown.dataset.facet;
      const values = collectFacetValues(cards, facet, manifest.order || {});
      if (!values.length) {
        dropdown.style.display = "none";
        return;
      }
      buildPanel(dropdown, facet, values, manifest.labels || {}, onChange);

      const toggle = dropdown.querySelector(".facet-toggle");
      const panel = dropdown.querySelector(".facet-panel");
      toggle.addEventListener("click", (evt) => {
        evt.stopPropagation();
        const isOpen = !panel.hidden;
        closeAllPanels(dropdowns);
        panel.hidden = isOpen;
        toggle.setAttribute("aria-expanded", String(!isOpen));
      });
    });

    document.addEventListener("click", (evt) => {
      const openDropdown = dropdowns.find(
        (d) => !d.querySelector(".facet-panel").hidden,
      );
      if (openDropdown && !openDropdown.contains(evt.target)) {
        closeAllPanels(dropdowns);
      }
    });

    document.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape") closeAllPanels(dropdowns);
    });

    if (searchInput) searchInput.addEventListener("input", onChange);

    restoreFromUrl(dropdowns, searchInput);

    const clearBtn = document.getElementById("filter-clear");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        dropdowns.forEach((dropdown) => {
          dropdown
            .querySelectorAll("input:checked")
            .forEach((i) => (i.checked = false));
        });
        if (searchInput) searchInput.value = "";
        applyFilters(cards, dropdowns, searchInput);
      });
    }

    applyFilters(cards, dropdowns, searchInput);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
