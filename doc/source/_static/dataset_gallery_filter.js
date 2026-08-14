// Client-side multi-select facet filter + name search for the dataset gallery.
// Builds a checkbox dropdown per facet from each card's `:class-card:`
// classes (e.g. "dtype-polydata"); a card matches when it has at least one
// checked value per facet with a selection (OR within a facet, AND across
// facets), and its search text contains the search query. Also drives the
// prev/next buttons, a more reliable way to move between cards than the
// carousel's native drag/swipe scrolling (finicky to hit on mobile).
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
    // "na" is handled separately (always first, if present) rather than via
    // the explicit order list, since that list only covers real values.
    const rest = Array.from(present).filter((v) => v !== "na");
    const explicitOrder = order[facet];
    const ordered = explicitOrder
      ? explicitOrder.filter((v) => rest.includes(v))
      : rest.sort();
    return present.has("na") ? ["na", ...ordered] : ordered;
  }

  function buildPanel(dropdown, facet, values, labels, onChange) {
    const panel = dropdown.querySelector(".facet-panel");
    // The panel floats over card content below it; without this, a click on
    // an option also bubbles out to whatever's underneath at those
    // coordinates (e.g. a card's link), triggering that too.
    panel.addEventListener("click", (evt) => evt.stopPropagation());
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

  function visibleCards(cards) {
    return cards.filter((c) => !c.classList.contains("gallery-hidden"));
  }

  // Index, among currently-visible cards, of whichever one is flush (or
  // closest) to the carousel's left edge - i.e. the one currently in view.
  function currentCardIndex(carousel, visible) {
    if (!carousel || !visible.length) return -1;
    const carouselLeft = carousel.getBoundingClientRect().left;
    let bestIndex = 0;
    let bestDelta = Infinity;
    visible.forEach((card, i) => {
      const delta = Math.abs(card.getBoundingClientRect().left - carouselLeft);
      if (delta < bestDelta) {
        bestDelta = delta;
        bestIndex = i;
      }
    });
    return bestIndex;
  }

  // Scrolls only the carousel horizontally, never the page - unlike
  // card.scrollIntoView(), which can also shift the page vertically to bring
  // more of a tall card into view.
  function scrollToCard(carousel, card, behavior) {
    if (!carousel) return;
    const delta =
      card.getBoundingClientRect().left - carousel.getBoundingClientRect().left;
    carousel.scrollTo({
      left: carousel.scrollLeft + delta,
      behavior: behavior || "auto",
    });
  }

  function updateNavButtons(carousel, cards, prevBtn, nextBtn) {
    if (!prevBtn || !nextBtn) return;
    const visible = visibleCards(cards);
    const idx = currentCardIndex(carousel, visible);
    prevBtn.disabled = idx <= 0;
    nextBtn.disabled = idx === -1 || idx >= visible.length - 1;
  }

  function applyFilters(
    cards,
    dropdowns,
    searchInput,
    carousel,
    prevBtn,
    nextBtn,
  ) {
    // Remember what's currently in view so it's still in view afterwards,
    // as long as this filter change doesn't itself exclude it.
    const anchorCard = carousel
      ? visibleCards(cards)[currentCardIndex(carousel, visibleCards(cards))]
      : undefined;

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

    if (
      carousel &&
      anchorCard &&
      !anchorCard.classList.contains("gallery-hidden")
    ) {
      scrollToCard(carousel, anchorCard, "auto");
    }
    updateNavButtons(carousel, cards, prevBtn, nextBtn);

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
    const carousel = cards[0].closest(".sd-cards-carousel");
    const prevBtn = document.getElementById("gallery-prev");
    const nextBtn = document.getElementById("gallery-next");
    const onChange = () =>
      applyFilters(cards, dropdowns, searchInput, carousel, prevBtn, nextBtn);

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
        applyFilters(cards, dropdowns, searchInput, carousel, prevBtn, nextBtn);
      });
    }

    if (carousel && prevBtn && nextBtn) {
      prevBtn.addEventListener("click", () => {
        const visible = visibleCards(cards);
        const idx = currentCardIndex(carousel, visible);
        if (idx > 0) scrollToCard(carousel, visible[idx - 1], "smooth");
      });
      nextBtn.addEventListener("click", () => {
        const visible = visibleCards(cards);
        const idx = currentCardIndex(carousel, visible);
        if (idx !== -1 && idx < visible.length - 1) {
          scrollToCard(carousel, visible[idx + 1], "smooth");
        }
      });
      let scrollTimer;
      carousel.addEventListener(
        "scroll",
        () => {
          clearTimeout(scrollTimer);
          scrollTimer = setTimeout(
            () => updateNavButtons(carousel, cards, prevBtn, nextBtn),
            100,
          );
        },
        { passive: true },
      );
    }

    applyFilters(cards, dropdowns, searchInput, carousel, prevBtn, nextBtn);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
