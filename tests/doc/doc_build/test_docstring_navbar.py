"""Test that docstring sections are hoisted into the "on this page" navbar.

``conf.py`` patches numpydoc to emit real headings instead of rubrics, then hoists
those sections out of the autodoc ``desc`` node so Sphinx's TocTreeCollector can see
them. Both halves are needed, and neither fails loudly if it breaks -- the navbar
entries just silently disappear.
"""

from __future__ import annotations

from pathlib import Path
import re

from conftest import BUILD_HTML_DIR

# Minimum number of API pages expected to gain a docstring-section entry. The
# real count is in the thousands; this only needs to be high enough that a
# silent regression can't slip through.
MIN_PAGES_WITH_HOISTED_SECTIONS = 100

# A generated page for a single object whose docstring has Notes and Examples.
API_PAGE = 'pyvista.PolyDataFilters.decimate.html'

_PAGE_TOC_RE = re.compile(
    r'<nav\b[^>]*\bclass="[^"]*\bpage-toc\b[^"]*"[^>]*>(.*?)</nav>', re.DOTALL
)
_HREF_RE = re.compile(r'href="#([^"]+)"')


def page_toc_anchors(html: str) -> list[str]:
    """Return the anchors linked from a page's "on this page" navbar."""
    match = _PAGE_TOC_RE.search(html)
    return _HREF_RE.findall(match.group(1)) if match else []


def find_api_page(filename: str) -> Path:
    """Return a generated single-object API page.

    Fails rather than skips when the page is missing: skipping would silently
    stop testing the feature, and nobody would know to update the test.
    """
    page = next(Path(BUILD_HTML_DIR).rglob(filename), None)
    assert page is not None, (
        f'{filename} not found under {BUILD_HTML_DIR}. If the API doc layout changed, point '
        f'this test at another single-object page with Notes and Examples sections.'
    )
    return page


def test_docstring_sections_are_hoisted_into_page_toc():
    """Confirm Notes and Examples sections are hoisted into the navbar."""
    html = find_api_page(API_PAGE).read_text()
    anchors = page_toc_anchors(html)

    assert 'notes' in anchors
    assert 'examples' in anchors


def test_hoisted_sections_are_not_rubrics():
    """Confirm hoisted sections render as real headings, not rubrics."""
    html = find_api_page(API_PAGE).read_text()

    # A rubric would render as <p class="rubric">Examples</p> and never reach
    # the navbar. Real headings carry an id so they can be linked.
    assert '<p class="rubric">Examples</p>' not in html
    assert 'id="examples"' in html


def test_multi_object_page_does_not_hoist_sections():
    """Confirm pages documenting several objects via ``:members:`` skip hoisting."""
    # `helpers.rst` documents several objects on one page via `:members:`, so
    # hoisting is skipped there to avoid colliding sections at page level.
    page = Path(BUILD_HTML_DIR) / 'api' / 'core' / 'helpers.html'
    assert page.is_file(), (
        f'{page} not found. If the API doc layout changed, point this test at another '
        f'page that documents several objects via `:members:`.'
    )
    anchors = page_toc_anchors(page.read_text())

    assert 'notes' not in anchors
    assert 'examples' not in anchors


def test_page_toc_anchors_resolve():
    """Confirm every navbar anchor has a matching id on the page."""
    html = find_api_page(API_PAGE).read_text()

    for anchor in page_toc_anchors(html):
        assert f'id="{anchor}"' in html, f'navbar links to #{anchor} but no such id exists'


def test_docstring_sections_hoisted_across_api_pages():
    """Confirm enough API pages gained a hoisted docstring-section entry."""
    api_pages = list(Path(BUILD_HTML_DIR).rglob('_autosummary/*.html'))
    assert api_pages, (
        f'no generated API pages found under {BUILD_HTML_DIR}. If autosummary no longer '
        f'writes to `_autosummary`, update this glob.'
    )

    pages_with_sections = [
        path
        for path in api_pages
        if {'notes', 'examples'} & set(page_toc_anchors(path.read_text()))
    ]

    assert len(pages_with_sections) > MIN_PAGES_WITH_HOISTED_SECTIONS
