"""Test the images generated from building the documentation."""

from __future__ import annotations

from pathlib import Path
import re
from xml.etree.ElementTree import parse

import pytest

ROOT_DIR = str(Path(__file__).parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
HTML_DIR = str(Path(BUILD_DIR) / 'html')


# Same value as `sphinx_gallery_conf['junit']` in `conf.py`
SPHINX_GALLERY_CONF_JUNIT = Path('sphinx-gallery') / 'junit-results.xml'
SPHINX_GALLERY_EXAMPLE_MAX_TIME = 150.0  # Measured in seconds
XML_FILE = HTML_DIR / SPHINX_GALLERY_CONF_JUNIT
assert XML_FILE.is_file()


xml_root = parse(XML_FILE).getroot()
test_cases = [dict(case.attrib) for case in xml_root.iterfind('testcase')]
test_ids = [case['classname'] for case in test_cases]


@pytest.mark.parametrize('testcase', test_cases, ids=test_ids)
def test_sphinx_gallery_execution_times(testcase):
    if float(testcase['time']) > SPHINX_GALLERY_EXAMPLE_MAX_TIME:
        pytest.fail(
            f'Gallery example {testcase["name"]!r} from {testcase["file"]!r}\n'
            f'Took too long to run: '
            f'Duration {testcase["time"]}s > {SPHINX_GALLERY_EXAMPLE_MAX_TIME}s',
        )


# -- docstring sections in the "on this page" navbar --------------------------
# `conf.py` patches numpydoc to emit real headings instead of rubrics, then
# hoists those sections out of the autodoc `desc` node so Sphinx's
# TocTreeCollector can see them. Both halves are needed, and neither fails
# loudly if it breaks -- the navbar entries just silently disappear.

# Minimum number of API pages expected to gain a docstring-section entry. The
# real count is in the thousands; this only needs to be high enough that a
# silent regression can't slip through.
MIN_PAGES_WITH_HOISTED_SECTIONS = 100

_PAGE_TOC_RE = re.compile(r'<nav class="bd-toc-nav page-toc">(.*?)</nav>', re.DOTALL)
_HREF_RE = re.compile(r'href="#([^"]+)"')


def page_toc_anchors(html: str) -> list[str]:
    """Return the anchors linked from a page's "on this page" navbar."""
    match = _PAGE_TOC_RE.search(html)
    return _HREF_RE.findall(match.group(1)) if match else []


def find_api_page(filename: str) -> Path:
    """Return a generated single-object API page, skipping if docs moved."""
    page = next(Path(HTML_DIR).rglob(filename), None)
    if page is None:
        pytest.skip(f'{filename} not found; the API doc layout may have changed')
    return page


def test_docstring_sections_are_hoisted_into_page_toc():
    # `PolyData.decimate` has both a Notes and an Examples section.
    html = find_api_page('pyvista.PolyData.decimate.html').read_text()
    anchors = page_toc_anchors(html)

    assert 'notes' in anchors
    assert 'examples' in anchors


def test_hoisted_sections_are_not_rubrics():
    html = find_api_page('pyvista.PolyData.decimate.html').read_text()

    # A rubric would render as <p class="rubric">Examples</p> and never reach
    # the navbar. Real headings carry an id so they can be linked.
    assert '<p class="rubric">Examples</p>' not in html
    assert 'id="examples"' in html


def test_multi_object_page_does_not_hoist_sections():
    # `helpers.rst` documents several objects on one page via `:members:`, so
    # hoisting is skipped there to avoid colliding sections at page level.
    page = Path(HTML_DIR) / 'api' / 'core' / 'helpers.html'
    if not page.is_file():
        pytest.skip('helpers.html not found; the API doc layout may have changed')
    anchors = page_toc_anchors(page.read_text())

    assert 'notes' not in anchors
    assert 'examples' not in anchors


def test_page_toc_anchors_resolve():
    html = find_api_page('pyvista.PolyData.decimate.html').read_text()

    for anchor in page_toc_anchors(html):
        assert f'id="{anchor}"' in html, f'navbar links to #{anchor} but no such id exists'


def test_docstring_sections_hoisted_across_api_pages():
    pages_with_sections = [
        path
        for path in Path(HTML_DIR).rglob('_autosummary/*.html')
        if {'notes', 'examples'} & set(page_toc_anchors(path.read_text()))
    ]

    assert len(pages_with_sections) > MIN_PAGES_WITH_HOISTED_SECTIONS
