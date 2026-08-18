"""Test the images generated from building the documentation."""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re
from typing import NamedTuple
from xml.etree import ElementTree as ET

import pytest

ROOT_DIR = str(Path(__file__).parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
HTML_DIR = str(Path(BUILD_DIR) / 'html')


# Same value as `sphinx_gallery_conf['junit']` in `conf.py`
SPHINX_GALLERY_CONF_JUNIT = Path('sphinx-gallery') / 'junit-results.xml'
SPHINX_GALLERY_EXAMPLE_MAX_TIME = 150.0  # Measured in seconds
XML_FILE = HTML_DIR / SPHINX_GALLERY_CONF_JUNIT


def load_test_cases() -> list[dict[str, str]]:
    """Return the sphinx-gallery junit test cases, or none if the docs aren't built.

    Parametrization happens at collection time, so this can't raise on a missing
    file without failing every test in the module.
    ``test_sphinx_gallery_junit_results_exist`` reports that instead.
    """
    if not XML_FILE.is_file():
        return []
    return [dict(case.attrib) for case in ET.parse(XML_FILE).getroot().iterfind('testcase')]


test_cases = load_test_cases()
test_ids = [case['classname'] for case in test_cases]


def test_top_level_module_target():
    index_html = (Path(HTML_DIR) / 'index.html').read_text(encoding='utf-8')

    assert 'id="module-pyvista"' in index_html


def test_sphinx_gallery_junit_results_exist():
    assert XML_FILE.is_file(), f'{XML_FILE} not found. Build the documentation first.'


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

# A generated page for a single object whose docstring has Notes and Examples.
API_PAGE = 'pyvista.PolyDataFilters.decimate.html'

_PAGE_TOC_RE = re.compile(r'<nav class="bd-toc-nav page-toc">(.*?)</nav>', re.DOTALL)
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
    page = next(Path(HTML_DIR).rglob(filename), None)
    assert page is not None, (
        f'{filename} not found under {HTML_DIR}. If the API doc layout changed, point '
        f'this test at another single-object page with Notes and Examples sections.'
    )
    return page


def test_docstring_sections_are_hoisted_into_page_toc():
    html = find_api_page(API_PAGE).read_text()
    anchors = page_toc_anchors(html)

    assert 'notes' in anchors
    assert 'examples' in anchors


def test_hoisted_sections_are_not_rubrics():
    html = find_api_page(API_PAGE).read_text()

    # A rubric would render as <p class="rubric">Examples</p> and never reach
    # the navbar. Real headings carry an id so they can be linked.
    assert '<p class="rubric">Examples</p>' not in html
    assert 'id="examples"' in html


def test_multi_object_page_does_not_hoist_sections():
    # `helpers.rst` documents several objects on one page via `:members:`, so
    # hoisting is skipped there to avoid colliding sections at page level.
    page = Path(HTML_DIR) / 'api' / 'core' / 'helpers.html'
    assert page.is_file(), (
        f'{page} not found. If the API doc layout changed, point this test at another '
        f'page that documents several objects via `:members:`.'
    )
    anchors = page_toc_anchors(page.read_text())

    assert 'notes' not in anchors
    assert 'examples' not in anchors


def test_page_toc_anchors_resolve():
    html = find_api_page(API_PAGE).read_text()

    for anchor in page_toc_anchors(html):
        assert f'id="{anchor}"' in html, f'navbar links to #{anchor} but no such id exists'


def test_docstring_sections_hoisted_across_api_pages():
    api_pages = list(Path(HTML_DIR).rglob('_autosummary/*.html'))
    assert api_pages, (
        f'no generated API pages found under {HTML_DIR}. If autosummary no longer '
        f'writes to `_autosummary`, update this glob.'
    )

    pages_with_sections = [
        path
        for path in api_pages
        if {'notes', 'examples'} & set(page_toc_anchors(path.read_text()))
    ]

    assert len(pages_with_sections) > MIN_PAGES_WITH_HOISTED_SECTIONS


def test_contributing_edit_button_points_to_contributing():
    html = (Path(HTML_DIR) / 'contributing.html').read_text(encoding='utf-8')
    assert 'https://github.com/pyvista/pyvista/edit/main/CONTRIBUTING.rst' in html


# -- cross-references between the API and gallery examples --------------------
# Two directions of the same relationship, kept together: does an example reference
# the API (checked statically, from its own source) and does the API reference back
# to it (checked here against the built HTML, since it relies on sphinx-autocodelink's
# "Used in" backreferences -- `autocodelink_autodoc_backrefs` -- generated dynamically
# at build time, so it can't be checked statically like the other direction can).

EXAMPLES_SRC_DIR = Path(ROOT_DIR) / 'examples'

_CROSSREF_RE = re.compile(r':(meth|func|class|mod|attr|exc|data|ref|obj):`[^`]+`')
_ANCHOR_RE = re.compile(r'^\s*\.\.\s+_(.+?):\s*$', re.MULTILINE)
_BACKREF_LIST_RE = re.compile(r'<ul class="sphinx-autocodelink-index">(.*?)</ul>', re.DOTALL)
_BACKREF_HREF_RE = re.compile(r'href="([^"]*)"')


class _ExampleCase(NamedTuple):
    test_id: str
    file_path: Path
    has_crossref_to_api: bool
    anchor: str | None


def find_example_files() -> list[Path]:
    """Return every gallery example source file."""
    return sorted(EXAMPLES_SRC_DIR.rglob('*.py'))


def analyze_example_file(file_path: Path) -> tuple[bool, str | None]:
    """Check a file for a cross-reference to the API, and return its first anchor."""
    content = file_path.read_text(encoding='utf-8')
    has_crossref = bool(_CROSSREF_RE.search(content))
    anchor_match = _ANCHOR_RE.search(content)
    return has_crossref, anchor_match.group(1) if anchor_match else None


def generate_example_cases() -> list[_ExampleCase]:
    cases = []
    for file_path in find_example_files():
        has_crossref_to_api, anchor = analyze_example_file(file_path)
        cases.append(
            _ExampleCase(
                test_id=str(file_path.relative_to(ROOT_DIR)),
                file_path=file_path,
                has_crossref_to_api=has_crossref_to_api,
                anchor=anchor,
            )
        )
    return cases


EXAMPLE_CASES = generate_example_cases()
EXAMPLE_CASE_IDS = [case.test_id for case in EXAMPLE_CASES]


def example_html_page(file_path: Path) -> Path:
    """Return the built HTML page Sphinx-Gallery generates for an example."""
    return Path(HTML_DIR) / file_path.relative_to(EXAMPLES_SRC_DIR).with_suffix('.html')


def load_backref_target_names() -> set[str]:
    """Return the filename of every page linked from a "Used in" list, across all built pages."""
    names = set()
    for page in Path(HTML_DIR).rglob('*.html'):
        content = page.read_text(encoding='utf-8')
        for list_block in _BACKREF_LIST_RE.findall(content):
            names.update(Path(href).name for href in _BACKREF_HREF_RE.findall(list_block))
    return names


#: Computed once at collection, not per parametrized example -- scanning every built
#: page is too expensive to repeat hundreds of times over.
BACKREF_TARGET_NAMES = load_backref_target_names() if Path(HTML_DIR).is_dir() else set()


@pytest.mark.parametrize('case', EXAMPLE_CASES, ids=EXAMPLE_CASE_IDS)
def test_example_has_cross_reference_to_api(case):
    if not case.has_crossref_to_api:
        msg = (
            "Example must include at least one cross-reference to PyVista's core or "
            'plotting API.\n '
            'E.g. if the example shows how to use `my_function`, then include a reference to '
            '`my_function`.\n'
            'E.g. use :class:`~pyvista.Plotter` to reference the `Plotter` class.\n'
            'E.g. use :meth:`~pyvista.DataSetFilters.transform` to reference the '
            '`transform` filter.\n'
        )
        pytest.fail(msg)


@pytest.mark.parametrize('case', EXAMPLE_CASES, ids=EXAMPLE_CASE_IDS)
def test_example_has_cross_reference_from_api(case):
    if case.file_path.name == 'add_example.py':
        pytest.skip('This is a meta-example for dev purposes.')

    page = example_html_page(case.file_path)
    assert page.is_file(), f'{page} not found. Build the documentation first.'

    if page.name not in BACKREF_TARGET_NAMES:
        msg = (
            "Example must be linked from PyVista's core or plotting API via a "
            '"Used in" backreference.\n'
            'E.g. if the example shows how to use `my_function` with dataset '
            '`download_some_dataset`, add a call to one (or both) in the example, so '
            'sphinx-autocodelink records the reference automatically.'
        )
        pytest.fail(msg)


@pytest.mark.parametrize('case', EXAMPLE_CASES, ids=EXAMPLE_CASE_IDS)
def test_example_anchor(case):
    def format_anchor(anchor):
        return f'.. _{anchor}:'

    expected_anchor = f'{case.file_path.stem}_example'
    if case.anchor is None:
        msg = (
            'Example is missing a reference anchor. Expected to find the anchor\n'
            f'{format_anchor(expected_anchor)!r} at the top of the file.'
        )
        raise pytest.fail(msg)

    if case.anchor != expected_anchor:
        msg = (
            f'Example has an incorrect reference anchor at the top of the file.\n'
            f'Actual: {format_anchor(case.anchor)!r}\n'
            f'Expected: {format_anchor(expected_anchor)!r}'
        )
        raise pytest.fail(msg)


# -- Open Graph link previews -------------------------------------------------
# Sanity checks against the real documentation build.

# Same value as `ogp_site_url` in `conf.py`
OGP_SITE_URL = 'https://docs.pyvista.org/'

_META_TAG = re.compile(r'<meta\b[^>]*>')
_META_KEY = re.compile(r'\b(?:property|name)="([^"]+)"')
_META_CONTENT = re.compile(r'\bcontent="([^"]*)"')
_PAGE_IMAGE = re.compile(r'<img\b[^>]*\bsrc="[^"]*/_images/([^"]+)"')


def meta_tags(page: Path) -> dict[str, str]:
    """Return a built page's ``<meta>`` tags, keyed by ``property`` or ``name``."""
    tags: dict[str, str] = {}
    for tag in _META_TAG.findall(page.read_text(encoding='utf-8')):
        key = _META_KEY.search(tag)
        content = _META_CONTENT.search(tag)
        if key is not None and content is not None:
            tags.setdefault(key.group(1), html.unescape(content.group(1)))
    return tags


def page_images(page: Path) -> list[str]:
    """Return the filenames of the images a page shows, in the order it shows them."""
    images = _PAGE_IMAGE.findall(page.read_text(encoding='utf-8'))
    assert images, f'{page} shows no images'
    return images


@dataclass(frozen=True)
class OpenGraphPage:
    id: str
    path: str
    description: str
    #: One-based position of the expected preview among the images the page shows.
    #: Ignored when `image` is set.
    image_number: int = 1
    #: Exact expected `og:image` URL, for a page whose preview isn't one of its
    #: own images (e.g. the root page's `ogp_image` fallback).
    image: str | None = None


OPENGRAPH_PAGES = (
    OpenGraphPage(
        id='prose',
        # A hand-written page selecting its image with ``.. autoopengraph_thumbnail::``
        path='user-guide/what-is-a-mesh.html',
        description='In PyVista, a mesh is any spatially referenced information',
        image_number=7,
    ),
    OpenGraphPage(
        id='gallery',
        # A gallery example selecting its image with ``sphinx_gallery_thumbnail_number``
        path='examples/00-load/create_circular_arc.html',
        description='Generate arc geometry with pyvista.CircularArc()',
        image_number=2,
    ),
    OpenGraphPage(
        id='api',
        # ``.. autoopengraph_thumbnail::`` inside an Examples section numpydoc wraps
        # in ``.. pyvista-plot::`` on its own
        path='api/core/_autosummary/pyvista.ImageDataFilters.crop.html',
        description='Crop this image to remove points at its boundaries.',
        image_number=4,
    ),
    OpenGraphPage(
        id='autoenum-gallery',
        # Images are plain ``.. image::`` directives rather than the plot directive,
        # to confirm selection counts image nodes generically
        path='api/utilities/_autosummary/pyvista.CellType.html',
        description='Define types of cells.',
        image_number=13,
    ),
    OpenGraphPage(
        id='root',
        # Opts out of selecting one of its own images with
        # ``.. autoopengraph_thumbnail:: none``, so its preview is the site-wide
        # default rather than one of its own real content images below the fold
        path='index.html',
        description=(
            'PyVista is the foundational Python library for 3D visualization and mesh '
            'analysis in scientific computing and engineering.'
        ),
        image=f'{OGP_SITE_URL}_static/pyvista_banner_small.png',
    ),
)


@pytest.mark.parametrize('page', OPENGRAPH_PAGES, ids=lambda page: page.id)
def test_opengraph_description(page: OpenGraphPage):
    path = Path(HTML_DIR) / page.path
    assert path.is_file(), f'{path} not found. Build the documentation first.'

    description = meta_tags(path).get('og:description')

    assert description is not None, f'{page.path} has no og:description'
    assert description.startswith(page.description)


@pytest.mark.parametrize('page', OPENGRAPH_PAGES, ids=lambda page: page.id)
def test_opengraph_image(page: OpenGraphPage):
    path = Path(HTML_DIR) / page.path
    assert path.is_file(), f'{path} not found. Build the documentation first.'

    if page.image is not None:
        expected = page.image
    else:
        expected = f'{OGP_SITE_URL}_images/{page_images(path)[page.image_number - 1]}'

    assert meta_tags(path).get('og:image') == expected
