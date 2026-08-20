"""Sanity checks against the sphinx-gallery examples in the built documentation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import NamedTuple
from xml.etree import ElementTree as ET

from conftest import BUILD_HTML_DIR
from conftest import PYVISTA_ROOT_DIR
import pytest

# Same value as `sphinx_gallery_conf['junit']` in `conf.py`
SPHINX_GALLERY_CONF_JUNIT = Path('sphinx-gallery') / 'junit-results.xml'
SPHINX_GALLERY_EXAMPLE_MAX_TIME = 150.0  # Measured in seconds
XML_FILE = BUILD_HTML_DIR / SPHINX_GALLERY_CONF_JUNIT


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
    """Confirm the index page anchors the top-level ``pyvista`` module."""
    index_html = (Path(BUILD_HTML_DIR) / 'index.html').read_text(encoding='utf-8')

    assert 'id="module-pyvista"' in index_html


def test_sphinx_gallery_junit_results_exist():
    """Confirm sphinx-gallery wrote its junit results."""
    assert XML_FILE.is_file(), f'{XML_FILE} not found. Build the documentation first.'


@pytest.mark.parametrize('testcase', test_cases, ids=test_ids)
def test_sphinx_gallery_execution_times(testcase):
    """Confirm no gallery example took too long to run."""
    if float(testcase['time']) > SPHINX_GALLERY_EXAMPLE_MAX_TIME:
        pytest.fail(
            f'Gallery example {testcase["name"]!r} from {testcase["file"]!r}\n'
            f'Took too long to run: '
            f'Duration {testcase["time"]}s > {SPHINX_GALLERY_EXAMPLE_MAX_TIME}s',
        )


# -- cross-references between the API and gallery examples --------------------
# Two directions of the same relationship, kept together: does an example reference
# the API (checked statically, from its own source) and does the API reference back
# to it (checked here against the built HTML, since it relies on sphinx-autocodelink's
# "Used In" backreferences -- `autocodelink_autodoc_backrefs` -- generated dynamically
# at build time, so it can't be checked statically like the other direction can).

EXAMPLES_SRC_DIR = PYVISTA_ROOT_DIR / 'examples'

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
                test_id=str(file_path.relative_to(PYVISTA_ROOT_DIR)),
                file_path=file_path,
                has_crossref_to_api=has_crossref_to_api,
                anchor=anchor,
            )
        )
    return cases


EXAMPLE_CASES = generate_example_cases()
EXAMPLE_CASE_IDS = [case.test_id for case in EXAMPLE_CASES]


def example_html_page(file_path: Path) -> Path:
    """Return the built HTML page Sphinx-Gallery generates for an example.

    ``gallery_dirs: ['examples']`` in ``conf.py`` mirrors the source ``examples/`` dir
    under the same name in the built output, so that path segment is kept, not stripped.
    """
    return Path(BUILD_HTML_DIR) / file_path.relative_to(PYVISTA_ROOT_DIR).with_suffix('.html')


def load_backref_target_names() -> set[str]:
    """Return the filename of every page linked from a "Used In" list, across all built pages."""
    names = set()
    for page in Path(BUILD_HTML_DIR).rglob('*.html'):
        content = page.read_text(encoding='utf-8')
        for list_block in _BACKREF_LIST_RE.findall(content):
            names.update(Path(href).name for href in _BACKREF_HREF_RE.findall(list_block))
    return names


#: Computed once at collection, not per parametrized example -- scanning every built
#: page is too expensive to repeat hundreds of times over.
BACKREF_TARGET_NAMES = load_backref_target_names() if Path(BUILD_HTML_DIR).is_dir() else set()


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
            '"Used In" backreference.\n'
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


# -- no "See Also" entry duplicates an example already in "Used In" -----------
# numpydoc's own "See Also" section and the raw `.. seealso::` directive both render as
# the same `.. admonition:: seealso` markup. A hand-written link to a gallery example
# there is only a problem once sphinx-autocodelink's own "Used In" section already shows
# the same example -- at that point it's pure duplication, and unlike "Used In", it never
# gets rechecked against what the example actually does. A "See Also" example that "Used
# In" doesn't (yet) cover is left alone -- that's real information, not redundancy.

_SEE_ALSO_RE = re.compile(r'<div class="admonition seealso">(.*?)</div>', re.DOTALL)
_BACKREFS_SECTION_RE = re.compile(
    r'<section class="sphinx-autocodelink-backrefs"[^>]*>.*?</section>', re.DOTALL
)
_EXAMPLE_HREF_RE = re.compile(r'href="([^"]*/examples/[^"]*)"')


def _example_hrefs(blocks: list[str]) -> set[str]:
    """Return the basename of every example page linked from `blocks`."""
    return {Path(href).name for block in blocks for href in _EXAMPLE_HREF_RE.findall(block)}


def test_see_also_does_not_duplicate_used_in_examples():
    pages = sorted(Path(BUILD_HTML_DIR).rglob('*.html'))
    assert pages, f'no built pages found under {BUILD_HTML_DIR}. Build the documentation first.'

    failures = {}
    for page in pages:
        html_text = page.read_text(encoding='utf-8')
        see_also_examples = _example_hrefs(_SEE_ALSO_RE.findall(html_text))
        used_in_examples = _example_hrefs(_BACKREFS_SECTION_RE.findall(html_text))
        duplicated = see_also_examples & used_in_examples
        if duplicated:
            failures[page.stem] = duplicated

    note = (
        'Remove the "See Also" entry for each example below -- it duplicates that same '
        'example already listed in the page\'s own "Used In" section:\n'
    )
    assert not failures, note + '\n'.join(
        f'{name}: {", ".join(sorted(examples))}' for name, examples in sorted(failures.items())
    )
