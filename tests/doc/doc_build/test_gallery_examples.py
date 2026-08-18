"""Sanity checks against the sphinx-gallery examples in the built documentation."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from conftest import BUILD_HTML_DIR
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
