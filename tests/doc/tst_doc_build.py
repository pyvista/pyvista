"""Test the images generated from building the documentation."""

from __future__ import annotations

from pathlib import Path
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
