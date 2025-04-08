from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

# Max execution time in seconds, tests fail if greater than this value
MAX_TIME = 60.0
# Project root directory relative to this test file
PROJECT_DIR = Path(__file__).parent.parent.parent
# Sphinx build directory
HTML_DIR = PROJECT_DIR / 'doc' / '_build' / 'html'
# Same value as `sphinx_gallery_conf['junit']` in `conf.py`
SPHINX_GALLERY_CONF_JUNIT = (Path('sphinx-gallery') / 'junit-results.xml',)


def load_test_cases_from_xml(path) -> list[dict[str, str | int | float | bool | None]]:
    """Parse test cases from the generated JUnit XML file."""
    tree = ET.parse(path)
    root = tree.getroot()
    cases = []
    for testcase in root.findall('testcase'):
        name = testcase.attrib['name']
        time = float(testcase.attrib['time'])
        skipped_elem = testcase.find('skipped')
        skipped = skipped_elem is not None
        skip_msg = skipped_elem.attrib.get('message') if skipped else None
        cases.append(
            {
                'name': name,
                'file': testcase.attrib['file'],
                'classname': testcase.attrib['classname'],
                'line': int(testcase.attrib['line']),
                'time': time,
                'skipped': skipped,
                'skip_message': skip_msg,
            }
        )
    return cases


TEST_CASES = load_test_cases_from_xml(HTML_DIR / SPHINX_GALLERY_CONF_JUNIT)
IDS = [case['classname'] for case in TEST_CASES]


@pytest.mark.parametrize('testcase', TEST_CASES, ids=IDS)
def test_gallery_example_execution_time(testcase):
    if testcase['skipped']:
        pytest.skip(testcase['skip_message'] or 'Skipped test.')

    msg = (
        f"Sphinx gallery example '{testcase['name']}' from file\n"
        f'\t{testcase["file"]}\n'
        f'has an execution time: {testcase["time"]} seconds\n'
        f'which exceeds the maximum allowed: {MAX_TIME} seconds.'
    )
    assert testcase['time'] < MAX_TIME, msg
