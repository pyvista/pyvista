from __future__ import annotations

from pathlib import Path
import re
from typing import NamedTuple
from typing import Optional

import pytest

ROOT_DIR = str(Path(__file__).parent.parent.parent)
EXAMPLES_DIR = str(Path(ROOT_DIR) / 'examples')

# Regex patterns
CROSSREF_PATTERN = re.compile(r':(meth|func|class|mod|attr|exc|data|ref|obj):`[^`]+`')
ANCHOR_PATTERN = re.compile(r'^\s*\.\.\s+_(.+?):\s*$', re.MULTILINE)


class _TestCaseTuple(NamedTuple):
    test_id: str
    file_path: str
    has_crossref: bool
    anchor: Optional[str]


def find_py_files(root_dir: str) -> list[str]:
    """Return a list of all .py files under the given root directory."""
    return [str(p) for p in Path(root_dir).rglob('*.py')]


def analyze_file(file_path: str) -> tuple[bool, str]:
    """Check a file for cross-references and return the first anchor."""
    with Path(file_path).open(encoding='utf-8') as f:
        content = f.read()

    has_crossref = bool(CROSSREF_PATTERN.search(content))
    anchor_match = ANCHOR_PATTERN.search(content)
    anchor = anchor_match.group(1) if anchor_match else None

    return has_crossref, anchor


def generate_test_cases() -> list[_TestCaseTuple]:
    test_cases = []
    example_files = find_py_files(EXAMPLES_DIR)
    for file_path in example_files:
        has_crossref, anchor = analyze_file(file_path)
        test_id = str(Path(file_path).relative_to(ROOT_DIR))
        test_cases.append(
            _TestCaseTuple(
                test_id=test_id, file_path=file_path, has_crossref=has_crossref, anchor=anchor
            )
        )
    return test_cases


TEST_CASES = generate_test_cases()
TEST_CASE_IDS = [case.test_id for case in TEST_CASES]


@pytest.mark.parametrize('test_case', TEST_CASES, ids=TEST_CASE_IDS)
def test_example_has_cross_reference(test_case):
    if not test_case.has_crossref:
        msg = (
            "Example must include at least one cross-reference to PyVista's API.\n "
            'E.g. if the example shows how to use `my_function`, then include a reference to `my_function`.\n'
            'E.g. use :class:`~pyvista.Plotter` to reference the `Plotter` class.\n'
            'E.g. use :meth:`~pyvista.DataSetFilters.transform` to reference the `transform` filter.\n'
        )
        pytest.fail(msg)


def test_example_filename_is_snake_case(): ...


@pytest.mark.parametrize('test_case', TEST_CASES, ids=TEST_CASE_IDS)
def test_example_anchor(test_case):
    def format_anchor(anchor):
        return f'.. _{anchor}:'

    expected_anchor = f'{Path(test_case.file_path).stem}_example'
    if test_case.anchor is None:
        msg = (
            'Example is missing a reference anchor. Expected to find the anchor:\n'
            f'{format_anchor(expected_anchor)}\n'
            'at the top of the file.'
        )
        raise pytest.fail(msg)

    if test_case.anchor != expected_anchor:
        msg = (
            f'Example has an incorrect anchor: {format_anchor(test_case.anchor)!r}\n'
            f'Expected: {format_anchor(expected_anchor)!r}'
        )
        raise pytest.fail(msg)
