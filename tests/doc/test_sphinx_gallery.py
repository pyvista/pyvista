from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import NamedTuple
from typing import Optional

import pytest

ROOT_DIR = str(Path(__file__).parent.parent.parent)
EXAMPLES_DIR = str(Path(ROOT_DIR) / 'examples')
PYVISTA_DIR = str(Path(ROOT_DIR) / 'pyvista')
DOC_DIR = str(Path(ROOT_DIR) / 'doc')


class _TestCaseTuple(NamedTuple):
    test_id: str
    file_path: str
    has_crossref_to_api: bool
    has_crossref_from_api: bool
    anchor: Optional[str]


def find_files_with_extension(root_dir: str, ext: str) -> list[str]:
    """Return a list of files with the given extension under root_dir."""
    return [str(p) for p in Path(root_dir).rglob(f'*{ext}')]


def count_ref_tags(py_root: str, rst_root: str) -> dict[str, int]:
    """Count :ref:`...` targets in both .py and .rst files."""
    pattern = re.compile(r':ref:`(?:[^<`]*<)?([^>`]+)>?`')
    ref_counter = Counter()

    # Process .py files
    for file_path in find_files_with_extension(py_root, '.py'):
        ref_counter.update(extract_ref_targets_from_file(file_path, pattern))

    # Process .rst files
    for file_path in find_files_with_extension(rst_root, '.rst'):
        ref_counter.update(extract_ref_targets_from_file(file_path, pattern))

    return dict(ref_counter)


def extract_ref_targets_from_file(file_path: str, pattern: re.Pattern) -> list[str]:
    """Extract all :ref:`...` targets from a single file using the given pattern."""
    with Path(file_path).open(encoding='utf-8') as f:
        content = f.read()
    return pattern.findall(content)


def analyze_gallery_example_file(file_path: str) -> tuple[bool, str]:
    """Check a file for cross-references and return the first anchor."""
    with Path(file_path).open(encoding='utf-8') as f:
        content = f.read()

    crossref_pattern = re.compile(r':(meth|func|class|mod|attr|exc|data|ref|obj):`[^`]+`')
    anchor_pattern = re.compile(r'^\s*\.\.\s+_(.+?):\s*$', re.MULTILINE)

    has_crossref = bool(crossref_pattern.search(content))
    anchor_match = anchor_pattern.search(content)
    anchor = anchor_match.group(1) if anchor_match else None

    return has_crossref, anchor


def generate_test_cases() -> list[_TestCaseTuple]:
    test_cases = []
    example_files = find_files_with_extension(EXAMPLES_DIR, '.py')
    ref_tags = count_ref_tags(py_root=PYVISTA_DIR, rst_root=DOC_DIR)
    for file_path in example_files:
        has_crossref_to_api, anchor = analyze_gallery_example_file(file_path)
        has_crossref_from_api = bool(ref_tags.get(anchor, None))
        test_id = str(Path(file_path).relative_to(ROOT_DIR))
        test_cases.append(
            _TestCaseTuple(
                test_id=test_id,
                file_path=file_path,
                has_crossref_to_api=has_crossref_to_api,
                has_crossref_from_api=has_crossref_from_api,
                anchor=anchor,
            )
        )
    return test_cases


TEST_CASES = generate_test_cases()
TEST_CASE_IDS = [case.test_id for case in TEST_CASES]


@pytest.mark.parametrize('test_case', TEST_CASES, ids=TEST_CASE_IDS)
def test_example_has_cross_reference_to_api(test_case):
    if not test_case.has_crossref_to_api:
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


@pytest.mark.parametrize('test_case', TEST_CASES, ids=TEST_CASE_IDS)
def test_example_has_cross_reference_from_api(test_case):
    if test_case.file_path.endswith('add_example.py'):
        pytest.skip('This is a meta-example for dev purposes.')

    if not test_case.has_crossref_from_api:
        msg = (
            "Example must include at least one cross-reference from PyVista's core or "
            'plotting API.\n'
            'E.g. if the example shows how to use `my_function` with dataset '
            '`download_some_dataset`\n'
            f'then consider including a reference:\n'
            f'    :ref:`{test_case.anchor}`\n'
            f'in the docstring of `my_function` and/or `download_some_dataset`.'
        )
        pytest.fail(msg)


@pytest.mark.parametrize('test_case', TEST_CASES, ids=TEST_CASE_IDS)
def test_example_anchor(test_case):
    def format_anchor(anchor):
        return f'.. _{anchor}:'

    expected_anchor = f'{Path(test_case.file_path).stem}_example'
    if test_case.anchor is None:
        msg = (
            'Example is missing a reference anchor. Expected to find the anchor\n'
            f'{format_anchor(expected_anchor)!r} at the top of the file.'
        )
        raise pytest.fail(msg)

    if test_case.anchor != expected_anchor:
        msg = (
            f'Example has an incorrect reference anchor at the top of the file.\n'
            f'Actual: {format_anchor(test_case.anchor)!r}\n'
            f'Expected: {format_anchor(expected_anchor)!r}'
        )
        raise pytest.fail(msg)
