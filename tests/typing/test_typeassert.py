"""Unit tests for the typeassert framework itself.

These use plain Python types rather than PyVista's, since the framework knows
nothing about the project under test.
"""

from __future__ import annotations

import pytest

from tests.typing.typeassert import CaseSkipped
from tests.typing.typeassert import assert_types
from tests.typing.typeassert import collect_case_file

IMPORT = 'from tests.typing.typeassert import assert_types\n'


def write_case_file(tmp_path, body):
    """Write a case file and return its parsed form."""
    path = tmp_path / 'cases.py'
    path.write_text(IMPORT + body, encoding='utf-8')
    return collect_case_file(path)


def test_assert_types_returns_the_value():
    assert assert_types([1], list) == [1]


def test_assert_types_rejects_a_mismatch():
    with pytest.raises(AssertionError, match='not assignable'):
        assert_types([1], list[str])


def test_assert_types_walks_containers():
    # The element at index 1 is the one that does not belong.
    with pytest.raises(AssertionError, match='not assignable'):
        assert_types([1, 'two', 3], list[int])


def test_collects_each_assertion_as_a_case(tmp_path):
    body = 'assert_types(len([1]), int)\nassert_types(str(1), str)\n'
    case_file = write_case_file(tmp_path, body)
    assert [case.id for case in case_file.cases] == ['len([1]) -> int', 'str(1) -> str']
    assert case_file.error is None


def test_setup_lines_are_everything_that_is_not_a_case(tmp_path):
    case_file = write_case_file(tmp_path, 'X = 1\nassert_types(X, int)\n')
    assert case_file.setup_lines == frozenset({1, 2})
    assert case_file.cases[0].lines == frozenset({3})


def test_a_case_cannot_see_another_cases_state(tmp_path):
    body = 'VALUES = []\nassert_types(VALUES.append(1), None)\nassert_types(len(VALUES), int)\n'
    case_file = write_case_file(tmp_path, body)
    namespace = case_file.setup_namespace()
    case_file.run(case_file.cases[1])
    # The append in the first case must not be visible to the second.
    assert namespace['VALUES'] == []


def test_a_failing_case_raises(tmp_path):
    case_file = write_case_file(tmp_path, 'assert_types(len([1]), str)\n')
    with pytest.raises(AssertionError, match='not assignable'):
        case_file.run(case_file.cases[0])


def test_nested_assertions_are_rejected(tmp_path):
    body = 'def helper():\n    assert_types(len([1]), int)\n\nassert_types(str(1), str)\n'
    case_file = write_case_file(tmp_path, body)
    assert 'must be a statement at module level' in case_file.error


def test_a_syntax_error_is_reported_rather_than_raised(tmp_path):
    case_file = write_case_file(tmp_path, 'def broken(\n')
    assert 'never closed' in case_file.error
    assert case_file.cases == ()


def test_skip_runtime_skips_only_the_named_case(tmp_path):
    body = (
        "SKIP_RUNTIME = {'len([1])': 'a reason'}\n"
        'assert_types(len([1]), int)\n'
        'assert_types(str(1), str)\n'
    )
    case_file = write_case_file(tmp_path, body)
    with pytest.raises(CaseSkipped, match='a reason'):
        case_file.run(case_file.cases[0])
    case_file.run(case_file.cases[1])


def test_a_skip_naming_no_case_is_reported(tmp_path):
    body = "SKIP_RUNTIME = {'gone()': 'a reason'}\nassert_types(len([1]), int)\n"
    case_file = write_case_file(tmp_path, body)
    assert case_file.unknown_skips(case_file.setup_namespace()) == ['gone()']
