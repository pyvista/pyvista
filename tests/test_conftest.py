from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import pytest
from pytest_cases import case
from pytest_cases import filters as ft
from pytest_cases import parametrize
from pytest_cases import parametrize_with_cases

import pyvista

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

pytest_plugins = 'pytester'


class PytesterStdoutParser:
    def __init__(self, regex: re.Pattern[str] | str):
        self.regex = re.compile(regex) if isinstance(regex, str) else regex

    def parse(self, results: pytest.RunResult):
        return [
            m.groupdict()
            for line in results.stdout.str().splitlines()
            if (m := re.match(self.regex, line))
        ]


class _ReportDescriptor:
    def __init__(self):
        self._status = None

    def __set_name__(self, owner, name: str):
        self._status = name

    def __get__(self, obj: RunResultsReport, objtype=None):
        return [t['name'] for t in obj.results if t['status'] == self._status.upper()]


class RunResultsReport:
    passed = _ReportDescriptor()
    skipped = _ReportDescriptor()
    failed = _ReportDescriptor()
    error = _ReportDescriptor()
    xpassed = _ReportDescriptor()
    xfailed = _ReportDescriptor()

    def __init__(self, results: list[dict[str, str]]):
        self.results = results


@pytest.fixture
def results_parser(monkeypatch: pytest.MonkeyPatch):
    """
    Results parser for all tests ran with a --verbose flag.
    It enables to get the test name (last part of the test path)
    as well as the status.

    Results can be passed to the `RunResultsReport` class to better interact
    with them.
    """
    monkeypatch.setenv('PYTEST_ADDOPTS', '-v')
    regex = re.compile(
        r'.*(?P<name>test_[\w\[\]]*) (?P<status>\w*) .*\[.*%\]$',
    )
    return PytesterStdoutParser(regex=regex)


@pytest.fixture(autouse=True)
def _load_current_config(
    pytestconfig: pytest.Config,
    pytester: pytest.Pytester,
):
    with (pytestconfig.rootpath / 'pyproject.toml').open('r') as file:
        toml = pytester.makepyprojecttoml(file.read())

    with (pytestconfig.rootpath / 'tests/conftest.py').open('r') as file:
        conftest = pytester.makeconftest(file.read())

    yield
    toml.unlink()
    conftest.unlink()


def test_warnings_turned_to_errors(
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,
):
    tests = """
    import pytest, warnings

    def test_warning():
        warnings.warn("foo",Warning)

    def test_no_warnings():
        ...
    """
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(
        passed=1,
        failed=1,
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)
    assert 'test_warning' in report.failed
    assert 'test_no_warnings' in report.passed


@pytest.mark.parametrize('greater', [True, False])
@pytest.mark.skip('Skip for patch release')
def test_warning_vtk(
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,
    monkeypatch: pytest.MonkeyPatch,
    greater: bool,
):
    tests = """
    import pytest, warnings

    def test_warning():
        msg = "`np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here."
        warnings.warn(msg, DeprecationWarning)

    """  # noqa: E501
    monkeypatch.setattr(pyvista, 'vtk_version_info', (9, 0) if not greater else (9, 1))

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(
        passed=1 if not greater else 0,
        failed=0 if not greater else 1,
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)
    assert 'test_warning' in (report.failed if greater else report.passed)


@pytest.mark.parametrize('cml', [True, False])
def test_downloads_mark(
    cml,
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,
):
    tests = """
    import pytest

    @pytest.mark.needs_download
    def test_downloads():
        ...

    def test_no_downloads():
        ...
    """
    cml = '--test_downloads' if cml else ''
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p, cml)

    results.assert_outcomes(
        skipped=0 if cml else 1,
        passed=2 if cml else 1,
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)

    assert 'test_no_downloads' in report.passed
    assert 'test_downloads' in (report.passed if cml else report.skipped)


class CasesNeedsVtk:
    @case
    @parametrize(greater=[True, False])
    def case_min_only(self, greater: bool, monkeypatch: pytest.MonkeyPatch):
        """Test when using the args, with both tuple and variadic *args and the kwargs"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 1)
        def test_greater_9_1(): ...

        @pytest.mark.needs_vtk_version((9, 1))
        def test_greater_9_1_tuple(): ...

        @pytest.mark.needs_vtk_version((9, 1, 0))
        def test_greater_9_1_tuple_2(): ...

        @pytest.mark.needs_vtk_version(at_least=(9, 1))
        def test_greater_9_1_kwargs(): ...

        @pytest.mark.needs_vtk_version(at_least=(9, 2))
        def test_greater_9_2(): ...
        """

        value = (8, 2, 0) if greater else (9, 2, 0)
        monkeypatch.setattr(pyvista, 'vtk_version_info', value)

        return tests, dict(passed=0 if greater else 5, skipped=5 if greater else 0)

    @case
    @parametrize(lower=[True, False])
    def case_max_only(self, lower: bool, monkeypatch: pytest.MonkeyPatch):
        """Test when using the max kwargs"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(less_than=(9, 1))
        def test_smaller_9_1(): ...

        @pytest.mark.needs_vtk_version(less_than=(9,))
        def test_smaller_9(): ...

        @pytest.mark.needs_vtk_version(less_than=(9, 1, 2))
        def test_smaller_9_1_2(): ...

        """

        value = (8, 2, 0) if lower else (9, 2, 0)
        monkeypatch.setattr(pyvista, 'vtk_version_info', value)

        return tests, dict(passed=3 if lower else 0, skipped=0 if lower else 3)

    @case
    def case_max_equal(self, monkeypatch: pytest.MonkeyPatch):
        """Test when the max version equals the current"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(less_than=(9, 1))
        def test_smaller_9_1_tuple1(): ...

        @pytest.mark.needs_vtk_version(less_than=(9, 1, 0))
        def test_smaller_9_1_tuple2(): ...

        """

        monkeypatch.setattr(pyvista, 'vtk_version_info', (9, 1, 0))

        return tests, dict(skipped=2)

    @case
    def case_min_equal(self, monkeypatch: pytest.MonkeyPatch):
        """Test when the min version equals the current"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(9, 1))
        def test_smaller_9_1_tuple1(): ...

        """

        monkeypatch.setattr(pyvista, 'vtk_version_info', (9, 1, 0))

        return tests, dict(passed=1)

    def case_multiple_decorating(self, monkeypatch: pytest.MonkeyPatch):
        """Test when decorating multiple times"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 1)
        @pytest.mark.needs_vtk_version(less_than=(9, 3))
        def test_1(): ...

        @pytest.mark.needs_vtk_version(less_than=(9, 3))
        @pytest.mark.needs_vtk_version(9, 1)
        def test_2(): ...

        """

        monkeypatch.setattr(pyvista, 'vtk_version_info', (8, 2, 0))

        return tests, dict(skipped=2)

    @case
    @parametrize(between=[True, False])
    def case_min_max(self, between: bool, monkeypatch: pytest.MonkeyPatch):
        """Test when using both min and max kwargs"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(8, 2), less_than=(9, 1))
        def test_between(): ...

        @pytest.mark.needs_vtk_version((8, 2), less_than=(9, 1))
        def test_between_tuple(): ...

        @pytest.mark.needs_vtk_version(8, 2, less_than=(9, 1))
        def test_between_variadic(): ...

        @pytest.mark.needs_vtk_version(8, 2, less_than=(9, 2))
        def test_between_variadic_max_equals(): ...

        @pytest.mark.needs_vtk_version(9, less_than=(9, 2))
        def test_between_variadic_min_equals(): ...

        """

        value = (9, 0, 0) if between else (9, 2, 0)
        monkeypatch.setattr(pyvista, 'vtk_version_info', value)

        return tests, dict(passed=5 if between else 0, skipped=0 if between else 5)

    @case(tags='raises')
    def case_raises_signature(self):
        """Test when not specifying any version, or using bad signature"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version
        def test_1(): ...

        @pytest.mark.needs_vtk_version(foo=1)
        def test_2(): ...

        """

        return tests, dict(errors=2)

    @case(tags='raises')
    def case_raises_both_args_min_kwargs(self):
        """Test when specifying both args and min kwargs"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 2, 1, at_least=(8, 1))
        def test_1(): ...

        @pytest.mark.needs_vtk_version((9, 2), at_least=(9, 1))
        def test_2(): ...
        """

        return tests, dict(errors=2)

    @case(tags='raises')
    def case_min_greater_max(self):
        """Test when specifying min > max"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 2, 1, less_than=(8, 1))
        def test_1(): ...

        @pytest.mark.needs_vtk_version((9, 2, 1), less_than=(8, 1))
        def test_2(): ...
        """

        return tests, dict(errors=2)

    @case(tags='raises')
    def case_version_tuple_wrong(self):
        """Test when specifying a tuple of len > 3"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 2, 1, 2)
        def test_1(): ...

        @pytest.mark.needs_vtk_version(less_than=(8, 1, 0, 1))
        def test_2(): ...
        """

        return tests, dict(errors=2)

    @case(tags='reason')
    def case_reason_default(self, monkeypatch: pytest.MonkeyPatch):
        """Test the reason kwargs"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 1)
        def test_1(): ...

        @pytest.mark.needs_vtk_version(9, 1, less_than=(9, 2))
        def test_2(): ...

        @pytest.mark.needs_vtk_version(less_than=(8, 1))
        def test_3(): ...

        """

        monkeypatch.setattr(pyvista, 'vtk_version_info', (8, 2))

        return tests, [
            r'SKIPPED.*Test needs VTK version >= \(9, 1, 0\), current is \(8, 2\)',
            r'SKIPPED.*Test needs \(9, 1, 0\) <= VTK version < \(9, 2, 0\), current is \(8, 2\).',
            r'SKIPPED.*Test needs VTK version < \(8, 1, 0\), current is \(8, 2\).',
        ]

    @case(tags='reason')
    def case_reason_custom(self, monkeypatch: pytest.MonkeyPatch):
        """Test the custom reason kwargs"""

        tests = """
        import pytest

        @pytest.mark.needs_vtk_version(9, 1, reason="foo")
        def test(): ...

        """

        monkeypatch.setattr(pyvista, 'vtk_version_info', (8, 2))

        return tests, ['SKIPPED.*foo']


@parametrize_with_cases(
    'tests, outcome', cases=CasesNeedsVtk, filter=~ft.has_tag('raises') & ~ft.has_tag('reason')
)
def test_needs_vtk_version(tests: str, outcome: dict, pytester: pytest.Pytester):
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(**outcome)


@parametrize_with_cases('tests, outcome', cases=CasesNeedsVtk, has_tag='raises')
def test_needs_vtk_version_raises(tests: str, outcome: dict, pytester: pytest.Pytester):
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(**outcome)


@parametrize_with_cases('tests, match', cases=CasesNeedsVtk, has_tag='reason')
def test_needs_vtk_version_reason(tests: str, match: list[str], pytester: pytest.Pytester):
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.stdout.re_match_lines(match)


@pytest.mark.skipif(os.name != 'nt', reason='Needs Windows platform to run')
def test_skip_windows(
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,
):
    tests = """
    import pytest

    @pytest.mark.skip_windows
    def test_skipped():
        ...

    def test_not_skipped():
        ...

    @pytest.mark.skip_windows(foo=1)
    def test_skipped_wrong():
        ...
    """
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(skipped=1, passed=1, errors=1)
    results.stdout.re_match_lines(
        [
            r'.*Marker `skip_windows` called with incorrect arguments\.',
            r".*Signature should be: @pytest\.mark\.skip_windows\(reason: str = 'Test fails on "
            r"Windows'\)",
        ]
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)

    assert 'test_not_skipped' in report.passed
    assert 'test_skipped' in report.skipped
    assert 'test_skipped_wrong' in report.error


@pytest.fixture
def _patch_uses_egl(mocker: MockerFixture):
    from pyvista.plotting.utilities import gl_checks

    m = mocker.patch.object(gl_checks, 'uses_egl')
    m.return_value = True


@pytest.mark.usefixtures('_patch_uses_egl')
def test_skip_egl(
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,
):
    tests = """
    import pytest

    @pytest.mark.skip_egl
    def test_skipped():
        ...

    @pytest.mark.skip_egl(reason="foo")
    def test_skipped_message():
        ...

    @pytest.mark.skip_egl("bar")
    def test_skipped_message_args():
        ...

    def test_not_skipped():
        ...

    @pytest.mark.skip_egl(foo=1)
    def test_skipped_wrong():
        ...

    """

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.stdout.re_match_lines(
        [
            r'.*Marker `skip_egl` called with incorrect arguments\.',
            r'.*Signature should be: @pytest\.mark\.skip_egl\(reason.*\)',
        ]
    )

    results.assert_outcomes(
        skipped=3,
        passed=1,
        errors=1,
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)

    assert 'test_not_skipped' in report.passed
    assert 'test_skipped' in report.skipped
    assert 'test_skipped_message_args' in report.skipped
    assert 'test_skipped_message' in report.skipped
    assert 'test_skipped_wrong' in report.error


@pytest.fixture
def _patch_mac_system(mocker: MockerFixture):
    import platform

    m = mocker.patch.object(platform, 'system')
    m.return_value = 'Darwin'


@pytest.mark.usefixtures('_patch_mac_system')
@parametrize(processor=['foo', None], machine=['bar', None])
def test_skip_mac(
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,
    mocker: MockerFixture,
    processor: str | None,
    machine: str | None,
):
    tests = """
    import pytest

    @pytest.mark.skip_mac
    def test_skipped():
        ...

    def test_not_skipped():
        ...

    @pytest.mark.skip_mac(foo=1)
    def test_skipped_wrong():
        ...

    @pytest.mark.skip_mac(processor="foo", machine="bar")
    def test_skipped_platform_machine():
        ...

    """

    import platform

    m = mocker.patch.object(platform, 'processor')
    m.return_value = processor

    m = mocker.patch.object(platform, 'machine')
    m.return_value = machine

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.stdout.re_match_lines(
        [
            r'.*Marker `skip_mac` called with incorrect arguments\.',
            r'.*Signature should be: @pytest\.mark\.skip_mac\(reason.*processor.*machine.*\)',
        ]
    )

    skipped = 1
    skipped += 1 if (processor is not None and machine is not None) else 0

    passed = 2
    passed -= 1 if (processor is not None and machine is not None) else 0

    results.assert_outcomes(
        skipped=skipped,
        passed=passed,
        errors=1,
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)

    assert 'test_not_skipped' in report.passed
    assert 'test_skipped' in report.skipped
    assert 'test_skipped_wrong' in report.error
    if processor is not None and machine is not None:
        assert 'test_skipped_platform_machine' in report.skipped
    else:
        assert 'test_skipped_platform_machine' in report.passed
