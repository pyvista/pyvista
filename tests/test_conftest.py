from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import pytest
from pytest_cases import case
from pytest_cases import filters as ft
from pytest_cases import parametrize
from pytest_cases import parametrize_with_cases

import pyvista as pv

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


class CasesRequireVtkVersion:
    """Cases for the ``require_vtk_version`` marker.

    The version constraints themselves are covered by the tests for
    :func:`pyvista.require_vtk_version` in ``tests/core/test_utilities.py``. These
    cases only cover the behavior which is specific to the pytest marker.
    """

    @case
    def case_skipped_or_passed(self, monkeypatch: pytest.MonkeyPatch):
        """Test that an unsatisfied constraint skips the test instead of raising."""

        tests = """
        import pytest

        @pytest.mark.require_vtk_version(9, 1)
        def test_at_least(): ...

        @pytest.mark.require_vtk_version(9, 3)
        def test_at_least_unsatisfied(): ...

        @pytest.mark.require_vtk_version(less_than=(9, 3))
        def test_less_than(): ...

        @pytest.mark.require_vtk_version(less_than=(9, 1))
        def test_less_than_unsatisfied(): ...
        """

        monkeypatch.setattr(pv, 'vtk_version_info', (9, 2, 0))
        monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', (8, 0, 0))

        return tests, dict(passed=2, skipped=2)

    @case
    def case_multiple_markers(self, monkeypatch: pytest.MonkeyPatch):
        """Test that every marker is evaluated when a test is decorated several times."""

        tests = """
        import pytest

        @pytest.mark.require_vtk_version(9, 1)
        @pytest.mark.require_vtk_version(less_than=(9, 3))
        def test_1(): ...

        @pytest.mark.require_vtk_version(less_than=(9, 3))
        @pytest.mark.require_vtk_version(9, 1)
        def test_2(): ...
        """

        monkeypatch.setattr(pv, 'vtk_version_info', (8, 2, 0))
        monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', (8, 0, 0))

        return tests, dict(skipped=2)

    @case(tags='raises')
    def case_raises_signature(self):
        """Test when not specifying any version, or using bad signature."""

        tests = """
        import pytest

        @pytest.mark.require_vtk_version
        def test_1(): ...

        @pytest.mark.require_vtk_version(foo=1)
        def test_2(): ...
        """

        return tests, dict(errors=2)

    @case(tags='raises')
    def case_raises_obsolete_constraint(self, monkeypatch: pytest.MonkeyPatch):
        """Test that an obsolete marker errors out instead of being skipped.

        The warning emitted for an obsolete constraint is turned into an error by the
        ``filterwarnings`` setting of the test suite.
        """

        tests = """
        import pytest

        @pytest.mark.require_vtk_version(9, 0, 0)
        def test_1(): ...

        @pytest.mark.require_vtk_version(less_than=(8, 9, 9))
        def test_2(): ...
        """

        monkeypatch.setattr(pv, 'vtk_version_info', (9, 0, 0))

        return tests, dict(errors=2)


@parametrize_with_cases(
    'tests, outcome', cases=CasesRequireVtkVersion, filter=~ft.has_tag('raises')
)
def test_require_vtk_version(tests: str, outcome: dict, pytester: pytest.Pytester):
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(**outcome)


@parametrize_with_cases('tests, outcome', cases=CasesRequireVtkVersion, has_tag='raises')
def test_require_vtk_version_raises(tests: str, outcome: dict, pytester: pytest.Pytester):
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(**outcome)


def test_require_vtk_version_skip_reason(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the skip reason is the message of the raised error."""

    tests = """
    import pytest

    @pytest.mark.require_vtk_version(9, 1)
    def test_default_reason(): ...

    @pytest.mark.require_vtk_version(9, 1, reason="foo")
    def test_custom_reason(): ...
    """

    monkeypatch.setattr(pv, 'vtk_version_info', (8, 2, 0))
    monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', (8, 0, 0))

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(skipped=2)
    results.stdout.re_match_lines(
        [
            (
                r'SKIPPED.*This feature requires VTK version 9\.1\.0 or greater\. '
                r'The installed version is 8\.2\.0\.'
            ),
            r'SKIPPED.*foo',
        ]
    )


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
            (
                r'.*Signature should be: @pytest\.mark\.skip_windows'
                r"\(reason: str = 'Test fails on Windows'\)"
            ),
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
