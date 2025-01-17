from __future__ import annotations

import re

import pytest

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
    errors = _ReportDescriptor()
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

    if cml:
        assert 'test_downloads' in report.passed
        assert 'test_no_downloads' in report.passed
    else:
        assert 'test_downloads' in report.skipped
        assert 'test_no_downloads' in report.passed


@pytest.mark.parametrize('greater', [True, False])
def test_needs_vtk_version_tuple(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    greater: bool,
):
    tests = """
    import pytest

    @pytest.mark.needs_vtk_version(9,1)
    def test_greater_9_1():
        ...

    @pytest.mark.needs_vtk_version((9,1))
    def test_greater_9_1_tuple():
        ...

    """
    import pyvista

    value = (8, 2) if greater else (9, 2)
    monkeypatch.setattr(pyvista, 'vtk_version_info', value)

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(
        passed=0 if greater else 2,
        skipped=2 if greater else 0,
    )


@pytest.mark.parametrize('version', [(9, 0), (9, 2)])
def test_needs_vtk_version(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int],
):
    tests = """
    import pytest

    @pytest.mark.needs_vtk_version(9,1)
    def test_greater_9_1():
        ...

    """
    import pyvista

    monkeypatch.setattr(pyvista, 'vtk_version_info', version)

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p, '-ra')

    results.assert_outcomes(
        passed=1 if version == (9, 2) else 0,
        skipped=0 if version == (9, 2) else 1,
    )

    if version == (9, 0):
        results.stdout.re_match_lines([r'SKIPPED.*Test needs VTK 9.1 or newer'])
