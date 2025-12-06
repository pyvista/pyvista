from __future__ import annotations

import pytest

import pyvista as pv

pytest_plugins = 'pytester'

from typing import TYPE_CHECKING

from tests.test_conftest import PytesterStdoutParser
from tests.test_conftest import RunResultsReport
from tests.test_conftest import results_parser  # noqa: F401

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def _load_current_config(
    pytestconfig: pytest.Config,
    pytester: pytest.Pytester,
):
    with (pytestconfig.rootpath / 'pyproject.toml').open('r') as file:
        toml = pytester.makepyprojecttoml(file.read())

    with (pytestconfig.rootpath / 'tests/plotting/conftest.py').open('r') as file:
        conftest = pytester.makeconftest(file.read())

    yield
    toml.unlink()
    conftest.unlink()


@pytest.mark.parametrize('support_plotting', [True, False])
def test_skip_plotting_mark(
    support_plotting: bool,
    pytester: pytest.Pytester,
    results_parser: PytesterStdoutParser,  # noqa: F811
    mocker: MockerFixture,
):
    tests = """
    import pytest

    @pytest.mark.skip_plotting
    def test_plotting():
        ...

    def test_no_plotting():
        ...
    """
    mock: MagicMock = mocker.patch.object(pv.plotting, 'system_supports_plotting')
    mock.return_value = support_plotting

    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    results.assert_outcomes(
        skipped=0 if support_plotting else 1,
        passed=2 if support_plotting else 1,
    )

    results = results_parser.parse(results=results)
    report = RunResultsReport(results)

    assert 'test_no_plotting' in report.passed
    assert 'test_plotting' in (report.passed if support_plotting else report.skipped)
