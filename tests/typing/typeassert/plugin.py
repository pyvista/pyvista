"""Pytest integration: collect case files and run each case as its own test.

A project wires this up from a conftest next to its cases::

    from pathlib import Path

    from typeassert.plugin import collect_cases_from

    CASES_DIR = Path(__file__).parent / 'cases'


    def pytest_collect_file(file_path, parent):
        return collect_cases_from(file_path, parent, CASES_DIR)

Each case file then collects as a test file of its own, one test per case for
the runtime half and one for the static half, plus a `setup` test covering the
lines that are not cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ._cases import CaseSkipped
from ._cases import collect_case_file
from ._mypy import run_mypy

if TYPE_CHECKING:
    from ._cases import Case
    from ._cases import CaseFile
    from ._mypy import Diagnostic

_DIAGNOSTICS = '_typeassert_diagnostics'


def collect_cases_from(file_path: Path, parent, cases_dir: Path):
    """Return a collector for `file_path`, or `None` if it is not a case file."""
    if file_path.suffix != '.py' or file_path.parent != Path(cases_dir):
        return None
    return CaseFileCollector.from_parent(parent, path=file_path, cases_dir=Path(cases_dir))


def _diagnostics(config, cases_dir: Path) -> dict[Path, list[Diagnostic]]:
    """Type-check the cases once per session and cache the result on the config."""
    cached = getattr(config, _DIAGNOSTICS, None)
    if cached is None:
        root = Path(config.rootpath)
        package = '.'.join(Path(cases_dir).relative_to(root).parts)
        # A cache dir per xdist worker: the run is cheap once warm, and sharing one
        # cache between concurrent workers is what makes it stale.
        worker = getattr(config, 'workerinput', {}).get('workerid', 'master')
        cached = run_mypy(
            package, root=root, cache_dir=root / '.mypy_cache' / f'typeassert-{worker}'
        )
        setattr(config, _DIAGNOSTICS, cached)
    return cached


def _report(case_file: CaseFile, errors: list[Diagnostic]) -> str:
    """Return the checker's messages against the source lines they came from."""
    source = case_file.path.read_text(encoding='utf-8').splitlines()
    body = '\n'.join(
        f'{case_file.name}:{error.line}: {error.message}\n\t{source[error.line - 1].strip()}'
        for error in errors
    )
    return f'Type checking reported {len(errors)} error(s):\n{body}'


class CaseFileCollector(pytest.File):
    """Collects one case file as a test file."""

    def __init__(self, *args, cases_dir: Path, **kwargs) -> None:
        """Record which directory this file was collected from."""
        super().__init__(*args, **kwargs)
        self.cases_dir = cases_dir

    def collect(self):
        """Yield a test for the file's setup, and two for each of its cases."""
        case_file = collect_case_file(Path(self.path))
        yield SetupItem.from_parent(self, name='setup', case_file=case_file)
        for case in case_file.cases:
            for item_type, suffix in ((RuntimeItem, 'runtime'), (StaticItem, 'static')):
                yield item_type.from_parent(
                    self, name=f'{case.id} [{suffix}]', case_file=case_file, case=case
                )


class _Item(pytest.Item):
    """Shared plumbing for the tests a case file collects."""

    def __init__(self, *args, case_file: CaseFile, **kwargs) -> None:
        """Record the case file this test belongs to."""
        super().__init__(*args, **kwargs)
        self.case_file = case_file

    @property
    def cases_dir(self) -> Path:
        """Return the directory the case file was collected from."""
        return self.parent.cases_dir

    def reportinfo(self):
        """Locate this test in its case file."""
        return self.path, None, self.name


class SetupItem(_Item):
    """Checks a case file's setup: that it is well formed, runs, and type-checks."""

    def runtest(self) -> None:
        """Assert the file parses, its setup executes, and nothing else is wrong."""
        # Report the file's own problem before asking for a type-check of it: a file
        # the checker cannot parse fails the whole run, which would mask the reason.
        if self.case_file.error is not None:
            pytest.fail(self.case_file.error, pytrace=False)

        namespace = self.case_file.setup_namespace()
        unknown = self.case_file.unknown_skips(namespace)
        if unknown:
            listed = '\n'.join(f'  {key}' for key in unknown)
            pytest.fail(
                f'SKIP_RUNTIME names expressions that no case in this file makes, so the '
                f'skip no longer applies to anything:\n{listed}',
                pytrace=False,
            )

        errors = [
            diagnostic
            for diagnostic in _diagnostics(self.config, self.cases_dir).get(
                self.case_file.path.resolve(), []
            )
            if diagnostic.line in self.case_file.setup_lines
        ]
        if errors:
            pytest.fail(_report(self.case_file, errors), pytrace=False)


class RuntimeItem(_Item):
    """Checks the value a case builds against the type the case expects."""

    def __init__(self, *args, case: Case, **kwargs) -> None:
        """Record the case this test runs."""
        super().__init__(*args, **kwargs)
        self.case = case

    def runtest(self) -> None:
        """Run this case, and only it."""
        try:
            self.case_file.run(self.case)
        except CaseSkipped as skipped:
            pytest.skip(str(skipped))


class StaticItem(_Item):
    """Checks the type a checker infers for a case against the type it expects."""

    def __init__(self, *args, case: Case, **kwargs) -> None:
        """Record the case this test checks."""
        super().__init__(*args, **kwargs)
        self.case = case

    def runtest(self) -> None:
        """Assert the checker reports nothing on this case's lines."""
        errors = [
            diagnostic
            for diagnostic in _diagnostics(self.config, self.cases_dir).get(
                self.case_file.path.resolve(), []
            )
            if diagnostic.line in self.case.lines
        ]
        if errors:
            pytest.fail(_report(self.case_file, errors), pytrace=False)
