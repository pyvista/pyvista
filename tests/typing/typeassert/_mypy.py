"""Run Mypy over the case files and read its diagnostics back."""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# `path:line:col: severity: message`, with the column absent on whole-file diagnostics.
_DIAGNOSTIC = re.compile(r'^(?P<path>.+?):(?P<line>\d+):(?:\d+:)? (?P<severity>\w+): (?P<msg>.*)$')


class MypyError(Exception):
    """Raised when Mypy could not run at all, as opposed to reporting diagnostics."""


@dataclass(frozen=True)
class Diagnostic:
    """One error Mypy reported, at one line of one file."""

    path: Path
    line: int
    message: str


def run_mypy(package: str, *, root: Path, cache_dir: Path) -> dict[Path, list[Diagnostic]]:
    """Type-check `package` from `root` and return its errors keyed by file.

    Mypy runs as a separate process so that a crash surfaces as a failed call
    rather than taking the test session down with it.
    """
    # `--follow-imports=silent` types the symbols the cases use without reporting the
    # host project's own diagnostics, which vary by platform and dependency versions.
    args = [
        sys.executable,
        '-m',
        'mypy',
        '--follow-imports=silent',
        '--no-color-output',
        '--no-error-summary',
        '--no-pretty',
        '--show-traceback',
        f'--cache-dir={cache_dir}',
        '--package',
        package,
    ]
    process = subprocess.run(args, capture_output=True, cwd=root, text=True, check=False)
    # Mypy exits 1 when it reports diagnostics and 2 when it could not run.
    if process.returncode > 1 or process.stderr:
        msg = f'Mypy failed to run:\n{" ".join(args)}\n\n{process.stderr}{process.stdout}'
        raise MypyError(msg)

    diagnostics: dict[Path, list[Diagnostic]] = {}
    for line in process.stdout.splitlines():
        match = _DIAGNOSTIC.match(line)
        if match is None or match['severity'] != 'error':
            continue
        path = (root / match['path']).resolve()
        diagnostics.setdefault(path, []).append(
            Diagnostic(path=path, line=int(match['line']), message=match['msg'])
        )
    return diagnostics
