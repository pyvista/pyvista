from __future__ import annotations

import re
import subprocess
import sys

import pytest

import pyvista as pv


def remove_date_field(text):
    """Remove the 'Date:' line from the report output."""
    text = re.sub(r'^  Date: .*\n', '', text, flags=re.MULTILINE)
    return text.lstrip()  # remove any leading whitespace/newlines


@pytest.mark.parametrize('include_args', [True, False])
def test_report(tmp_path, include_args):
    """Test that the CLI call to `pyvista report`  matches `pyvista.Report()` in python."""
    cli_args = [sys.executable, '-m', 'pyvista', 'report']
    python_kwargs = {}
    if include_args:
        cli_args.extend(
            ['--gpu=False', '--sort=true', '--additional', 'mypy', 'scipy', '--text-width', '100']
        )
        python_kwargs = dict(gpu=False, sort=True, additional=['mypy', 'scipy'])

    expected = str(pv.Report(**python_kwargs))

    result = subprocess.run(
        cli_args,
        check=False,
        capture_output=True,
        encoding='utf-8',
        cwd=tmp_path,
    )
    actual = result.stdout.strip()

    # Helpful error if subprocess failed
    assert result.returncode == 0, (
        f'Subprocess failed with exit code {result.returncode}\n'
        f'STDOUT:\n{result.stdout}\n'
        f'STDERR:\n{result.stderr}'
    )

    # Remove Date field (time may be off by 1 second)
    expected_no_date = remove_date_field(expected)
    actual_no_date = remove_date_field(actual)

    assert actual_no_date == expected_no_date
