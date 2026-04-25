from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

THIS_PATH = Path(__file__).parent.absolute()
EXAMPLES_DIR = Path(__file__).parent.parent.absolute()


def collect_example_files():
    test_files = []
    for dirpath, _, filenames in os.walk(EXAMPLES_DIR):
        if THIS_PATH.match(dirpath) or dirpath.endswith('__pycache__'):
            continue
        for filename in filenames:
            full_path = Path(dirpath) / filename
            if not filename.endswith('.py'):
                continue
            # Use relative path and cast to str for better repr in pytest output
            rel_path = full_path.relative_to(EXAMPLES_DIR)
            test_files.append(str(rel_path))
    return test_files


@pytest.mark.parametrize('test_file', collect_example_files())
def test_serve(test_file):
    returncode = subprocess.run(
        ['python', EXAMPLES_DIR / test_file, '--serve', '--timeout', '1', '--port', '0'],
        check=False,
    ).returncode
    assert returncode == 0
