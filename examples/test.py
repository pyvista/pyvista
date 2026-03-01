from __future__ import annotations

import os
from pathlib import Path

import pytest

THIS_FILE = Path(__file__).absolute()
EXAMPLES_DIR = Path(__file__).parent.absolute()


def collect_example_files():
    test_files = []
    for dirpath, _, filenames in os.walk(EXAMPLES_DIR):
        for filename in filenames:
            full_path = Path(dirpath) / filename
            if full_path == THIS_FILE or not filename.endswith('.py'):
                continue
            # Use relative path and cast to str for better repr in pytest output
            rel_path = full_path.relative_to(EXAMPLES_DIR)
            test_files.append(str(rel_path))
    return test_files


@pytest.mark.parametrize("test_file", collect_example_files())
def test_example(test_file):
    exit_code = os.system(f"python {test_file}")
    assert exit_code == 0, f"Test failed: {test_file}"
