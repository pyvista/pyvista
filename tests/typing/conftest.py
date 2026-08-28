"""Collect the typing cases in this directory as tests."""

from __future__ import annotations

from pathlib import Path

from tests.typing.typeassert import collect_cases_from

CASES_DIR = Path(__file__).parent / 'cases'


def pytest_collect_file(file_path, parent):
    """Turn each case file into a test file of its own."""
    return collect_cases_from(file_path, parent, CASES_DIR)
