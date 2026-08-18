"""Shared paths and marker for tests that inspect the built documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
HTML_DIR = str(Path(BUILD_DIR) / 'html')

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply `needs_docs_build` to every test collected under this directory.

    `items` covers the whole session, not just this directory, since collection
    hooks aren't scoped to the conftest.py that defines them.
    """
    for item in items:
        if _THIS_DIR in item.path.parents:
            item.add_marker(pytest.mark.needs_docs_build)
