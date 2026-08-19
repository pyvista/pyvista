"""Shared paths and marker for tests that inspect the built documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

_root_dir = Path(__file__).parent.parent.parent.parent
BUILD_HTML_DIR = str(_root_dir / 'doc' / '_build' / 'html')

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply `needs_doc_build` to every test collected under this directory.

    `items` covers the whole session, not just this directory, since collection
    hooks aren't scoped to the conftest.py that defines them.
    """
    for item in items:
        if _THIS_DIR in item.path.parents:
            item.add_marker(pytest.mark.needs_doc_build)
