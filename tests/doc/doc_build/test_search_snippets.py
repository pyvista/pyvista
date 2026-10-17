"""Test the search-result snippet index written alongside the built documentation."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import BUILD_HTML_DIR


def test_api_page_snippet_is_the_docstring_summary():
    """A member page's snippet is its summary line, not a hoisted docstring section."""
    index = json.loads((Path(BUILD_HTML_DIR) / 'searchsummaries.json').read_text())

    snippet = index['api/core/_autosummary/pyvista.DataObjectFilters.transform']

    assert snippet.startswith('Transform this mesh with a 4x4 transform.')
