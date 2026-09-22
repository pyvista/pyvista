"""Test that type aliases written by name link to their documentation."""

from __future__ import annotations

from pathlib import Path
import re

from conftest import BUILD_HTML_DIR
import pytest

SIGNATURE_ALIASES = [
    ('pyvista.DataObjectFilters.translate.html', 'VectorLike'),  # subscripted
    ('pyvista.DataObjectFilters.validate_mesh.html', 'MeshValidationFields'),  # in a union
    ('pyvista.Plotter.add_mesh.html', 'ColorLike'),
]
DOCSTRING_ALIASES = [
    *SIGNATURE_ALIASES,
    # documented outside the ``pyvista`` module
    ('pyvista.plotting.picking.PickingComponent.enable_cell_picking.html', 'ColorLike'),
]


def split_api_page(filename: str) -> tuple[str, str]:
    """Return the signature and the rest of a generated single-object API page."""
    page = next(Path(BUILD_HTML_DIR).rglob(filename), None)
    assert page is not None, f'{filename} not found under {BUILD_HTML_DIR}'
    html = page.read_text()
    start = html.index('<dt class="sig sig-object py"')
    end = html.index('</dt>', start)
    return html[start:end], html[end:]


def link_anchors(html: str, text: str) -> list[str]:
    """Return the anchors of internal links displayed as ``text``."""
    pattern = rf'<a class="reference internal" href="[^"]*#([^"]+)"[^>]*>(?:<[^>]+>)*{text}<'
    return re.findall(pattern, html)


@pytest.mark.parametrize(('filename', 'alias'), SIGNATURE_ALIASES)
def test_signature_links_type_alias(filename, alias):
    """Confirm a type alias in a signature links to its documentation."""
    signature, _ = split_api_page(filename)
    assert f'pyvista.{alias}' in link_anchors(signature, alias)


@pytest.mark.parametrize(('filename', 'alias'), DOCSTRING_ALIASES)
def test_docstring_links_type_alias(filename, alias):
    """Confirm a type alias in a docstring parameter type links to its documentation."""
    _, docstring = split_api_page(filename)
    assert f'pyvista.{alias}' in link_anchors(docstring, alias)
