"""Test that type aliases written by name link to their documentation."""

from __future__ import annotations

from pathlib import Path
import re

from conftest import BUILD_HTML_DIR
import pytest

SIGNATURE_ALIASES = [
    ('pyvista.DataObjectFilters.translate.html', 'VectorLike'),  # subscripted
    ('pyvista.DataObjectFilters.validate_mesh.html', 'MeshValidationFields'),  # in a union
    ('pyvista.Plotter.add_bounding_box.html', 'ColorLike'),  # on its own
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


@pytest.mark.parametrize(
    ('page', 'alias'),
    [('api/core/typing.html', 'CellsLike'), ('api/utilities/colors.html', 'ColorLike')],
)
def test_type_alias_drops_inherited_docstring(page, alias):
    """Confirm a type alias does not show the docstring of its ``typing`` origin."""
    html = (Path(BUILD_HTML_DIR) / page).read_text()
    assert f'pyvista.{alias}' in html
    assert 'Represent a union type' not in html
    assert 'Type aliases are created through the type statement' not in html
