"""Test that type aliases written by name link to their documentation."""

from __future__ import annotations

import html
from pathlib import Path
import re

from conftest import BUILD_HTML_DIR
import pytest

SIGNATURE_ALIASES = [
    ('pyvista.DataObjectFilters.translate.html', 'VectorLikeFloat'),
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
    text = page.read_text()
    start = text.index('<dt class="sig sig-object py"')
    end = text.index('</dt>', start)
    return text[start:end], text[end:]


def link_anchors(page: str, text: str) -> list[str]:
    """Return the anchors of internal links displayed as ``text``."""
    pattern = rf'<a class="reference internal" href="[^"]*#([^"]+)"[^>]*>(?:<[^>]+>)*{text}<'
    return re.findall(pattern, page)


@pytest.mark.parametrize(('filename', 'alias'), SIGNATURE_ALIASES)
def test_signature_links_type_alias(filename, alias):
    """Confirm a type alias in a signature links to its documentation."""
    signature, _ = split_api_page(filename)
    assert f'pyvista.typing.{alias}' in link_anchors(signature, alias)


@pytest.mark.parametrize(('filename', 'alias'), DOCSTRING_ALIASES)
def test_docstring_links_type_alias(filename, alias):
    """Confirm a type alias in a docstring parameter type links to its documentation."""
    _, docstring = split_api_page(filename)
    assert f'pyvista.typing.{alias}' in link_anchors(docstring, alias)


def test_type_alias_drops_inherited_docstring():
    """Confirm a type alias does not show the docstring of its ``typing`` origin."""
    page = typing_page()
    assert 'pyvista.typing.ColorLike' in page
    assert 'Represent a union type' not in page
    assert 'Type aliases are created through the type statement' not in page


def typing_page() -> str:
    """Return the HTML of the typing page."""
    return (Path(BUILD_HTML_DIR) / 'api' / 'core' / 'typing.html').read_text()


def type_alias_value(page: str, alias: str) -> str:
    """Return the HTML of the value block of a type alias."""
    start = page.index(f'id="pyvista.typing.{alias}"')
    assert 'type-alias-value' in page[page.rindex('<pre', 0, start) : start]
    return page[start : page.index('</pre>', start)]


@pytest.mark.parametrize(
    ('alias', 'value'),
    [
        ('CellsLike', 'MatrixLikeInt | VectorLikeInt'),
        ('TransformLike', 'RotationLike | vtkMatrix4x4 | vtkTransform'),
        ('LineStyle', "Literal['', '-', '--', ':', '-.', '-..']"),
        (
            'VectorLikeInt',
            (
                'Sequence[int]\n'
                '| Sequence[np.integer | np.bool]\n'
                '| np.ndarray[tuple[int], np.dtype[np.integer | np.bool]]'
            ),
        ),
    ],
)
def test_type_alias_value_replaces_signature(alias, value):
    """Confirm a type alias shows its value in place of its signature, keeping its anchor."""
    page = typing_page()
    assert f'<dt class="sig sig-object py" id="pyvista.typing.{alias}"' not in page
    text = re.sub(r'<[^>]+>', '', type_alias_value(page, alias).split('>', 1)[1])
    assert html.unescape(text).strip() == value


@pytest.mark.parametrize(
    ('alias', 'anchor'),
    [
        ('CellsLike', 'pyvista.typing.MatrixLikeInt'),
        ('VectorLikeInt', 'numpy.integer'),
        ('VectorLikeInt', 'collections.abc.Sequence'),
    ],
)
def test_type_alias_value_links_types(alias, anchor):
    """Confirm the value of a type alias links the types it names."""
    assert re.search(rf'href="[^"]*#{re.escape(anchor)}"', type_alias_value(typing_page(), alias))


def test_type_alias_drops_alias_line():
    """Confirm no type alias repeats its value as an "alias of" line."""
    assert 'alias of' not in typing_page()


@pytest.mark.parametrize('alias', ['NamedColors', 'NamedColormaps'])
def test_type_alias_hidden_value(alias):
    """Confirm a type alias documented without its value keeps its anchor but no signature."""
    page = typing_page()
    assert f'<dt class="sig sig-object py" id="pyvista.typing.{alias}"' not in page
    start = page.index(f'id="pyvista.typing.{alias}"')
    assert page[page.rindex('<', 0, start) : start].startswith('<dd')
    assert 'type-alias-value' not in page[start : page.index('</dd>', start)]
