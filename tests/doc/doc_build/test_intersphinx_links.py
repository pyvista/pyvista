"""Test that intersphinx links point at the project that documents each object."""

from __future__ import annotations

from pathlib import Path
import re

from conftest import BUILD_HTML_DIR
import pytest

_HREF_RE = re.compile(r'href="([^"]+)"')


@pytest.mark.parametrize(
    ('page', 'target'),
    [
        pytest.param(
            'pyvista.DataObjectFilters.validate_mesh.html',
            '/library/collections.abc.html#collections.abc.Sequence',
            id='Sequence-in-signature',
        ),
        pytest.param(
            'pyvista.PolyData.verts.html',
            '/reference/arrays.scalars.html#numpy.int64',
            id='int64-in-example-code',
        ),
        pytest.param(
            'pyvista.from_trimesh.html',
            'trimesh.org/trimesh.html#trimesh.Trimesh',
            id='Trimesh',
        ),
    ],
)
def test_link_target(page, target):
    """Confirm a name links to the docs of the project that defines it."""
    path = next(Path(BUILD_HTML_DIR).rglob(page), None)
    assert path is not None, f'{page} not found under {BUILD_HTML_DIR}'

    hrefs = _HREF_RE.findall(path.read_text(encoding='utf-8'))

    assert any(href.endswith(target) for href in hrefs), f'{page} has no link to {target}'


def test_no_links_to_trimesh_reexports():
    """Confirm names ``trimesh.typed`` re-exports, like ``Sequence``, never link to trimesh."""
    pages = sorted(Path(BUILD_HTML_DIR).rglob('*.html'))
    assert pages, f'no built pages found under {BUILD_HTML_DIR}. Build the documentation first.'

    linked = [
        str(path.relative_to(BUILD_HTML_DIR))
        for path in pages
        if b'trimesh.org/trimesh.typed.html' in path.read_bytes()
    ]

    assert not linked
