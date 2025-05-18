from __future__ import annotations

from http import HTTPStatus
import subprocess
import textwrap

import pytest
import requests

from doc.source.vtk_role import find_member_anchor

class_template_url = 'https://vtk.org/doc/nightly/html/class{}.html'
VTK_POLY_DATA_CLASS_URL = class_template_url.format('vtkPolyData')
VTK_IMAGE_DATA_CLASS_URL = class_template_url.format('vtkImageData')

GET_DIMENSIONS_ANCHOR = 'a3cbcab15f8744efeb5300e21dcfbe9af'
GET_DIMENSIONS_URL = VTK_IMAGE_DATA_CLASS_URL + f'#{GET_DIMENSIONS_ANCHOR}'

SET_EXTENT_ANCHOR = 'a6e4c45a06e756c2d9d72f2312e773cb9'
SET_EXTENT_URL = VTK_IMAGE_DATA_CLASS_URL + f'#{SET_EXTENT_ANCHOR}'


@pytest.fixture(scope='module')
def vtk_polydata_html():
    """Fixture that fetches HTML for vtkPolyData once per test module."""
    response = requests.get(VTK_POLY_DATA_CLASS_URL, timeout=3)
    response.raise_for_status()
    return response.text


def test_find_member_anchor(vtk_polydata_html):
    anchor = find_member_anchor(vtk_polydata_html, 'Foo')
    assert anchor is None

    anchor = find_member_anchor(vtk_polydata_html, 'GetVerts')
    assert isinstance(anchor, str)

    # Confirm that the anchor appears in the HTML
    assert f'id="{anchor}"' in vtk_polydata_html

    # Confirm that the final URL with anchor resolves
    full_url = f'{VTK_POLY_DATA_CLASS_URL}#{anchor}'
    response = requests.get(full_url, timeout=3, allow_redirects=True)
    assert response.status_code == HTTPStatus.OK


@pytest.fixture
def temp_doc_project(tmp_path):
    """Set up a minimal Sphinx doc project using the :vtk: role in a docstring."""
    src = tmp_path / 'src'
    src.mkdir()

    # example.py with :vtk: usage
    (src / 'example.py').write_text(
        textwrap.dedent("""
        def foo():
            \"\"\"Example function referencing :vtk:`vtkPolyData`

            We can also reference members, e.g.:

            - :vtk:`vtkImageData.GetDimensions`
            - :vtk:`vtkImageData.SetExtent`

            \"\"\"
            pass
        """)
    )

    # conf.py with VTKRole registration
    (src / 'conf.py').write_text(
        textwrap.dedent("""
        import os
        import sys
        sys.path.insert(0, os.path.abspath("."))
        extensions = ["sphinx.ext.autodoc"]

        from doc.source.vtk_role import VTKRole

        def setup(app):
            app.add_role('vtk', VTKRole())
        """)
    )

    # index.rst that includes the example module
    (src / 'index.rst').write_text(
        textwrap.dedent("""
        API Reference
        =============

        .. automodule:: example
           :members:
           :undoc-members:
        """)
    )

    return src


def test_vtk_role_generates_valid_link(temp_doc_project):
    build_dir = temp_doc_project.parent / '_build'
    build_html_dir = build_dir / 'html'

    result = subprocess.run(
        [
            'sphinx-build',
            '-b',
            'html',
            str(temp_doc_project),
            str(build_html_dir),
            '-W',  # Warnings as errors
            '--keep-going',
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, 'Sphinx build failed, stderr: {stderr}'

    # Check the main index.html
    index_html = build_html_dir / 'index.html'
    assert index_html.exists()

    # Confirm the vtk url is in the docs
    html = index_html.read_text(encoding='utf-8')
    assert VTK_POLY_DATA_CLASS_URL in html
    assert GET_DIMENSIONS_URL in html
    assert SET_EXTENT_URL in html
