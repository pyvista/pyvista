"""Sphinx role for linking to VTK documentation."""

from __future__ import annotations

from http import HTTPStatus
import re
import subprocess
import sys
import textwrap

from bs4 import BeautifulSoup
import pytest
import requests

from pyvista.ext.vtk_role import _find_member_anchor
from pyvista.ext.vtk_role import _vtk_class_url

GET_DIMENSIONS_ANCHOR = 'a3cbcab15f8744efeb5300e21dcfbe9af'
GET_DIMENSIONS_URL = f'{_vtk_class_url("vtkImageData")}#{GET_DIMENSIONS_ANCHOR}'
SET_EXTENT_ANCHOR = 'a6e4c45a06e756c2d9d72f2312e773cb9'
SET_EXTENT_URL = f'{_vtk_class_url("vtkImageData")}#{SET_EXTENT_ANCHOR}'

EVENT_IDS_ANCHOR = 'a59a8690330ebcb1af6b66b0f3121f8fe'
EVENT_IDS_URL = f'{_vtk_class_url("vtkCommand")}#{EVENT_IDS_ANCHOR}'

ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[.*?m')


@pytest.fixture(scope='module')
def vtk_polydata_html():
    """Fixture that fetches HTML for vtkPolyData once per test module."""
    response = requests.get(_vtk_class_url('vtkPolyData'), timeout=3)
    response.raise_for_status()
    return response.text


def test_find_member_anchor(vtk_polydata_html):
    anchor = _find_member_anchor(vtk_polydata_html, 'Foo')
    assert anchor is None

    anchor = _find_member_anchor(vtk_polydata_html, 'GetVerts')
    assert isinstance(anchor, str)

    # Confirm that the anchor appears in the HTML
    assert f'id="{anchor}"' in vtk_polydata_html

    # Confirm that the final URL with anchor resolves
    full_url = f'{_vtk_class_url("vtkPolyData")}#{anchor}'
    response = requests.get(full_url, timeout=3, allow_redirects=True)
    assert response.status_code == HTTPStatus.OK


def make_temp_doc_project(tmp_path, sample_text: str):
    """Set up a minimal Sphinx project that uses the :vtk: role directly in index.rst."""
    src = tmp_path / 'src'
    src.mkdir()

    # conf.py with the extension enabled
    (src / 'conf.py').write_text("""extensions = ['pyvista.ext.vtk_role']""")

    # Write index.rst with sample text
    lines = [
        'Test Page',
        '=========',
        '',
        sample_text.strip(),
        '',
    ]
    (src / 'index.rst').write_text('\n'.join(lines))

    return src


@pytest.mark.parametrize(
    ('code_block', 'expected_links', 'expected_warning'),
    [
        (  # Valid cases (get/set methods and enum)
            textwrap.dedent("""
            :vtk:`vtkImageData.GetDimensions`.
            :vtk:`vtkImageData.SetExtent`
            :vtk:`vtkCommand.EventIds`
            """),
            {
                GET_DIMENSIONS_URL: 'vtkImageData.GetDimensions',
                SET_EXTENT_URL: 'vtkImageData.SetExtent',
                EVENT_IDS_URL: 'vtkCommand.EventIds',
            },
            None,
        ),
        (  # Use an explicit title
            ':vtk:`Get Image Dimensions<vtkImageData.GetDimensions>`',
            {GET_DIMENSIONS_URL: 'Get Image Dimensions'},
            None,
        ),
        (  # Use a tilde
            ':vtk:`~vtkImageData.GetDimensions`',
            {GET_DIMENSIONS_URL: 'GetDimensions'},
            None,
        ),
        (  # Valid class but too many member parts
            ':vtk:`vtkImageData.GetDimensions.SomethingElse`',
            {
                GET_DIMENSIONS_URL: 'vtkImageData.GetDimensions.SomethingElse',
            },
            "Too many nested members in VTK reference: 'vtkImageData.GetDimensions.SomethingElse'. Interpreting as 'vtkImageData.GetDimensions', ignoring: 'SomethingElse'",
        ),
        (  # Valid class, invalid method
            ':vtk:`vtkImageData.FakeMethod`',
            {_vtk_class_url('vtkImageData'): 'vtkImageData.FakeMethod'},
            "VTK method anchor not found for: 'vtkImageData.FakeMethod' → https://vtk.org/doc/nightly/html/classvtkImageData.html#<anchor>, the class URL is used instead.",
        ),
        (  # Invalid class
            ':vtk:`NonExistentClass`',
            {_vtk_class_url('NonExistentClass'): 'NonExistentClass'},
            "Invalid VTK class reference: 'NonExistentClass' → https://vtk.org/doc/nightly/html/classNonExistentClass.html",
        ),
        (  # Test caching with valid class and invalid member
            textwrap.dedent("""
            :vtk:`vtkImageData`
            :vtk:`vtkImageData`
            :vtk:`vtkImageData.FakeEnum`
            :vtk:`vtkImageData.FakeEnum`
            """),
            {
                # Only one URL expected: the url for a bad member falls back to the class URL
                _vtk_class_url('vtkImageData'): 'vtkImageData',
            },
            "VTK method anchor not found for: 'vtkImageData.FakeEnum' → https://vtk.org/doc/nightly/html/classvtkImageData.html#<anchor>, the class URL is used instead.",
        ),
        (  # Test caching with invalid class and invalid member
            textwrap.dedent("""
           :vtk:`vtkFooBar`
           :vtk:`vtkFooBar`
           :vtk:`vtkFooBar.Baz`
           :vtk:`vtkFooBar.Baz`
           """),
            {
                _vtk_class_url('vtkFooBar'): 'vtkFooBar',
            },
            "Invalid VTK class reference: 'vtkFooBar' → https://vtk.org/doc/nightly/html/classvtkFooBar.html",
        ),
    ],
)
def test_vtk_role(tmp_path, code_block, expected_links, expected_warning):
    doc_project = make_temp_doc_project(tmp_path, code_block)
    build_dir = tmp_path / '_build'
    build_html_dir = build_dir / 'html'

    result = subprocess.run(  # noqa: UP022
        [
            sys.executable,
            '-msphinx',
            '-b',
            'html',
            str(doc_project),
            str(build_html_dir),
            '-W',
            '--keep-going',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # need to explicitly decode the output with UTF-8 to avoid UnicodeDecodeError
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    print('STDOUT:\n', stdout)
    print('STDERR:\n', stderr)

    if expected_warning:
        assert result.returncode != 0, 'Expected warning but build succeeded'

        # Verify warning message. Skip check on Windows due to Unicode/color output differences
        if not sys.platform.startswith('win'):
            assert expected_warning in stderr, (
                f'Expected warning:\n{expected_warning!r}\n\nBut got:\n{stderr}'
            )
    else:
        assert result.returncode == 0, 'Unexpected failure in Sphinx build'

    index_html = build_html_dir / 'index.html'
    assert index_html.exists()
    html = index_html.read_text(encoding='utf-8')

    # Parse HTML and validate all expected links
    soup = BeautifulSoup(html, 'html.parser')
    for href, expected_text in expected_links.items():
        link = soup.find('a', href=href)
        assert link is not None, f'Expected link with href="{href}" not found'
        assert link.text == expected_text, (
            f'Expected link text "{expected_text}", got "{link.text}"'
        )
