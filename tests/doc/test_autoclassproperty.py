"""Tests for pyvista.ext._autoclassproperty: unit tests, plus a real build of a tiny project."""

from __future__ import annotations

import subprocess
import sys

import pytest
from sphinx.ext.autosummary import generate

import pyvista as pv
from pyvista.core.utilities.fileio import _FileIOBase
from pyvista.core.utilities.misc import _classproperty
from pyvista.ext import _autoclassproperty as autoclassproperty


def test_attrgetter_returns_classproperty():
    prop = autoclassproperty.classproperty_attrgetter(_FileIOBase, 'extensions')
    assert isinstance(prop, _classproperty)
    assert prop.__doc__.startswith('Return the file extensions')


def test_attrgetter_finds_inherited_classproperty():
    prop = autoclassproperty.classproperty_attrgetter(pv.PLYReader, 'extensions')
    assert prop is vars(_FileIOBase)['extensions']


def test_attrgetter_defers_to_getattr():
    assert autoclassproperty.classproperty_attrgetter(pv.PLYReader, 'path') is pv.PLYReader.path
    assert autoclassproperty.classproperty_attrgetter(int, 'missing', 3) == 3


def test_attrgetter_respects_override():
    class Child(_FileIOBase):
        extensions = ('.foo',)

    assert autoclassproperty.classproperty_attrgetter(Child, 'extensions') == ('.foo',)


@pytest.fixture
def patched_content(monkeypatch):
    """Patch a recording stand-in for ``generate_autosummary_content``; return its calls."""
    calls = []
    monkeypatch.setattr(
        generate, 'generate_autosummary_content', lambda *args, **_kwargs: calls.append(args)
    )
    autoclassproperty._patch_autosummary_content()
    return calls


@pytest.mark.parametrize(
    ('name', 'parent', 'expected'),
    [
        ('mod._FileIOBase.extensions', _FileIOBase, vars(_FileIOBase)['extensions']),
        ('mod._FileIOBase.path', _FileIOBase, 'value'),
        ('mod.extensions', pv, 'value'),
    ],
    ids=['classproperty', 'property', 'module-parent'],
)
def test_patch_passes_classproperty(patched_content, name, parent, expected):
    generate.generate_autosummary_content(name, 'value', parent)
    assert patched_content[-1][1] is expected


@pytest.mark.usefixtures('patched_content')
def test_patch_is_applied_once():
    patched = generate.generate_autosummary_content
    autoclassproperty._patch_autosummary_content()
    assert generate.generate_autosummary_content is patched


def test_runtime_value_unchanged():
    assert pv.PLYReader.extensions == ('.ply',)


@pytest.fixture(scope='module')
def tiny_build(tmp_path_factory):
    """Build one autosummary page per ``_FileIOBase`` class property."""
    tmp_path = tmp_path_factory.mktemp('autoclassproperty')
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    (source_dir / 'conf.py').write_text(
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',"
        " 'pyvista.ext._autoclassproperty']\n"
        'autosummary_generate = True\n',
        encoding='utf-8',
    )
    (source_dir / 'index.rst').write_text(
        'Test\n====\n\n.. currentmodule:: pyvista.core.utilities.fileio\n\n'
        '.. autosummary::\n   :toctree: _autosummary\n\n'
        '   _FileIOBase.extensions\n   _FileIOBase.extension_patterns\n',
        encoding='utf-8',
    )
    build_dir = tmp_path / 'build'
    proc = subprocess.run(
        [sys.executable, '-msphinx', '-b', 'text', str(source_dir), str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return source_dir, build_dir


@pytest.mark.parametrize('name', ['extensions', 'extension_patterns'])
def test_tiny_build(tiny_build, name):
    source_dir, build_dir = tiny_build
    stem = f'pyvista.core.utilities.fileio._FileIOBase.{name}'

    stub = (source_dir / '_autosummary' / f'{stem}.rst').read_text(encoding='utf-8')
    assert f'.. autoproperty:: _FileIOBase.{name}' in stub

    page = (build_dir / '_autosummary' / f'{stem}.txt').read_text(encoding='utf-8')
    assert f'property _FileIOBase.{name}:' in page
    assert '= ()' not in page
    assert 'associated with this class' in page
