"""Tests for pyvista.ext._embed_py_file."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from sphinx.application import Sphinx
from sphinx.util.docutils import docutils_namespace

from pyvista.ext import _embed_py_file

SCRIPT = 'import pyvista as pv\n\nMESH = pv.Sphere()\n'


class _StubApp:
    """Minimal stand-in recording the directive a Sphinx extension registers."""

    def __init__(self):
        self.directives = {}

    def add_directive(self, name, cls):
        """Record the registered directive."""
        self.directives[name] = cls


@pytest.fixture
def build(tmp_path, monkeypatch):
    """Return a callable that builds a one-page project and returns (html, warnings, app)."""

    def _build(body, downloader):
        monkeypatch.setattr(_embed_py_file, 'download_file', downloader)
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'conf.py').write_text("extensions = ['pyvista.ext._embed_py_file']\n")
        (src / 'index.rst').write_text(body, encoding='utf-8')
        out = tmp_path / 'out'
        warnings = io.StringIO()
        with docutils_namespace():
            app = Sphinx(
                srcdir=str(src),
                confdir=str(src),
                outdir=str(out),
                doctreedir=str(out / '.doctrees'),
                buildername='html',
                status=None,
                warning=warnings,
                freshenv=True,
            )
            app.build()
        html = (out / 'index.html').read_text(encoding='utf-8')
        return html, warnings.getvalue(), app

    return _build


def test_embeds_the_downloaded_file_as_python(build, tmp_path):
    script = tmp_path / 'sample.py'
    script.write_text(SCRIPT, encoding='utf-8')

    html, warnings, _ = build(
        'Page\n====\n\n.. embed-py-file:: sample/sample.py\n', lambda name: str(script)
    )

    assert 'highlight-python' in html
    assert 'MESH' in html
    assert warnings == ''


def test_embedded_file_is_a_build_dependency(build, tmp_path):
    script = tmp_path / 'sample.py'
    script.write_text(SCRIPT, encoding='utf-8')

    _, _, app = build(
        'Page\n====\n\n.. embed-py-file:: sample/sample.py\n', lambda name: str(script)
    )

    deps = {Path(dep).resolve() for dep in app.env.dependencies['index']}
    assert script.resolve() in deps


def test_download_failure_warns_and_embeds_nothing(build):
    def _fail(name):
        msg = f'no such file: {name}'
        raise FileNotFoundError(msg)

    html, warnings, _ = build('Page\n====\n\n.. embed-py-file:: sample/gone.py\n', _fail)

    assert 'Failed to embed sample/gone.py' in warnings
    assert 'no such file' in warnings
    assert 'highlight-python' not in html


def test_setup_registers_the_directive():
    app = _StubApp()
    _embed_py_file.setup(app)
    assert app.directives == {'embed-py-file': _embed_py_file.EmbedPyFileDirective}


def test_setup_declares_parallel_safety():
    meta = _embed_py_file.setup(_StubApp())
    assert meta['parallel_read_safe']
    assert meta['parallel_write_safe']
