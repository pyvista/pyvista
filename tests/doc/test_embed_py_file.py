"""Tests for pyvista.ext._embed_py_file.

Builds run in a subprocess; an in-process one leaves the ``sphinx`` logger taken over
for the rest of the session.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from pyvista.ext import _embed_py_file

if TYPE_CHECKING:
    from pathlib import Path

TARGET_VARNAME = 'PYVISTA_TEST_EMBED_TARGET'

CONF = '''\
"""Minimal Sphinx project for building a page via pyvista.ext._embed_py_file."""

from __future__ import annotations

import os

from pyvista.ext import _embed_py_file

extensions = ['pyvista.ext._embed_py_file']

_target = os.environ.get('PYVISTA_TEST_EMBED_TARGET')


def _download(name):
    """Return a local file instead of downloading one."""
    if _target is None:
        msg = f'no such file: {name}'
        raise FileNotFoundError(msg)
    return _target


_embed_py_file.download_file = _download
'''

INDEX = 'Page\n====\n\n.. embed-py-file:: sample/sample.py\n'

SCRIPT = 'import pyvista as pv\n\nMOOOOSE = pv.Sphere()\n'


class _StubApp:
    """Minimal stand-in recording the directive a Sphinx extension registers."""

    def __init__(self):
        self.directives = {}

    def add_directive(self, name, cls):
        """Record the registered directive."""
        self.directives[name] = cls


def _build(src: Path, out: Path, target: Path | None) -> subprocess.CompletedProcess:
    """Build the one-page project, embedding ``target`` when one is given."""
    src.mkdir(parents=True, exist_ok=True)
    (src / 'conf.py').write_text(CONF, encoding='utf-8')
    (src / 'index.rst').write_text(INDEX, encoding='utf-8')
    env = dict(os.environ)
    env.pop(TARGET_VARNAME, None)
    if target is not None:
        env[TARGET_VARNAME] = str(target)
    return subprocess.run(
        [sys.executable, '-msphinx', '-b', 'html', str(src), str(out)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture(scope='module')
def embedded(tmp_path_factory):
    """Build the project once with a file to embed; return (process, html dir)."""
    tmp_path = tmp_path_factory.mktemp('embed_py_file')
    script = tmp_path / 'sample.py'
    script.write_text(SCRIPT, encoding='utf-8')
    out = tmp_path / 'out'
    proc = _build(tmp_path / 'src', out, script)
    assert proc.returncode == 0, proc.stderr
    return proc, out


def test_embeds_the_downloaded_file_as_python(embedded):
    proc, out = embedded
    html = (out / 'index.html').read_text(encoding='utf-8')

    assert 'highlight-python' in html
    assert 'MOOOOSE' in html
    assert proc.stderr == ''


def test_embedded_file_is_kept_out_of_the_search_index(embedded):
    _, out = embedded
    search = (out / 'searchindex.js').read_text(encoding='utf-8')

    assert 'moooose' not in search.lower()


def test_download_failure_warns_and_embeds_nothing(tmp_path):
    out = tmp_path / 'out'
    proc = _build(tmp_path / 'src', out, None)

    assert proc.returncode == 0, proc.stderr
    assert 'Failed to embed sample/sample.py' in proc.stderr
    assert 'no such file' in proc.stderr
    assert 'highlight-python' not in (out / 'index.html').read_text(encoding='utf-8')


def test_rebuild_picks_up_a_changed_file(tmp_path):
    script = tmp_path / 'sample.py'
    script.write_text(SCRIPT, encoding='utf-8')
    src = tmp_path / 'src'
    out = tmp_path / 'out'
    assert _build(src, out, script).returncode == 0

    script.write_text('BUFFALO = 1\n', encoding='utf-8')
    stat = script.stat()
    os.utime(script, (stat.st_atime + 10, stat.st_mtime + 10))
    proc = _build(src, out, script)

    assert proc.returncode == 0, proc.stderr
    html = (out / 'index.html').read_text(encoding='utf-8')
    assert 'BUFFALO' in html
    assert 'MOOOOSE' not in html


def test_setup_registers_the_directive():
    app = _StubApp()
    _embed_py_file.setup(app)
    assert app.directives == {'embed-py-file': _embed_py_file.EmbedPyFileDirective}


def test_setup_declares_parallel_safety():
    meta = _embed_py_file.setup(_StubApp())
    assert meta['parallel_read_safe']
    assert meta['parallel_write_safe']
