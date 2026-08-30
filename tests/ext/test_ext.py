"""Test functions from plotting extension."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pyvista as pv
from pyvista.ext import plot_directive
from pyvista.ext import viewer_directive
from pyvista.ext.plot_directive import hash_plot_code


@pytest.fixture(autouse=True)
def _restore_gallery_globals(monkeypatch):
    """Put back the globals ``plot_directive.setup`` sets.

    Loading the extension is what turns gallery mode on -- see
    ``test_setup_enables_gallery_mode`` -- and every ``setup`` call in this module makes
    that happen in-process, so without this the flag stays on for every test after it in
    the worker. With it on, ``BasePlotter.show`` exports a vtksz through trame, which
    launches the process-lifetime ``pyvista-jupyter`` server and leaves a
    ``vtkWebApplication`` behind for the leak check to blame on an unrelated test
    (pyvista/pyvista#8929, reported against ``test_command_glob[shell-plot]``).
    """
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', pv.BUILDING_GALLERY)
    monkeypatch.setattr(pv, 'OFF_SCREEN', pv.OFF_SCREEN)


def test_hash_plot_code_consistency():
    code = 'import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])'
    options = {}

    hash1 = hash_plot_code(code, options)
    hash2 = hash_plot_code(code, options)
    assert hash1 == hash2
    assert len(hash1) == 16

    different_code = 'plt.plot([4, 5, 6])'
    hash3 = hash_plot_code(different_code, options)
    assert hash1 != hash3


def test_hash_plot_code_normalization():
    code_with_noise = (
        'import matplotlib.pyplot as plt  # plotting lib\n\nplt.plot([1, 2, 3])  # make plot\n\n'
    )
    code_clean = 'import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])'
    doctest_code = '>>> import matplotlib.pyplot as plt\n>>> plt.plot([1, 2, 3])'
    options = {}

    hash1 = hash_plot_code(code_with_noise, options)
    hash2 = hash_plot_code(code_clean, options)
    hash3 = hash_plot_code(doctest_code, options)
    assert hash1 == hash2 == hash3


def test_hash_plot_code_context_option():
    code = 'plt.plot([1, 2, 3])'

    hash_no_context = hash_plot_code(code, {})
    hash_with_context = hash_plot_code(code, {'context': True})
    hash_other_option = hash_plot_code(code, {'other': True})

    assert hash_no_context != hash_with_context
    assert hash_no_context == hash_other_option


class _Builder:
    def __init__(self, target_uri):
        self.target_uri = target_uri

    def get_target_uri(self, docname):
        assert docname == 'guide/example'
        return self.target_uri


@pytest.mark.parametrize(
    ('target_uri', 'expected_viewer_uri'),
    [
        ('guide/example.html', '../_static/viewer.html'),
        ('guide/example/', '../../_static/viewer.html'),
    ],
)
def test_offline_viewer_paths_use_builder_target_uri(
    tmp_path, monkeypatch, target_uri, expected_viewer_uri
):
    monkeypatch.setattr(viewer_directive, 'HTML_VIEWER_PATH', '/tmp/viewer.html')
    out_dir = tmp_path / '_build' / 'html'
    dest_file = out_dir / '_images' / 'plot_directive' / 'guide' / 'scene.vtksz'
    dest_file.parent.mkdir(parents=True)
    dest_file.touch()
    env = SimpleNamespace(
        docname='guide/example',
        app=SimpleNamespace(outdir=out_dir, builder=_Builder(target_uri)),
    )

    viewer_uri, asset_uri = viewer_directive._offline_viewer_paths(env, dest_file)

    assert viewer_uri == expected_viewer_uri
    assert asset_uri == '../_images/plot_directive/guide/scene.vtksz'


def test_offline_viewer_paths_warns_for_asset_outside_images(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(viewer_directive, 'HTML_VIEWER_PATH', '/tmp/viewer.html')
    out_dir = tmp_path / '_build' / 'html'
    dest_file = out_dir / 'plot_directive' / 'guide' / 'scene.vtksz'
    dest_file.parent.mkdir(parents=True)
    dest_file.touch()
    env = SimpleNamespace(
        docname='guide/example',
        app=SimpleNamespace(outdir=out_dir, builder=_Builder('guide/example.html')),
    )

    with caplog.at_level('WARNING', logger=viewer_directive.__name__):
        viewer_uri, asset_uri = viewer_directive._offline_viewer_paths(env, dest_file)

    assert viewer_uri is None
    assert asset_uri is None
    assert 'is not under outdir/_images; cannot compute asset URI' in caplog.text


def test_record_namespace_is_none_when_sphinx_autocodelink_unimportable(monkeypatch):
    # `sys.modules[name] = None` is the standard way to force `import name` to raise
    # ImportError, without the package actually needing to be uninstalled.
    monkeypatch.setitem(sys.modules, 'sphinx_autocodelink', None)
    try:
        importlib.reload(plot_directive)
        assert plot_directive.record_namespace is None
    finally:
        # Undo now (rather than waiting for monkeypatch's own teardown) so this reload
        # picks the real import back up -- otherwise plot_directive stays reloaded with
        # record_namespace=None for every test that runs after this one.
        monkeypatch.undo()
        importlib.reload(plot_directive)
    assert plot_directive.record_namespace is not None


class _FakeSphinxApp:
    """Enough of Sphinx's ``Application`` for exercising ``plot_directive.setup``."""

    def __init__(self):
        self.config = SimpleNamespace()
        self.confdir = ''
        self.directives = {}
        self.connected = {}
        self.config_values = {}
        self.setup_extension_calls = []

    def add_directive(self, name, directive):
        self.directives[name] = directive

    def connect(self, event, handler):
        self.connected.setdefault(event, []).append(handler)

    def add_config_value(self, name, default, rebuild):  # noqa: ARG002 -- matches Sphinx's signature
        self.config_values[name] = default

    def setup_extension(self, name):
        self.setup_extension_calls.append(name)


def test_setup_depends_on_sphinx_autocodelink_when_available():
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    assert app.setup_extension_calls == ['sphinx_autocodelink']


def test_setup_skips_sphinx_autocodelink_when_unavailable(monkeypatch):
    monkeypatch.setattr(plot_directive, 'record_namespace', None)
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    assert app.setup_extension_calls == []


def test_autocodelink_raises_when_enabled_without_package(monkeypatch):
    monkeypatch.setattr(plot_directive, 'record_namespace', None)
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    check_autocodelink_available = app.connected['config-inited'][0]

    config = SimpleNamespace(pyvista_plot_autocodelink=True)
    with pytest.raises(RuntimeError, match='sphinx-autocodelink'):
        check_autocodelink_available(app, config)


@pytest.mark.parametrize('enabled', [True, False])
def test_autocodelink_does_not_raise_when_package_available(enabled):
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    check_autocodelink_available = app.connected['config-inited'][0]

    config = SimpleNamespace(pyvista_plot_autocodelink=enabled)
    check_autocodelink_available(app, config)  # does not raise


def test_autocodelink_does_not_raise_when_disabled_without_package(monkeypatch):
    monkeypatch.setattr(plot_directive, 'record_namespace', None)
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    check_autocodelink_available = app.connected['config-inited'][0]

    config = SimpleNamespace(pyvista_plot_autocodelink=False)
    check_autocodelink_available(app, config)  # does not raise


def test_import_does_not_enable_gallery_mode():
    """Importing the extension must not change how plotters behave process-wide.

    ``BUILDING_GALLERY`` makes ``_ALL_PLOTTERS`` hold every plotter strongly rather
    than through a weak proxy, so setting it here at import time meant that merely
    importing this module -- which collecting this file does -- leaked a plotter for
    every test that ran afterwards in the same session. A documentation build gets
    the flag from :func:`~pyvista.ext.plot_directive.setup` instead.
    """
    code = (
        'import pyvista as pv;'
        'import pyvista.ext.plot_directive;'
        'print(pv.BUILDING_GALLERY, pv.OFF_SCREEN)'
    )
    # A subprocess because this process imported the module long ago, and with the
    # environment pinned because both flags also have an environment default.
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, 'PYVISTA_BUILDING_GALLERY': 'false', 'PYVISTA_OFF_SCREEN': 'false'},
    )
    assert result.stdout.split() == ['False', 'False']


def test_setup_enables_gallery_mode(monkeypatch):
    """Loading the extension is what turns gallery mode on."""
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', False)
    monkeypatch.setattr(pv, 'OFF_SCREEN', False)

    plot_directive.setup(MagicMock())

    assert pv.BUILDING_GALLERY
    assert pv.OFF_SCREEN


_TWO_PIECE_CODE = '>>> import pyvista as pv\n>>> pv.Sphere().plot()\n>>> pv.Cube().plot()\n'


def _fake_render(output_dir, output_base):
    """Write the files ``render_figures`` would produce for ``_TWO_PIECE_CODE``."""
    files = {
        f'{output_base}_00_00.png': b'static',
        f'{output_base}_00_00.vtksz': b'interactive',
        f'{output_base}_01_00.png': b'cube',
    }
    for name, content in files.items():
        (output_dir / name).write_bytes(content)
    _, pieces = plot_directive._split_code_at_show(_TWO_PIECE_CODE)
    return [
        (pieces[0], [plot_directive.ImageFile(str(output_dir), f'{output_base}_00_00.vtksz')]),
        (pieces[1], [plot_directive.ImageFile(str(output_dir), f'{output_base}_01_00.png')]),
    ]


def test_figure_cache_round_trip(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    entry = tmp_path / 'cache' / 'abc123'
    records = [{'accessed': 'pv.Sphere', 'candidates': ['pyvista.Sphere'], 'counts_as_use': True}]

    results = _fake_render(first, 'page-abc123')
    plot_directive._store_cached_figures(
        entry, results=results, output_base='page-abc123', records=records
    )

    cached = plot_directive._load_cached_figures(
        entry, code=_TWO_PIECE_CODE, output_dir=str(second), output_base='other-abc123'
    )
    assert cached is not None
    cached_results, cached_records = cached
    assert cached_records == records
    assert [[img.basename for img in images] for _, images in cached_results] == [
        ['other-abc123_00_00.vtksz'],
        ['other-abc123_01_00.png'],
    ]
    assert [piece for piece, _ in cached_results] == [piece for piece, _ in results]
    assert (second / 'other-abc123_00_00.vtksz').read_bytes() == b'interactive'
    # the interactive scene's static companion is carried along
    assert (second / 'other-abc123_00_00.png').read_bytes() == b'static'
    assert (second / 'other-abc123_01_00.png').read_bytes() == b'cube'


def test_figure_cache_misses(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    entry = tmp_path / 'cache' / 'abc123'

    def load(code):
        return plot_directive._load_cached_figures(
            entry, code=code, output_dir=str(out), output_base='page'
        )

    assert load(_TWO_PIECE_CODE) is None  # no entry yet

    results = _fake_render(out, 'page-abc123')
    plot_directive._store_cached_figures(
        entry, results=results, output_base='page-abc123', records=None
    )

    one_piece = '>>> import pyvista as pv\n>>> pv.Sphere().plot()\n'
    assert load(one_piece) is None  # code splits into a different number of pieces

    (entry / '01_00.png').unlink()
    assert load(_TWO_PIECE_CODE) is None  # a listed file is missing from the entry


def test_figure_cache_store_loses_race(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    entry = tmp_path / 'cache' / 'abc123'
    results = _fake_render(out, 'page-abc123')
    plot_directive._store_cached_figures(
        entry, results=results, output_base='page-abc123', records=None
    )
    manifest = (entry / 'manifest.json').read_bytes()

    plot_directive._store_cached_figures(
        entry, results=results, output_base='page-abc123', records=['changed']
    )
    assert (entry / 'manifest.json').read_bytes() == manifest
    assert list(entry.parent.glob('*.tmp')) == []


def test_setup_connects_figure_cache_clear():
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    assert plot_directive._clear_figure_cache in app.connected['builder-inited']
