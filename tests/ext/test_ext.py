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


def test_split_code_at_show_ends_a_piece_at_a_commented_show():
    code = '>>> pl.show()  # doctest: +SKIP\n>>> pl = pv.Plotter()\n'
    _, pieces = plot_directive._split_code_at_show(code)
    assert pieces == ['>>> pl.show()  # doctest: +SKIP', '>>> pl = pv.Plotter()\n']


def test_split_code_at_show_ends_a_piece_at_a_comment_holding_a_quote():
    code = ">>> pl.show()  # don't rely on this\n>>> pl = pv.Plotter()\n"
    _, pieces = plot_directive._split_code_at_show(code)
    assert pieces == [">>> pl.show()  # don't rely on this", '>>> pl = pv.Plotter()\n']


@pytest.mark.parametrize('quote', ["'", '"'])
def test_split_code_at_show_keeps_a_hash_inside_a_string(quote):
    show = f'>>> mesh.plot(color={quote}#ff0000{quote})'
    _, pieces = plot_directive._split_code_at_show(f'{show}\n>>> a = 1\n')
    assert pieces == [show, '>>> a = 1\n']


DOCTEST_WITH_SKIP = '>>> a = 1\n>>> explode()  # doctest: +SKIP\n>>> b = 2\n'


def test_executable_piece_filters_when_a_statement_is_skipped():
    filtered = plot_directive._executable_piece(DOCTEST_WITH_SKIP, is_doctest=True)
    assert filtered == 'a = 1\nb = 2\n'


def test_executable_piece_filters_a_skipped_multiline_statement():
    piece = '>>> total = sum(\n...     [1, 2]\n... )  # doctest: +SKIP\n>>> a = 1\n'
    filtered = plot_directive._executable_piece(piece, is_doctest=True)
    assert 'sum' not in filtered
    assert 'a = 1' in filtered


@pytest.mark.parametrize('marker', ['# doctest: +SKIP', '# doctest:+SKIP', '#doctest: +SKIP'])
def test_executable_piece_matches_skip_spacing_variants(marker):
    piece = f'>>> a = 1\n>>> explode()  {marker}\n'
    assert plot_directive._executable_piece(piece, is_doctest=True) == 'a = 1\n'


def test_executable_piece_none_without_a_skip():
    assert plot_directive._executable_piece('>>> a = 1\n', is_doctest=True) is None


def test_executable_piece_none_for_non_doctest():
    assert plot_directive._executable_piece("x = 'doctest: +SKIP'", is_doctest=False) is None


def _render(code, tmp_path):
    """Call render_figures with the minimal config the code path needs."""
    config = SimpleNamespace(
        pyvista_plot_setup=None, pyvista_plot_cleanup=None, pyvista_plot_autocodelink=False
    )
    return plot_directive.render_figures(
        code=code,
        code_path='<test>',
        output_dir=str(tmp_path),
        output_base='out',
        context=False,
        function_name=None,
        config=config,
        force_static=True,
    )


def test_render_figures_runs_the_example_after_a_skipped_show(tmp_path, caplog):
    # the example after a skipped show binds its own plotter instead of the closed one
    code = (
        '>>> import pyvista as pv\n'
        '>>> pl = pv.Plotter()\n'
        '>>> pl.show()  # doctest: +SKIP\n'
        '\n'
        'Prose between the two examples.\n'
        '\n'
        '>>> pl = pv.Plotter()\n'
        '>>> pl.enable_terrain_style()\n'
        '>>> pl.show()  # doctest: +SKIP\n'
    )
    _render(code, tmp_path)
    assert not [r for r in caplog.records if 'doctest: +SKIP' in r.message]


def test_render_figures_warns_when_the_filtered_remainder_raises(tmp_path, caplog):
    # a failure among the statements alongside a skip warns instead of raising
    code = ">>> raise RuntimeError('kaboom')\n>>> boom()  # doctest: +SKIP\n"
    results = _render(code, tmp_path)
    assert len(results) == 1
    warnings = [record for record in caplog.records if record.levelname == 'WARNING']
    assert any('doctest: +SKIP' in r.message and 'kaboom' in r.message for r in warnings)


def test_render_figures_still_raises_for_a_piece_without_skips(tmp_path):
    with pytest.raises(plot_directive.PlotError, match='kaboom'):
        _render(">>> raise RuntimeError('kaboom')\n", tmp_path)


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
