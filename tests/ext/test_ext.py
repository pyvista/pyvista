"""Test functions from plotting extension."""

from __future__ import annotations

import functools
import importlib
import os
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pyvista as pv
from pyvista.ext import _embed_py_file
from pyvista.ext import _plot_subprocess
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
    monkeypatch.setattr(viewer_directive, 'HTML_VIEWER_PATH', str(tmp_path / 'viewer.html'))
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
    monkeypatch.setattr(viewer_directive, 'HTML_VIEWER_PATH', str(tmp_path / 'viewer.html'))
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


def test_embed_py_file_warns_for_a_multi_file_dataset(monkeypatch, caplog):
    monkeypatch.setattr(_embed_py_file, 'download_file', lambda _name: ['one.py', 'two.py'])
    state_machine = SimpleNamespace(reporter=None)
    directive = _embed_py_file.EmbedPyFileDirective(
        'pyvista-embed-py-file', ['many'], {}, [], 1, 0, '', None, state_machine
    )

    with caplog.at_level('WARNING', logger=_embed_py_file.__name__):
        assert directive.run() == []

    assert 'many downloads to more than one file' in caplog.text


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
        self.config_rebuilds = {}
        self.setup_extension_calls = []

    def add_directive(self, name, directive):
        self.directives[name] = directive

    def connect(self, event, handler):
        self.connected.setdefault(event, []).append(handler)

    def add_config_value(self, name, default, rebuild):
        self.config_values[name] = default
        self.config_rebuilds[name] = rebuild

    def setup_extension(self, name):
        self.setup_extension_calls.append(name)


@pytest.mark.parametrize(
    ('config_value', 'options'),
    [(True, {}), (False, {'force_static': None})],
)
def test_run_uses_force_static_config(monkeypatch, tmp_path, config_value, options):
    captured = {}

    def fake_render_figures(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(plot_directive, 'render_figures', fake_render_figures)

    config = SimpleNamespace(
        pyvista_plot_force_static=config_value,
        pyvista_plot_use_counter=False,
        pyvista_plot_include_source=True,
        pyvista_plot_skip=False,
        pyvista_plot_skip_optional=False,
    )
    app = SimpleNamespace(
        builder=SimpleNamespace(outdir=tmp_path / 'html', srcdir=tmp_path / 'src'),
        confdir=str(tmp_path),
        doctreedir=tmp_path / 'doctrees',
    )
    env = SimpleNamespace(app=app, config=config)
    document = SimpleNamespace(
        settings=SimpleNamespace(env=env),
        attributes={'source': str(tmp_path / 'src' / 'index.rst')},
    )
    state_machine = SimpleNamespace(document=document)

    plot_directive.run([], [], dict(options), state_machine, SimpleNamespace(), 1)

    assert captured['force_static'] is True


def test_setup_depends_on_sphinx_autocodelink_when_available():
    app = _FakeSphinxApp()
    plot_directive.setup(app)
    assert app.config_values['pyvista_plot_force_static'] is False
    assert app.config_rebuilds['pyvista_plot_force_static'] == 'env'
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


def _job(code: str, tmp_path) -> dict:
    """Return the ``_execute_pieces`` keyword arguments for ``code``."""
    is_doctest, code_pieces = plot_directive._split_code_at_show(code)
    return {
        'code_pieces': code_pieces,
        'is_doctest': is_doctest,
        'code_setup': None,
        'code_cleanup': None,
        'code_path': '<test>',
        'output_dir': str(tmp_path),
        'output_base': 'out',
        'context': False,
        'function_name': None,
        'force_static': True,
    }


@pytest.fixture
def render_process():
    """Yield a render process and end it afterwards."""
    process = _plot_subprocess.RenderProcess()
    yield process
    process.close()


def test_render_process_runs_a_job(monkeypatch, tmp_path):
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    process = _plot_subprocess.RenderProcess()
    code = '>>> import pyvista as pv\n>>> pv.Sphere().plot()\n'
    results, warnings, records = process.run(_job(code, tmp_path), want_records=False)
    process.close()
    assert process._proc.returncode == 0
    assert records is None
    assert warnings == []
    assert len(results) == 1
    assert [(tmp_path / name).is_file() for name in results[0][1]] == [True]


def test_render_process_reports_a_failing_job(render_process, tmp_path):
    process = render_process
    with pytest.raises(RuntimeError, match='kaboom'):
        process.run(_job(">>> raise RuntimeError('kaboom')\n", tmp_path), want_records=False)


def test_render_process_reports_its_exit(render_process, tmp_path):
    process = render_process
    process._proc.kill()
    process._proc.wait()
    with pytest.raises(RuntimeError, match='exited with code'):
        process.run(_job('>>> 1\n', tmp_path), want_records=False)
    with pytest.raises(RuntimeError, match='exited with code'):
        process.run(_job('>>> 1\n', tmp_path), want_records=False)


def test_render_process_reports_an_exit_during_a_job(render_process, tmp_path):
    with pytest.raises(RuntimeError, match='exited with code 3'):
        render_process.run(_job('>>> __import__("os")._exit(3)\n', tmp_path), want_records=False)


def test_get_render_process_replaces_an_exited_process(monkeypatch):
    monkeypatch.setattr(_plot_subprocess, '_render_process', None)
    monkeypatch.setattr(_plot_subprocess.atexit, 'register', lambda _close: None)
    first = _plot_subprocess.get_render_process()
    first._proc.kill()
    first.close()
    second = _plot_subprocess.get_render_process()
    assert second is not first
    second.close()


def test_render_process_relays_warnings_and_records(monkeypatch, tmp_path):
    from sphinx_autocodelink import _records_for
    from sphinx_autocodelink import _to_jsonable

    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    code = (
        '>>> import pyvista as pv\n'
        '>>> mesh = pv.Sphere()\n'
        ">>> raise RuntimeError('kaboom')\n"
        '>>> boom()  # doctest: +SKIP\n'
    )
    job = _job(code, tmp_path)
    expected_results, expected_warnings, ns, source = plot_directive._execute_pieces(**job)
    expected_records = [_to_jsonable(record) for record in _records_for(source, ns)]
    process = _plot_subprocess.RenderProcess()
    results, warnings, records = process.run(job, want_records=True)
    process.close()
    assert results == expected_results
    # the render process may import pyvista from another path than this process
    strip_paths = functools.partial(re.sub, r'File ".*?"', 'File "..."')
    assert list(map(strip_paths, warnings)) == list(map(strip_paths, expected_warnings))
    assert any('kaboom' in warning for warning in warnings)
    assert records == expected_records
    assert any('Sphere' in str(record) for record in records)


def test_render_process_applies_the_theme_and_error_file(monkeypatch, tmp_path):
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    pv.set_plot_theme('document')
    pv.global_theme.font.size = 37
    pv.set_error_output_file(tmp_path / 'errors.txt')
    code = (
        '>>> import pyvista as pv\n'
        '>>> from pathlib import Path\n'
        f'>>> _ = Path({str(tmp_path / "theme.txt")!r}).write_text(\n'
        "...     f'{pv.global_theme.name} {pv.global_theme.font.size}'\n"
        '... )\n'
        '>>> from pyvista import _vtk\n'
        '>>> _vtk.vtkOutputWindow.GetInstance().DisplayErrorText("relayed error")\n'
    )
    process = _plot_subprocess.RenderProcess()
    process.run(_job(code, tmp_path), want_records=False)
    process.close()
    assert (tmp_path / 'theme.txt').read_text() == 'document 37'
    assert 'relayed error' in (tmp_path / 'errors.txt').read_text()


def test_get_render_process_starts_one_process(monkeypatch):
    monkeypatch.setattr(_plot_subprocess, '_render_process', None)
    alive = SimpleNamespace(_proc=SimpleNamespace(poll=lambda: None), close=lambda: None)
    monkeypatch.setattr(_plot_subprocess, 'RenderProcess', lambda: alive)
    monkeypatch.setattr(_plot_subprocess.atexit, 'register', lambda _close: None)
    assert _plot_subprocess.get_render_process() is _plot_subprocess.get_render_process()


def test_in_forked_worker(monkeypatch):
    assert not _plot_subprocess.in_forked_worker()
    monkeypatch.setattr(_plot_subprocess.multiprocessing, 'parent_process', object)
    assert _plot_subprocess.in_forked_worker()


@pytest.mark.parametrize('inside_autodoc', [True, False])
def test_store_records_uses_the_docstring_category_inside_autodoc(monkeypatch, inside_autodoc):
    import sphinx_autocodelink

    stored = MagicMock()
    monkeypatch.setattr(sphinx_autocodelink, '_store_records', stored)
    monkeypatch.setattr(sphinx_autocodelink, 'is_inside_autodoc_desc', lambda _state: True)
    env = SimpleNamespace(docname='doc')
    _plot_subprocess.store_records(env, [], state=object() if inside_autodoc else None)
    category = sphinx_autocodelink.DEFAULT_DOCSTRING_EXAMPLE_CATEGORY if inside_autodoc else ''
    stored.assert_called_once_with(env, 'doc', [], category=category)


def test_execute_pieces_runs_setup_and_cleanup(tmp_path):
    job = _job('>>> seen = marker\n', tmp_path)
    job['code_setup'] = 'marker = 1'
    job['code_cleanup'] = 'del marker'
    _results, _messages, ns, _source = plot_directive._execute_pieces(**job)
    assert ns['seen'] == 1
    assert 'marker' not in ns


def test_render_figures_keeps_a_nested_plot_directive_whole(tmp_path):
    code = '>>> x = 1\n>>> pl.show()\n\n.. pyvista-plot::\n\n   >>> y = 2\n'
    results = _render(code, tmp_path)
    assert [piece for piece, _images in results] == [code]


def test_render_figures_uses_the_render_process(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(plot_directive, 'in_forked_worker', lambda: True)
    process = SimpleNamespace(run=lambda _job, **_kwargs: ([('code', ['a.png'])], ['w'], None))
    monkeypatch.setattr(plot_directive, 'get_render_process', lambda: process)
    results = _render('>>> 1\n', tmp_path)
    assert [(code, [image.basename for image in images]) for code, images in results] == [
        ('code', ['a.png'])
    ]
    assert 'w' in caplog.text


def test_render_figures_stores_render_process_records(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_directive, 'in_forked_worker', lambda: True)
    process = SimpleNamespace(run=lambda _job, **_kwargs: ([], [], [{'a': 1}]))
    monkeypatch.setattr(plot_directive, 'get_render_process', lambda: process)
    stored = MagicMock()
    monkeypatch.setattr(plot_directive, 'store_records', stored)
    config = SimpleNamespace(
        pyvista_plot_setup=None, pyvista_plot_cleanup=None, pyvista_plot_autocodelink=True
    )
    env = SimpleNamespace(docname='doc')
    plot_directive.render_figures(
        code='>>> 1\n',
        code_path='<test>',
        output_dir=str(tmp_path),
        output_base='out',
        context=False,
        function_name=None,
        config=config,
        force_static=True,
        env=env,
    )
    stored.assert_called_once_with(env, [{'a': 1}], None)


def test_render_figures_raises_plot_error_from_the_render_process(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_directive, 'in_forked_worker', lambda: True)

    def run(_job, **_kwargs):
        msg = 'kaboom'
        raise RuntimeError(msg)

    monkeypatch.setattr(plot_directive, 'get_render_process', lambda: SimpleNamespace(run=run))
    with pytest.raises(plot_directive.PlotError, match='kaboom'):
        _render('>>> 1\n', tmp_path)


def test_embed_py_file_downloads_in_a_subprocess_in_a_forked_worker(monkeypatch):
    monkeypatch.setattr(_embed_py_file, 'in_forked_worker', lambda: True)
    run = MagicMock(return_value=SimpleNamespace(stdout='noise\n["one.py"]\n'))
    monkeypatch.setattr(_embed_py_file.subprocess, 'run', run)
    assert _embed_py_file._download('name') == ['one.py']
    assert run.call_args.args[0][-1] == 'name'
