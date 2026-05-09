"""Tests for Marimo notebook support."""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from pyvista.jupyter import _is_marimo
from pyvista.jupyter import _resolve_backend

has_ipython = bool(importlib.util.find_spec('IPython'))
skip_no_ipython = pytest.mark.skipif(not has_ipython, reason='Requires IPython package')


def test_is_marimo_returns_false_outside_marimo():
    assert _is_marimo() is False


def test_is_marimo_returns_false_when_marimo_not_installed():
    with patch.dict('sys.modules', {'marimo': None}):
        assert _is_marimo() is False


def test_is_marimo_returns_true_when_running_in_notebook():
    mock_marimo = MagicMock()
    mock_marimo.running_in_notebook.return_value = True
    with patch.dict('sys.modules', {'marimo': mock_marimo}):
        assert _is_marimo() is True


def test_is_marimo_returns_false_when_not_in_notebook():
    mock_marimo = MagicMock()
    mock_marimo.running_in_notebook.return_value = False
    with patch.dict('sys.modules', {'marimo': mock_marimo}):
        assert _is_marimo() is False


@skip_no_ipython
def test_resolve_backend_returns_html_in_marimo():
    mock_marimo = MagicMock()
    mock_marimo.running_in_notebook.return_value = True
    with patch.dict('sys.modules', {'marimo': mock_marimo}):
        backend = _resolve_backend()
    assert backend == 'html'


@skip_no_ipython
def test_resolve_backend_returns_trame_outside_marimo():
    assert _resolve_backend() == 'trame'


@skip_no_ipython
def test_plain_html_widget_repr_html():
    from pyvista.trame.jupyter import PlainHtmlWidget

    plotter = MagicMock()
    scene = MagicMock()
    scene.getvalue.return_value = '<html>test</html>'
    plotter.export_html.return_value = scene

    widget = PlainHtmlWidget(plotter, width='100%', height='600px')
    html = widget._repr_html_()
    assert '<iframe' in html
    assert 'pyvista' in html


@skip_no_ipython
def test_show_trame_html_mode_uses_plain_widget_without_ipywidgets():
    from unittest.mock import patch as _patch

    import pyvista.trame.jupyter as trame_mod
    from pyvista.trame.jupyter import PlainHtmlWidget

    plotter = MagicMock()
    plotter.render_window = MagicMock()
    plotter._window_size_unset = True

    scene = MagicMock()
    scene.getvalue.return_value = '<html/>'
    plotter.export_html.return_value = scene

    with _patch.object(trame_mod, 'HTML', object):
        result = trame_mod.show_trame(plotter, mode='html')

    assert isinstance(result, PlainHtmlWidget)
