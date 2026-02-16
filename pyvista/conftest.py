"""Close all plotters to help control memory usage for our doctests."""

from __future__ import annotations

import os

import matplotlib as mpl
import pytest

import pyvista as pv


@pytest.fixture(autouse=True, scope='session')
def matplotlib_headless():
    """Use a non-interactive Matplotlib backend to avoid Tk issues on Windows CI."""
    if 'CI' in os.environ:
        mpl.use('Agg')


@pytest.fixture(autouse=True)
def autoclose_plotters():
    """Close all plotters."""
    yield
    pv.close_all()


@pytest.fixture(autouse=True)
def reset_global_theme():
    """Reset global_theme."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    pv.set_plot_theme('document_build')
    yield
    pv.set_plot_theme('document_build')


@pytest.fixture(autouse=True)
def catch_vtk_errors():
    """Raise a RuntimeError when vtk errors are emitted."""
    with pv.VtkErrorCatcher(raise_errors=True, emit_warnings=True):
        yield
