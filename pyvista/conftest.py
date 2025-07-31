"""Close all plotters to help control memory usage for our doctests."""

from __future__ import annotations

import pytest

import pyvista


@pytest.fixture(autouse=True)
def autoclose_plotters():
    """Close all plotters."""
    yield
    pyvista.close_all()


@pytest.fixture(autouse=True)
def reset_global_theme():
    """Reset global_theme."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    pyvista.set_plot_theme('document_build')
    yield
    pyvista.set_plot_theme('document_build')


@pytest.fixture(autouse=True)
def catch_vtk_errors(request):
    """Raise a RuntimeError when vtk errors are emitted."""
    if request.node.get_closest_marker('no_vtk_error_catcher'):
        yield  # skip the context manager
    else:
        with pyvista.VtkErrorCatcher(raise_errors=True):
            yield
