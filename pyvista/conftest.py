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


def pytest_collection_modifyitems(config, items):
    """Modify collected tests."""
    enabled = True  # "CI" in os.environ
    for item in items:
        item.add_marker(pytest.mark.fail_slow('5s', enabled=enabled))
