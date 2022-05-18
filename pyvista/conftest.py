"""Close all plotters to help control memory usage for our doctests."""

import pytest

import pyvista


@pytest.fixture(autouse=True)
def autoclose_plotters():
    """Close all plotters."""
    yield
    pyvista.close_all()


@pytest.fixture(autouse=True)
def reset_gloal_theme():
    """Reset global_theme."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    yield
    pyvista.global_theme.restore_defaults()
