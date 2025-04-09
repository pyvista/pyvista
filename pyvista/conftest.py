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
def set_default_theme():
    """Reset the theme for every test."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    pyvista.global_theme.load_theme('document')
    pyvista.global_theme.resample_environment_texture = True  # Speed up CI
    yield
    pyvista.global_theme.load_theme('document')
