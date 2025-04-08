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
    """Reset the testing theme for every test."""
    pyvista.global_theme.load_theme(pyvista.plotting.themes._TestingTheme())
    yield
    pyvista.global_theme.load_theme(pyvista.plotting.themes._TestingTheme())
