"""Close all plotters to help control memory usage for our doctests."""

from __future__ import annotations

import pytest

import pyvista


@pytest.fixture(autouse=True)
def autoclose_plotters():  # noqa: PT004
    """Close all plotters."""
    yield
    pyvista.close_all()


@pytest.fixture(autouse=True)
def reset_gloal_theme():  # noqa: PT004
    """Reset global_theme."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    yield
    pyvista.global_theme.restore_defaults()
