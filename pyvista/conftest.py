"""Close all plotters to help control memory usage for our doctest."""

import pytest

import pyvista


@pytest.fixture(autouse=True)
def autoclose_plotters():  # pragma: no cover
    """Close all plotters."""
    yield
    pyvista.close_all()
