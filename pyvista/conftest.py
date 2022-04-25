"""Fixtures for doctest."""
import pytest

import pyvista


@pytest.fixture(autouse=True)
def close_all_plotters():
    """Close all plotters after doctests."""
    yield
    pyvista.close_all()
