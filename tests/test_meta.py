"""Meta-tests for the test suite.

Module for tests that test the test setup itself, and in general
anything that's beyond testing actual library code.

"""

from __future__ import annotations


def test_mpl_backend():
    """Check if the backend is correctly set for testing."""
    import matplotlib as mpl

    assert mpl.get_backend() == 'agg'
