"""Meta-tests for the test suite.

Module for tests that test the test setup itself, and in general
anything that's beyond testing actual library code.

"""


def test_mpl_backend():
    """Check if the backend is correctly set for testing."""
    import matplotlib

    assert matplotlib.get_backend() == 'agg'
