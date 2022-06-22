"""Meta-tests for the test suite.

Module for tests that test the test setup itself, and in general
anything that's beyond testing actual library code.

"""


def test_mpl_backend():
    """Check if the backend is correctly set for testing."""
    # only fail if matplotlib is otherwise available
    try:
        import matplotlib
    except ImportError:
        return

    assert matplotlib.get_backend() == 'agg'
