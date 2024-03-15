"""Conftest for jupyter tests."""

import pytest


@pytest.fixture(autouse=True)
def skip_check_gc(skip_check_gc):  # noqa: PT004
    """A large number of tests here fail gc."""
