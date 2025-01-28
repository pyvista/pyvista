"""Conftest for jupyter tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def skip_check_gc(skip_check_gc):
    """A large number of tests here fail gc."""
