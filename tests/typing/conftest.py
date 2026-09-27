"""Configuration for the typing cases."""

from __future__ import annotations

import pytest
from type_assert.plugin import SetupItem
from type_assert.plugin import StaticItem

from tests.conftest import NUMPY_VERSION_INFO


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip the type checker runs of the cases for NumPy stubs without rank typing."""
    if isinstance(item, (SetupItem, StaticItem)) and NUMPY_VERSION_INFO < (2, 3, 0):
        pytest.skip('static typing needs NumPy >= 2.3')
