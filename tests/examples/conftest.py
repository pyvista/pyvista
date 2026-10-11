"""Fixtures shared by the examples tests."""

from __future__ import annotations

import pytest

from pyvista.examples import downloads


def _download_state():
    """Return the module-level download settings a test must leave as it found them."""
    return {
        'FETCHER': downloads.FETCHER,
        'FETCHER.base_url': downloads.FETCHER.base_url,
        'FETCHER.path': downloads.FETCHER.path,
        'SOURCE': downloads.SOURCE,
        'USER_DATA_PATH': downloads.USER_DATA_PATH,
        '_FILE_CACHE': downloads._FILE_CACHE,
    }


@pytest.fixture(autouse=True)
def _guard_download_state():
    """Fail a test that leaks a change to the download source or cache settings."""
    before = _download_state()
    yield
    after = _download_state()
    leaked = {key: (before[key], after[key]) for key in before if before[key] != after[key]}
    assert not leaked, f'Test leaked changes to `pyvista.examples.downloads`: {leaked}'
