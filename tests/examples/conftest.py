"""Fixtures shared by the examples tests."""

from __future__ import annotations

import copy

import pytest

from pyvista.examples import downloads


def _shadows_own_method(obj, name, value):
    """Return whether ``value`` is just the class's own method bound to ``obj``."""
    return getattr(value, '__self__', None) is obj and value.__func__ is getattr(
        type(obj), name, None
    )


def _download_state():
    """Return the module-level download settings a test must leave as it found them."""
    return {
        'FETCHER': downloads.FETCHER,
        **{
            f'FETCHER.{name}': copy.copy(value)
            for name, value in vars(downloads.FETCHER).items()
            if name != 'registry'  # downloads add to it
            and not _shadows_own_method(downloads.FETCHER, name, value)
        },
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
    leaked = [
        f'{key}: {before.get(key)!r} -> {after.get(key)!r}'
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    ]
    assert not leaked, 'Test leaked changes to `pyvista.examples.downloads`:\n' + '\n'.join(leaked)
