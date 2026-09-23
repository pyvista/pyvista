"""A public submodule, accessed as an attribute of its package."""

from __future__ import annotations

from autocodelink_samples._internal import Derived


def make() -> Derived:
    """Return a ``Derived``, reachable only via this submodule."""
    return Derived()
