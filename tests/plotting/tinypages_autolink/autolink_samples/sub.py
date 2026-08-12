"""A genuinely public submodule, accessed as an attribute of its package.

Mirrors ``pyvista.examples``: a real, user-facing submodule whose functions
are what a reader actually wants linked, reached via ``import package.sub``
or ``from package import sub`` rather than a direct import of the function.
"""

from __future__ import annotations

from autolink_samples._internal import Derived


def make() -> Derived:
    """Return a ``Derived``, reachable only via this submodule."""
    return Derived()
