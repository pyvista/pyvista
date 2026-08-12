"""Where the real implementations live.

Deliberately a *different* module than where they're documented (see
``__init__.py``), the same as pyvista's own ``pyvista.core.dataset`` module
holding ``DataSet``, which is only ever documented as ``pyvista.DataSet``.
"""

from __future__ import annotations


class Base:
    """Base class, not documented anywhere on its own -- only ``Derived`` is."""

    def meth(self) -> None:
        """A method inherited by ``Derived``, never overridden there."""


class Derived(Base):
    """Documented as ``autolink_samples.Derived``, but actually defined here."""
