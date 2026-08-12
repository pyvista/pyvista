"""Real implementations, documented under a different module (see ``__init__.py``)."""

from __future__ import annotations


class Base:
    """Base class, not documented anywhere on its own -- only ``Derived`` is."""

    def meth(self) -> None:
        """A method inherited by ``Derived``, never overridden there."""


class Derived(Base):
    """Documented as ``autolink_samples.Derived``, but actually defined here."""
