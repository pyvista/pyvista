"""Fixtures for :mod:`pyvista.ext._autolink`.

Self-contained: nothing here depends on pyvista's own API being documented.
Each Examples section imports pyvista solely to trigger ``conf.py``'s
``_str_examples`` override into ``.. pyvista-plot::``.
"""

from __future__ import annotations

from autolink_samples._internal import Derived  # re-exported, mirrors pyvista's pattern

__all__ = ['Derived']


class Widget:
    """A simple widget."""

    def draw(self) -> None:
        """Draw the widget."""


def make_widget_or_string(as_string=False):
    """Return either a ``Widget`` or a ``str``.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autolink_samples import make_widget_or_string
    >>> widget = make_widget_or_string()
    >>> widget.draw()

    """
    return 'nope' if as_string else Widget()


def make_derived():
    """Create a ``Derived`` and call its inherited method.

    ``Derived`` inherits ``meth`` from ``Base``, defined in a different,
    undocumented module.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autolink_samples import make_derived
    >>> obj = make_derived()
    >>> obj.meth()

    """
    return Derived()


def multi_block_examples():
    """Test that state carries across separate ``>>>`` groups on one page.

    Examples
    --------
    Create a widget.

    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autolink_samples import make_widget_or_string
    >>> widget = make_widget_or_string()

    Draw it.

    >>> widget.draw()

    """
    return


def make_partial_method():
    """Reference a ``functools.partial`` wrapping a bound method.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from functools import partial
    >>> from autolink_samples import make_widget_or_string
    >>> widget = make_widget_or_string()
    >>> method = partial(widget.draw)
    >>> method()

    """
    return


def module_attribute_chain():
    """Test a function reached as an attribute of an imported submodule.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autolink_samples import sub
    >>> obj = sub.make()
    >>> obj.meth()

    """
    return
