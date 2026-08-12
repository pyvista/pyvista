"""Fixtures for :mod:`pyvista.ext._autolink`, built as their own tiny Sphinx site.

Deliberately self-contained -- nothing here depends on pyvista's own API
being documented anywhere in this tiny test site -- so these test the
resolver itself, not whether the site happens to document enough of
pyvista to give it something real to find. Each function's docstring says
what it targets.

Built under ``tests/plotting/tinypages_autolink/`` rather than alongside
``tests/plotting/tinypages/samples.py`` -- see that site's ``conf.py`` for why.

Every Examples section below imports pyvista even though nothing here uses
it, purely because that's what triggers ``conf.py``'s ``_str_examples``
override to route the block through ``.. pyvista-plot::`` -- the same
requirement ``tinypages/samples.py`` already documents on ``make_sphere``.
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

    A static type checker sees ``Widget | str`` and has to guess -- or, as
    sphinx-codeautolink did on Python 3.14, silently resolve to
    ``typing.Union`` itself. This only has to ask the real, already-executed
    object what it is.

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
    undocumented module -- the exact re-export mismatch that broke every
    method past the first call under static analysis.

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

    ``widget`` is defined in one doctest group and used from a separate one
    below it, a common numpydoc style (several short examples separated by
    prose). The static extension needed ``codeautolink_concat_default``
    turned on for this; here it's just how the plot directive already runs
    a whole Examples section as one continuous script.

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

    ``functools.partial`` instances pass ``inspect.isroutine`` (they're
    method descriptors) but have no ``__qualname__``, which used to crash
    the resolver outright instead of just yielding no link for them.

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

    ``sub.make`` is flattened onto ``autolink_samples`` for documentation
    purposes but actually lives one package deeper -- the same re-export
    mismatch classes have, applied to a module (e.g.
    ``pyvista.examples.load_hexbeam`` reached via ``from pyvista import
    examples``).

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autolink_samples import sub
    >>> obj = sub.make()
    >>> obj.meth()

    """
    return
