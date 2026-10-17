"""Fixtures for ``pyvista_plot_autocodelink``.

Self-contained: nothing here depends on pyvista's own API being documented.
Each Examples section imports pyvista solely to trigger ``conf.py``'s
``_str_examples`` override into ``.. pyvista-plot::``.
"""

from __future__ import annotations

from autocodelink_samples._internal import Derived  # re-exported, mirrors pyvista's pattern

__all__ = ['Derived']


class Widget:
    """A simple widget."""

    def draw(self) -> None:
        """Draw the widget."""


def make_widget() -> Widget:
    """Return a ``Widget``."""
    return Widget()


def call_chain_example():
    """Test a call with no intermediate variable resolving its trailing attribute.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autocodelink_samples import make_widget
    >>> make_widget().draw()

    """
    return


def make_widget_or_string(as_string=False):
    """Return either a ``Widget`` or a ``str``.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autocodelink_samples import make_widget_or_string
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
    >>> from autocodelink_samples import make_derived
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
    >>> from autocodelink_samples import make_widget_or_string
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
    >>> from autocodelink_samples import make_widget_or_string
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
    >>> from autocodelink_samples import sub
    >>> obj = sub.make()
    >>> obj.meth()

    """
    return


def never_referenced():
    """Nothing in this fixture site calls this -- for the empty-backrefs case."""
    return


def hoist_target():
    """Referenced by :func:`call_hoist_target`; alone on its own page for hoisting.

    Has its own Examples section, so its injected "Used In" section lands
    right after numpydoc's own Examples section in the docstring -- the
    exact position hoist_docstring_sections must still find it in. Also has
    its own See Also section, testing that it renders as a real section (not
    a ``.. seealso::`` admonition) hoisted to page level after Examples, not
    in numpydoc's own default position before it.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> pv.Sphere()  # doctest: +SKIP

    See Also
    --------
    call_hoist_target
        Calls this function.

    """
    return


def call_hoist_target():
    """Call ``hoist_target``, giving it a non-empty backreferences section.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autocodelink_samples import hoist_target
    >>> hoist_target()

    """
    return


def raw_seealso_target():
    """Referenced by :func:`call_raw_seealso_target`.

    Mirrors ``pyvista.examples.downloads``: a literal ``.. seealso::``
    admonition written directly in the Examples text, rather than numpydoc's
    own "See Also" section syntax. Tests that it too gets promoted to a real,
    hoisted section, landing after Examples and before Used In.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> pv.Sphere()  # doctest: +SKIP

    .. seealso::

        :func:`hoist_target`
            A different way to link to another function.

    """
    return


def call_raw_seealso_target():
    """Call ``raw_seealso_target``, giving it a non-empty backreferences section.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from autocodelink_samples import raw_seealso_target
    >>> raw_seealso_target()

    """
    return
