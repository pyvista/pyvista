"""Individual docstring-conversion edge cases for examples_download.py.

Each function's "Examples" section exercises exactly one RST construct that
the examples_download extension needs to handle, with trivial dummy code
(``import sys``) so the generated download is easy to eyeball.
"""

from __future__ import annotations


def case_dropdown():
    """Docstring with a dropdown in its Examples section.

    Examples
    --------
    >>> import sys

    .. dropdown:: Click me

        This hidden content should be removed completely, not even as
        a comment.
    """


def case_tabset():
    """Docstring with a tab-set in its Examples section.

    Examples
    --------
    >>> import sys

    .. tab-set::

        .. tab-item:: Static Scene

            Static content should be removed completely.

        .. tab-item:: Interactive Scene

            .. raw:: html

                <iframe src="viewer.html"></iframe>
    """


def case_note():
    """Docstring with a note in its Examples section.

    Examples
    --------
    >>> import sys

    .. note::

        This is a note with useful information.
    """


def case_warning():
    """Docstring with a warning in its Examples section.

    Examples
    --------
    >>> import sys

    .. warning::

        This is a warning about something dangerous.
    """


def case_multi_paragraph_note():
    """Docstring with a two-paragraph note.

    Examples
    --------
    >>> import sys

    .. note::

        First paragraph of the note.

        Second paragraph of the note.
    """


def case_generic_admonition():
    """Docstring with a generic, custom-titled admonition.

    Examples
    --------
    >>> import sys

    .. admonition:: Custom Title

        Body of the custom admonition.
    """


def case_versionadded():
    """Docstring with a versionadded directive.

    Examples
    --------
    >>> import sys

    .. versionadded:: 0.45

        Added this nice new feature.
    """


def case_versionchanged():
    """Docstring with a versionchanged directive.

    Examples
    --------
    >>> import sys

    .. versionchanged:: 0.47

        Behavior changed slightly.
    """


def case_deprecated():
    """Docstring with a deprecated directive.

    Examples
    --------
    >>> import sys

    .. deprecated:: 0.50

        Use something_else instead.
    """


def case_xref_plain():
    """Docstring with a plain class/meth/func/attr cross-reference.

    Examples
    --------
    See :class:`docstring_cases.Sample` and
    :meth:`docstring_cases.Sample.show` and
    :attr:`docstring_cases.Sample.value`.

    >>> import sys
    """


def case_xref_explicit_title():
    """Docstring with an explicit-title cross-reference.

    Examples
    --------
    See the :class:`Sample class <docstring_cases.Sample>` for details.

    >>> import sys
    """


def case_ref_plain():
    """Docstring with a plain :ref: link.

    Examples
    --------
    See :ref:`some-target` for background.

    >>> import sys
    """


def case_ref_explicit_title():
    """Docstring with an explicit-title :ref: link.

    Examples
    --------
    See the :ref:`background section <some-target>` for details.

    >>> import sys
    """


def case_inline_literal():
    """Docstring with inline double-backtick literals.

    Examples
    --------
    Set ``some_variable = True`` before calling this.

    >>> import sys
    """


def case_combined():
    """Docstring combining several constructs in one Examples section.

    Examples
    --------
    This example uses :class:`docstring_cases.Sample` and sets
    ``some_variable = True``.

    .. note::

        Remember to close the plotter when done.

    >>> import sys
    >>> x = 1
    3

    .. dropdown:: More details

        Extra detail that should be dropped entirely.
    """


def case_no_examples():
    """Docstring with no Examples section at all.

    Should never produce a download link -- this is the core on/off
    decision the whole extension hinges on.
    """


def case_prose_only():
    """Docstring with an Examples section that has no real code.

    Examples
    --------
    This just talks about the function without any code at all. Nothing
    here should ever be executed, so no download link should appear.
    """


def case_pyvista_plot_wrapped():
    """Docstring whose Examples section imports pyvista.

    This is the main compatibility case with ``plot_directive.py``: since
    this Examples section imports pyvista, the ``_str_examples``
    monkeypatch (see this fixture's ``conf.py``) auto-wraps it in a
    ``.. pyvista-plot::`` call, which itself wraps its generated source in
    a ``.. container:: pyvista-plot-source`` node. This case exists to
    confirm that container doesn't confuse the Examples-heading scan --
    the code inside it should still come through as real, executable code,
    and the figure it renders should be dropped, not turned into a comment.

    Examples
    --------
    >>> import pyvista as pv
    >>> pv.Sphere().plot()
    """


class Sample:
    """A sample class used as a cross-reference target above."""

    value = 1

    def show(self):
        """Show it."""
