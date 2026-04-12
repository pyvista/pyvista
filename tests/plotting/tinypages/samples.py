from __future__ import annotations

import pyvista as pv


def make_sphere():
    """Make a sphere.

    Test that the pyvista-plot directive works correctly when there is
    something to plot.

    Examples
    --------
    >>> import pyvista as pv  # must import pyvista for the plotting directive to work
    >>> from samples import make_sphere
    >>> sphere = make_sphere()
    >>> sphere.plot()

    """
    return pv.Sphere()


def make_sphere_second():
    """Make a sphere.

    Test that the pyvista-plot directive works correctly when it is part of the doctest.

    Examples
    --------
    .. pyvista-plot::
      :include-source: False

      >>> import pyvista as pv  # must import pyvista for the plotting directive to work
      >>> from samples import make_sphere
      >>> sphere = make_sphere()
      >>> sphere.plot()

    """
    return pv.Sphere()


def example_with_empty_plotter():
    """Do not do anything.

    Test that the pyvista-plot directive works correctly when there is
    nothing to plot and the plotter has been created but not shown.

    Examples
    --------
    >>> import pyvista as pv
    >>> pl = pv.Plotter()

    """
    return


def example_with_closed_plotter():
    """Do not do anything.

    Test that the pyvista-plot directive works correctly when there is
    nothing to plot and the plotter has been created but not shown.

    Examples
    --------
    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> pl.close()

    """
    return
