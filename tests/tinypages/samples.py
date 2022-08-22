import pyvista as pv


def make_sphere():
    """Make a sphere.

    Test that the pyvista-plot directive works correctly when there is
    something to plot.

    Examples
    --------
    >>> import pyvista  # must import pyvista for the plotting directive to work
    >>> from samples import make_sphere
    >>> sphere = make_sphere()
    >>> sphere.plot()

    """
    return pv.Sphere()


def do_nothing():
    """Do not do anything.

    Test that the pyvista-plot directive works correctly when there is
    nothing to plot and the plotter has been created but not shown.

    Examples
    --------
    >>> import pyvista
    >>> pl = pyvista.Plotter()

    """
    return
