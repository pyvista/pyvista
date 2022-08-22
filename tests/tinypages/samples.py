import pyvista as pv


def make_sphere():
    """Make a sphere.

    Examples
    --------
    >>> import pyvista  # must import pyvista for the plotting directive to work
    >>> from samples import make_sphere
    >>> sphere = make_sphere()
    >>> sphere.plot()

    """
    return pv.Sphere()
