import pyvista


def not_called():
    """Never called if plot_directive works as expected."""
    raise NotImplementedError


def plot_poly():
    """The function that should be executed."""
    pyvista.Polygon().plot()
