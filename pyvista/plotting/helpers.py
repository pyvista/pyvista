"""This module contains some convenience helper functions."""

from typing import Tuple

import numpy as np

import pyvista
from pyvista.core.utilities.helpers import is_pyvista_dataset


def plot_arrows(cent, direction, **kwargs):
    """Plot arrows as vectors.

    Parameters
    ----------
    cent : array_like[float]
        Accepts a single 3d point or array of 3d points.

    direction : array_like[float]
        Accepts a single 3d point or array of 3d vectors.
        Must contain the same number of items as ``cent``.

    **kwargs : dict, optional
        See :func:`pyvista.plot`.

    Returns
    -------
    tuple
        See the returns of :func:`pyvista.plot`.

    See Also
    --------
    pyvista.plot

    Examples
    --------
    Plot a single random arrow.

    >>> import numpy as np
    >>> import pyvista
    >>> cent = np.random.random(3)
    >>> direction = np.random.random(3)
    >>> pyvista.plot_arrows(cent, direction)

    Plot 100 random arrows.

    >>> import numpy as np
    >>> import pyvista
    >>> cent = np.random.random((100, 3))
    >>> direction = np.random.random((100, 3))
    >>> pyvista.plot_arrows(cent, direction)

    """
    return pyvista.plot([cent, direction], **kwargs)


def plot_compare_four(
    data_a,
    data_b,
    data_c,
    data_d,
    display_kwargs=None,
    plotter_kwargs=None,
    show_kwargs=None,
    screenshot=None,
    camera_position=None,
    outline=None,
    outline_color='k',
    labels=('A', 'B', 'C', 'D'),
    link=True,
    notebook=None,
):
    """Plot a 2 by 2 comparison of data objects.

    Parameters
    ----------
    data_a : pyvista.DataSet
        The data object to display in the top-left corner.
    data_b : pyvista.DataSet
        The data object to display in the top-right corner.
    data_c : pyvista.DataSet
        The data object to display in the bottom-left corner.
    data_d : pyvista.DataSet
        The data object to display in the bottom-right corner.
    display_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``add_mesh`` method.
    plotter_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``Plotter`` constructor.
    show_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``show`` method.
    screenshot : str or bool, default: None
        File name or path to save screenshot of the plot, or ``True`` to return
        a screenshot array.
    camera_position : list, default: None
        The camera position to use in the plot.
    outline : pyvista.DataSet, default: None
        An outline to plot around the data objects.
    outline_color : str, default: 'k'
        The color of the outline.
    labels : tuple of str, default: ('A', 'B', 'C', 'D')
        The labels to display for each data object.
    link : bool, default: True
        If ``True``, link the views of the subplots.
    notebook : bool, default: None
        If ``True``, display the plot in a Jupyter notebook.

    Returns
    -------
    pyvista.Plotter
        The plotter object.
    """
    datasets = [[data_a, data_b], [data_c, data_d]]
    labels = [labels[0:2], labels[2:4]]

    if plotter_kwargs is None:
        plotter_kwargs = {}
    if display_kwargs is None:
        display_kwargs = {}
    if show_kwargs is None:
        show_kwargs = {}

    plotter_kwargs['notebook'] = notebook

    pl = pyvista.Plotter(shape=(2, 2), **plotter_kwargs)

    for i in range(2):
        for j in range(2):
            pl.subplot(i, j)
            pl.add_mesh(datasets[i][j], **display_kwargs)
            pl.add_text(labels[i][j])
            if is_pyvista_dataset(outline):
                pl.add_mesh(outline, color=outline_color)
            if camera_position is not None:
                pl.camera_position = camera_position

    if link:
        pl.link_views()
        # when linked, camera must be reset such that the view range
        # of all subrender windows matches
        pl.reset_camera()

    return pl.show(screenshot=screenshot, **show_kwargs)


def view_vectors(view: str, negative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Given a plane to view, return vectors for setting up camera.

    Parameters
    ----------
    view : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}
        Plane to return vectors for.

    negative : bool, default: False
        Whether to view from opposite direction.

    Returns
    -------
    vec : numpy.ndarray
        ``[x, y, z]`` vector that points in the viewing direction.

    viewup : numpy.ndarray
        ``[x, y, z]`` vector that points to the viewup direction.

    """
    if view == 'xy':
        vec = np.array([0, 0, 1])
        viewup = np.array([0, 1, 0])
    elif view == 'yx':
        vec = np.array([0, 0, -1])
        viewup = np.array([1, 0, 0])
    elif view == 'xz':
        vec = np.array([0, -1, 0])
        viewup = np.array([0, 0, 1])
    elif view == 'zx':
        vec = np.array([0, 1, 0])
        viewup = np.array([1, 0, 0])
    elif view == 'yz':
        vec = np.array([1, 0, 0])
        viewup = np.array([0, 0, 1])
    elif view == 'zy':
        vec = np.array([-1, 0, 0])
        viewup = np.array([0, 1, 0])
    else:
        raise ValueError(
            f"Unexpected value for direction {view}\n"
            "    Expected: 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'"
        )

    if negative:
        vec *= -1
    return vec, viewup
