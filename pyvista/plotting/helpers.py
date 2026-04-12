"""Convenience helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.helpers import is_pyvista_dataset

if TYPE_CHECKING:
    from pyvista.core._typing_core import NumpyArray


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
    >>> import pyvista as pv
    >>> rng = np.random.default_rng(seed=0)
    >>> cent = rng.random(3)
    >>> direction = rng.random(3)
    >>> pv.plot_arrows(cent, direction)

    Plot 100 random arrows.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> cent = rng.random((100, 3))
    >>> direction = rng.random((100, 3))
    >>> pv.plot_arrows(cent, direction)

    """
    return pv.plot([cent, direction], **kwargs)


@_deprecate_positional_args(allowed=['data_a', 'data_b', 'data_c', 'data_d'], n_allowed=4)
def plot_compare_four(  # noqa: PLR0917
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
    link: bool = True,  # noqa: FBT001, FBT002
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

    pl = pv.Plotter(shape=(2, 2), **plotter_kwargs)

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


@_deprecate_positional_args(allowed=['view'])
def view_vectors(view: str, negative: bool = False) -> tuple[NumpyArray[int], NumpyArray[int]]:  # noqa: FBT001, FBT002
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
        msg = (
            f'Unexpected value for direction {view}\n'
            "    Expected: 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'"
        )
        raise ValueError(msg)

    if negative:
        vec *= -1
    return vec, viewup
