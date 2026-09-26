"""Convenience helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.helpers import is_pyvista_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet

    from ._typing import ColorLike

_ViewOptions = Literal['xy', 'yx', 'xz', 'zx', 'yz', 'zy']

# The direction each plane is viewed from, paired with its up vector.
_VIEW_VECTORS: dict[_ViewOptions, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    'xy': ((0, 0, 1), (0, 1, 0)),
    'yx': ((0, 0, -1), (1, 0, 0)),
    'xz': ((0, -1, 0), (0, 0, 1)),
    'zx': ((0, 1, 0), (1, 0, 0)),
    'yz': ((1, 0, 0), (0, 0, 1)),
    'zy': ((-1, 0, 0), (0, 1, 0)),
}


def plot_arrows(
    cent: VectorLike[float] | MatrixLike[float],
    direction: VectorLike[float] | MatrixLike[float],
    **kwargs: Any,
) -> Any:
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
    pyvista.plot_compare
    pyvista.Plotter

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
    return pv.plot([np.asarray(cent), np.asarray(direction)], **kwargs)


def plot_compare_four(  # noqa: PLR0917  # pragma: no cover
    data_a: DataSet,
    data_b: DataSet,
    data_c: DataSet,
    data_d: DataSet,
    *,
    display_kwargs: dict[str, Any] | None = None,
    plotter_kwargs: dict[str, Any] | None = None,
    show_kwargs: dict[str, Any] | None = None,
    screenshot: str | bool = False,
    camera_position: Any = None,
    outline: DataSet | None = None,
    outline_color: ColorLike = 'k',
    labels: Sequence[str] = ('A', 'B', 'C', 'D'),
    link: bool = True,
    notebook: bool | None = None,
) -> Any:
    """Plot a 2 by 2 comparison of data objects.

    .. deprecated:: 0.49
        Use :func:`~pyvista.plot_compare` instead, which supports any number of
        data objects::

            plot_compare([data_a, data_b, data_c, data_d])

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

    screenshot : str | bool, default: False
        File name or path to save screenshot of the plot, or ``True`` to return
        a screenshot array.

    camera_position : list, default: None
        The camera position to use in the plot.

    outline : pyvista.DataSet, default: None
        An outline to plot around the data objects.

    outline_color : str, default: 'k'
        The color of the outline.

    labels : tuple[str, str, str, str], default: ('A', 'B', 'C', 'D')
        The labels to display for each data object.

    link : bool, default: True
        If ``True``, link the views of the subplots.

    notebook : bool, default: None
        If ``True``, display the plot in a Jupyter notebook.

    Returns
    -------
    tuple
        See the returns of :func:`pyvista.Plotter.show`.

    See Also
    --------
    pyvista.plot_compare
    pyvista.plot
    pyvista.Plotter

    """
    # Deprecated on 0.49.0, estimated removal on 0.52.0
    warn_external(
        '`plot_compare_four` is deprecated. Use `plot_compare` instead, '
        'which supports any number of data objects.',
        PyVistaDeprecationWarning,
    )
    if pv.version_info >= (0, 52):  # pragma: no cover
        msg = 'Remove this deprecated function.'
        raise RuntimeError(msg)

    datasets = [[data_a, data_b], [data_c, data_d]]
    corner_labels = [labels[0:2], labels[2:4]]

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
            pl.add_text(corner_labels[i][j])
            if is_pyvista_dataset(outline):
                pl.add_mesh(outline, color=outline_color)
            if camera_position is not None:
                pl.camera_position = camera_position

    if link:
        pl.link_views()
        # when linked, camera must be reset such that the view range
        # of all subrender windows matches
        if camera_position is None:
            pl.reset_camera()

    return pl.show(screenshot=screenshot, **show_kwargs)


def view_vectors(
    view: _ViewOptions, *, negative: bool = False
) -> tuple[NDArray[np.signedinteger], NDArray[np.signedinteger]]:
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
        ``[x, y, z]`` vector that points to the ``viewup`` direction.

    """
    _validation.check_contains(list(get_args(_ViewOptions)), must_contain=view, name='view')
    direction, up = _VIEW_VECTORS[view]
    vec = np.array(direction)
    if negative:
        vec *= -1
    return vec, np.array(up)
