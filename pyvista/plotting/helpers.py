"""Convenience helper functions."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
import math
import string
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.helpers import is_pyvista_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import PartitionedDataSet
    from pyvista.core._typing_core import NumpyArray
    from pyvista.plotting._typing import CameraPositionOptions
    from pyvista.plotting._typing import PlottableType


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
    return pv.plot([cent, direction], **kwargs)


# Sentinel for the default labels, since `None` means no labels at all
_AUTO_LABELS: Any = object()


def _generate_labels(n_labels: int) -> list[str]:
    """Generate labels ``'A'``, ``'B'``, ..., ``'Z'``, ``'AA'``, ``'AB'``, ... ."""
    labels = []
    for index in range(n_labels):
        label = ''
        value = index
        while value >= 0:
            label = string.ascii_uppercase[value % 26] + label
            value = value // 26 - 1
        labels.append(label)
    return labels


def _unpack_datasets(datasets: Any) -> tuple[list[Any], list[str] | None]:
    """Return the datasets to compare along with any labels defined by the input."""
    if isinstance(datasets, Mapping):
        return list(datasets.values()), [str(key) for key in datasets]
    if isinstance(datasets, pv.MultiBlock):
        return list(datasets), datasets.keys()
    if is_pyvista_dataset(datasets):
        msg = (
            f'Expected a sequence of datasets, got a single {type(datasets).__name__} instead.\n'
            'Use a sequence, e.g. `[dataset_a, dataset_b]`, to compare multiple datasets.'
        )
        raise TypeError(msg)
    if isinstance(datasets, str) or not isinstance(datasets, Iterable):
        msg = f'Expected a sequence of datasets, got {type(datasets).__name__} instead.'
        raise TypeError(msg)
    return list(datasets), None


def _validate_labels(labels: Any, *, names: list[str] | None, n_datasets: int) -> list[str] | None:
    """Return one label per dataset, or ``None`` if no labels should be shown."""
    if labels is _AUTO_LABELS:
        return _generate_labels(n_datasets) if names is None else names
    if labels is None:
        return None
    if isinstance(labels, str):
        msg = (
            f'Labels must be a sequence of strings or None, got {labels!r} instead.\n'
            'A single string is not a valid sequence of labels.'
        )
        raise TypeError(msg)
    labels = list(labels)
    if len(labels) != n_datasets:
        msg = f'Number of labels ({len(labels)}) must match the number of datasets ({n_datasets}).'
        raise ValueError(msg)
    return labels


def _validate_reference_mesh(reference_mesh: Any) -> None:
    """Raise if the reference mesh is not a dataset."""
    if reference_mesh is not None and not is_pyvista_dataset(reference_mesh):
        msg = f'Reference mesh must be a dataset, got {type(reference_mesh).__name__} instead.'
        raise TypeError(msg)


def _auto_shape(n_datasets: int) -> tuple[int, int]:
    """Return the ``(n_rows, n_cols)`` grid to use when no shape is given."""
    # Use as few rows as the square root allows so that the grid is never
    # taller than it is wide, e.g. (1, 3) for three datasets, (2, 2) for four
    n_rows = math.isqrt(n_datasets)
    return n_rows, math.ceil(n_datasets / n_rows)


def _union_bounds(renderers: Sequence[Any]) -> tuple[float, ...]:
    """Return the bounds enclosing all of the renderers."""
    bounds = np.array([renderer.bounds for renderer in renderers])
    return (
        bounds[:, 0].min(),
        bounds[:, 1].max(),
        bounds[:, 2].min(),
        bounds[:, 3].max(),
        bounds[:, 4].min(),
        bounds[:, 5].max(),
    )


# A dataset smaller than this fraction of all of them together is barely visible when
# the subplots share a camera fit to all of them
_MIN_RELATIVE_SIZE = 0.05

# Datasets which are each at least half the size of all of them together occupy the
# same space at a comparable scale, so a camera fit to all of them suits each one
_LINK_RELATIVE_SIZE = 0.5


def _bounds_length(bounds: Sequence[float]) -> float:
    """Return the diagonal length of the bounds, as :attr:`~pyvista.DataSet.length` does."""
    return float(
        np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    )


def _relative_size(renderers: Sequence[Any]) -> float:
    """Return the size of the smallest dataset relative to all of them together."""
    union = _bounds_length(_union_bounds(renderers))
    return min(renderer.length for renderer in renderers) / union if union > 0 else 1.0


def _warn_if_dataset_is_too_small(relative_size: float) -> None:
    """Warn when sharing a camera leaves one of the datasets too small to make out."""
    if relative_size < _MIN_RELATIVE_SIZE:
        msg = (
            f'The smallest dataset is {relative_size:.1%} of the size of all of the datasets '
            f'together, so it may be too small to make out when the subplots share a camera. '
            f'Use `link=False` to fit each subplot to its own dataset instead.'
        )
        warn_external(msg)


def _from_plotter_kwargs(plotter_kwargs: dict[str, Any], name: str, value: Any) -> Any:
    """Return the value of an argument which may instead be given in ``plotter_kwargs``.

    Both this function and the ``Plotter`` accept these arguments, so allow either
    one to define them, but not both, where the two could contradict each other.
    """
    if name not in plotter_kwargs:
        return value
    if value is not None:
        msg = (
            f'{name.capitalize()} was given both as the {name!r} argument and in '
            "'plotter_kwargs'. Use one or the other."
        )
        raise TypeError(msg)
    return plotter_kwargs.pop(name)


def _subplot_args(shape: tuple[int, ...], index: int) -> tuple[int, ...]:
    """Return the ``subplot`` arguments for the index within the layout."""
    # Layouts defined by a string descriptor are 1D and take a single index
    return (index,) if len(shape) == 1 else divmod(index, shape[1])


def plot_compare(
    datasets: Sequence[PlottableType] | Mapping[str, PlottableType] | MultiBlock,
    *,
    display_kwargs: dict[str, Any] | None = None,
    plotter_kwargs: dict[str, Any] | None = None,
    show_kwargs: dict[str, Any] | None = None,
    text_kwargs: dict[str, Any] | None = None,
    screenshot: str | bool | None = None,
    cpos: CameraPositionOptions | None = None,
    reference_mesh: DataSet | MultiBlock | PartitionedDataSet | None = None,
    reference_kwargs: dict[str, Any] | None = None,
    labels: Sequence[str] | None = _AUTO_LABELS,
    shape: Sequence[int] | str | None = None,
    link: bool | None = None,
    show_axes: bool | None = None,
    show_bounds: bool = False,
    zoom: float | str | None = None,
    notebook: bool | None = None,
):
    """Plot a grid comparison of any number of data objects.

    Each data object is shown in its own subplot. By default, the subplots are arranged
    in a compact grid which is never taller than it is wide, e.g. ``(1, 2)`` for two
    datasets, ``(1, 3)`` for three, ``(2, 2)`` for four, and ``(2, 3)`` for five or six.
    Any leftover subplots are left empty. Use ``shape`` to control the layout explicitly.

    .. versionadded:: 0.49

    Parameters
    ----------
    datasets : Sequence[DataSet] | Mapping[str, DataSet] | MultiBlock
        The data objects to compare. At least two datasets are required. If a
        mapping or a :class:`~pyvista.MultiBlock` is given, its keys are used as
        the default ``labels``.

    display_kwargs : dict, optional
        Additional keyword arguments to pass to the ``add_mesh`` method.

    plotter_kwargs : dict, optional
        Additional keyword arguments to pass to the ``Plotter`` constructor. A
        ``'shape'`` or ``'notebook'`` given here is used as the argument of the
        same name below, and it is an error to give either in both places.

    show_kwargs : dict, optional
        Additional keyword arguments to pass to the ``show`` method.

    text_kwargs : dict, optional
        Additional keyword arguments to pass to the ``add_text`` method used to
        show the ``labels``, e.g. ``{'font_size': 24}``. Has no effect when
        ``labels`` is ``None``.

    screenshot : str | bool, optional
        File name or path to save screenshot of the plot, or ``True`` to return
        a screenshot array.

    cpos : list, optional
        The camera position to use in the plot.

    reference_mesh : DataSet | MultiBlock, optional
        A mesh to draw in every subplot to give the comparison a common frame of
        reference, e.g. an outline of the dataset the compared results are
        derived from. The same mesh is drawn in each subplot, so it does not
        follow the bounds of the individual datasets.

    reference_kwargs : dict, optional
        Additional keyword arguments to pass to the ``add_mesh`` method used to
        show the ``reference_mesh``. Defaults to ``{'color': 'k'}``.

    labels : Sequence[str] | None, optional
        The labels to display for each data object. Must have the same length as
        ``datasets``. By default, the keys of ``datasets`` are used when it is a
        mapping or a :class:`~pyvista.MultiBlock`, and the labels ``'A'``,
        ``'B'``, ``'C'``, ... are generated otherwise. Set to ``None`` to disable
        labels. A single string is not a valid sequence of labels and raises an
        error.

    shape : Sequence[int] | str, optional
        The shape of the subplot layout, in any form accepted by
        :class:`~pyvista.Plotter`. Either a ``(n_rows, n_cols)`` sequence, or a
        string descriptor such as ``'3|1'`` for three subplots on the left and
        one on the right, or ``'4/2'`` for four on top and two on the bottom.
        Must define at least as many subplots as there are datasets. By default,
        the compact grid described above is used.

    link : bool, optional
        If ``True``, link the views of the subplots so that they share a single
        camera. The shared camera is fit to the bounds of every dataset, so the
        datasets are shown at a common scale and the framing does not depend on
        the order they are given in. If ``False``, each subplot keeps its own
        camera and is fit to its own dataset.

        By default, the views are linked when every dataset is at least half the
        size of all of them together, which means they occupy the same space at a
        comparable scale and one camera suits them all. Datasets which are much
        smaller than the rest, or which are far apart, are not linked, since a
        shared camera would leave some of them too small to make out.

        In every case the camera is only fit when ``cpos`` is ``None`` or a
        string, since a fully-specified camera position is used as given.

    show_axes : bool, optional
        Show the axes orientation widget in every subplot. By default, the
        :attr:`~pyvista.plotting.themes.Theme.axes` setting of the theme is
        used, as it is by :func:`pyvista.plot`.

    show_bounds : bool, default: False
        Show the bounds axes in every subplot.

    zoom : float | str, optional
        Camera zoom, applied after the camera is fit to the datasets. Either
        ``'tight'`` or a float, where a value greater than 1 is a zoom-in.

    notebook : bool, optional
        If ``True``, display the plot in a Jupyter notebook.

    Returns
    -------
    cpos : CameraPosition
        See the returns of :meth:`pyvista.Plotter.show`.

    See Also
    --------
    pyvista.plot
    pyvista.plot_arrows
    pyvista.Plotter
    pyvista.Plotter.subplot

    Examples
    --------
    Compare three filtered versions of a dataset.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.load_airplane()
    >>> pv.plot_compare(
    ...     [mesh.clip('x'), mesh.clip('y'), mesh.clip('z')],
    ...     display_kwargs={'color': 'w'},
    ... )

    Use a dictionary to label each dataset and set the camera position explicitly.

    >>> pv.plot_compare(
    ...     {
    ...         'clip x': mesh.clip('x'),
    ...         'clip y': mesh.clip('y'),
    ...         'clip z': mesh.clip('z'),
    ...     },
    ...     display_kwargs={'color': 'w'},
    ...     cpos='xy',
    ... )

    A :class:`~pyvista.MultiBlock` is compared block-by-block, and its block
    names are used as labels.

    >>> blocks = pv.MultiBlock(
    ...     {'sphere': pv.Sphere(), 'cube': pv.Cube(), 'cone': pv.Cone()}
    ... )
    >>> pv.plot_compare(blocks)

    Control the shape of the plot explicitly.

    >>> pv.plot_compare(blocks, shape=(3, 1))

    A shape with more subplots than datasets shows blank plots.

    >>> pv.plot_compare(blocks, shape=(2, 2))

    Use a string descriptor to plot one on top, two on the bottom.

    >>> pv.plot_compare(blocks, shape='2/1')

    """
    datasets, names = _unpack_datasets(datasets)

    n_datasets = len(datasets)
    if n_datasets < 2:
        msg = f'At least two datasets are required for comparison, got {n_datasets} instead.'
        raise ValueError(msg)

    labels = _validate_labels(labels, names=names, n_datasets=n_datasets)
    _validate_reference_mesh(reference_mesh)

    plotter_kwargs = {} if plotter_kwargs is None else dict(plotter_kwargs)
    shape = _from_plotter_kwargs(plotter_kwargs, 'shape', shape)
    notebook = _from_plotter_kwargs(plotter_kwargs, 'notebook', notebook)

    if shape is None:
        shape = _auto_shape(n_datasets)

    display_kwargs = {} if display_kwargs is None else display_kwargs
    show_kwargs = {} if show_kwargs is None else show_kwargs
    text_kwargs = {} if text_kwargs is None else text_kwargs
    reference_kwargs = {'color': 'k'} if reference_kwargs is None else reference_kwargs

    plotter_kwargs['notebook'] = notebook

    # The shape itself is validated by the plotter
    pl = pv.Plotter(shape=shape, **plotter_kwargs)

    n_subplots = len(pl.renderers)
    if n_subplots < n_datasets:
        pl.close()
        msg = (
            f'Shape {shape!r} defines {n_subplots} subplot(s) which is not enough '
            f'for {n_datasets} datasets.'
        )
        raise ValueError(msg)

    for index, dataset in enumerate(datasets):
        pl.subplot(*_subplot_args(pl.renderers.shape, index))
        pl.add_mesh(dataset, **display_kwargs)
        if labels is not None:
            pl.add_text(labels[index], **text_kwargs)
        if reference_mesh is not None:
            pl.add_mesh(reference_mesh, **reference_kwargs)
        if cpos is not None:
            pl.camera_position = cpos

    # Empty subplots are skipped throughout, since an empty renderer reports default
    # bounds rather than no bounds at all, and has nothing to decorate
    renderers = list(pl.renderers)[:n_datasets]
    relative_size = _relative_size(renderers)

    if link is None:
        link = relative_size >= _LINK_RELATIVE_SIZE

    if link:
        pl.link_views()
        _warn_if_dataset_is_too_small(relative_size)

    # Do not reset when a fully-specified cpos is provided.
    if cpos is None or isinstance(cpos, str):
        # Linked views share a single camera, so it is fit to the bounds of every
        # dataset at once. This shows the datasets at a common scale and keeps the
        # framing from depending on the order the datasets are given in. Each subplot
        # has its own camera otherwise, and is fit to its own dataset.
        if link:
            if cpos is None:
                # Apply the default view direction now. It is otherwise applied lazily,
                # which would both override the fit below and leave the camera pointing
                # down an axis instead. Setting it also marks the camera as set.
                pl.camera_position = pl.get_default_cam_pos()
            pl.renderer.reset_camera(bounds=_union_bounds(renderers))
        else:
            # Every renderer has its own camera, so reset each to fit each dataset
            # independently
            for renderer in renderers:
                renderer.reset_camera()

    if show_axes is None:
        show_axes = pl.theme.axes.show
    if show_axes:
        for renderer in renderers:
            # Match `pyvista.plot`, which draws box axes when the theme asks for them
            renderer.add_box_axes() if pl.theme.axes.box else renderer.add_axes()

    if show_bounds:
        for renderer in renderers:
            renderer.show_bounds()

    if zoom is not None:
        # Linked subplots share one camera, so zooming each would compound the zoom
        for renderer in renderers[:1] if link else renderers:
            renderer.camera.zoom(zoom)

    return pl.show(screenshot=screenshot, **show_kwargs)


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

    display_kwargs : dict, optional
        Additional keyword arguments to pass to the ``add_mesh`` method.

    plotter_kwargs : dict, optional
        Additional keyword arguments to pass to the ``Plotter`` constructor.

    show_kwargs : dict, optional
        Additional keyword arguments to pass to the ``show`` method.

    screenshot : str | bool, optional
        File name or path to save screenshot of the plot, or ``True`` to return
        a screenshot array.

    camera_position : list, optional
        The camera position to use in the plot.

    outline : pyvista.DataSet, optional
        An outline to plot around the data objects.

    outline_color : str, default: 'k'
        The color of the outline.

    labels : tuple[str, str, str, str], default: ('A', 'B', 'C', 'D')
        The labels to display for each data object.

    link : bool, default: True
        If ``True``, link the views of the subplots.

    notebook : bool, optional
        If ``True``, display the plot in a Jupyter notebook.

    Returns
    -------
    pyvista.Plotter
        The plotter object.

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

    return plot_compare(
        [data_a, data_b, data_c, data_d],
        display_kwargs=display_kwargs,
        plotter_kwargs=plotter_kwargs,
        show_kwargs=show_kwargs,
        screenshot=screenshot,
        cpos=camera_position,
        # Non-dataset outlines were silently ignored by this function, so keep
        # ignoring them here rather than raising as `plot_compare` now does
        reference_mesh=outline if is_pyvista_dataset(outline) else None,
        reference_kwargs={'color': outline_color},
        labels=labels,
        link=link,
        notebook=notebook,
    )


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
