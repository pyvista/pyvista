"""Convenience helper functions."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
import math
import string
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

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
    if labels is None:
        return None
    if isinstance(labels, str):
        if labels != 'auto':
            msg = (
                f"Labels must be a sequence of strings, 'auto', or None, got {labels!r} instead.\n"
                'A single string is not a valid sequence of labels.'
            )
            raise TypeError(msg)
        return _generate_labels(n_datasets) if names is None else names
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
    camera_position: CameraPositionOptions | None = None,
    reference_mesh: DataSet | MultiBlock | PartitionedDataSet | None = None,
    reference_kwargs: dict[str, Any] | None = None,
    labels: Sequence[str] | Literal['auto'] | None = 'auto',
    shape: Sequence[int] | str | None = None,
    link: bool = True,
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

    display_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``add_mesh`` method.

    plotter_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``Plotter`` constructor.

    show_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``show`` method.

    text_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``add_text`` method used to
        show the ``labels``, e.g. ``{'font_size': 24}``. Has no effect when
        ``labels`` is ``None``.

    screenshot : str | bool, default: None
        File name or path to save screenshot of the plot, or ``True`` to return
        a screenshot array.

    camera_position : list, default: None
        The camera position to use in the plot.

    reference_mesh : DataSet | MultiBlock, default: None
        A mesh to draw in every subplot to give the comparison a common frame of
        reference, e.g. an outline of the dataset the compared results are
        derived from. The same mesh is drawn in each subplot, so it does not
        follow the bounds of the individual datasets.

    reference_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``add_mesh`` method used to
        show the ``reference_mesh``. Defaults to ``{'color': 'k'}``.

    labels : Sequence[str] | 'auto' | None, default: 'auto'
        The labels to display for each data object. Must have the same length as
        ``datasets``. If ``'auto'``, the keys of ``datasets`` are used when it is
        a mapping or a :class:`~pyvista.MultiBlock`, and the labels ``'A'``,
        ``'B'``, ``'C'``, ... are generated otherwise. Set to ``None`` to disable
        labels. A string other than ``'auto'`` is not a valid sequence of labels
        and raises an error.

    shape : Sequence[int] | str, default: None
        The shape of the subplot layout, in any form accepted by
        :class:`~pyvista.Plotter`. Either a ``(n_rows, n_cols)`` sequence, or a
        string descriptor such as ``'3|1'`` for three subplots on the left and
        one on the right, or ``'4/2'`` for four on top and two on the bottom.
        Must define at least as many subplots as there are datasets. By default,
        or with ``'auto'``, the compact grid described above is used.

    link : bool, default: True
        If ``True``, link the views of the subplots.

    notebook : bool, default: None
        If ``True``, display the plot in a Jupyter notebook.

    Returns
    -------
    cpos : CameraPosition
        See the returns of :meth:`pyvista.Plotter.show`.

    See Also
    --------
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
    ...     camera_position='xy',
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
    if shape is None or (isinstance(shape, str) and shape == 'auto'):
        shape = _auto_shape(n_datasets)

    _validate_reference_mesh(reference_mesh)

    plotter_kwargs = {} if plotter_kwargs is None else dict(plotter_kwargs)
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
        if camera_position is not None:
            pl.camera_position = camera_position

    if link:
        pl.link_views()
        # When linked, camera must be reset such that the view range of all subplots match.
        # Do not reset when a fully-specific cpos is provided.
        if camera_position is None or isinstance(camera_position, str):
            pl.reset_camera()

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

    display_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``add_mesh`` method.

    plotter_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``Plotter`` constructor.

    show_kwargs : dict, default: None
        Additional keyword arguments to pass to the ``show`` method.

    screenshot : str | bool, default: None
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
    pyvista.Plotter
        The plotter object.

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
        camera_position=camera_position,
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
