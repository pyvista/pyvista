"""Compare any number of data objects side by side."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
import math
import string
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
import weakref

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.utilities.helpers import is_pyvista_dataset
from pyvista.plotting.text import _TEXT_POSITIONS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import PartitionedDataSet
    from pyvista.plotting._typing import CameraPositionOptions
    from pyvista.plotting._typing import PlottableType
    from pyvista.plotting.text import TextPositionOptions


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


def _normalized(dataset: Any) -> Any:
    """Return the dataset resized to a length of one and centered on the origin."""
    resize = getattr(dataset, 'resize', None) if is_pyvista_dataset(dataset) else None
    if resize is None:
        msg = (
            f'Cannot normalize {type(dataset).__name__}, which cannot be resized. '
            'Convert it to a dataset which can, or use `normalize=False`.'
        )
        raise TypeError(msg)
    # `resize` returns a new dataset, leaving the one which was given as it is
    return resize(length=1.0, center=(0.0, 0.0, 0.0))


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


def _warn_if_dataset_is_too_small(relative_size: float, of_what: str, remedy: str) -> None:
    """Warn when one of the datasets is too small to make out in its subplot."""
    if relative_size < _MIN_RELATIVE_SIZE:
        msg = (
            f'The smallest dataset is {relative_size:.1%} of the size of {of_what}, so it '
            f'may be too small to make out. {remedy}'
        )
        warn_external(msg)


def _from_kwargs(
    kwargs: dict[str, Any], key: str, value: Any, *, name: str, kwargs_name: str
) -> Any:
    """Return the value of an argument which may instead be given in a keyword dict.

    Both this function and the method the keywords are forwarded to accept these
    arguments, so allow either one to define them, but not both, where the two could
    contradict each other.
    """
    if key not in kwargs:
        return value
    if value is not None:
        msg = (
            f'{name.replace("_", " ").capitalize()} was given both as the {name!r} '
            f'argument and in {kwargs_name!r}. Use one or the other.'
        )
        raise TypeError(msg)
    return kwargs.pop(key)


def _subplot_args(shape: tuple[int, ...], index: int) -> tuple[int, ...]:
    """Return the ``subplot`` arguments for the index within the layout."""
    # Layouts defined by a string descriptor are 1D and take a single index
    return (index,) if len(shape) == 1 else divmod(index, shape[1])


# Draw each label as large as it fits in its own subplot, so labels of different
# lengths, or subplots of different sizes, are drawn at different sizes
_BEST_FIT = 'best_fit'

# Draw every label at the size of the one which has to be smallest to fit, so that they
# are all the same size no matter how long they are or which subplot they are in
_UNIFORM = 'uniform'

_LABEL_SIZE_MODES = (_BEST_FIT, _UNIFORM)

# Labels are measured at this size and scaled from it, since the width of a string is
# proportional to its font size. It is large enough that the rounding of the measured
# width does not skew the result.
_REFERENCE_FONT_SIZE = 100

# The fraction of the width of a subplot a label may occupy. The rest keeps the label
# clear of the edge of the subplot and of the label of the subplot beside it.
_LABEL_WIDTH_FRACTION = 0.9

# A label drawn any smaller than this is too small to read, so shorten the text
# instead. Roughly a third of the size the theme asks for.
_MIN_LABEL_SIZE = 14

# What the middle of a label too long to be drawn at a readable size is replaced with
_ELLIPSIS = '…'

# The name the label of a subplot is drawn under, which is how it is found again to be
# fitted to the subplot each time the window is rendered at a new size
_LABEL_NAME = 'plot_compare_label'

# `add_text` draws text at twice the font size it is given, so the font size of the
# theme is expressed in the same units as a fitted size by doubling it as well
_POINTS_PER_FONT_SIZE = 2


def _validate_label_size(label_size: Any) -> float | Literal['best_fit', 'uniform'] | None:
    """Return the label size as either a font size or the name of a sizing mode."""
    if label_size is None:
        return None
    modes = ', '.join(repr(mode) for mode in _LABEL_SIZE_MODES)
    if isinstance(label_size, str):
        if label_size in _LABEL_SIZE_MODES:
            return cast('Literal["best_fit", "uniform"]', label_size)
        msg = f'Label size {label_size!r} is not a font size, {modes} or None.'
        raise ValueError(msg)
    if isinstance(label_size, bool) or not isinstance(label_size, (int, float, np.number)):
        msg = (
            f'Label size must be a font size, {modes} or None, '
            f'got {type(label_size).__name__} instead.'
        )
        raise TypeError(msg)
    if label_size <= 0:
        msg = f'Label size must be greater than zero, got {label_size} instead.'
        raise ValueError(msg)
    return float(label_size)


def _validate_label_position(label_position: Any) -> TextPositionOptions | None:
    """Return the position to draw the labels in, which is one of the named places."""
    if label_position is None or label_position in _TEXT_POSITIONS:
        return label_position
    positions = ', '.join(repr(position) for position in _TEXT_POSITIONS)
    coordinate = '' if isinstance(label_position, str) else " Give a coordinate in 'label_kwargs'."
    msg = f'Label position must be one of {positions} or None, got {label_position!r} instead.{coordinate}'  # noqa: E501
    raise ValueError(msg)


def _text_width(text: str, prop: Any, *, size: float, dpi: int) -> float:
    """Return the width in pixels of the text drawn at the given font size."""
    # A `pyvista.TextProperty` loads the theme into a property shared by every one of
    # them, which measuring has no business doing, so measure with a plain VTK one
    measured = _vtk.vtkTextProperty()
    # Copy the property the text is actually drawn with, so that the font family and
    # style it defines are measured rather than the defaults
    measured.ShallowCopy(prop)
    measured.SetFontSize(int(size))
    bounds = [0, 0, 0, 0]
    # The text is measured rather than drawn, so no render window is needed for it.
    # The renderer is made here and dropped again rather than kept, since a text
    # renderer of its own is not something a plot has any business outliving.
    _vtk.vtkMathTextFreeTypeTextRenderer().GetBoundingBox(measured, text, bounds, dpi)
    return bounds[1] - bounds[0]


def _fitting_size(text: str, prop: Any, *, width: float, dpi: int) -> float:
    """Return the largest font size at which the text fits within the width."""
    measured = _text_width(text, prop, size=_REFERENCE_FONT_SIZE, dpi=dpi)
    # An empty label has no width to fit, so it never constrains the size
    return math.inf if measured <= 0 else _REFERENCE_FONT_SIZE * width / measured


def _ellipsize(text: str, n_kept: int) -> str:
    """Return the text with all but ``n_kept`` of its middle characters elided."""
    head = math.ceil(n_kept / 2)
    tail = n_kept // 2
    return text[:head] + _ELLIPSIS + (text[len(text) - tail :] if tail else '')


def _shorten(text: str, prop: Any, *, width: float, dpi: int, size: float) -> str:
    """Return the longest elision of the text which fits the width at the given size."""
    if _text_width(text, prop, size=size, dpi=dpi) <= width:
        # The label fits as it is, so there is nothing to elide
        return text
    # The elided text only grows as more of it is kept, so bisect for the most it can
    # keep rather than measuring every length
    low, high = 0, len(text) - 1
    while low < high:
        n_kept = (low + high + 1) // 2
        if _text_width(_ellipsize(text, n_kept), prop, size=size, dpi=dpi) <= width:
            low = n_kept
        else:
            high = n_kept - 1
    return _ellipsize(text, low)


def _fit_labels(
    actors: Sequence[pv.Text],
    labels: Sequence[str],
    renderers: Sequence[Any],
    *,
    uniform: bool,
    ceiling: float,
    dpi: int,
) -> None:
    """Draw every label at the largest size which fits in its subplot."""
    widths = [renderer.GetSize()[0] * _LABEL_WIDTH_FRACTION for renderer in renderers]
    sizes = [
        min(ceiling, _fitting_size(label, actor.prop, width=width, dpi=dpi))
        for label, actor, width in zip(labels, actors, widths, strict=True)
    ]
    if uniform:
        sizes = [min(sizes)] * len(sizes)

    for actor, label, width, size in zip(actors, labels, widths, sizes, strict=True):
        if size < _MIN_LABEL_SIZE:
            # The label is unreadable at the size it takes to fit, so draw it at the
            # smallest readable size and shorten the text until that fits instead
            size = _MIN_LABEL_SIZE  # noqa: PLW2901
            label = _shorten(label, actor.prop, width=width, dpi=dpi, size=size)  # noqa: PLW2901
        actor.prop.font_size = int(size)
        actor.input = label


def _fit_labels_on_render(
    plotter: pv.Plotter,
    labels: Sequence[str],
    *,
    name: str,
    uniform: bool | None,
) -> None:
    """Fit the labels to their subplots before every render which needs it again.

    The size which fits depends on the size of the subplots, which is only settled
    once the window is shown, and changes again whenever the window is resized.
    """
    fitted: list[Any] = []
    # The render window holds this callback for as long as it lives, so hold nothing
    # of the plotter it belongs to in return, and look up what is needed instead.
    # Anything held here would outlive the plotter it was drawn by.
    reference = weakref.ref(plotter)

    def fit(*_args: Any) -> None:
        plotter = reference()
        render_window = None if plotter is None else plotter.render_window
        if plotter is None or render_window is None:  # pragma: no cover
            return
        dpi = render_window.GetDPI()
        renderers = list(plotter.renderers)[: len(labels)]
        actors = [renderer.actors.get(name) for renderer in renderers]
        if not all(actors):
            # A label has been removed or drawn over since it was added, so there is
            # nothing left to fit rather than anything to complain about mid-render
            return
        sizes = [renderer.GetSize() for renderer in renderers]
        if fitted == [dpi, sizes]:
            # Nothing the fitted sizes depend on has changed since the last render
            return
        fitted[:] = [dpi, sizes]
        widths = [width for width, _ in sizes]
        _fit_labels(
            actors,
            labels,
            renderers,
            # Subplots of the same width share a size which suits all of the labels.
            # Sharing one between subplots of different widths would instead pin every
            # label to the size which fits in the narrowest of them. A grid divides the
            # window between its subplots, which leaves a pixel of it over now and
            # then, so widths within a pixel of each other count as the same width.
            uniform=max(widths) - min(widths) <= 1 if uniform is None else uniform,
            ceiling=plotter.theme.font.size * _POINTS_PER_FONT_SIZE,
            dpi=dpi,
        )

    # `StartEvent` is emitted before each render, when the subplots have already been
    # given the size they are about to be drawn at
    plotter.render_window.AddObserver(_vtk.vtkCommand.StartEvent, fit)  # type: ignore[union-attr]


def plot_compare(  # noqa: ANN201
    datasets: Sequence[PlottableType] | Mapping[str, PlottableType] | MultiBlock,
    *,
    display_kwargs: dict[str, Any] | None = None,
    plotter_kwargs: dict[str, Any] | None = None,
    show_kwargs: dict[str, Any] | None = None,
    label_kwargs: dict[str, Any] | None = None,
    screenshot: str | bool | None = None,
    cpos: CameraPositionOptions | None = None,
    reference_mesh: DataSet | MultiBlock | PartitionedDataSet | None = None,
    reference_kwargs: dict[str, Any] | None = None,
    labels: Sequence[str] | None = _AUTO_LABELS,
    label_size: float | Literal['best_fit', 'uniform'] | None = None,
    label_position: TextPositionOptions | None = None,
    shape: Sequence[int] | str | None = None,
    normalize: bool = False,
    link: bool | None = None,
    show_axes: bool | None = None,
    show_bounds: bool = False,
    zoom: float | str | None = None,
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
        Additional keyword arguments to pass to the
        :meth:`~pyvista.Plotter.add_mesh` method.

    plotter_kwargs : dict, optional
        Additional keyword arguments to pass to the :class:`~pyvista.Plotter`
        constructor. A ``'shape'`` given here is used as the ``shape`` argument
        below, but not in both places.

    show_kwargs : dict, optional
        Additional keyword arguments to pass to the :meth:`~pyvista.Plotter.show`
        method.

    label_kwargs : dict, optional
        Additional keyword arguments for the :class:`~pyvista.Text` actor which
        draws each of the ``labels``, e.g. ``{'color': 'red'}``. Takes what
        :meth:`~pyvista.Plotter.add_text` takes. Has no effect when ``labels``
        is ``None``.

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
        Additional keyword arguments to pass to the
        :meth:`~pyvista.Plotter.add_mesh` method used to show the
        ``reference_mesh``. Defaults to ``{'color': 'k'}``.

    labels : Sequence[str] | None, optional
        The labels to display for each data object. Must have the same length as
        ``datasets``. By default, the keys of ``datasets`` are used when it is a
        mapping or a :class:`~pyvista.MultiBlock`, and the labels ``'A'``,
        ``'B'``, ``'C'``, ... are generated otherwise. Set to ``None`` to disable
        labels. A single string is not a valid sequence of labels and raises an
        error.

        If the input has keys `and` ``labels`` are provided, the provided
        ``labels`` take precedence and are used instead of its keys.

    label_size : float | str, optional
        The size to draw the ``labels`` at, as either a font size or how to work
        one out. A font size is used as given, and may be too large for a label
        to fit in its subplot. The sizes which are worked out are:

        * ``'best_fit'``: draw each label as large as it fits in its own
          subplot, up to the font size of the theme. Labels of different lengths,
          and labels in subplots of different sizes, are drawn at different sizes.
        * ``'uniform'``: draw every label at the size of the one which has to be
          smallest to fit, so that they are all the same size no matter how long
          they are or which subplot they are in.

        By default, ``'uniform'`` is used when the subplots are all the same
        width, and ``'best_fit'`` otherwise, since one size shared between
        subplots of different widths is pinned to whatever fits the narrowest of
        them. A label too long to fit at a readable size has its middle elided.

        The size is worked out again whenever the window is resized. It may also
        be given as ``'font_size'`` in ``label_kwargs``, but not in both places.
        Has no effect when ``labels`` is ``None``.

        Labels are drawn by a :class:`~pyvista.Text` actor, which draws text at
        the size it is given.

        .. versionadded:: 0.49

    label_position : str, optional
        Where in its subplot to draw each of the ``labels``, as one of the places
        :meth:`~pyvista.Plotter.add_text` names: ``'upper_left'``,
        ``'upper_right'``, ``'lower_left'``, ``'lower_right'``, ``'upper_edge'``,
        ``'lower_edge'``, ``'left_edge'`` or ``'right_edge'``. Defaults to
        ``'upper_left'``.

        A coordinate may be given as ``'position'`` in ``label_kwargs`` instead,
        along with ``'viewport'`` to read it as a fraction of the size of the
        subplot rather than as pixels, but not in both places. Has no effect when
        ``labels`` is ``None``.

        .. versionadded:: 0.49

    shape : Sequence[int] | str, optional
        The shape of the subplot layout, in any form accepted by
        :class:`~pyvista.Plotter`. Either a ``(n_rows, n_cols)`` sequence, or a
        string descriptor such as ``'3|1'`` for three subplots on the left and
        one on the right, or ``'4/2'`` for four on top and two on the bottom.
        Must define at least as many subplots as there are datasets. By default,
        the compact grid described above is used.

    normalize : bool, default: False
        Resize every dataset to a diagonal :attr:`~pyvista.DataSet.length` of one,
        centered on the origin, so that datasets of very different sizes are
        compared shape by shape. The datasets given are left as they are, and the
        resized copies of them are what is drawn. A ``reference_mesh`` is resized
        along with them, so that it is the same frame of reference for each.

        Normalized datasets are all the same size and in the same place, so they
        are linked by default, which datasets of very different sizes are not.

        .. versionadded:: 0.49

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

    Datasets of very different sizes are compared shape by shape by normalizing
    them. The airplane is some forty times the size of the ant, which is a speck
    beside it otherwise.

    >>> pv.plot_compare(
    ...     {
    ...         'airplane': examples.load_airplane(),
    ...         'ant': examples.load_ant(),
    ...     },
    ...     normalize=True,
    ... )

    Anything the :class:`~pyvista.Plotter` itself takes is given to it through
    ``plotter_kwargs``. Draw a border around each subplot to tell them apart.

    >>> pv.plot_compare(
    ...     blocks,
    ...     plotter_kwargs={'border': True, 'border_color': 'grey'},
    ... )

    Plot on a dark background by giving the plotter a theme of its own, which
    also decides the color the labels are drawn in.

    >>> pv.plot_compare(blocks, plotter_kwargs={'theme': pv.themes.DarkTheme()})

    """
    datasets, names = _unpack_datasets(datasets)

    n_datasets = len(datasets)
    if n_datasets < 2:
        msg = f'At least two datasets are required for comparison, got {n_datasets} instead.'
        raise ValueError(msg)

    labels = _validate_labels(labels, names=names, n_datasets=n_datasets)
    _validate_reference_mesh(reference_mesh)

    if normalize:
        datasets = [_normalized(dataset) for dataset in datasets]
        if reference_mesh is not None:
            reference_mesh = _normalized(reference_mesh)

    plotter_kwargs = {} if plotter_kwargs is None else dict(plotter_kwargs)
    shape = _from_kwargs(
        plotter_kwargs, 'shape', shape, name='shape', kwargs_name='plotter_kwargs'
    )

    if shape is None:
        shape = _auto_shape(n_datasets)

    display_kwargs = {} if display_kwargs is None else display_kwargs
    show_kwargs = {} if show_kwargs is None else show_kwargs
    label_kwargs = {} if label_kwargs is None else dict(label_kwargs)
    reference_kwargs = {'color': 'k'} if reference_kwargs is None else reference_kwargs

    label_size = _validate_label_size(
        _from_kwargs(
            label_kwargs, 'font_size', label_size, name='label_size', kwargs_name='label_kwargs'
        )
    )
    # A coordinate is only ever given in the keywords, so validate the argument on its
    # own before the two are reconciled
    label_position = _from_kwargs(
        label_kwargs,
        'position',
        _validate_label_position(label_position),
        name='label_position',
        kwargs_name='label_kwargs',
    )
    if label_position is not None:
        label_kwargs['position'] = label_position

    # A font size is drawn as given, and only the sizes which are worked out are fitted
    fitted = not isinstance(label_size, float)
    if not fitted:
        label_kwargs['font_size'] = label_size
    elif label_kwargs.get('name') is None:
        # Name the labels so they can be found again to be fitted on every render
        label_kwargs['name'] = _LABEL_NAME

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
            pl._add_text_actor(labels[index], **label_kwargs)
        if cpos is not None:
            pl.camera_position = cpos

    # Empty subplots are skipped throughout, since an empty renderer reports default
    # bounds rather than no bounds at all, and has nothing to decorate
    renderers = list(pl.renderers)[:n_datasets]

    # Measure the datasets before the reference mesh is added, since it goes in every
    # subplot and would otherwise make them all report the same bounds as each other
    smallest = min(renderer.length for renderer in renderers)
    linked_on_purpose = link is not None
    if link is None:
        link = _relative_size(renderers) >= _LINK_RELATIVE_SIZE

    if link:
        pl.link_views()
        # Linking on its own is only worth warning about when it was asked for, since
        # datasets which are linked automatically are of a comparable size already
        if linked_on_purpose:
            _warn_if_dataset_is_too_small(
                _relative_size(renderers),
                'all of the datasets together, which the shared camera has to fit',
                'Use `link=False` to fit each subplot to its own dataset, or '
                '`normalize=True` to resize them all to the same size.',
            )

    if reference_mesh is not None:
        _warn_if_dataset_is_too_small(
            smallest / reference_mesh.length if reference_mesh.length > 0 else 1.0,
            'the reference mesh, which every subplot has to fit as well as its own dataset',
            'Use a smaller `reference_mesh`.',
        )
        for index in range(n_datasets):
            pl.subplot(*_subplot_args(pl.renderers.shape, index))
            pl.add_mesh(reference_mesh, **reference_kwargs)

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

    if fitted and labels is not None:
        _fit_labels_on_render(
            pl,
            labels,
            name=label_kwargs['name'],
            uniform=None if label_size is None else label_size == _UNIFORM,
        )

    return pl.show(screenshot=screenshot, **show_kwargs)
