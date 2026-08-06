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


class _Sentinel:
    """A default which stands for something other than a value a caller can give.

    Prints as the name it is given, so that a signature shows what it stands for
    rather than the address of an object.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


# The default labels, which `None` cannot stand for since it means no labels at all
_AUTO_LABELS: Any = _Sentinel('auto')


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


def _union_of_bounds(bounds: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return the bounds enclosing every one of the given bounds."""
    stacked = np.array(bounds)
    return (
        stacked[:, 0].min(),
        stacked[:, 1].max(),
        stacked[:, 2].min(),
        stacked[:, 3].max(),
        stacked[:, 4].min(),
        stacked[:, 5].max(),
    )


def _union_bounds(renderers: Sequence[Any]) -> tuple[float, ...]:
    """Return the bounds enclosing all of the renderers."""
    return _union_of_bounds([renderer.bounds for renderer in renderers])


def _fix_clipping_range_on_render(
    plotter: pv.Plotter, bounds: tuple[float, ...], n_renderers: int
) -> None:
    """Reset the clipping range to the given bounds before every future render.

    Linked renderers share one camera but not one clipping range: an interactor style
    narrows the range before every render it drives, from `ComputeVisiblePropBounds`
    of whichever renderer the interaction is in, which is one subplot's worth of
    bounds where every subplot needs fitting. Undo that here, every time, from the
    same bounds the camera itself was fit to, rather than from an actor standing in
    for them, which would leave `renderer.bounds` itself, and anything that relies on
    it, reporting more than what is actually in each subplot.
    """
    # The render window holds this callback for as long as it lives, so hold nothing of
    # the plotter in return: anything held here would outlive the plot
    reference = weakref.ref(plotter)

    def fix(*_args: Any) -> None:
        plotter = reference()
        render_window = None if plotter is None else plotter.render_window
        if plotter is None or render_window is None:  # pragma: no cover
            return
        for renderer in list(plotter.renderers)[:n_renderers]:
            renderer.ResetCameraClippingRange(*bounds)

    # `StartEvent` is emitted before each render, ahead of whatever clipping range an
    # interactor style narrowed it to while handling the interaction that led here
    plotter.render_window.AddObserver(_vtk.vtkCommand.StartEvent, fix)  # type: ignore[union-attr]


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


def _relative_size(
    renderers: Sequence[Any], *, reference_bounds: Sequence[float] | None = None
) -> float:
    """Return the size of the smallest dataset relative to all of them together.

    A ``reference_mesh`` is drawn in every subplot alongside its own dataset, so what
    a subplot actually has to fit is the two together rather than the dataset alone.
    Give its bounds to size each subplot by that instead.
    """
    own_bounds = [renderer.bounds for renderer in renderers]
    if reference_bounds is not None:
        own_bounds = [_union_of_bounds([bounds, reference_bounds]) for bounds in own_bounds]
    union = _bounds_length(_union_of_bounds(own_bounds))
    return min(_bounds_length(bounds) for bounds in own_bounds) / union if union > 0 else 1.0


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


# Draw each label as large as it fits in its own subplot
_BEST_FIT = 'best_fit'

# Draw every label at the size of the one which has to be smallest to fit
_UNIFORM = 'uniform'

_LABEL_SIZE_MODES = (_BEST_FIT, _UNIFORM)

# Labels are measured at this size and scaled from it, since the width of a string is
# proportional to its font size. Large enough that rounding the width does not skew it.
_REFERENCE_FONT_SIZE = 100

# The fraction of the width of a subplot a label may occupy, the rest keeping it clear
# of the edge and of the label beside it
_LABEL_WIDTH_FRACTION = 0.9

# A label drawn any smaller than this is too small to read, so shorten the text
# instead. Roughly a third of the size the theme asks for.
_MIN_LABEL_SIZE = 14

# What the middle of a label too long to be drawn at a readable size is replaced with
_ELLIPSIS = '…'

# The name a label is drawn under, which is how it is found again to be refitted
_LABEL_NAME = 'plot_compare_label'

# Text is drawn at twice the font size it is given, so doubling a font size puts it in
# the same units as a fitted size
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


def _text_width(text: str, prop: Any, *, size: float, dpi: int, measurer: Any) -> float:
    """Return the width in pixels of the text drawn at the given font size."""
    # A `pyvista.TextProperty` loads the theme into a property shared by all of them,
    # which measuring has no business doing, so measure with a plain VTK one
    measured = _vtk.vtkTextProperty()
    # Copy what the text is drawn with, to measure its font rather than the default
    measured.ShallowCopy(prop)
    measured.SetFontSize(int(size))
    bounds = [0, 0, 0, 0]
    measurer.GetBoundingBox(measured, text, bounds, dpi)
    return bounds[1] - bounds[0]


def _fitting_size(text: str, prop: Any, *, width: float, dpi: int, measurer: Any) -> float:
    """Return the largest font size at which the text fits within the width."""
    measured = _text_width(text, prop, size=_REFERENCE_FONT_SIZE, dpi=dpi, measurer=measurer)
    # An empty label has no width to fit, so it never constrains the size
    return math.inf if measured <= 0 else _REFERENCE_FONT_SIZE * width / measured


def _ellipsize(text: str, n_kept: int) -> str:
    """Return the text with all but ``n_kept`` of its middle characters elided."""
    head = math.ceil(n_kept / 2)
    tail = n_kept // 2
    return text[:head] + _ELLIPSIS + (text[len(text) - tail :] if tail else '')


def _shorten(text: str, prop: Any, *, width: float, dpi: int, size: float, measurer: Any) -> str:
    """Return the longest elision of the text which fits the width at the given size."""
    if _text_width(text, prop, size=size, dpi=dpi, measurer=measurer) <= width:
        # The label fits as it is, so there is nothing to elide
        return text
    # The elided text only grows as more of it is kept, so bisect for the most it can
    # keep rather than measuring every length
    low, high = 0, len(text) - 1
    while low < high:
        n_kept = (low + high + 1) // 2
        kept = _ellipsize(text, n_kept)
        if _text_width(kept, prop, size=size, dpi=dpi, measurer=measurer) <= width:
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
    measurer: Any,
) -> None:
    """Draw every label at the largest size which fits in its subplot."""
    widths = [renderer.GetSize()[0] * _LABEL_WIDTH_FRACTION for renderer in renderers]
    sizes = [
        min(ceiling, _fitting_size(label, actor.prop, width=width, dpi=dpi, measurer=measurer))
        for label, actor, width in zip(labels, actors, widths, strict=True)
    ]
    if uniform:
        sizes = [min(sizes)] * len(sizes)

    for actor, label, width, size in zip(actors, labels, widths, sizes, strict=True):
        if size < _MIN_LABEL_SIZE:
            # The label is unreadable at the size it takes to fit, so draw it at the
            # smallest readable size and shorten the text until that fits instead
            size = _MIN_LABEL_SIZE  # noqa: PLW2901
            label = _shorten(  # noqa: PLW2901
                label, actor.prop, width=width, dpi=dpi, size=size, measurer=measurer
            )
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
    # Measuring text needs a text renderer, which is made here rather than where the
    # measuring happens, since that is called from within a render, and making one of
    # these brings up the backends which draw text
    measurer = _vtk.vtkMathTextFreeTypeTextRenderer()
    # The render window holds this callback for as long as it lives, so hold nothing of
    # the plotter in return: anything held here would outlive the plot
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
            # A shared size is pinned to whatever fits the narrowest subplot, so it
            # is only shared between subplots of the same width. A grid leaves a pixel
            # of the window over now and then, so widths within a pixel are the same.
            uniform=max(widths) - min(widths) <= 1 if uniform is None else uniform,
            ceiling=plotter.theme.font.size * _POINTS_PER_FONT_SIZE,
            dpi=dpi,
            measurer=measurer,
        )

    # `StartEvent` is emitted before each render, when the subplots already have the
    # size they are about to be drawn at
    plotter.render_window.AddObserver(_vtk.vtkCommand.StartEvent, fit)  # type: ignore[union-attr]


def plot_compare(  # noqa: ANN201
    datasets: Sequence[PlottableType] | Mapping[str, PlottableType],
    *,
    dataset_kwargs: dict[str, Any] | None = None,
    labels: Sequence[str] | None = _AUTO_LABELS,
    label_size: float | Literal['best_fit', 'uniform'] | None = None,
    label_position: TextPositionOptions | None = None,
    label_kwargs: dict[str, Any] | None = None,
    reference_mesh: DataSet | MultiBlock | PartitionedDataSet | None = None,
    reference_kwargs: dict[str, Any] | None = None,
    shape: Sequence[int] | str | None = None,
    normalize: bool = False,
    link: bool | None = None,
    cpos: CameraPositionOptions | None = None,
    zoom: float | str | None = None,
    show_axes: bool | None = None,
    show_bounds: bool = False,
    screenshot: str | bool | None = None,
    plotter_kwargs: dict[str, Any] | None = None,
    show_kwargs: dict[str, Any] | None = None,
):
    """Plot a grid comparison of any number of data objects.

    .. note::
        This function is also available via command-line interface. See
        :ref:`pyvista compare <cli_compare>` for details.

    Each data object is shown in its own subplot. By default, the subplots are arranged
    in a compact grid which is never taller than it is wide, e.g. ``(1, 2)`` for two
    datasets, ``(1, 3)`` for three, ``(2, 2)`` for four, and ``(2, 3)`` for five or six.
    Any leftover subplots are left empty. Use ``shape`` to control the layout explicitly.

    .. versionadded:: 0.49

    Parameters
    ----------
    datasets : Sequence[PlottableType] | Mapping[str, PlottableType]
        The data objects to compare, each of which is anything
        :meth:`~pyvista.Plotter.add_mesh` can draw. At least two datasets are
        required. If a mapping or a :class:`~pyvista.MultiBlock` is given, its
        keys are used as the default ``labels``.

    dataset_kwargs : dict, optional
        Additional keyword arguments passed to :meth:`~pyvista.Plotter.add_mesh`.
        The same arguments are used for each dataset.

    labels : Sequence[str] | None, optional
        The labels to display for each data object. Must have the same length as
        ``datasets``. By default, the keys of ``datasets`` are used when it is a
        mapping or a :class:`~pyvista.MultiBlock`, and the labels ``'A'``,
        ``'B'``, ``'C'``, ... are generated otherwise. Set to ``None`` to disable
        labels.

        If the input has keys `and` ``labels`` are provided, the provided
        ``labels`` take precedence and are used instead of its keys.

    label_size : float | str, optional
        The size of the ``labels``, either as a literal font size integer or as
        a string denoting how to work one out. A font size is used as given, and
        the labels will have the same constant size regardless of the window size
        of the plot. The label sizes which are worked out are:

        * ``'best_fit'``: draw each label as large as it fits in its own
          subplot, up to the font size of the theme. Labels of different lengths,
          and labels in subplots of different sizes, are drawn at different sizes.
        * ``'uniform'``: draw every label at the size of the one which has to be
          smallest to fit, so that they are all the same size no matter how long
          they are or which subplot they are in.

        By default, ``'uniform'`` is used when ``shape`` is a grid and all subplots
        have the same width; `'best_fit'`` is used otherwise when ``shape`` is a
        string descriptor and the subplots have different widths. A label too long
        to fit at a readable size has its middle elided, e.g. ``this is a very long label``
        may become ``this is...g label``.

        With the ``'best_fit'`` and ``'uniform'`` options, the actual font size is
        dynamically re-computed whenever the window is resized. Has no effect when
        ``labels`` is ``None``.

    label_position : str, optional
        Where in its subplot to draw each of the ``labels``: ``'upper_left'``,
        ``'upper_right'``, ``'lower_left'``, ``'lower_right'``, ``'upper_edge'``,
        ``'lower_edge'``, ``'left_edge'`` or ``'right_edge'``.

        Defaults to ``'upper_left'``. Has no effect when ``labels`` is ``None``.

    label_kwargs : dict, optional
        Additional keyword arguments for the :class:`~pyvista.Text` actor which
        draws each of the ``labels``, e.g. ``{'color': 'red'}``. Takes what
        :meth:`~pyvista.Plotter.add_text` takes. Has no effect when ``labels``
        is ``None``.

    reference_mesh : DataSet | MultiBlock, optional
        A mesh to draw in every subplot to give the comparison a common frame of
        reference, e.g. an outline of the dataset the compared results are
        derived from. The same mesh is drawn in each subplot, so it does not
        follow the bounds of the individual datasets. See the warning in
        ``normalize`` before using both.

    reference_kwargs : dict, optional
        Additional keyword arguments to pass to the
        :meth:`~pyvista.Plotter.add_mesh` method used to show the
        ``reference_mesh``. Defaults to ``{'color': 'k'}``.

    shape : Sequence[int] | str, optional
        The shape of the subplot layout, in any form accepted by
        :class:`~pyvista.Plotter`. Either a ``(n_rows, n_cols)`` sequence, or a
        string descriptor such as ``'3|1'`` for three subplots on the left and
        one on the right, or ``'4/2'`` for four on top and two on the bottom.
        Must define at least as many subplots as there are datasets. By default,
        the compact grid described above in the summary is used.

    normalize : bool, default: False
        Resize every dataset to a diagonal :attr:`~pyvista.DataSet.length` of one,
        centered on the origin, so that datasets of very different sizes are
        compared shape by shape. The datasets given are left as they are, and the
        resized copies of them are what is drawn, as is a ``reference_mesh``.

        Normalized datasets are all the same size and in the same place, so they
        are linked by default, which datasets of very different sizes are not.

        .. warning::

            A ``reference_mesh`` says much less about normalized datasets. Each
            of them is resized by a factor of its own, so the one mesh drawn in
            every subplot no longer relates them to each other, and is drawn at
            the size of each rather than around it.

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
        shared camera would leave some of them too small to make out. What each
        subplot has to fit is the dataset and the ``reference_mesh`` together when
        one is given, since the same mesh is drawn alongside every dataset.

        In every case the camera is only fit when ``cpos`` is ``None`` or a
        string, since a fully-specified camera position is used as given.

    cpos : CameraPositionOptions, optional
        The camera position to use in every subplot, as a list of the position,
        the focal point and the view up, or as one of the views
        :attr:`~pyvista.Plotter.camera_position` names, e.g. ``'xy'`` or
        ``'iso'``. A view is fit to the datasets, and a fully specified position
        is used as it is.

    zoom : float | str, optional
        Camera zoom, applied after the camera is fit to the datasets. Either
        ``'tight'`` or a float, where a value greater than ``1`` is a zoom-in.

    show_axes : bool, optional
        Show the axes orientation widget in every subplot. By default, the
        :attr:`~pyvista.plotting.themes.Theme.axes` setting of the theme is used.

    show_bounds : bool, default: False
        Show the bounds axes in every subplot.

    screenshot : str | bool, optional
        File name or path to save screenshot of the plot, or ``True`` to return
        a screenshot array.

    plotter_kwargs : dict, optional
        Additional keyword arguments to pass to the :class:`~pyvista.Plotter`
        constructor.

    show_kwargs : dict, optional
        Additional keyword arguments to pass to the :meth:`~pyvista.Plotter.show`
        method.

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
    ...     dataset_kwargs={'color': 'w'},
    ... )

    Use a dictionary to label each dataset and set the camera position explicitly.

    >>> pv.plot_compare(
    ...     {
    ...         'clip x': mesh.clip('x'),
    ...         'clip y': mesh.clip('y'),
    ...         'clip z': mesh.clip('z'),
    ...     },
    ...     dataset_kwargs={'color': 'w'},
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

    dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
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
        pl.add_mesh(dataset, **dataset_kwargs)
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
    # A reference mesh is drawn in every subplot, so what a subplot has to fit is the
    # dataset and the reference together, which is what decides whether to link too. An
    # outline enclosing every dataset, for instance, makes every subplot's own bounds
    # the same regardless of how different the datasets themselves are, which is
    # exactly the case linking suits.
    reference_bounds = None if reference_mesh is None else reference_mesh.bounds
    if link is None:
        link = _relative_size(renderers, reference_bounds=reference_bounds) >= _LINK_RELATIVE_SIZE

    if link:
        pl.link_views()
        # A dataset which is small beside the others is not made any easier to see by
        # a reference mesh which happens to be large enough to enclose all of them, so
        # warn from the size of the datasets on their own, whether linking was asked
        # for or decided from the reference mesh being one they are all small beside.
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
            bounds = _union_bounds(renderers)
            pl.renderer.ResetCamera(*bounds)
            pl.renderer.ResetCameraClippingRange(*bounds)
            # `reset_camera` above only reaches this render; an interactor style
            # narrows the clipping range again before every render it drives, from
            # `ComputeVisiblePropBounds` of whichever renderer the interaction is in,
            # which is one subplot's worth of bounds where every subplot needs
            # fitting. Undo that on every future render too, from the same bounds
            # rather than an actor standing in for them, which would leave
            # `renderer.bounds`, and anything relying on it, reporting more than what
            # is actually in each subplot.
            _fix_clipping_range_on_render(pl, bounds, n_datasets)
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
