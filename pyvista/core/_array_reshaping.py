"""Reshape data arrays between their flat VTK layout and a shaped NumPy layout."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TypeVar

import numpy as np

from ._structured_data import _grid_shape
from ._structured_data import _StructuredDataMixin
from .utilities.arrays import FieldAssociation
from .utilities.misc import _NoNewAttrMixin

_ArrayReshapingPresets = Literal['grid', 'zyx', 'channels_first', 'image', 'tensor']

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from typing import Self

    from numpy.typing import NDArray

    from ._typing_core import _AnyArrayLike
    from .dataobject import DataObject
    from .dataset import DataSet
    from .datasetattributes import DataSetAttributes

    _DataSetT = TypeVar('_DataSetT', bound=DataSet)

    class _ArrayFilter(Protocol):
        """Signature of the filters returned by :meth:`ArrayReshaper.wrap`."""

        def __call__(  # numpydoc ignore=PR01,RT01
            self,
            dataset: _DataSetT,
            /,
            scalars: str | None = None,
            *,
            preference: Literal['point', 'cell', 'field'] = 'point',
            output_name: str | None = None,
            inplace: bool = False,
        ) -> _DataSetT:
            """Filter ``dataset``."""

    class _ReshapeFunction(Protocol):
        """Signature of the functions wrapped by :class:`ArrayReshaper`."""

        def __call__(  # numpydoc ignore=PR01,RT01
            self, array: NDArray[Any], /, *, dataset: DataObject, association: FieldAssociation
        ) -> NDArray[Any]:
            """Convert ``array``."""


class ArrayReshaper(_NoNewAttrMixin):
    """Convert data arrays between their flat VTK layout and a shaped layout.

    A reshaper is a pair of functions. Use a preset by name, build one for a
    structured grid with :meth:`grid`, or pass two functions for any other
    layout. Used by :attr:`DataSetAttributes.reshaped
    <pyvista.DataSetAttributes.reshaped>`, with the default set by
    :attr:`pyvista.global_config.array_reshaping
    <pyvista.core.config.Config.array_reshaping>`.

    .. versionadded:: 0.50

    Parameters
    ----------
    reshape : callable
        Function that takes the flat array and returns the shaped array. It
        is called as ``reshape(array, dataset=dataset, association=association)``.

    flatten : callable
        Function that takes a shaped array and returns the flat array to
        store, with the same call signature as ``reshape``.

    name : str, optional
        Name shown in the representation of the reshaper.

    See Also
    --------
    pyvista.DataSetAttributes.reshaped
        Shaped view of the data arrays.

    Examples
    --------
    Show every point array of a mesh as a column vector.

    >>> import pyvista as pv
    >>> column = pv.ArrayReshaper(
    ...     reshape=lambda array, **_: array.reshape(-1, 1),
    ...     flatten=lambda array, **_: array.ravel(),
    ...     name='column',
    ... )
    >>> mesh = pv.Cube()
    >>> mesh.point_data['data'] = range(mesh.n_points)
    >>> mesh.point_data.reshaped(column)['data'].shape
    (8, 1)

    List the presets.

    >>> pv.ArrayReshaper.presets()
    ('grid', 'zyx', 'channels_first', 'image', 'tensor')

    """

    def __init__(
        self: Self,
        *,
        reshape: _ReshapeFunction,
        flatten: _ReshapeFunction,
        name: str | None = None,
    ) -> None:
        """Initialize the reshaper."""
        self._reshape = reshape
        self._flatten = flatten
        self._name = name

    @property
    def name(self: Self) -> str | None:  # numpydoc ignore=RT01
        """Return the name of the reshaper."""
        return self._name

    def __repr__(self: Self) -> str:
        """Return the representation of the reshaper."""
        return f'{type(self).__name__}({self.name!r})'

    @classmethod
    def presets(cls) -> tuple[_ArrayReshapingPresets, ...]:
        """Return the names of the preset reshapers.

        ``'grid'``
            Arrays of structured grids take the grid shape, indexed
            ``array[i, j, k]`` along x, y, and z, with any component axis last.

        ``'zyx'``
            As ``'grid'``, indexed ``array[k, j, i]``.

        ``'channels_first'``
            As ``'grid'``, with the component axis first.

        ``'image'``
            Images as ``(row, column, channel)`` with the top row first, as read
            by Pillow, OpenCV, or imageio. A stack of frames along z is
            ``(frame, row, column, channel)``.

        ``'tensor'``
            Arrays with nine components are 3x3 matrices, on any dataset.

        Returns
        -------
        tuple[str, ...]
            Preset names.

        """
        return tuple(_PRESETS)

    @classmethod
    def grid(
        cls,
        *,
        axes: str = 'xyz',
        components: Literal['last', 'first'] = 'last',
        squeeze: bool = False,
        origin: Literal['lower', 'upper'] = 'lower',
        name: str | None = None,
    ) -> ArrayReshaper:
        """Return a reshaper that shapes arrays of structured grids to the grid dimensions.

        Point arrays take the shape of the grid ``dimensions``, cell arrays one
        less along each axis, and multi-component arrays add a component axis.
        The shaped arrays are views, so editing them in place edits the dataset.
        Setting accepts the shaped layout, the same layout with or without its
        axes of length one, and a flat array with one row per point or cell.

        Parameters
        ----------
        axes : str, default: 'xyz'
            Index order of the shaped array as a permutation of ``'xyz'``. With
            ``'xyz'``, ``array[i, j, k]`` is the value at index ``i`` along x,
            ``j`` along y, and ``k`` along z, matching the grid ``dimensions``
            and ``numpy.meshgrid(..., indexing='ij')``. With ``'zyx'``, the
            order is reversed to ``array[k, j, i]``, as used by many image
            libraries.

        components : str, default: 'last'
            Position of the component axis of multi-component arrays,
            ``'last'`` or ``'first'``.

        squeeze : bool, default: False
            Drop grid axes of length one, so that the arrays of a 2D image have
            two spatial axes.

        origin : str, default: 'lower'
            Position of the first row, as in :func:`matplotlib.pyplot.imshow`. The
            rows run along the second to last spatial axis. ``'lower'`` keeps
            the first row at the lowest index along that axis, and ``'upper'``
            puts it at the highest, as in images read by Pillow, OpenCV, or imageio.

        name : str, optional
            Name shown in the representation of the reshaper.

        Returns
        -------
        ArrayReshaper
            Reshaper for the grid layout.

        Examples
        --------
        Set a 3D array as point data of an image and get it back with its shape.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> image = pv.ImageData(dimensions=(4, 3, 2))
        >>> volume = np.arange(24).reshape(4, 3, 2)
        >>> image.point_data.reshaped['volume'] = volume
        >>> image.point_data.reshaped['volume'].shape
        (4, 3, 2)

        The flat array follows the VTK point order, where x varies fastest.

        >>> image.point_data['volume'][:5]
        pyvista_ndarray([ 0,  6, 12, 18,  2])

        Index the arrays of a 2D image as ``array[row, column]``.

        >>> image = pv.ImageData(dimensions=(4, 3, 1))
        >>> image.point_data['data'] = range(image.n_points)
        >>> rows = pv.ArrayReshaper.grid(axes='zyx', squeeze=True)
        >>> image.point_data.reshaped(rows)['data']
        pyvista_ndarray([[ 0,  1,  2,  3],
                         [ 4,  5,  6,  7],
                         [ 8,  9, 10, 11]])

        Put the top row first, as in image files read with Pillow or OpenCV.

        >>> top_first = pv.ArrayReshaper.grid(axes='zyx', squeeze=True, origin='upper')
        >>> image.point_data.reshaped(top_first)['data']
        pyvista_ndarray([[ 8,  9, 10, 11],
                         [ 4,  5,  6,  7],
                         [ 0,  1,  2,  3]])

        Read scans with several modalities stored as ``(channel, z, y, x)``, the
        ``(C, D, H, W)`` layout of PyTorch.

        >>> image = pv.ImageData(dimensions=(4, 3, 2))
        >>> scans = pv.ArrayReshaper.grid(axes='zyx', components='first')
        >>> image.point_data.reshaped(scans)['scan'] = np.zeros((2, 2, 3, 4))
        >>> image.point_data['scan'].shape
        (24, 2)

        """
        layout = _GridLayout(axes=axes, components=components, squeeze=squeeze, origin=origin)
        return cls(reshape=layout.reshape, flatten=layout.flatten, name=name or repr(layout))

    def reshape(
        self: Self, array: NDArray[Any], *, dataset: DataObject, association: FieldAssociation
    ) -> NDArray[Any]:
        """Return the shaped form of a flat data array.

        Parameters
        ----------
        array : numpy.ndarray
            Flat data array as stored by VTK.

        dataset : pyvista.DataObject
            Dataset that owns the array.

        association : pyvista.core.utilities.arrays.FieldAssociation
            Association of the array.

        Returns
        -------
        numpy.ndarray
            Shaped array.

        """
        return self._reshape(array, dataset=dataset, association=association)

    def flatten(
        self: Self, array: NDArray[Any], *, dataset: DataObject, association: FieldAssociation
    ) -> NDArray[Any]:
        """Return the flat form of a shaped data array.

        Parameters
        ----------
        array : numpy.ndarray
            Shaped data array.

        dataset : pyvista.DataObject
            Dataset the array is set on.

        association : pyvista.core.utilities.arrays.FieldAssociation
            Association of the array.

        Returns
        -------
        numpy.ndarray
            Flat array to store.

        """
        return self._flatten(array, dataset=dataset, association=association)

    def wrap(self: Self, function: Callable[[NDArray[Any]], NDArray[Any]]) -> _ArrayFilter:
        """Return a filter that runs a function of shaped arrays on a dataset.

        The filter reshapes an array of the dataset, passes it to ``function``,
        and stores the result flattened. Like other filters, it returns a copy
        unless ``inplace=True``, so :meth:`~pyvista.CompositeFilters.generic_filter`
        runs it on every block.

        Parameters
        ----------
        function : callable
            Function that takes a shaped array and returns one in the same layout.

        Returns
        -------
        callable
            Filter called as ``filter(dataset, scalars=None, *,
            preference='point', output_name=None, inplace=False)``. ``scalars``
            defaults to the active scalars, ``preference`` chooses between point
            and cell arrays of the same name, and ``output_name`` defaults to
            ``scalars``.

        See Also
        --------
        pyvista.CompositeFilters.generic_filter
            Run the filter on every block of a :class:`~pyvista.MultiBlock`.

        Examples
        --------
        Smooth an image along x with :func:`scipy.ndimage.gaussian_filter1d`,
        which needs the array in its 3D shape.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from scipy import ndimage
        >>> smooth_x = pv.ArrayReshaper.grid().wrap(
        ...     lambda a: ndimage.gaussian_filter1d(a, 1, axis=0)
        ... )
        >>> image = pv.ImageData(dimensions=(4, 3, 2))
        >>> image.point_data['x'] = image.points[:, 0]
        >>> smoothed = smooth_x(image, 'x')
        >>> np.round(smoothed.point_data['x'][:4], 2)
        array([0.43, 1.06, 1.94, 2.57])

        The input is unchanged unless ``inplace=True``.

        >>> image.point_data['x'][:4]
        pyvista_ndarray([0., 1., 2., 3.])
        >>> _ = smooth_x(image, 'x', output_name='smooth_x', inplace=True)
        >>> image.point_data.keys()
        ['x', 'smooth_x']

        Run the filter on every block of a :class:`~pyvista.MultiBlock`.

        >>> blocks = pv.MultiBlock([image, image.copy()])
        >>> smoothed_blocks = blocks.generic_filter(smooth_x, 'x')
        >>> [np.allclose(block['x'], smoothed['x']) for block in smoothed_blocks]
        [True, True]

        """
        from .utilities.arrays import get_array_association  # noqa: PLC0415
        from .utilities.arrays import set_default_active_scalars  # noqa: PLC0415

        def filtered(  # numpydoc ignore=PR01,RT01
            dataset: _DataSetT,
            /,
            scalars: str | None = None,
            *,
            preference: Literal['point', 'cell', 'field'] = 'point',
            output_name: str | None = None,
            inplace: bool = False,
        ) -> _DataSetT:
            """Run ``function`` on an array of ``dataset``."""
            output = dataset if inplace else dataset.copy()
            if scalars is None:
                association, scalars = set_default_active_scalars(output)
            else:
                association = get_array_association(
                    output, scalars, preference=preference, err=True
                )
            attributes = {
                FieldAssociation.POINT: output.point_data,
                FieldAssociation.CELL: output.cell_data,
            }.get(association, output.field_data)
            view = ReshapedArrays(attributes, self)
            view[scalars if output_name is None else output_name] = function(view[scalars])
            return output

        return filtered


def _point_ids_out_of_grid_order(dataset: object) -> bool:
    """Return whether ``dataset`` is an explicit structured grid with scrambled point ids."""
    from .cell import _get_connectivity  # noqa: PLC0415
    from .pointset import ExplicitStructuredGrid  # noqa: PLC0415

    if not isinstance(dataset, ExplicitStructuredGrid):
        return False
    nx, ny, nz = dataset.dimensions
    ids = np.arange(nx * ny * nz).reshape((nx, ny, nz), order='F')
    lo, hi = slice(None, -1), slice(1, None)
    corners = [ids[i, j, k] for k in (lo, hi) for i, j in ((lo, lo), (hi, lo), (hi, hi), (lo, hi))]
    expected = np.stack(corners, axis=-1).reshape(-1, 8, order='F')
    connectivity = _get_connectivity(dataset.GetCells())
    return connectivity.size != expected.size or not np.array_equal(
        connectivity.reshape(-1, 8), expected
    )


class _GridLayout:
    """Reshape flat grid arrays to the grid shape in an index order."""  # numpydoc ignore=PR01

    def __init__(
        self,
        *,
        axes: str,
        components: Literal['last', 'first'],
        squeeze: bool,
        origin: Literal['lower', 'upper'],
    ) -> None:
        """Validate and store the layout."""
        if sorted(axes) != ['x', 'y', 'z']:
            msg = f"`axes` must be a permutation of 'xyz', got {axes!r}."
            raise ValueError(msg)
        if components not in ('last', 'first'):
            msg = f"`components` must be 'last' or 'first', got {components!r}."
            raise ValueError(msg)
        if origin not in ('lower', 'upper'):
            msg = f"`origin` must be 'lower' or 'upper', got {origin!r}."
            raise ValueError(msg)
        self.axes = axes
        self.components = components
        self.squeeze = squeeze
        self.origin = origin

    def __repr__(self) -> str:
        """Return the layout as a call to :meth:`ArrayReshaper.grid`."""
        return (
            f'grid(axes={self.axes!r}, components={self.components!r}, '
            f'squeeze={self.squeeze}, origin={self.origin!r})'
        )

    def _xyz_shape(
        self, dataset: DataObject, association: FieldAssociation
    ) -> tuple[int, int, int]:
        """Return the grid shape of the arrays in x, y, z order."""
        if not isinstance(dataset, _StructuredDataMixin):
            msg = (
                f'ArrayReshaper({self!r}) reshapes arrays of structured grids only, '
                f'not arrays of {type(dataset).__name__}.'
            )
            raise TypeError(msg)
        if association not in (FieldAssociation.POINT, FieldAssociation.CELL):
            kind = 'field' if association == FieldAssociation.NONE else association.name.lower()
            msg = (
                f'ArrayReshaper({self!r}) reshapes point and cell arrays only, not {kind} arrays.'
            )
            raise TypeError(msg)
        cells = association == FieldAssociation.CELL
        shape = _grid_shape(dataset.dimensions, cells=cells)
        n_values = dataset.n_cells if cells else dataset.n_points
        if n_values != np.prod(shape):
            msg = (
                f'ArrayReshaper({self!r}) cannot reshape the {association.name.lower()} '
                f'arrays of this {type(dataset).__name__}: it has {n_values} values, '
                'not one per grid index.'
            )
            raise TypeError(msg)
        if not cells and _point_ids_out_of_grid_order(dataset):
            msg = (
                f'ArrayReshaper({self!r}) cannot reshape the point arrays of this '
                'ExplicitStructuredGrid: its point ids are not in grid order.'
            )
            raise TypeError(msg)
        return shape

    def _layout(
        self, dataset: DataObject, association: FieldAssociation, *, squeeze: bool
    ) -> tuple[tuple[int, int, int], list[int], list[int]]:
        """Return the grid shape, the kept axes in index order, and the axes to flip."""
        xyz_shape = self._xyz_shape(dataset, association)
        kept = [i for i, n in enumerate(xyz_shape) if not (squeeze and n == 1)]
        order = [i for i in ('xyz'.index(a) for a in self.axes) if i in kept]
        flipped = [len(order) - 2] if self.origin == 'upper' and len(order) > 1 else []
        return xyz_shape, order, flipped

    def reshape(  # numpydoc ignore=PR01,RT01
        self, array: NDArray[Any], /, *, dataset: DataObject, association: FieldAssociation
    ) -> NDArray[Any]:
        """Return the grid-shaped view of a flat array."""
        xyz_shape, order, flipped = self._layout(dataset, association, squeeze=self.squeeze)
        n_components = array.ndim - 1
        shaped = array.reshape((*xyz_shape, *array.shape[1:]), order='F')
        shaped = shaped.squeeze(axis=tuple(i for i in range(3) if i not in order))
        positions = [sorted(order).index(i) for i in order]
        shaped = shaped.transpose((*positions, *range(len(order), len(order) + n_components)))
        for axis in flipped:
            shaped = np.flip(shaped, axis=axis)
        if self.components == 'first' and n_components:
            shaped = np.moveaxis(shaped, -1, 0)
        return shaped

    def flatten(  # numpydoc ignore=PR01,RT01
        self, array: NDArray[Any], /, *, dataset: DataObject, association: FieldAssociation
    ) -> NDArray[Any]:
        """Return the flat form of a grid-shaped array."""
        xyz_shape = self._xyz_shape(dataset, association)
        n_values = int(np.prod(xyz_shape))
        is_flat = array.ndim in (1, 2) and array.shape[0] == n_values
        flat = self._flatten_grid(array, dataset, association, squeeze=self.squeeze)
        if flat is None and not is_flat:
            flat = self._flatten_grid(array, dataset, association, squeeze=not self.squeeze)
        if flat is not None:
            return flat
        if is_flat:
            return array
        _, order, _ = self._layout(dataset, association, squeeze=self.squeeze)
        spatial = ', '.join(str(xyz_shape[i]) for i in order)
        with_components = (
            f'({spatial}, n_components)'
            if self.components == 'last'
            else f'(n_components, {spatial})'
        )
        msg = (
            f'Array of shape {array.shape} does not fit ArrayReshaper({self!r}) for this '
            f'{type(dataset).__name__}. Expected a shape of ({spatial}), {with_components}, '
            f'or a flat array with {n_values} rows.'
        )
        raise ValueError(msg)

    def _flatten_grid(
        self,
        array: NDArray[Any],
        dataset: DataObject,
        association: FieldAssociation,
        *,
        squeeze: bool,
    ) -> NDArray[Any] | None:
        """Return the flat form of an array in this layout, or ``None`` if it does not fit."""
        xyz_shape, order, flipped = self._layout(dataset, association, squeeze=squeeze)
        n_spatial = len(order)
        first = self.components == 'first'
        n_components = array.ndim - n_spatial
        if n_components not in (0, 1):
            return None
        spatial = tuple(xyz_shape[i] for i in order)
        grid_axes = array.shape[n_components:] if first else array.shape[:n_spatial]
        if grid_axes != spatial:
            return None
        if first and n_components:
            array = np.moveaxis(array, 0, -1)
        for axis in flipped:
            array = np.flip(array, axis=axis)
        inverse = [order.index(i) for i in sorted(order)]
        array = array.transpose((*inverse, *range(n_spatial, n_spatial + n_components)))
        array = np.expand_dims(array, tuple(i for i in range(3) if i not in order))
        return array.reshape((-1, *array.shape[3:]), order='F')


def _reshape_tensors(array: NDArray[Any], **_: Any) -> NDArray[Any]:
    """Return 9-component arrays as 3x3 matrices, and other arrays unchanged."""
    if array.ndim == 2 and array.shape[1] == 9:
        # VTK stores each tensor in column-major order
        return array.reshape(-1, 3, 3).transpose(0, 2, 1)
    return array


def _flatten_tensors(array: NDArray[Any], **_: Any) -> NDArray[Any]:
    """Return 3x3 matrices as 9-component arrays, and other arrays unchanged."""
    if array.ndim == 3 and array.shape[1:] == (3, 3):
        return array.transpose(0, 2, 1).reshape(-1, 9)
    return array


_PRESETS: dict[_ArrayReshapingPresets, ArrayReshaper] = {
    'grid': ArrayReshaper.grid(name='grid'),
    'zyx': ArrayReshaper.grid(axes='zyx', name='zyx'),
    'channels_first': ArrayReshaper.grid(components='first', name='channels_first'),
    'image': ArrayReshaper.grid(axes='zyx', squeeze=True, origin='upper', name='image'),
    'tensor': ArrayReshaper(reshape=_reshape_tensors, flatten=_flatten_tensors, name='tensor'),
}


def _resolve_reshaper(reshaper: _ArrayReshapingPresets | ArrayReshaper) -> ArrayReshaper:
    """Return the reshaper for a preset name or reshaper."""
    if isinstance(reshaper, ArrayReshaper):
        return reshaper
    if isinstance(reshaper, str):
        try:
            return _PRESETS[reshaper]
        except KeyError:
            msg = f'Unknown array reshaping preset {reshaper!r}. Choose from {list(_PRESETS)}.'
            raise ValueError(msg) from None
    msg = (  # type: ignore[unreachable]
        'Array reshaping must be a preset name or an ArrayReshaper, '
        f'got {type(reshaper).__name__}.'
    )
    raise TypeError(msg)


class ReshapedArrays(_NoNewAttrMixin, MutableMapping[str, Any]):
    """Shaped view of the data arrays of a :class:`~pyvista.DataSetAttributes`.

    Getting an array returns it reshaped, and setting an array flattens it
    first. Call the view with a preset name or reshaper for another layout.

    .. versionadded:: 0.50

    Parameters
    ----------
    attributes : pyvista.DataSetAttributes
        Attributes holding the arrays.

    reshaper : str | pyvista.ArrayReshaper, optional
        Preset name or reshaper. Defaults to
        :attr:`pyvista.global_config.array_reshaping
        <pyvista.core.config.Config.array_reshaping>` at the time of each access.

    Examples
    --------
    >>> import pyvista as pv
    >>> image = pv.ImageData(dimensions=(4, 3, 2))
    >>> image.point_data['data'] = range(image.n_points)
    >>> image.point_data.reshaped['data'].shape
    (4, 3, 2)

    Call the view for another layout.

    >>> image.point_data.reshaped('zyx')['data'].shape
    (2, 3, 4)

    """

    def __init__(
        self: Self,
        attributes: DataSetAttributes,
        reshaper: _ArrayReshapingPresets | ArrayReshaper | None = None,
    ) -> None:
        """Initialize the view."""
        self._attributes = attributes
        self._reshaper = None if reshaper is None else _resolve_reshaper(reshaper)

    @property
    def reshaper(self: Self) -> ArrayReshaper:  # numpydoc ignore=RT01
        """Return the reshaper used by this view."""
        if self._reshaper is not None:
            return self._reshaper
        from .config import global_config  # noqa: PLC0415

        return _resolve_reshaper(global_config.array_reshaping)

    def __call__(self: Self, reshaper: _ArrayReshapingPresets | ArrayReshaper) -> ReshapedArrays:
        """Return a view of the same arrays with another reshaper.

        Parameters
        ----------
        reshaper : str | pyvista.ArrayReshaper
            Preset name or reshaper.

        Returns
        -------
        ReshapedArrays
            View that uses ``reshaper``.

        """
        return type(self)(self._attributes, reshaper)

    def _dataset(self: Self) -> DataObject:
        """Return the dataset that owns the arrays."""
        return self._attributes.dataset  # type: ignore[return-value]

    def __getitem__(self: Self, key: str) -> NDArray[Any]:
        """Return the named array reshaped."""
        array = self._attributes[key]
        return self.reshaper.reshape(
            array, dataset=self._dataset(), association=self._attributes.association
        )

    def __setitem__(self: Self, key: str, value: _AnyArrayLike) -> None:
        """Flatten an array and store it under ``key``."""
        array = np.asarray(value)
        self._attributes[key] = self.reshaper.flatten(
            array, dataset=self._dataset(), association=self._attributes.association
        )

    def __delitem__(self: Self, key: str) -> None:
        """Remove the named array."""
        del self._attributes[key]

    def __iter__(self: Self) -> Iterator[str]:
        """Iterate over the array names."""
        return iter(self._attributes)

    def __len__(self: Self) -> int:
        """Return the number of arrays."""
        return len(self._attributes)

    def __contains__(self: Self, key: object) -> bool:
        """Return whether an array with this name exists."""
        return isinstance(key, str) and key in self._attributes

    def _ipython_key_completions_(self: Self) -> list[str]:
        """Tab completion of IPython."""
        return self._attributes.keys()

    def __repr__(self: Self) -> str:
        """Return the names and reshaped shapes of the arrays."""
        reshaper = self.reshaper
        lines = [
            'pyvista ReshapedArrays',
            f'Association     : {self._attributes.association.name}',
            f'Reshaper        : {reshaper!r}',
        ]
        if not self:
            lines.append('Contains arrays : None')
            return '\n'.join(lines)
        lines.append('Contains arrays :')
        for name in self:
            try:
                array = self[name]
            except (TypeError, ValueError) as error:
                lines.append(f'    {name[:23]:<24}{type(error).__name__}')
                continue
            lines.append(f'    {name[:23]:<24}{array.dtype!s:<11}{array.shape!s}')
        return '\n'.join(lines)
