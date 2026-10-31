"""Reshape data arrays between their flat VTK layout and a shaped NumPy layout."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Protocol

import numpy as np

from ._structured_data import _grid_shape
from ._structured_data import _StructuredDataMixin
from .utilities.arrays import FieldAssociation
from .utilities.misc import _NoNewAttrMixin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from numpy.typing import NDArray

    from ._typing_core import _AnyArrayLike
    from .dataobject import DataObject
    from .datasetattributes import DataSetAttributes
    from .pyvista_ndarray import pyvista_ndarray

    class _ReshapeFunction(Protocol):
        """Signature of the functions wrapped by :class:`ArrayReshaper`."""

        def __call__(  # numpydoc ignore=PR01,RT01
            self, array: NDArray[Any], /, *, dataset: DataObject, association: FieldAssociation
        ) -> NDArray[Any]:
            """Convert ``array``."""


class ArrayReshaper(_NoNewAttrMixin):
    """Convert data arrays between their flat VTK layout and a shaped layout.

    Used by :attr:`DataSetAttributes.reshaped <pyvista.DataSetAttributes.reshaped>`.
    The default is set with :attr:`pyvista.global_config.array_reshaping
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
    GridReshaper
        Reshaper for arrays of structured grids.
    pyvista.DataSetAttributes.reshaped
        Shaped view of the data arrays.

    Examples
    --------
    Show every point array of a mesh as a column vector.

    >>> import pyvista as pv
    >>> reshaper = pv.ArrayReshaper(
    ...     reshape=lambda array, **_: array.reshape(-1, 1),
    ...     flatten=lambda array, **_: array.ravel(),
    ...     name='column',
    ... )
    >>> mesh = pv.Cube()
    >>> mesh.point_data['data'] = range(mesh.n_points)
    >>> mesh.point_data.reshaped.using(reshaper)['data'].shape
    (8, 1)

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
        self.name = name

    def __repr__(self: Self) -> str:
        """Return the representation of the reshaper."""
        return f'{type(self).__name__}(name={self.name!r})'

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


class GridReshaper(ArrayReshaper):
    """Reshape arrays of structured grids to the grid dimensions.

    Point arrays take the shape of the grid ``dimensions``, cell arrays one less
    along each axis, and multi-component arrays add a component axis. The shaped
    arrays are views, so editing them in place edits the dataset.

    .. versionadded:: 0.50

    Parameters
    ----------
    axes : str, default: 'xyz'
        Index order of the shaped array. With ``'xyz'``, ``array[i, j, k]``
        is the value at index ``i`` along x, ``j`` along y, and ``k`` along z,
        matching :attr:`~pyvista.core._structured_data._StructuredDataMixin.dimensions` and
        ``numpy.meshgrid(..., indexing='ij')``. With ``'zyx'``, the order is
        reversed to ``array[k, j, i]``, as used by many image libraries.

    components : str, default: 'last'
        Position of the component axis of multi-component arrays, ``'last'``
        or ``'first'``.

    squeeze : bool, default: False
        Drop grid axes of length one, so that the arrays of a 2D image have
        two spatial axes.

    origin : str, default: 'lower'
        Position of the first row along the vertical axis of the image, as in
        :func:`matplotlib.pyplot.imshow`. The vertical axis is the second grid
        axis longer than one, so y for an image in the xy plane. Use
        ``'upper'`` for images whose first row is the top row, as read by
        Pillow, OpenCV or imageio.

    name : str, optional
        Name shown in the representation of the reshaper.

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

    Use the reverse index order for the arrays of a 2D image.

    >>> image = pv.ImageData(dimensions=(4, 3, 1))
    >>> image.point_data['data'] = range(image.n_points)
    >>> reshaper = pv.GridReshaper(axes='zyx', squeeze=True)
    >>> image.point_data.reshaped.using(reshaper)['data']
    pyvista_ndarray([[ 0,  1,  2,  3],
                     [ 4,  5,  6,  7],
                     [ 8,  9, 10, 11]])

    Put the top row first, as in image files read with Pillow or OpenCV.

    >>> reshaper = pv.GridReshaper(axes='zyx', squeeze=True, origin='upper')
    >>> image.point_data.reshaped.using(reshaper)['data']
    pyvista_ndarray([[ 8,  9, 10, 11],
                     [ 4,  5,  6,  7],
                     [ 0,  1,  2,  3]])

    Read scans with several modalities stored as ``(channel, z, y, x)``, as in PyTorch or MONAI.

    >>> image = pv.ImageData(dimensions=(4, 3, 2))
    >>> reshaper = pv.GridReshaper(axes='zyx', components='first')
    >>> image.point_data.reshaped.using(reshaper)['scan'] = np.zeros((2, 2, 3, 4))
    >>> image.point_data['scan'].shape
    (24, 2)

    """

    def __init__(
        self: Self,
        *,
        axes: Literal['xyz', 'zyx'] = 'xyz',
        components: Literal['last', 'first'] = 'last',
        squeeze: bool = False,
        origin: Literal['lower', 'upper'] = 'lower',
        name: str | None = None,
    ) -> None:
        """Initialize the grid reshaper."""
        if axes not in ('xyz', 'zyx'):
            msg = f"`axes` must be 'xyz' or 'zyx', got {axes!r}."
            raise ValueError(msg)
        if components not in ('last', 'first'):
            msg = f"`components` must be 'last' or 'first', got {components!r}."
            raise ValueError(msg)
        if origin not in ('lower', 'upper'):
            msg = f"`origin` must be 'lower' or 'upper', got {origin!r}."
            raise ValueError(msg)
        self._axes = axes
        self._components = components
        self._squeeze = squeeze
        self._origin = origin
        self.name = name

    @property
    def axes(self: Self) -> Literal['xyz', 'zyx']:  # numpydoc ignore=RT01
        """Return the index order of the shaped array."""
        return self._axes

    @property
    def components(self: Self) -> Literal['last', 'first']:  # numpydoc ignore=RT01
        """Return the position of the component axis."""
        return self._components

    @property
    def squeeze(self: Self) -> bool:  # numpydoc ignore=RT01
        """Return whether grid axes of length one are dropped."""
        return self._squeeze

    @property
    def origin(self: Self) -> Literal['lower', 'upper']:  # numpydoc ignore=RT01
        """Return the position of the first row along the vertical axis."""
        return self._origin

    def __repr__(self: Self) -> str:
        """Return the representation of the reshaper."""
        args = (
            f'axes={self.axes!r}, components={self.components!r}, '
            f'squeeze={self.squeeze}, origin={self.origin!r}'
        )
        if self.name is not None:
            args += f', name={self.name!r}'
        return f'{type(self).__name__}({args})'

    def _spatial_shape(
        self: Self, dataset: DataObject, association: FieldAssociation, *, squeeze: bool
    ) -> tuple[tuple[int, ...], int | None]:
        """Return the grid shape of the arrays in index order and the axis to flip."""
        if not isinstance(dataset, _StructuredDataMixin):
            msg = (
                f'{self!r} reshapes arrays of structured grids only, '
                f'not arrays of {type(dataset).__name__}.'
            )
            raise TypeError(msg)
        if association not in (FieldAssociation.POINT, FieldAssociation.CELL):
            msg = f'{self!r} reshapes point and cell arrays only, not {association.name} arrays.'
            raise TypeError(msg)
        cells = association == FieldAssociation.CELL
        shape = _grid_shape(dataset.dimensions, cells=cells)
        n_values = dataset.n_cells if cells else dataset.n_points
        if n_values != np.prod(shape):
            msg = (
                f'{self!r} cannot reshape the {association.name.lower()} arrays of this '
                f'{type(dataset).__name__}: it has {n_values} values, not one per grid index.'
            )
            raise TypeError(msg)
        axes = list(enumerate(shape))
        if squeeze:
            axes = [axis for axis in axes if axis[1] != 1]
        long_axes = [label for label, n in axes if n != 1]
        vertical = long_axes[1] if len(long_axes) > 1 else None
        if self.axes == 'zyx':
            axes.reverse()
        labels = [label for label, _ in axes]
        flipped = (
            labels.index(vertical) if self.origin == 'upper' and vertical is not None else None
        )
        return tuple(n for _, n in axes), flipped

    def reshape(
        self: Self, array: NDArray[Any], *, dataset: DataObject, association: FieldAssociation
    ) -> NDArray[Any]:
        """Return the grid-shaped form of a flat data array.

        Parameters
        ----------
        array : numpy.ndarray
            Flat data array as stored by VTK.

        dataset : pyvista.DataObject
            Structured grid that owns the array.

        association : pyvista.core.utilities.arrays.FieldAssociation
            Association of the array.

        Returns
        -------
        numpy.ndarray
            Grid-shaped view of the array.

        """
        spatial, flipped = self._spatial_shape(dataset, association, squeeze=self.squeeze)
        order: Literal['C', 'F'] = 'F' if self.axes == 'xyz' else 'C'
        shaped = array.reshape((*spatial, *array.shape[1:]), order=order)
        if flipped is not None:
            shaped = np.flip(shaped, axis=flipped)
        if self.components == 'first' and array.ndim > 1:
            shaped = np.moveaxis(shaped, -1, 0)
        return shaped

    def flatten(
        self: Self, array: NDArray[Any], *, dataset: DataObject, association: FieldAssociation
    ) -> NDArray[Any]:
        """Return the flat form of a grid-shaped data array.

        The grid shape without its axes of length one is also accepted, and a
        flat array with one row per point or cell is returned unchanged.

        Parameters
        ----------
        array : numpy.ndarray
            Grid-shaped data array.

        dataset : pyvista.DataObject
            Structured grid the array is set on.

        association : pyvista.core.utilities.arrays.FieldAssociation
            Association of the array.

        Returns
        -------
        numpy.ndarray
            Flat array to store.

        """
        spatial, _ = self._spatial_shape(dataset, association, squeeze=self.squeeze)
        n_values = int(np.prod(spatial))
        is_flat = array.ndim in (1, 2) and array.shape[0] == n_values
        for squeeze in dict.fromkeys((self.squeeze, True)):
            if squeeze != self.squeeze and is_flat:
                return array
            flat = self._flatten_grid(array, dataset, association, squeeze=squeeze)
            if flat is not None:
                return flat
        if is_flat:
            return array
        first = self.components == 'first'
        expected = (
            f'{spatial}, {(*spatial, "n")}' if not first else f'{spatial}, {("n", *spatial)}'
        )
        msg = (
            f'Array of shape {array.shape} does not fit {self!r} for this '
            f'{type(dataset).__name__}. Expected a shape of {expected}, or a flat '
            f'array with {n_values} rows.'
        )
        raise ValueError(msg)

    def _flatten_grid(
        self: Self,
        array: NDArray[Any],
        dataset: DataObject,
        association: FieldAssociation,
        *,
        squeeze: bool,
    ) -> NDArray[Any] | None:
        """Return the flat form of an array in this grid layout, or ``None`` if it does not fit."""
        spatial, flipped = self._spatial_shape(dataset, association, squeeze=squeeze)
        n_spatial = len(spatial)
        first = self.components == 'first'
        n_components = array.ndim - n_spatial
        if n_components not in (0, 1):
            return None
        grid_axes = array.shape[n_components:] if first else array.shape[:n_spatial]
        if grid_axes != spatial:
            return None
        if first and n_components:
            array = np.moveaxis(array, 0, -1)
        if flipped is not None:
            array = np.flip(array, axis=flipped)
        order: Literal['C', 'F'] = 'F' if self.axes == 'xyz' else 'C'
        return array.reshape((-1, *array.shape[n_spatial:]), order=order)


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


_PRESETS: dict[str, ArrayReshaper] = {
    'grid': GridReshaper(name='grid'),
    'zyx': GridReshaper(axes='zyx', name='zyx'),
    'channels_first': GridReshaper(components='first', name='channels_first'),
    'image': GridReshaper(axes='zyx', squeeze=True, origin='upper', name='image'),
    'tensor': ArrayReshaper(reshape=_reshape_tensors, flatten=_flatten_tensors, name='tensor'),
}


def _resolve_reshaper(reshaper: str | ArrayReshaper) -> ArrayReshaper:
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

    Getting an array returns it reshaped, and setting an array flattens it first.

    .. versionadded:: 0.50

    Parameters
    ----------
    attributes : pyvista.DataSetAttributes
        Attributes holding the arrays.

    reshaper : str | pyvista.ArrayReshaper, optional
        Preset name or reshaper. Defaults to
        :attr:`pyvista.global_config.array_reshaping
        <pyvista.core.config.Config.array_reshaping>` at the time of each access.

    """

    def __init__(
        self: Self, attributes: DataSetAttributes, reshaper: str | ArrayReshaper | None = None
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

    def using(self: Self, reshaper: str | ArrayReshaper) -> ReshapedArrays:
        """Return a view of the same arrays with another reshaper.

        Parameters
        ----------
        reshaper : str | pyvista.ArrayReshaper
            Preset name or reshaper.

        Returns
        -------
        ReshapedArrays
            View that uses ``reshaper``.

        Examples
        --------
        >>> import pyvista as pv
        >>> image = pv.ImageData(dimensions=(4, 3, 2))
        >>> image.point_data['data'] = range(image.n_points)
        >>> image.point_data.reshaped.using('zyx')['data'].shape
        (2, 3, 4)

        """
        return type(self)(self._attributes, reshaper)

    def _dataset(self: Self) -> DataObject:
        """Return the dataset that owns the arrays."""
        return self._attributes.dataset  # type: ignore[return-value]

    def __getitem__(self: Self, key: str) -> pyvista_ndarray:
        """Return the named array reshaped."""
        array = self._attributes[key]
        return self.reshaper.reshape(  # type: ignore[return-value]
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
