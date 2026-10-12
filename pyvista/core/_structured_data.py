"""Shared index-space properties of structured datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import pyvista_validation as _validation

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLikeInt


def _extent_from(offset: VectorLikeInt, dims: VectorLikeInt) -> list[int]:
    """Return the extent starting at ``offset`` with dimensions ``dims``."""
    return [int(n) for o, d in zip(offset, dims, strict=True) for n in (o, o + d - 1)]


class _StructuredDataMixin:
    """Add ``dimensions``, ``extent`` and ``offset`` properties to structured datasets.

    .. note::
        This class is a private internal implementation detail. It is documented
        solely so that its public members, which are inherited by public classes,
        are visible in the documentation.

    """

    #: Whether the dimensions are defined by the dataset's coordinates or cells.
    _has_fixed_dimensions: ClassVar[bool] = False

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Return or set the grid's dimensions.

        These are effectively the number of points along each of the
        three dataset axes. They are all positive, or all zero for an empty grid.
        Setting the dimensions keeps the grid's :attr:`offset`.

        The dimensions of a :class:`~pyvista.RectilinearGrid` and an
        :class:`~pyvista.ExplicitStructuredGrid` are defined by their coordinates and
        cells and cannot be set. The dimensions of a :class:`~pyvista.StructuredGrid`
        are checked against its number of points when they are set, so set its points
        first.

        Returns
        -------
        tuple[int, int, int]
            Dimensions of the grid.

        Examples
        --------
        Create a uniform grid with dimensions ``(2, 3, 4)``.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(2, 3, 4))
        >>> grid.dimensions
        (2, 3, 4)
        >>> grid.plot(show_edges=True)

        Set the dimensions to ``(3, 4, 5)``.

        >>> grid.dimensions = (3, 4, 5)
        >>> grid.plot(show_edges=True)

        """
        extent = self.extent
        return (extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1)

    @dimensions.setter
    def dimensions(self, dims: VectorLikeInt) -> None:
        if self._has_fixed_dimensions:
            name = type(self).__name__
            article = 'an' if name[0] in 'AEIOU' else 'a'
            msg = (
                f'The dimensions of {article} `{name}` are implicitly defined and thus '
                'cannot be set.'
            )
            raise AttributeError(msg)
        dims_ = _validation.validate_array3(
            dims, must_be_integer=True, must_be_nonnegative=True, dtype_out=int, name='Dimensions'
        )
        self.extent = _extent_from(self.offset, dims_)

    @property
    def extent(self) -> tuple[int, int, int, int, int, int]:  # numpydoc ignore=RT01
        """Return or set the extent of the grid.

        The extent is simply the first and last indices for each of the three axes.
        It encodes information about the grid's :attr:`offset` and :attr:`dimensions`,
        and setting it is subject to the same constraints on the dimensions.

        .. versionchanged:: 0.50
            Available on :class:`~pyvista.RectilinearGrid`,
            :class:`~pyvista.StructuredGrid` and :class:`~pyvista.ExplicitStructuredGrid`.

        Examples
        --------
        Create an :class:`~pyvista.ImageData` and show its extent.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(10, 10, 10))
        >>> grid.extent
        (0, 9, 0, 9, 0, 9)

        >>> grid.extent = (2, 5, 2, 5, 2, 5)
        >>> grid.extent
        (2, 5, 2, 5, 2, 5)

        For an :class:`~pyvista.ImageData`, this also modifies the :attr:`offset`,
        :attr:`dimensions`, and :attr:`~pyvista.DataSet.bounds`. Since
        we use default spacing of 1 here, the bounds match the extent exactly.

        >>> grid.offset
        (2, 2, 2)

        >>> grid.dimensions
        (4, 4, 4)

        >>> grid.bounds
        BoundsTuple(x_min = 2.0,
                    x_max = 5.0,
                    y_min = 2.0,
                    y_max = 5.0,
                    z_min = 2.0,
                    z_max = 5.0)

        """
        return self.GetExtent()  # type: ignore[attr-defined]

    @extent.setter
    def extent(self, new_extent: VectorLikeInt) -> None:
        new_extent_ = _validation.validate_arrayN(
            new_extent,
            must_be_integer=True,
            must_have_length=6,
            to_list=True,
            dtype_out=int,
            name='Extent',
        )
        if new_extent_ == list(self.extent):
            return
        new_dims = (
            new_extent_[1] - new_extent_[0] + 1,
            new_extent_[3] - new_extent_[2] + 1,
            new_extent_[5] - new_extent_[4] + 1,
        )
        _validation.check_nonnegative(new_dims, name='Dimensions')
        if 0 in new_dims and any(new_dims):
            msg = f'Dimensions {new_dims} must be all zero (empty) or all positive.'
            raise ValueError(msg)
        if new_dims != self.dimensions:
            self._check_dimensions(new_dims)
        self._set_extent(new_extent_)

    def _check_dimensions(self, dims: tuple[int, int, int]) -> None:
        """Raise if the dimensions cannot change to ``dims``."""
        if self._has_fixed_dimensions:
            msg = (
                f'Cannot change the dimensions of `{type(self).__name__}` from '
                f'{self.dimensions} to {dims}. Set the offset to shift the extent instead.'
            )
            raise ValueError(msg)

    def _set_extent(self, extent: list[int]) -> None:
        """Set the extent of the underlying VTK dataset."""
        self.SetExtent(extent)  # type: ignore[attr-defined]

    @property
    def offset(self) -> tuple[int, int, int]:  # numpydoc ignore=RT01
        """Return or set the index offset of the grid.

        The offset is simply the first indices for each of the three axes
        and defines the minimum :attr:`extent` of the grid. Offset values
        can be positive or negative. Setting the offset keeps the grid's
        :attr:`dimensions`.

        For an :class:`~pyvista.ImageData`, the offset is relative to the
        :attr:`~pyvista.ImageData.origin` in physical space. For other grids, the points
        do not move.

        .. versionadded:: 0.45

        .. versionchanged:: 0.50
            Available on :class:`~pyvista.RectilinearGrid`,
            :class:`~pyvista.StructuredGrid` and :class:`~pyvista.ExplicitStructuredGrid`.

        Examples
        --------
        Create an :class:`~pyvista.ImageData` and show that the offset is zeros by default.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(10, 10, 10))
        >>> grid.offset
        (0, 0, 0)

        The offset defines the minimum extent.

        >>> grid.extent
        (0, 9, 0, 9, 0, 9)

        Set the offset to a new value for all axes.

        >>> grid.offset = 2
        >>> grid.offset
        (2, 2, 2)

        Show the extent again. Note how all values have increased by the offset value.

        >>> grid.extent
        (2, 11, 2, 11, 2, 11)

        Set the offset for each axis separately and show the extent again.

        >>> grid.offset = (-1, -2, -3)
        >>> grid.extent
        (-1, 8, -2, 7, -3, 6)

        Set the offset of a :class:`~pyvista.StructuredGrid`. Its points, and therefore its bounds,
        are unchanged.

        >>> grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_structured_grid()
        >>> grid.offset = (-1, -2, -3)
        >>> grid.extent
        (-1, 1, -2, 0, -3, -1)

        >>> grid.bounds
        BoundsTuple(x_min = 0.0,
                    x_max = 2.0,
                    y_min = 0.0,
                    y_max = 2.0,
                    z_min = 0.0,
                    z_max = 2.0)

        """
        return self.extent[::2]

    @offset.setter
    def offset(self, offset: int | VectorLikeInt) -> None:
        offset_ = _validation.validate_array3(
            offset, broadcast=True, must_be_integer=True, dtype_out=int, name='Offset'
        )
        self.extent = _extent_from(offset_, self.dimensions)
