"""Filters module with class to manage filters/algorithms for structured grid datasets."""

from __future__ import annotations

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _get_output
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.misc import abstract_class


@abstract_class
class StructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for structured grid datasets."""

    def extract_subset(self, voi, rate=(1, 1, 1), boundary=False):
        """Select piece (e.g., volume of interest).

        To use this filter set the VOI ivar which are i-j-k min/max
        indices that specify a rectangular region in the data. (Note
        that these are 0-offset.) You can also specify a sampling rate
        to subsample the data.

        Typical applications of this filter are to extract a slice
        from a volume for image processing, subsampling large volumes
        to reduce data size, or extracting regions of a volume with
        interesting data.

        Parameters
        ----------
        voi : sequence[int]
            Length 6 iterable of ints: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            These bounds specify the volume of interest in i-j-k min/max
            indices.

        rate : sequence[int], default: (1, 1, 1)
            Length 3 iterable of ints: ``(xrate, yrate, zrate)``.

        boundary : bool, default: False
            Control whether to enforce that the "boundary" of the grid
            is output in the subsampling process. (This only has
            effect when the rate in any direction is not equal to
            1). When this is on, the subsampling will always include
            the boundary of the grid even if the sample rate is
            not an even multiple of the grid dimensions.

        Returns
        -------
        pyvista.StructuredGrid
            StructuredGrid with extracted subset.

        Examples
        --------
        Split a grid in half.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_structured()
        >>> voi_1 = grid.extract_subset(
        ...     [0, 80, 0, 40, 0, 1], boundary=True
        ... )
        >>> voi_2 = grid.extract_subset(
        ...     [0, 80, 40, 80, 0, 1], boundary=True
        ... )

        For fun, add the two grids back together and show they are
        identical to the original grid.

        >>> joined = voi_1.concatenate(voi_2, axis=1)
        >>> assert np.allclose(grid.points, joined.points)

        """
        alg = _vtk.vtkExtractGrid()
        alg.SetVOI(voi)
        alg.SetInputDataObject(self)
        alg.SetSampleRate(rate)
        alg.SetIncludeBoundary(boundary)
        alg.Update()
        return _get_output(alg)

    def concatenate(self, other, axis, tolerance=0.0):
        """Concatenate a structured grid to this grid.

        Joins structured grids into a single structured grid.  Grids
        must be of compatible dimension, and must be coincident along
        the seam. Grids must have the same point and cell data.  Field
        data is ignored.

        Parameters
        ----------
        other : pyvista.StructuredGrid
            Structured grid to concatenate.

        axis : int
            Axis along which to concatenate.

        tolerance : float, default: 0.0
            Tolerance for point coincidence along joining seam.

        Returns
        -------
        pyvista.StructuredGrid
            Concatenated grid.

        Examples
        --------
        Split a grid in half and join them.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_structured()
        >>> voi_1 = grid.extract_subset(
        ...     [0, 80, 0, 40, 0, 1], boundary=True
        ... )
        >>> voi_2 = grid.extract_subset(
        ...     [0, 80, 40, 80, 0, 1], boundary=True
        ... )
        >>> joined = voi_1.concatenate(voi_2, axis=1)
        >>> f'{grid.dimensions} same as {joined.dimensions}'
        '(80, 80, 1) same as (80, 80, 1)'

        """
        if axis > 2:
            raise RuntimeError('Concatenation axis must be <= 2.')

        # check dimensions are compatible
        for i, (dim1, dim2) in enumerate(zip(self.dimensions, other.dimensions)):
            if i == axis:
                continue
            if dim1 != dim2:
                raise ValueError(
                    f'StructuredGrids with dimensions {self.dimensions} and {other.dimensions} '
                    'are not compatible.',
                )

        # check point/cell variables are the same
        if not set(self.point_data.keys()) == set(other.point_data.keys()):
            raise RuntimeError('Grid to concatenate has different point array names.')
        if not set(self.cell_data.keys()) == set(other.cell_data.keys()):
            raise RuntimeError('Grid to concatenate has different cell array names.')

        # check that points are coincident (within tolerance) along seam
        if not np.allclose(
            np.take(self.points_matrix, indices=-1, axis=axis),
            np.take(other.points_matrix, indices=0, axis=axis),
            atol=tolerance,
        ):
            raise RuntimeError(
                f'Grids cannot be joined along axis {axis}, as points '
                'are not coincident within tolerance of {tolerance}.',
            )

        # slice to cut off the repeated grid face
        slice_spec = [slice(None, None, None)] * 3
        slice_spec[axis] = slice(0, -1, None)
        slice_spec = tuple(slice_spec)  # trigger basic indexing

        # concatenate points, cutting off duplicate
        new_points = np.concatenate(
            (self.points_matrix[slice_spec], other.points_matrix),
            axis=axis,
        )

        # concatenate point arrays, cutting off duplicate
        new_point_data = {}
        for name, point_array in self.point_data.items():
            arr_1 = self._reshape_point_array(point_array)
            arr_2 = other._reshape_point_array(other.point_data[name])
            if not np.array_equal(
                np.take(arr_1, indices=-1, axis=axis),
                np.take(arr_2, indices=0, axis=axis),
            ):
                raise RuntimeError(
                    f'Grids cannot be joined along axis {axis}, as field '
                    '`{name}` is not identical along the seam.',
                )
            new_point_data[name] = np.concatenate((arr_1[slice_spec], arr_2), axis=axis).ravel(
                order='F',
            )

        new_dims = np.array(self.dimensions)
        new_dims[axis] += other.dimensions[axis] - 1

        # concatenate cell arrays
        new_cell_data = {}
        for name, cell_array in self.cell_data.items():
            arr_1 = self._reshape_cell_array(cell_array)
            arr_2 = other._reshape_cell_array(other.cell_data[name])
            new_cell_data[name] = np.concatenate((arr_1, arr_2), axis=axis).ravel(order='F')

        # assemble output
        joined = pyvista.StructuredGrid()
        joined.dimensions = list(new_dims)
        joined.points = new_points.reshape((-1, 3), order='F')
        joined.point_data.update(new_point_data)
        joined.cell_data.update(new_cell_data)

        return joined
