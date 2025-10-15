"""Filters with a class to manage filters/algorithms for uniform grid datasets."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
import operator
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast
from typing import get_args
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import set_default_active_scalars
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from pyvista import ImageData
    from pyvista import PolyData
    from pyvista import pyvista_ndarray
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike

_InterpolationOptions = Literal[
    'nearest',
    'linear',
    'cubic',
    'lanczos',
    'hamming',
    'blackman',
    'bspline',
    'bspline0',
    'bspline1',
    'bspline2',
    'bspline3',
    'bspline4',
    'bspline5',
    'bspline6',
    'bspline7',
    'bspline8',
    'bspline9',
]
_AxisOptions = Literal[0, 1, 2, 'x', 'y', 'z']
_ConcatenateModeOptions = Literal[
    'strict',
    'resample-off-axis',
    'resample-proportional',
    'crop-off-axis',
    'crop-match',
    'preserve-extents',
]
_ConcatenateDTypePolicyOptions = Literal['strict', 'promote', 'match']
_ConcatenateComponentPolicyOptions = Literal['strict', 'promote_rgba']


@abstract_class
class ImageDataFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for uniform grid datasets."""

    @_deprecate_positional_args
    def gaussian_smooth(  # noqa: PLR0917
        self,
        radius_factor=1.5,
        std_dev=2.0,
        scalars=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Smooth the data with a Gaussian kernel.

        Parameters
        ----------
        radius_factor : float | sequence[float], default: 1.5
            Unitless factor to limit the extent of the kernel.

        std_dev : float | sequence[float], default: 2.0
            Standard deviation of the kernel in pixel units.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Uniform grid with smoothed scalars.

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with
        :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`.

        Examples
        --------
        First, create sample data to smooth. Here, we use
        :func:`pyvista.perlin_noise() <pyvista.core.utilities.features.perlin_noise>`
        to create meaningful data.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (2, 5, 8), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, bounds=[0, 1, 0, 1, 0, 1], dim=(20, 20, 20)
        ... )
        >>> grid.plot(show_scalar_bar=False)

        Next, smooth the sample data.

        >>> smoothed = grid.gaussian_smooth()
        >>> smoothed.plot(show_scalar_bar=False)

        See :ref:`gaussian_smoothing_example` for a full example using this filter.

        """
        alg = _vtk.vtkImageGaussianSmooth()
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
            if field.value == 1:
                msg = 'If `scalars` not given, active scalars must be point array.'
                raise ValueError(msg)
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
            if field.value == 1:
                msg = 'Can only process point data, given `scalars` are cell data.'
                raise ValueError(msg)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        if isinstance(radius_factor, Iterable):
            alg.SetRadiusFactors(radius_factor)  # type: ignore[call-overload]
        else:
            alg.SetRadiusFactors(radius_factor, radius_factor, radius_factor)
        if isinstance(std_dev, Iterable):
            alg.SetStandardDeviations(std_dev)  # type: ignore[call-overload]
        else:
            alg.SetStandardDeviations(std_dev, std_dev, std_dev)
        _update_alg(alg, progress_bar=progress_bar, message='Performing Gaussian Smoothing')
        return _get_output(alg)

    @_deprecate_positional_args
    def median_smooth(  # noqa: PLR0917
        self,
        kernel_size=(3, 3, 3),
        scalars=None,
        preference='point',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Smooth data using a median filter.

        The Median filter that replaces each pixel with the median value from a
        rectangular neighborhood around that pixel. Neighborhoods can be no
        more than 3 dimensional. Setting one axis of the neighborhood
        kernelSize to 1 changes the filter into a 2D median.

        See :vtk:`vtkImageMedian3D` for more details.

        Parameters
        ----------
        kernel_size : sequence[int], default: (3, 3, 3)
            Size of the kernel in each dimension (units of voxels), for example
            ``(x_size, y_size, z_size)``. Default is a 3D median filter. If you
            want to do a 2D median filter, set the size to 1 in the dimension
            you don't want to filter over.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, default: "point"
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Uniform grid with smoothed scalars.

        Warnings
        --------
        Applying this filter to cell data will send the output to a new point
        array with the same name, overwriting any existing point data array
        with the same name.

        Examples
        --------
        First, create sample data to smooth. Here, we use
        :func:`pyvista.perlin_noise() <pyvista.core.utilities.features.perlin_noise>`
        to create meaningful data.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (2, 5, 8), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, bounds=[0, 1, 0, 1, 0, 1], dim=(20, 20, 20)
        ... )
        >>> grid.plot(show_scalar_bar=False)

        Next, smooth the sample data.

        >>> smoothed = grid.median_smooth(kernel_size=(10, 10, 10))
        >>> smoothed.plot(show_scalar_bar=False)

        """
        alg = _vtk.vtkImageMedian3D()
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
        else:
            field = self.get_array_association(scalars, preference=preference)  # type: ignore[attr-defined]
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetKernelSize(kernel_size[0], kernel_size[1], kernel_size[2])
        _update_alg(alg, progress_bar=progress_bar, message='Performing Median Smoothing')
        return _get_output(alg)

    def slice_index(  # type: ignore[misc]
        self: ImageData,
        i: int | VectorLike[int] | slice | None = None,
        j: int | VectorLike[int] | slice | None = None,
        k: int | VectorLike[int] | slice | None = None,
        *,
        index_mode: Literal['extent', 'dimensions'] = 'dimensions',
        strict_index: bool = False,
        rebase_coordinates: bool = False,
        progress_bar: bool = False,
    ) -> ImageData:
        """Extract a subset using IJK indices.

        This filter enables slicing :class:`~pyvista.ImageData` with Python-style indexing using
        IJK coordinates. It can be used to extract a single slice, multiple contiguous slices, or
        a volume of interest. Unlike other slicing filters, this filter returns
        :class:`~pyvista.ImageData`.

        .. note::
            Slicing by index is also possible using the "get index" operator ``[]``. See examples.

        .. versionadded::0.46

        Parameters
        ----------
        i, j, k : int | VectorLike[int] | slice, optional
            Indices to slice along the I, J, and K coordinate axes, respectively. Specify an
            integer for a single index, or two integers ``[start, stop)`` for a range of indices.

            .. note::

                Like regular Python slicing:

                - Half-open intervals are used, i.e. the ``start`` index is included in the range
                  but the ``stop`` index is not.
                - Negative indexing is supported.
                - An ``IndexError`` is raised when a single integer is specified as the index and
                  the index is out-of-bounds.
                - An ``IndexError`` is `not` raised when a range is specified as the index and
                  the index is out-of-bounds. This default can be overridden by setting
                  ``strict_index=True``.
                - A copy of the data is returned (modifying the sliced output does `not` affect
                  the input data).

        index_mode : 'extent' | 'dimensions', default: 'dimensions'
            Mode to use when determining the range of values to index from.

            - Use ``'dimensions'`` to index values in the range ``[0, dimensions - 1]``.
            - Use ``'extent'`` to index values based on the :class:`~pyvista.ImageData.extent`,
              i.e. ``[offset, offset + dimensions - 1]``.

            The main difference between these modes is the inclusion or exclusion of the
            :attr:`~pyvista.ImageData.offset`. ``dimensions`` is more pythonic and is how the
            object's data arrays themselves would be indexed, whereas ``'extent'`` respects VTK's
            definition of ``extent`` and considers the object's geometry.

        strict_index : bool, default: False
            Raise an ``IndexError`` if `any` of the indices are out of range. By default, an
            ``IndexError`` is only raised if a single integer index is out of range, but not when
            a range of indices are specified; set this to ``True`` to raise in error in both cases.

        rebase_coordinates : bool, default: False
            Rebase the coordinate reference of the extracted subset:

            - the :attr:`~pyvista.ImageData.origin` is set to the minimum bounds of the subset
            - the :attr:`~pyvista.ImageData.offset` is reset to ``(0, 0, 0)``

            The rebasing effectively applies a positive translation in world (XYZ) coordinates and
            a similar (i.e. inverse) negative translation in voxel (IJK) coordinates. As a result,
            the :attr:`~pyvista.DataSet.bounds` of the output are unchanged, but the coordinate
            reference frame is modified.

            Set this to ``False`` to leave the origin unmodified and keep the offset specified by
            the indexing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        ImageData
            Sliced mesh.

        See Also
        --------
        crop
        extract_subset
        :meth:`~pyvista.DataObjectFilters.slice`
        :meth:`~pyvista.DataObjectFilters.slice_implicit`
        :meth:`~pyvista.DataObjectFilters.slice_orthogonal`
        :meth:`~pyvista.DataObjectFilters.slice_along_axis`
        :meth:`~pyvista.DataObjectFilters.slice_along_line`

        Examples
        --------
        Create a :class:`~pyvista.ImageData` mesh and give it some point data.

        >>> import pyvista as pv
        >>> mesh = pv.ImageData(dimensions=(10, 10, 10))
        >>> mesh['data'] = range(mesh.n_points)

        Extract a single slice along the k-axis.

        >>> sliced = mesh.slice_index(k=5)
        >>> sliced.dimensions
        (10, 10, 1)

        Equivalently:

        >>> sliced2 = mesh[:, :, 5]
        >>> sliced == sliced2
        True

        Extract a volume of interest.

        >>> sliced = mesh.slice_index(i=[1, 3], j=[2, 5], k=[5, 10])
        >>> sliced.dimensions
        (2, 3, 5)

        Equivalently:

        >>> sliced2 = mesh[1:3, 2:5, 5:10]
        >>> sliced == sliced2
        True

        Use ``None`` to implicitly define the start and/or stop indices.

        >>> sliced = mesh.slice_index(i=[None, 3], j=[2, None], k=None)
        >>> sliced.dimensions
        (3, 8, 10)

        Equivalently:

        >>> sliced2 = mesh[:3, 2:, :]
        >>> sliced == sliced2
        True

        See :ref:`slice_example` for more examples using this filter.

        """

        def _set_default_start_and_stop(rng, default_start, default_stop):
            if isinstance(rng, slice):
                return rng
            out = (default_start, default_stop) if rng is None else np.asanyarray(rng).tolist()
            if isinstance(out, list) and len(out) >= 2:
                if out[0] is None:
                    out[0] = default_start
                if out[1] is None:
                    out[1] = default_stop
            return out

        if i is None and j is None and k is None:
            msg = 'No indices were provided for slicing.'
            raise TypeError(msg)

        lower = (0, 0, 0) if index_mode == 'dimensions' else self.offset
        indices = tuple(
            _set_default_start_and_stop(slc, low, dim)
            for slc, low, dim in zip((i, j, k), lower, self.dimensions)
        )
        voi = self._compute_voi_from_index(
            indices, index_mode=index_mode, strict_index=strict_index
        )
        return self.extract_subset(
            voi, rebase_coordinates=rebase_coordinates, progress_bar=progress_bar
        )

    @_deprecate_positional_args(allowed=['voi', 'rate'])
    def extract_subset(  # noqa: PLR0917
        self,
        voi,
        rate=(1, 1, 1),
        boundary: bool = False,  # noqa: FBT001, FBT002
        rebase_coordinates: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Select piece (e.g., volume of interest).

        To use this filter set the VOI ivar which are i-j-k min/max indices
        that specify a rectangular region in the data. (Note that these are
        0-offset.) You can also specify a sampling rate to subsample the
        data.

        Typical applications of this filter are to extract a slice from a
        volume for image processing, subsampling large volumes to reduce data
        size, or extracting regions of a volume with interesting data.

        Parameters
        ----------
        voi : sequence[int]
            Length 6 iterable of ints: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
            These bounds specify the volume of interest in i-j-k min/max
            indices.

        rate : sequence[int], default: (1, 1, 1)
            Length 3 iterable of ints: ``(xrate, yrate, zrate)``.

        boundary : bool, default: False
            Control whether to enforce that the "boundary" of the grid
            is output in the subsampling process. This only has effect
            when the rate in any direction is not equal to 1. When
            this is enabled, the subsampling will always include the
            boundary of the grid even though the sample rate is not an
            even multiple of the grid dimensions. By default this is
            disabled.

        rebase_coordinates : bool, default: True
            If ``True`` (default), reset the coordinate reference of the extracted subset:

            - the :attr:`~pyvista.ImageData.origin` is set to the minimum bounds of the subset
            - the :attr:`~pyvista.ImageData.offset` is reset to ``(0, 0, 0)``

            The rebasing effectively applies a positive translation in world (XYZ) coordinates and
            a similar (i.e. inverse) negative translation in voxel (IJK) coordinates. As a result,
            the :attr:`~pyvista.DataSet.bounds` of the output are unchanged, but the coordinate
            reference frame is modified.

            Set this to ``False`` to leave the origin unmodified and keep the offset specified by
            the ``voi`` parameter.

            .. versionadded:: 0.46

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            ImageData subset.

        See Also
        --------
        slice_index
        crop

        """
        alg = _vtk.vtkExtractVOI()
        alg.SetVOI(voi)
        alg.SetInputDataObject(self)
        alg.SetSampleRate(rate)
        alg.SetIncludeBoundary(boundary)
        _update_alg(alg, progress_bar=progress_bar, message='Extracting Subset')
        result = _get_output(alg)
        if rebase_coordinates:
            # Adjust for the confusing issue with the extents
            #   see https://gitlab.kitware.com/vtk/vtk/-/issues/17938
            result.origin = result.bounds[::2]
            result.offset = (0, 0, 0)
        return result

    @staticmethod
    def _clip_extent(extent: VectorLike[int], *, clip_to: VectorLike[int]) -> NumpyArray[int]:
        out = np.array(extent)
        for axis in range(3):
            min_ind = axis * 2
            max_ind = axis * 2 + 1

            out[min_ind] = np.max((clip_to[min_ind], extent[min_ind]))  # type: ignore[arg-type]
            out[max_ind] = np.min((clip_to[max_ind], extent[max_ind]))  # type: ignore[arg-type]
        return out

    def crop(  # type: ignore[misc]
        self: ImageData,
        *,
        factor: float | VectorLike[float] | None = None,
        margin: int | VectorLike[int] | None = None,
        offset: VectorLike[int] | None = None,
        dimensions: VectorLike[int] | None = None,
        extent: VectorLike[int] | None = None,
        normalized_bounds: VectorLike[float] | None = None,
        mask: str | ImageData | NumpyArray[float] | Literal[True] | None = None,
        padding: int | VectorLike[int] | None = None,
        background_value: float | None = None,
        keep_dimensions: bool = False,
        fill_value: float | VectorLike[float] | None = None,
        rebase_coordinates: bool = False,
        progress_bar: bool = False,
    ) -> ImageData:
        """Crop this image to remove points at its boundaries.

        This filter is useful for extracting a volume or region of interest. There are several ways
        to crop:

        #. Use ``factor`` to crop a portion of the image symmetrically.
        #. Use ``margin`` to remove points from the image border.
        #. Use ``dimensions`` (and optionally, ``offset``) to explicitly crop to the specified
           :attr:`~pyvista.ImageData.dimensions` and :attr:`~pyvista.ImageData.offset`.
        #. Use ``extent`` to explicitly crop to a specified :attr:`~pyvista.ImageData.extent`.
        #. Use ``normalized_bounds`` to crop a bounding box relative to the input size.
        #. Use ``mask``, ``padding``, and ``background_value`` to crop to this mesh using scalar
           values to define the cropping region.

        These methods are all independent, e.g. it is not possible to specify both ``factor`` and
        ``margin``.

        By default, the cropped output's :attr:`~pyvista.ImageData.dimensions` are typically less
        than the input's dimensions. Optionally, use ``keep_dimensions`` and ``fill_value`` to
        ensure the output dimensions always match the input.

        .. note::

            All cropping is performed using the image's ijk-indices, not physical xyz-bounds.

        .. versionadded:: 0.46

        Parameters
        ----------
        factor : float, optional
            Cropping factor in range ``[0.0, 1.0]`` which specifies the proportion of the image to
            keep along each axis. Use a single float for uniform cropping or a vector of three
            floats for cropping each xyz-axis independently. The crop is centered in the image.

        margin : int | VectorLike[int], optional
            Margin to remove from each side of each axis. Specify:

            - A single value to remove from all boundaries equally.
            - Two values, one for each ``(X, Y)`` axis, to remove margin from
              each axis independently.
            - Three values, one for each ``(X, Y, Z)`` axis, to remove margin from
              each axis independently.
            - Four values, one for each ``(-X, +X, -Y, +Y)`` boundary, to remove
              margin from each boundary independently.
            - Six values, one for each ``(-X, +X, -Y, +Y, -Z, +Z)`` boundary, to remove
              margin from each boundary independently.

        offset : VectorLike[int], optional
            Length-3 vector of integers specifying the :attr:`~pyvista.ImageData.offset` indices
            where the cropping region originates. If specified, then ``dimensions`` must also be
            provided.

        dimensions : VectorLike[int], optional
            Length-3 vector of integers specifying the :attr:`~pyvista.ImageData.dimensions` of
            the cropping region. ``offset`` may also be provided, but if it is not, the crop is
            centered in the image.

        extent : VectorLike[int], optional
            Length-6 vector of integers specifying the full :attr:`~pyvista.ImageData.extent` of
            the cropping region.

        normalized_bounds : VectorLike[float], optional
            Normalized bounds relative to the input. These are floats between ``0.0`` and ``1.0``
            that define a box relative to the input size. The input is cropped such that it fully
            fits within these bounds. Has the form ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        mask : str | ImageData | NumpyArray[float] | bool, optional
            Scalar values that define the cropping region. Set this option to:

            - a string denoting the name of scalars belonging to this mesh
            - ``True`` to use this mesh's default scalars
            - a separate image, in which case the other image's active scalars are used
            - a 1D or 2D (multi-component) array

            The length of the scalar array must equal the number of points.

            This mesh will be cropped to the bounds of the foreground values of the array, i.e.
            values that are not equal to the specified ``background_value``.

        padding : int | VectorLike[int], optional
            Padding to add to foreground region `before` cropping. Only valid when using a mask to
            crop the image. Specify:

            - A single value to pad all boundaries equally.
            - Two values, one for each ``(X, Y)`` axis, to apply symmetric padding to
              each axis independently.
            - Three values, one for each ``(X, Y, Z)`` axis, to apply symmetric padding
              to each axis independently.
            - Four values, one for each ``(-X, +X, -Y, +Y)`` boundary, to apply
              padding to each boundary independently.
            - Six values, one for each ``(-X, +X, -Y, +Y, -Z, +Z)`` boundary, to apply
              padding to each boundary independently.

            The specified value is the `maximum` padding that may be applied. If the padding
            extends beyond the actual extents of this mesh, it is clipped and does not extend
            outside the bounds of the image.

        background_value : float | VectorLike[float], optional
            Value or multi-component vector considered to be the background. Only valid when using
            a mask to crop the image.

        keep_dimensions : bool, default: False
            If ``True``, the cropped output is :meth:`padded <pad_image>` with ``fill_value`` to
            ensure the output dimensions match the input.

        fill_value : float | VectorLike[float], optional
            Value used when padding the cropped output if ``keep_dimensions`` is ``True``. May be
            a single float or a multi-component vector (e.g. RGB vector).

        rebase_coordinates : bool, default: False
            Rebase the coordinate reference of the cropped output:

            - the :attr:`~pyvista.ImageData.origin` is set to the minimum bounds of the subset
            - the :attr:`~pyvista.ImageData.offset` is reset to ``(0, 0, 0)``

            The rebasing effectively applies a positive translation in world (XYZ) coordinates and
            a similar (i.e. inverse) negative translation in voxel (IJK) coordinates. As a result,
            the :attr:`~pyvista.DataSet.bounds` of the output are unchanged, but the coordinate
            reference frame is modified.

            Set this to ``False`` to leave the origin unmodified and keep the offset used by the
            crop.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        ImageData
            Cropped image.

        See Also
        --------
        pad_image
            Add points to image boundaries. This is the inverse operation of the ``margin`` crop.

        resample
            Modify an image's dimensions and spacing.

        select_values
            Threshold-like filter which may be used to generate a mask for cropping.

        extract_subset
            Equivalent filter to ``crop(extent=voi, rebase_coordinates=True)``.

        :ref:`crop_labeled_example`
            Example cropping :class:`~pyvista.ImageData` using a segmentation mask.

        Examples
        --------
        Load a grayscale image.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> gray_image = examples.download_yinyang()
        >>> gray_image.dimensions
        (512, 342, 1)

        Define a custom plotting helper to show the image as pixel cells.

        >>> def image_plotter(image):
        ...     pixel_cells = image.points_to_cells()
        ...
        ...     pl = pv.Plotter()
        ...     pl.add_mesh(
        ...         pixel_cells,
        ...         cmap='gray',
        ...         clim=[0, 255],
        ...         lighting=False,
        ...         show_scalar_bar=False,
        ...     )
        ...     pl.view_xy()
        ...     pl.camera.tight()
        ...     return pl

        Plot the image for context.

        >>> image_plotter(gray_image).show()

        Crop the white border around the image using active scalars as a mask. Here we specify a
        background value of ``255`` to correspond to white pixels. If this was an RGB image, we
        could also specify ``(255, 255, 255)`` as the background value.

        >>> cropped = gray_image.crop(mask=True, background_value=255)
        >>> cropped.dimensions
        (237, 238, 1)
        >>> image_plotter(cropped).show()

        Use ``margin`` instead to remove 100 and 20 pixels from each side of the x- and y-axis,
        respectively.

        >>> cropped = gray_image.crop(margin=(100, 20))
        >>> cropped.dimensions
        (312, 302, 1)
        >>> image_plotter(cropped).show()

        Use ``offset`` to select a starting location for the crop (from the origin at the
        bottom-left corner) along with ``dimensions`` to define the crop size.

        >>> cropped = gray_image.crop(offset=(50, 20, 0), dimensions=(300, 200, 1))
        >>> cropped.dimensions
        (300, 200, 1)
        >>> image_plotter(cropped).show()

        Use ``extent`` directly instead of using ``dimensions`` and ``offset`` to yield the same
        result as above.

        >>> cropped = gray_image.crop(extent=(50, 349, 20, 219, 0, 0))
        >>> cropped.extent
        (50, 349, 20, 219, 0, 0)
        >>> image_plotter(cropped).show()

        Use ``factor`` to crop 40% of the image. This `keeps` 40% of the pixels along each axis,
        and `removes` 60% (i.e. 30% from each side).

        >>> cropped = gray_image.crop(factor=0.4)
        >>> cropped.dimensions
        (204, 136, 1)
        >>> image_plotter(cropped).show()

        Use ``normalized_bounds`` to crop from 40% to 80% of the image along the x-axis, and
        from 30% to 90% of the image along the y-axis.

        >>> cropped = gray_image.crop(normalized_bounds=[0.4, 0.8, 0.3, 0.9, 0.0, 1.0])
        >>> cropped.extent
        (205, 408, 103, 306, 0, 0)
        >>> image_plotter(cropped).show()

        """
        CORE_METHOD_KWARGS = dict(
            factor=factor,
            margin=margin,
            offset=offset,
            dimensions=dimensions,
            extent=extent,
            normalized_bounds=normalized_bounds,
            mask=mask,
        )
        SUPPORTING_KWARGS = dict(padding=padding, background_value=background_value)
        MUTUALLY_EXCLUSIVE_KWARGS = CORE_METHOD_KWARGS | SUPPORTING_KWARGS

        def _raise_error_kwargs_not_none(arg_name, also_exclude: Sequence[str] = ()):
            args_to_check = MUTUALLY_EXCLUSIVE_KWARGS.copy()
            for arg in [arg_name, *also_exclude]:
                args_to_check.pop(arg)

            for key, val in args_to_check.items():
                if val is not None:
                    msg = (
                        f'When cropping with {arg_name}, the following parameters cannot be set:\n'
                        f'{list(args_to_check.keys())}.\n'
                        f'Got: {key}={val}'
                    )
                    raise TypeError(msg)

        def _validate_scalars(mesh: ImageData, scalars: str | None = None):
            if scalars is None:
                field, scalars = set_default_active_scalars(mesh)
            else:
                field = mesh.get_array_association(scalars, preference='point')
            if field != FieldAssociation.POINT:
                msg = (
                    f"Scalars '{scalars}' must be associated with point data. "
                    f'Got {field.name.lower()} data instead.'
                )
                raise ValueError(msg)
            return field, scalars

        def _voi_from_mask(mask_: str | ImageData | NumpyArray[float] | bool):  # noqa: FBT001
            _raise_error_kwargs_not_none('mask', also_exclude=['background_value', 'padding'])
            # Validate scalars
            if isinstance(mask_, (str, bool)):
                mesh = self
                if isinstance(mask_, str):
                    scalars = mask_
                elif mask_ is True:
                    scalars = None
                else:
                    msg = 'mask cannot be `False`.'
                    raise ValueError(msg)
            elif isinstance(mask_, pyvista.ImageData):
                mesh = mask_
                scalars = None
            else:
                mesh = pyvista.ImageData(dimensions=self.dimensions, offset=self.offset)
                scalars = 'scalars'
                mesh[scalars] = mask_

            field, scalars_ = _validate_scalars(mesh, scalars)
            array = cast(
                'pyvista.pyvista_ndarray', get_array(mesh, name=scalars_, preference=field)
            )
            num_components = 1 if array.ndim == 1 else array.shape[1]

            # Create a binary foreground/background mask array
            default_background = 0.0
            background = default_background if background_value is None else background_value
            if num_components > 1:
                background = _validation.validate_arrayN(
                    background, name='background_value', must_have_length=(1, num_components)
                )
                mask_array = np.any(array != background, axis=1)
            else:
                background = _validation.validate_number(background, name='background_value')
                mask_array = array != background

            # Get foreground voi
            shaped_array = mask_array.reshape(mesh.dimensions[::-1])
            coords = np.argwhere(shaped_array)
            if coords.size == 0:
                msg = (
                    f'Crop with mask failed, no foreground values found in array '
                    f'{scalars_!r} using background value {background}.'
                )
                raise ValueError(msg)

            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0)
            voi = xmin, xmax, ymin, ymax, zmin, zmax

            if padding is not None:
                pad = _validate_padding(padding)
                voi = _pad_extent(voi, pad)

            # Add offset
            voi_array = np.array(voi)
            voi_array[[0, 1]] += mesh.offset[0]
            voi_array[[2, 3]] += mesh.offset[1]
            voi_array[[4, 5]] += mesh.offset[2]

            # Clip voi so it doesn't extend beyond the image's extent
            return ImageDataFilters._clip_extent(voi_array, clip_to=self.extent)

        def _voi_from_normalized_bounds(normalized_bounds_):
            _raise_error_kwargs_not_none('normalized_bounds')
            bounds = _validation.validate_arrayN(
                normalized_bounds_,
                must_have_dtype=float,
                must_be_in_range=[0.0, 1.0],
                must_have_length=6,
                name='normalized_bounds',
            )
            # Compute IJK bounds from normalized bounds
            # A small bias is added to account for numerical error
            # e.g. we want floor(0.29999999 * 10 + eps) = 3, not 2
            dims = np.array(self.dimensions)
            eps = 1e-6
            norm_starts = bounds[::2]
            norm_stops = bounds[1::2]
            starts = np.ceil(norm_starts * dims - eps).astype(int)
            stops = np.floor(norm_stops * dims + eps).astype(int) - 1

            # Ensure dimensions are not set to 0
            stops = np.maximum(stops, starts)

            # Apply image offset
            offset = self.offset
            xmin, ymin, zmin = starts + offset
            xmax, ymax, zmax = stops + offset

            return xmin, xmax, ymin, ymax, zmin, zmax

        def _voi_from_extent(extent_):
            _raise_error_kwargs_not_none('extent')
            return _validation.validate_arrayN(
                extent_,
                must_be_integer=True,
                must_have_length=6,
                dtype_out=int,
                name='extent',
            )

        def _voi_from_factor(factor_):
            _raise_error_kwargs_not_none('factor')
            valid_factor = _validation.validate_array3(
                factor_,
                broadcast=True,
                must_be_in_range=[0.0, 1.0],
                dtype_out=float,
                name='crop factor',
            )

            scale_dims = valid_factor * np.array(self.dimensions)
            new_dimensions = np.floor(scale_dims).astype(int)
            new_dimensions = np.maximum(new_dimensions, 1)  # avoid zero

            # Center of the current image in ijk coordinates
            center = self.offset + ((np.array(self.dimensions) - 1) // 2)
            # Offset to center the new cropped region around the original center
            new_offset = center - ((new_dimensions - 1) // 2)

            return pyvista.ImageData(dimensions=new_dimensions, offset=new_offset).extent

        def _voi_from_dimensions(dimensions_):
            valid_dims = _validation.validate_array3(
                dimensions_,
                broadcast=True,
                must_have_dtype=np.integer,
                must_be_in_range=[1, np.inf],
                dtype_out=int,
                name='dimensions',
            )

            new_dimensions = np.minimum(valid_dims, np.array(self.dimensions))

            # Center of the current image in ijk coordinates
            center = self.offset + ((np.array(self.dimensions) - 1) // 2)

            # Compute offset to center the new cropped region
            # When the difference is odd, place the extra point on the min side
            half_size = (new_dimensions - 1) // 2
            new_offset = center - half_size

            return pyvista.ImageData(dimensions=new_dimensions, offset=new_offset).extent

        def _voi_from_margin(margin_):
            _raise_error_kwargs_not_none('margin')
            padding = _validate_padding(margin_)
            # Do not pad singleton dims
            singleton_dims = np.array(self.dimensions) == 1
            mask = [x for pair in zip(singleton_dims, singleton_dims) for x in pair]
            padding[mask] = np.array(self.extent)[mask]
            return _pad_extent(self.extent, -padding)

        def _voi_from_dimensions_or_offset(dimensions_, offset_):
            _raise_error_kwargs_not_none('dimensions', also_exclude=['offset'])
            if dimensions_ is None:
                msg = 'Dimensions must also be specified when cropping with offset.'
                raise TypeError(msg)
            elif offset_ is None:
                return _voi_from_dimensions(dimensions_)
            else:
                return pyvista.ImageData(dimensions=dimensions, offset=offset).extent

        if factor is not None:
            voi = _voi_from_factor(factor)
        elif margin is not None:
            voi = _voi_from_margin(margin)
        elif mask is not None:
            voi = _voi_from_mask(mask)
        elif normalized_bounds is not None:
            voi = _voi_from_normalized_bounds(normalized_bounds)
        elif extent is not None:
            voi = _voi_from_extent(extent)
        elif dimensions is not None or offset is not None:
            voi = _voi_from_dimensions_or_offset(dimensions, offset)
        else:
            msg = (
                'No crop arguments provided. One of the following keywords must be provided:\n'
                f'{list(CORE_METHOD_KWARGS.keys())}'
            )
            raise TypeError(msg)

        # Ensure dimensions are all at least one
        voi = np.array(voi)
        voi[1] = max(voi[0:2])
        voi[3] = max(voi[2:4])
        voi[5] = max(voi[4:6])

        cropped = self.extract_subset(
            voi, rebase_coordinates=rebase_coordinates, progress_bar=progress_bar
        )
        if not keep_dimensions:
            return cropped

        # Compute padding required to make extents match the input
        off_before = self.offset
        off_after = cropped.offset
        dims_before = self.dimensions
        dims_after = cropped.dimensions
        padding = [
            off_after[0] - off_before[0],
            (off_before[0] + dims_before[0]) - (off_after[0] + dims_after[0]),
            off_after[1] - off_before[1],
            (off_before[1] + dims_before[1]) - (off_after[1] + dims_after[1]),
            off_after[2] - off_before[2],
            (off_before[2] + dims_before[2]) - (off_after[2] + dims_after[2]),
        ]

        # Pad the output
        fill = fill_value if fill_value is not None else 0
        result = cropped.pad_image(
            pad_value=fill,
            pad_size=padding,
            pad_all_scalars=True,
            dimensionality=self.dimensionality,
            progress_bar=progress_bar,
        )

        # The pad filter removes cell data, so copy it unchanged from input
        result.cell_data.update(self.cell_data)
        return result

    @_deprecate_positional_args(allowed=['dilate_value', 'erode_value'])
    def image_dilate_erode(  # noqa: PLR0917
        self,
        dilate_value=1.0,
        erode_value=0.0,
        kernel_size=(3, 3, 3),
        scalars=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Dilates one value and erodes another.

        .. deprecated:: 0.47.0
            :meth:`image_dilate_erode` is deprecated. Use :meth:`dilate`, :meth:`erode`,
            :meth:`open`, or :meth:`close` instead.

        ``image_dilate_erode`` will dilate one value and erode another. It uses
        an elliptical footprint, and only erodes/dilates on the boundary of the
        two values. The filter is restricted to the X, Y, and Z axes for now.
        It can degenerate to a 2 or 1-dimensional filter by setting the kernel
        size to 1 for a specific axis.

        Parameters
        ----------
        dilate_value : float, default: 1.0
            Dilate value in the dataset.

        erode_value : float, default: 0.0
            Erode value in the dataset.

        kernel_size : sequence[int], default: (3, 3, 3)
            Determines the size of the kernel along the three axes.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset that has been dilated/eroded on the boundary of the specified scalars.

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with
        :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`.

        Examples
        --------
        Demonstrate image dilate/erode on an example dataset. First, plot
        the example dataset with the active scalars.

        >>> from pyvista import examples
        >>> uni = examples.load_uniform()
        >>> uni.plot()

        Now, plot the image threshold with ``threshold=[400, 600]``. Note how
        values within the threshold are 1 and outside are 0.

        >>> ithresh = uni.image_threshold([400, 600])
        >>> ithresh.plot()

        Note how there is a hole in the thresholded image. Apply a closing
        filter with a large kernel to fill that hole in.

        >>> iclosed = ithresh.close(kernel_size=[5, 5, 5])
        >>> iclosed.plot()

        """
        warnings.warn(
            'image_dilate_erode is deprecated. Use dilate, erode, open, or close instead.',
            PyVistaDeprecationWarning,
            stacklevel=2,
        )

        alg = _vtk.vtkImageDilateErode3D()
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
            if field.value == 1:
                msg = 'If `scalars` not given, active scalars must be point array.'
                raise ValueError(msg)
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
            if field.value == 1:
                msg = 'Can only process point data, given `scalars` are cell data.'
                raise ValueError(msg)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetKernelSize(*kernel_size)
        alg.SetDilateValue(dilate_value)
        alg.SetErodeValue(erode_value)
        _update_alg(alg, progress_bar=progress_bar, message='Performing Dilation and Erosion')
        return _get_output(alg)

    def _get_binary_values(  # type: ignore[misc]
        self: ImageData,
        scalars: str,
        association: Literal[FieldAssociation.POINT],
        *,
        binary: bool | VectorLike[float] | None,
    ) -> tuple[float, float] | None:
        if binary is None:
            # Value is unset, so check if the scalars are actually binary
            array = self.get_array(scalars, association)
            min_val, max_val = self.get_data_range(array)
            # Binary if bool or two adjacent integers or two unique values
            # We rely on short-circuit evaluation to avoid the np.unique call unless necessary
            if array.dtype == np.bool_ or (
                np.issubdtype(array.dtype, np.integer) and (max_val - min_val) == 1
            ):
                return min_val, max_val
            else:
                unique = np.unique(array)
                if unique.size in [1, 2]:
                    return unique.min(), unique.max()
            return None  # Scalars are not binary

        elif binary is True:
            # Use the range to set the values
            return self.get_data_range(scalars, association)
        elif binary is False:
            # Do not return any values
            return None
        else:
            # Binary values are set explicitly
            return _validation.validate_data_range(binary, name='binary values')

    def _configure_dilate_erode_alg(  # type: ignore[misc]
        self: ImageData,
        *,
        kernel_size: int | VectorLike[int],
        scalars: str,
        association: Literal[FieldAssociation.POINT],
        binary_values: tuple[float, float] | None,
        operation: Literal['dilation', 'erosion'],
    ) -> (
        _vtk.vtkImageContinuousErode3D
        | _vtk.vtkImageContinuousDilate3D
        | _vtk.vtkImageDilateErode3D
    ):
        alg: (
            _vtk.vtkImageContinuousErode3D
            | _vtk.vtkImageContinuousDilate3D
            | _vtk.vtkImageDilateErode3D
        )

        if binary_values is not None:
            background_val, foreground_val = binary_values
            if operation == 'dilation':
                dilate_value = foreground_val
                erode_value = background_val
            else:
                dilate_value = background_val
                erode_value = foreground_val

            alg = _vtk.vtkImageDilateErode3D()
            alg.SetDilateValue(dilate_value)
            alg.SetErodeValue(erode_value)
        else:
            alg = (
                _vtk.vtkImageContinuousDilate3D()
                if operation == 'dilation'
                else _vtk.vtkImageContinuousErode3D()
            )

        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            association.value,
            scalars,
        )

        kernal_sz = _validation.validate_array3(kernel_size, broadcast=True, name='kernel_size')
        alg.SetKernelSize(*kernal_sz)
        return alg

    def _get_alg_output_from_input(self, alg, *, progress_bar: bool, operation: str):
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar=progress_bar, message=f'Performing {operation}')
        return _get_output(alg)

    def _validate_point_scalars(  # type: ignore[misc]
        self: ImageData, scalars: str | None = None
    ) -> tuple[Literal[FieldAssociation.POINT], str]:
        if scalars is None:
            field, scalars = set_default_active_scalars(self)
            if field == FieldAssociation.CELL:
                msg = 'If `scalars` not given, active scalars must be point array.'
                raise ValueError(msg)
        else:
            field = self.get_array_association(scalars, preference='point')
            if field == FieldAssociation.CELL:
                msg = 'Can only process point data, given `scalars` are cell data.'
                raise ValueError(msg)
        return cast('Literal[FieldAssociation.POINT]', field), scalars

    def dilate(  # type: ignore[misc]
        self: ImageData,
        kernel_size: int | VectorLike[int] = (3, 3, 3),
        scalars: str | None = None,
        *,
        binary: bool | VectorLike[float] | None = None,
        progress_bar: bool = False,
    ):
        """Morphologically dilate grayscale or binary data.

        This filter may be used to dilate grayscale images with continuous data, binary images
        with a single background and foreground value, or multi-label images.

        For binary inputs with two unique values, this filter uses :vtk:`vtkImageDilateErode3D`
        by default to perform fast binary dilation over an ellipsoidal neighborhood. Otherwise,
        the slower class :vtk:`vtkImageContinuousDilate3D` is used to perform generalized grayscale
        dilation by replacing each pixel with the maximum over an ellipsoidal neighborhood.

        Optionally, the ``binary`` keyword may be used to explicitly control the behavior of the
        filter.

        .. versionadded:: 0.47

        Parameters
        ----------
        kernel_size : int | VectorLike[int], default: (3, 3, 3)
            Determines the size of the kernel along the xyz-axes. Only non-singleton dimensions
            are dilated, e.g. a kernel size of ``(3, 3, 1)`` and ``(3, 3, 3)`` produce the same
            result for 2D images.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        binary : bool | VectorLike[float], optional
            Control if binary dilation or continuous dilation is used.

            If set, :vtk:`vtkImageDilateErode3D` is used to strictly dilate with two values.
            Set this to ``True`` to dilate the maximum value in ``scalars`` with its minimum value,
            or set it to two values ``[background_value, foreground_value]`` to dilate
            ``foreground_value`` with ``background_value`` explicitly.

            Set this to ``False`` to use :vtk:`vtkImageContinuousDilate3D` to perform continuous
            dilation.

            By default, ``binary`` is ``True`` if the input has two unique values, and ``False``
            otherwise.

            .. note::
                - If the input is a binary mask, setting ``binary=True`` produces the same output
                  as ``binary=False``, but the filter is much more performant.
                - Setting ``binary=[background_value, foreground_value]`` is useful to `isolate`
                  the dilation to two values, e.g. for multi-label segmentation masks.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset that has been dilated.

        See Also
        --------
        erode, open, close

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with
        :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`.

        Examples
        --------
        .. pyvista-plot::
            :force_static:

            Create a toy example with two non-zero grayscale foreground values
            :meth:`padded <pad_image>` with a background of zeros.

            >>> import pyvista as pv
            >>> im = pv.ImageData(dimensions=(2, 1, 1))
            >>> im['data'] = [128, 255]
            >>> im = im.pad_image(pad_value=0, pad_size=2, dimensionality=2)

            Define a custom plotter to plot pixels as cells.

            >>> def image_plotter(image):
            ...     pl = pv.Plotter()
            ...     pl.add_mesh(
            ...         image.points_to_cells(),
            ...         cmap='grey',
            ...         clim=[0, 255],
            ...         show_scalar_bar=False,
            ...         show_edges=True,
            ...         lighting=False,
            ...         line_width=3,
            ...     )
            ...     pl.camera_position = 'xy'
            ...     pl.camera.tight()
            ...     pl.enable_anti_aliasing()
            ...     return pl

            Show the image.

            >>> image_plotter(im).show()

            Dilate it with default settings. Observe that `both` foreground values are dilated.

            >>> dilated = im.dilate()
            >>> image_plotter(dilated).show()

            Use a larger kernel size.

            >>> dilated = im.dilate(kernel_size=5)
            >>> image_plotter(dilated).show()

            Use an asymmetric kernel.

            >>> dilated = im.dilate(kernel_size=(2, 4, 1))
            >>> image_plotter(dilated).show()

            Use binary dilation. By default, the max value (``255`` in this example) is dilated
            with the min value (``0`` in this example). All other values are unaffected.

            >>> dilated = im.dilate(binary=True)
            >>> image_plotter(dilated).show()

            Equivalently, set the binary values for the dilation explicitly.

            >>> dilated = im.dilate(binary=[0, 255])
            >>> image_plotter(dilated).show()

            Use binary dilation with the other foreground value instead.

            >>> dilated = im.dilate(binary=[0, 128])
            >>> image_plotter(dilated).show()

        """
        association, scalars = self._validate_point_scalars(scalars)
        binary_values = self._get_binary_values(scalars, association, binary=binary)
        operation: Literal['dilation'] = 'dilation'
        alg = self._configure_dilate_erode_alg(
            kernel_size=kernel_size,
            scalars=scalars,
            association=association,
            binary_values=binary_values,
            operation=operation,
        )
        return self._get_alg_output_from_input(alg, progress_bar=progress_bar, operation=operation)

    def erode(  # type: ignore[misc]
        self: ImageData,
        kernel_size: int | VectorLike[int] = (3, 3, 3),
        scalars: str | None = None,
        *,
        binary: bool | VectorLike[float] | None = None,
        progress_bar: bool = False,
    ):
        """Morphologically erode grayscale or binary data.

        This filter may be used to erode grayscale images with continuous data, binary images
        with a single background and foreground value, or multi-label images.

        For binary inputs with two unique values, this filter uses :vtk:`vtkImageDilateErode3D`
        by default to perform fast binary erosion over an ellipsoidal neighborhood. Otherwise,
        the slower class :vtk:`vtkImageContinuousErode3D` is used to perform generalized grayscale
        erosion by replacing each pixel with the minimum over an ellipsoidal neighborhood.

        Optionally, the ``binary`` keyword may be used to explicitly control the behavior of the
        filter.

        .. versionadded:: 0.47

        Parameters
        ----------
        kernel_size : int | VectorLike[int], default: (3, 3, 3)
            Determines the size of the kernel along the xyz-axes. Only non-singleton dimensions
            are eroded, e.g. a kernel size of ``(3, 3, 1)`` and ``(3, 3, 3)`` produce the same
            result for 2D images.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        binary : bool | VectorLike[float], optional
            Control if binary erosion or continuous erosion is used.

            If set, :vtk:`vtkImageDilateErode3D` is used to strictly erode with two values.
            Set this to ``True`` to erode the maximum value in ``scalars`` with its minimum value,
            or set it to two values ``[background_value, foreground_value]`` to erode
            ``foreground_value`` with ``background_value`` explicitly.

            Set this to ``False`` to use :vtk:`vtkImageContinuousErode3D` to perform continuous
            erosion.

            By default, ``binary`` is ``True`` if the input has two unique values, and ``False``
            otherwise.

            .. note::
                - If the input is a binary mask, setting ``binary=True`` produces the same output
                  as ``binary=False``, but the filter is much more performant.
                - Setting ``binary=[background_value, foreground_value]`` is useful to `isolate`
                  the erosion to two values, e.g. for multi-label segmentation masks.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset that has been eroded.

        See Also
        --------
        dilate, open, close

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with
        :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`.

        Examples
        --------
        .. pyvista-plot::
            :force_static:

            Create a toy example with two non-zero grayscale foreground regions
            using :meth:`pad_image`.

            >>> import pyvista as pv
            >>> # Create an initial background point
            >>> im = pv.ImageData(dimensions=(1, 1, 1))
            >>> im['data'] = [0]
            >>> # Add a foreground region
            >>> im = im.pad_image(pad_value=255, pad_size=(4, 3), dimensionality=2)
            >>> # Add a second foreground region
            >>> im = im.pad_image(pad_value=128, pad_size=(2, 0, 1, 0))
            >>> # Add background values to two sides
            >>> im = im.pad_image(pad_value=0, pad_size=(1, 0))

            Define a custom plotter to plot pixels as cells.

            >>> def image_plotter(image):
            ...     pl = pv.Plotter()
            ...     pl.add_mesh(
            ...         image.points_to_cells(),
            ...         cmap='grey',
            ...         clim=[0, 255],
            ...         show_scalar_bar=False,
            ...         show_edges=True,
            ...         lighting=False,
            ...         line_width=3,
            ...     )
            ...     pl.camera_position = 'xy'
            ...     pl.camera.tight()
            ...     pl.enable_anti_aliasing()
            ...     return pl

            Show the image.

            >>> image_plotter(im).show()

            Erode it with default settings. Observe that `both` foreground values are eroded.

            >>> eroded = im.erode()
            >>> image_plotter(eroded).show()

            Use a larger kernel size.

            >>> eroded = im.erode(kernel_size=5)
            >>> image_plotter(eroded).show()

            Use an asymmetric kernel.

            >>> eroded = im.erode(kernel_size=(2, 4, 1))
            >>> image_plotter(eroded).show()

            Use binary erosion. By default, the max value (``255`` in this example) is eroded
            with the min value (``0`` in this example). All other values are unaffected.

            >>> eroded = im.erode(binary=True)
            >>> image_plotter(eroded).show()

            Equivalently, set the binary values for the erosion explicitly.

            >>> eroded = im.erode(binary=[0, 255])
            >>> image_plotter(eroded).show()

            Use binary erosion with the other foreground value instead.

            >>> eroded = im.erode(binary=[0, 128])
            >>> image_plotter(eroded).show()

        """
        association, scalars = self._validate_point_scalars(scalars)
        binary_values = self._get_binary_values(scalars, association, binary=binary)
        operation: Literal['erosion'] = 'erosion'
        alg = self._configure_dilate_erode_alg(
            kernel_size=kernel_size,
            scalars=scalars,
            association=association,
            binary_values=binary_values,
            operation=operation,
        )
        return self._get_alg_output_from_input(alg, progress_bar=progress_bar, operation=operation)

    def open(  # type: ignore[misc]
        self: ImageData,
        kernel_size: int | VectorLike[int] = (3, 3, 3),
        scalars: str | None = None,
        *,
        binary: bool | VectorLike[float] | None = None,
        progress_bar: bool = False,
    ):
        """Perform morphological opening on continuous or binary data.

        Opening is an :meth:`erosion <erode>` followed by a :meth:`dilation <dilate>`.
        It is used to remove small objects/noise while preserving the shape and size of larger
        objects.

        .. versionadded:: 0.47

        Parameters
        ----------
        kernel_size : int | VectorLike[int], default: (3, 3, 3)
            Determines the size of the kernel along the xyz-axes. Only non-singleton dimensions
            are opened, e.g. a kernel size of ``(3, 3, 1)`` and ``(3, 3, 3)`` produce the same
            result for 2D images.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        binary : bool | VectorLike[float], optional
            Control if binary opening or continuous opening is used. Refer to
            :meth:`erode` and/or :meth:`dilate` for details about using this keyword.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset that has been opened.

        See Also
        --------
        close, erode, dilate

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with
        :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`.

        Examples
        --------
        Load a grayscale image :func:`~pyvista.examples.downloads.download_chest()` and show it
        for context.

        >>> from pyvista import examples
        >>> im = examples.download_chest()
        >>> clim = im.get_data_range()
        >>> kwargs = dict(
        ...     cmap='grey',
        ...     clim=clim,
        ...     lighting=False,
        ...     cpos='xy',
        ...     zoom='tight',
        ...     show_axes=False,
        ...     show_scalar_bar=False,
        ... )
        >>> im.plot(**kwargs)

        Use ``open`` to remove small objects in the lungs.

        >>> opened = im.open(kernel_size=15)
        >>> opened.plot(**kwargs)

        """
        # Opening: erosion followed by dilation
        # Note: we need to configure both algorithms before getting the output since
        # the selected erosion/dilation values may be affected by the alg update
        association, scalars = self._validate_point_scalars(scalars)
        binary_values = self._get_binary_values(scalars, association, binary=binary)

        erosion: Literal['erosion'] = 'erosion'
        erosion_alg = self._configure_dilate_erode_alg(
            kernel_size=kernel_size,
            scalars=scalars,
            association=association,
            binary_values=binary_values,
            operation=erosion,
        )

        dilation: Literal['dilation'] = 'dilation'
        dilation_alg = self._configure_dilate_erode_alg(
            kernel_size=kernel_size,
            scalars=scalars,
            association=association,
            binary_values=binary_values,
            operation=dilation,
        )

        # Get filter outputs: erode then dilate
        erosion_output = self._get_alg_output_from_input(
            erosion_alg, progress_bar=progress_bar, operation=erosion
        )
        return erosion_output._get_alg_output_from_input(
            dilation_alg, progress_bar=progress_bar, operation=dilation
        )

    def close(  # type: ignore[misc]
        self: ImageData,
        kernel_size: int | VectorLike[int] = (3, 3, 3),
        scalars: str | None = None,
        *,
        binary: bool | VectorLike[float] | None = None,
        progress_bar: bool = False,
    ):
        """Perform morphological closing on continuous or binary data.

        Closing is a :meth:`dilation <dilate>` followed by an :meth:`erosion <erode>`.
        It is used to fill small holes/gaps while preserving the shape and size of larger objects.

        .. versionadded:: 0.47

        Parameters
        ----------
        kernel_size : int | VectorLike[int], default: (3, 3, 3)
            Determines the size of the kernel along the xyz-axes. Only non-singleton dimensions
            are closed, e.g. a kernel size of ``(3, 3, 1)`` and ``(3, 3, 3)`` produce the same
            result for 2D images.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        binary : bool | VectorLike[float], optional
            Control if binary closing or continuous closing is used. Refer to
            :meth:`dilate` and/or :meth:`erode` for details about using this keyword.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset that has been closed.

        See Also
        --------
        open, erode, dilate

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with
        :func:`~pyvista.DataObjectFilters.cell_data_to_point_data`.

        Examples
        --------
        Load a binary image: :func:`~pyvista.examples.downloads.download_yinyang()`.

        >>> from pyvista import examples
        >>> im = examples.download_yinyang()

        Use ``close`` with a relatively small kernel to fill the top black edge of the yinyang.

        >>> closed = im.close(kernel_size=5)
        >>> kwargs = dict(
        ...     cmap='grey',
        ...     lighting=False,
        ...     cpos='xy',
        ...     zoom='tight',
        ...     show_axes=False,
        ...     show_scalar_bar=False,
        ... )
        >>> closed.plot(**kwargs)

        Use a much larger kernel to also fill the small black circle.

        >>> closed = im.close(kernel_size=25)
        >>> closed.plot(**kwargs)

        Since closing is the inverse of opening, we can alternatively use :meth:`open` to
        fill the white foreground values instead of the black background.

        >>> opened = im.open(kernel_size=25)
        >>> opened.plot(**kwargs)

        """
        # Closing: dilation followed by erosion
        # Note: we need to configure both algorithms before getting the output since
        # the selected erosion/dilation values may be affected by the alg update
        association, scalars = self._validate_point_scalars(scalars)
        binary_values = self._get_binary_values(scalars, association, binary=binary)

        dilation: Literal['dilation'] = 'dilation'
        dilation_alg = self._configure_dilate_erode_alg(
            kernel_size=kernel_size,
            scalars=scalars,
            association=association,
            binary_values=binary_values,
            operation=dilation,
        )

        erosion: Literal['erosion'] = 'erosion'
        erosion_alg = self._configure_dilate_erode_alg(
            kernel_size=kernel_size,
            scalars=scalars,
            association=association,
            binary_values=binary_values,
            operation=erosion,
        )

        # Get filter outputs: dilate then erode
        dilation_output = self._get_alg_output_from_input(
            dilation_alg, progress_bar=progress_bar, operation=erosion
        )
        return dilation_output._get_alg_output_from_input(
            erosion_alg, progress_bar=progress_bar, operation=dilation
        )

    @_deprecate_positional_args(allowed=['threshold'])
    def image_threshold(  # noqa: PLR0917
        self,
        threshold,
        in_value=1.0,
        out_value=0.0,
        scalars=None,
        preference='point',
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Apply a threshold to scalar values in a uniform grid.

        If a single value is given for threshold, scalar values above or equal
        to the threshold are ``'in'`` and scalar values below the threshold are ``'out'``.
        If two values are given for threshold (sequence) then values equal to
        or between the two values are ``'in'`` and values outside the range are ``'out'``.

        If ``None`` is given for ``in_value``, scalars that are ``'in'`` will not be replaced.
        If ``None`` is given for ``out_value``, scalars that are ``'out'`` will not be replaced.

        Warning: applying this filter to cell data will send the output to a
        new point array with the same name, overwriting any existing point data
        array with the same name.

        Parameters
        ----------
        threshold : float or sequence[float]
            Single value or (min, max) to be used for the data threshold.  If
            a sequence, then length must be 2. Threshold(s) for deciding which
            cells/points are ``'in'`` or ``'out'`` based on scalar data.

        in_value : float, default: 1.0
            Scalars that match the threshold criteria for ``'in'`` will be replaced with this.

        out_value : float, default: 0.0
            Scalars that match the threshold criteria for ``'out'`` will be replaced with this.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, default: "point"
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset with the specified scalars thresholded.

        See Also
        --------
        select_values
            Threshold-like method for keeping some values and replacing others.
        :meth:`~pyvista.DataSetFilters.threshold`
            General threshold method that returns a :class:`~pyvista.UnstructuredGrid`.

        Examples
        --------
        Demonstrate image threshold on an example dataset. First, plot
        the example dataset with the active scalars.

        >>> from pyvista import examples
        >>> uni = examples.load_uniform()
        >>> uni.plot()

        Now, plot the image threshold with ``threshold=100``. Note how
        values above the threshold are 1 and below are 0.

        >>> ithresh = uni.image_threshold(100)
        >>> ithresh.plot()

        See :ref:`image_representations_example` for more examples using this filter.

        """
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
        else:
            field = self.get_array_association(scalars, preference=preference)  # type: ignore[attr-defined]

        # For some systems integer scalars won't threshold
        # correctly. Cast to float to be robust.
        cast_dtype = np.issubdtype(
            array_dtype := self.active_scalars.dtype,  # type: ignore[attr-defined]
            int,
        ) and array_dtype != np.dtype(np.uint8)
        if cast_dtype:
            self[scalars] = self[scalars].astype(float, casting='safe')  # type: ignore[index]

        alg = _vtk.vtkImageThreshold()
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        # set the threshold(s) and mode
        threshold_val = np.atleast_1d(threshold)
        if (size := threshold_val.size) not in (1, 2):
            msg = f'Threshold must have one or two values, got {size}.'
            raise ValueError(msg)
        if size == 2:
            alg.ThresholdBetween(threshold_val[0], threshold_val[1])
        else:
            alg.ThresholdByUpper(threshold_val[0])
        # set the replacement values / modes
        if in_value is not None:
            alg.SetReplaceIn(True)
            alg.SetInValue(np.array(in_value).astype(array_dtype))  # type: ignore[arg-type]
        else:
            alg.SetReplaceIn(False)
        if out_value is not None:
            alg.SetReplaceOut(True)
            alg.SetOutValue(np.array(out_value).astype(array_dtype))  # type: ignore[arg-type]
        else:
            alg.SetReplaceOut(False)
        # run the algorithm
        _update_alg(alg, progress_bar=progress_bar, message='Performing Image Thresholding')
        output = _get_output(alg)
        if cast_dtype:
            self[scalars] = self[scalars].astype(array_dtype)  # type: ignore[index]
            output[scalars] = output[scalars].astype(array_dtype)
        return output

    @_deprecate_positional_args
    def fft(self, output_scalars_name=None, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Apply a fast Fourier transform (FFT) to the active scalars.

        The input can be real or complex data, but the output is always
        :attr:`numpy.complex128`. The filter is fastest for images that have
        power of two sizes.

        The filter uses a butterfly diagram for each prime factor of the
        dimension. This makes images with prime number dimensions (i.e. 17x17)
        much slower to compute. FFTs of multidimensional meshes (i.e volumes)
        are decomposed so that each axis executes serially.

        The frequencies of the output assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with applied FFT.

        See Also
        --------
        rfft : The reverse transform.
        low_pass : Low-pass filtering of FFT output.
        high_pass : High-pass filtering of FFT output.

        Examples
        --------
        Apply FFT to an example image.

        >>> from pyvista import examples
        >>> image = examples.download_moonlanding_image()
        >>> fft_image = image.fft()
        >>> fft_image.point_data  # doctest:+SKIP
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : PNGImage
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
        PNGImage                complex128 (298620,)          SCALARS

        See :ref:`image_fft_example` for a full example using this filter.

        """
        # check for active scalars, otherwise risk of segfault
        if self.point_data.active_scalars_name is None:  # type: ignore[attr-defined]
            try:
                set_default_active_scalars(self)  # type: ignore[arg-type]
            except MissingDataError:
                msg = 'FFT filter requires point scalars.'
                raise MissingDataError(msg) from None

            # possible only cell scalars were made active
            if self.point_data.active_scalars_name is None:  # type: ignore[attr-defined]
                msg = 'FFT filter requires point scalars.'
                raise MissingDataError(msg)

        alg = _vtk.vtkImageFFT()
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar=progress_bar, message='Performing Fast Fourier Transform')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,  # type: ignore[attr-defined]
            output_scalars_name,
        )
        return output

    @_deprecate_positional_args
    def rfft(self, output_scalars_name=None, progress_bar: bool = False):  # noqa: FBT001, FBT002
        """Apply a reverse fast Fourier transform (RFFT) to the active scalars.

        The input can be real or complex data, but the output is always
        :attr:`numpy.complex128`. The filter is fastest for images that have power
        of two sizes.

        The filter uses a butterfly diagram for each prime factor of the
        dimension. This makes images with prime number dimensions (i.e. 17x17)
        much slower to compute. FFTs of multidimensional meshes (i.e volumes)
        are decomposed so that each axis executes serially.

        The frequencies of the input assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with the applied reverse FFT.

        See Also
        --------
        fft : The direct transform.
        low_pass : Low-pass filtering of FFT output.
        high_pass : High-pass filtering of FFT output.

        Examples
        --------
        Apply reverse FFT to an example image.

        >>> from pyvista import examples
        >>> image = examples.download_moonlanding_image()
        >>> fft_image = image.fft()
        >>> image_again = fft_image.rfft()
        >>> image_again.point_data  # doctest:+SKIP
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : PNGImage
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            PNGImage                complex128 (298620,)            SCALARS

        See :ref:`image_fft_example` for a full example using this filter.

        """
        self._check_fft_scalars()
        alg = _vtk.vtkImageRFFT()
        alg.SetInputDataObject(self)
        _update_alg(
            alg, progress_bar=progress_bar, message='Performing Reverse Fast Fourier Transform.'
        )
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,  # type: ignore[attr-defined]
            output_scalars_name,
        )
        return output

    @_deprecate_positional_args(allowed=['x_cutoff', 'y_cutoff', 'z_cutoff'])
    def low_pass(  # noqa: PLR0917
        self,
        x_cutoff,
        y_cutoff,
        z_cutoff,
        order=1,
        output_scalars_name=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Perform a Butterworth low pass filter in the frequency domain.

        This filter requires that the :class:`ImageData` have a complex point
        scalars, usually generated after the :class:`ImageData` has been
        converted to the frequency domain by a :func:`ImageDataFilters.fft`
        filter.

        A :func:`ImageDataFilters.rfft` filter can be used to convert the
        output back into the spatial domain. This filter attenuates high
        frequency components.  Input and output are complex arrays with
        datatype :attr:`numpy.complex128`.

        The frequencies of the input assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        x_cutoff : float
            The cutoff frequency for the x-axis.

        y_cutoff : float
            The cutoff frequency for the y-axis.

        z_cutoff : float
            The cutoff frequency for the z-axis.

        order : int, default: 1
            The order of the cutoff curve. Given from the equation
            ``1 + (cutoff/freq(i, j))**(2*order)``.

        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with the applied low pass filter.

        See Also
        --------
        fft : Direct fast Fourier transform.
        rfft : Reverse fast Fourier transform.
        high_pass : High-pass filtering of FFT output.

        Examples
        --------
        See :ref:`image_fft_perlin_noise_example` for a full example using this filter.

        """
        self._check_fft_scalars()
        alg = _vtk.vtkImageButterworthLowPass()
        alg.SetInputDataObject(self)
        alg.SetCutOff(x_cutoff, y_cutoff, z_cutoff)
        alg.SetOrder(order)
        _update_alg(alg, progress_bar=progress_bar, message='Performing Low Pass Filter')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,  # type: ignore[attr-defined]
            output_scalars_name,
        )
        return output

    @_deprecate_positional_args(allowed=['x_cutoff', 'y_cutoff', 'z_cutoff'])
    def high_pass(  # noqa: PLR0917
        self,
        x_cutoff,
        y_cutoff,
        z_cutoff,
        order=1,
        output_scalars_name=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Perform a Butterworth high pass filter in the frequency domain.

        This filter requires that the :class:`ImageData` have a complex point
        scalars, usually generated after the :class:`ImageData` has been
        converted to the frequency domain by a :func:`ImageDataFilters.fft`
        filter.

        A :func:`ImageDataFilters.rfft` filter can be used to convert the
        output back into the spatial domain. This filter attenuates low
        frequency components.  Input and output are complex arrays with
        datatype :attr:`numpy.complex128`.

        The frequencies of the input assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        x_cutoff : float
            The cutoff frequency for the x-axis.

        y_cutoff : float
            The cutoff frequency for the y-axis.

        z_cutoff : float
            The cutoff frequency for the z-axis.

        order : int, default: 1
            The order of the cutoff curve. Given from the equation
            ``1/(1 + (cutoff/freq(i, j))**(2*order))``.

        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with the applied high pass filter.

        See Also
        --------
        fft : Direct fast Fourier transform.
        rfft : Reverse fast Fourier transform.
        low_pass : Low-pass filtering of FFT output.

        Examples
        --------
        See :ref:`image_fft_perlin_noise_example` for a full example using this filter.

        """
        self._check_fft_scalars()
        alg = _vtk.vtkImageButterworthHighPass()
        alg.SetInputDataObject(self)
        alg.SetCutOff(x_cutoff, y_cutoff, z_cutoff)
        alg.SetOrder(order)
        _update_alg(alg, progress_bar=progress_bar, message='Performing High Pass Filter')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,  # type: ignore[attr-defined]
            output_scalars_name,
        )
        return output

    def _change_fft_output_scalars(self, dataset, orig_name, out_name) -> None:
        """Modify the name and dtype of the output scalars for an FFT filter."""
        name = orig_name if out_name is None else out_name
        pdata = dataset.point_data
        if pdata.active_scalars_name != name:
            pdata[name] = pdata.pop(pdata.active_scalars_name)

        # always view the datatype of the point_data as complex128
        dataset._association_complex_names['POINT'].add(name)

    def _check_fft_scalars(self):
        """Check for complex active scalars.

        This is necessary for rfft, low_pass, and high_pass filters.

        """
        # check for complex active point scalars, otherwise the risk of segfault
        if self.point_data.active_scalars_name is None:  # type: ignore[attr-defined]
            possible_scalars = self.point_data.keys()  # type: ignore[attr-defined]
            if len(possible_scalars) == 1:
                self.set_active_scalars(possible_scalars[0], preference='point')  # type: ignore[attr-defined]
            elif len(possible_scalars) > 1:
                msg = (
                    'There are multiple point scalars available. Set one to be '
                    'active with `point_data.active_scalars_name = `'
                )
                raise AmbiguousDataError(msg)
            else:
                msg = 'FFT filters require point scalars.'
                raise MissingDataError(msg)

        if not np.issubdtype(self.point_data.active_scalars.dtype, np.complexfloating):  # type: ignore[attr-defined]
            msg = (
                'Active scalars must be complex data for this filter, represented '
                'as an array with a datatype of `numpy.complex64` or '
                '`numpy.complex128`.'
            )
            raise ValueError(msg)

    def _flip_uniform(self, axis) -> pyvista.ImageData:
        """Flip the uniform grid along a specified axis and return a uniform grid.

        This varies from :func:`DataSet.flip_x` because it returns a ImageData.

        """
        alg = _vtk.vtkImageFlip()
        alg.SetInputData(self)
        alg.SetFilteredAxes(axis)
        alg.Update()
        return cast('pyvista.ImageData', wrap(alg.GetOutput()))

    @_deprecate_positional_args
    def contour_labeled(  # noqa: PLR0917
        self,
        n_labels: int | None = None,
        smoothing: bool = False,  # noqa: FBT001, FBT002
        smoothing_num_iterations: int = 50,
        smoothing_relaxation_factor: float = 0.5,
        smoothing_constraint_distance: float = 1,
        output_mesh_type: Literal['quads', 'triangles'] = 'quads',
        output_style: Literal['default', 'boundary'] = 'default',
        scalars: str | None = None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> pyvista.PolyData:
        """Generate labeled contours from 3D label maps.

        SurfaceNets algorithm is used to extract contours preserving sharp
        boundaries for the selected labels from the label maps.
        Optionally, the boundaries can be smoothened to reduce the staircase
        appearance in case of low resolution input label maps.

        This filter requires that the :class:`ImageData` has integer point
        scalars, such as multi-label maps generated from image segmentation.

        .. note::
           Requires ``vtk>=9.3.0``.

        .. deprecated:: 0.45
            This filter produces unexpected results and is deprecated.
            Use :meth:`~pyvista.ImageDataFilters.contour_labels` instead.
            See https://github.com/pyvista/pyvista/issues/5981 for details.

            To replicate the default behavior from this filter, call `contour_labels`
            with the following arguments:

            .. code-block:: python

                image.contour_labels(
                    boundary_style='strict_external',  # old filter strictly uses external polygons
                    smoothing=False,  # old filter does not apply smoothing
                    output_mesh_type='quads',  # old filter generates quads
                    pad_background=False,  # old filter generates open surfaces at input edges
                    orient_faces=False,  # old filter does not orient faces
                    simplify_output=False,  # old filter returns multi-component scalars
                )

        Parameters
        ----------
        n_labels : int, optional
            Number of labels to be extracted (all are extracted if None is given).

        smoothing : bool, default: False
            Apply smoothing to the meshes.

        smoothing_num_iterations : int, default: 50
            Number of smoothing iterations.

        smoothing_relaxation_factor : float, default: 0.5
            Relaxation factor of the smoothing.

        smoothing_constraint_distance : float, default: 1
            Constraint distance of the smoothing.

        output_mesh_type : str, default: 'quads'
            Type of the output mesh. Must be either ``'quads'``, or ``'triangles'``.

        output_style : str, default: 'default'
            Style of the output mesh. Must be either ``'default'`` or ``'boundary'``.
            When ``'default'`` is specified, the filter produces a mesh with both
            interior and exterior polygons. When ``'boundary'`` is selected, only
            polygons on the border with the background are produced (without interior
            polygons). Note that style ``'selected'`` is currently not implemented.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            :class:`pyvista.PolyData` Labeled mesh with the segments labeled.

        References
        ----------
        Sarah F. Frisken, SurfaceNets for Multi-Label Segmentations with Preservation
        of Sharp Boundaries, Journal of Computer Graphics Techniques (JCGT), vol. 11,
        no. 1, 34-54, 2022. Available online http://jcgt.org/published/0011/01/03/

        https://www.kitware.com/really-fast-isocontouring/

        Examples
        --------
        See :ref:`contouring_example` for a full example using this filter.

        See Also
        --------
        pyvista.DataSetFilters.contour
            Generalized contouring method which uses MarchingCubes or FlyingEdges.

        pyvista.DataSetFilters.pack_labels
            Function used internally by SurfaceNets to generate contiguous label data.

        """
        warnings.warn(
            'This filter produces unexpected results and is deprecated. '
            'Use `contour_labels` instead.'
            '\nRefer to the documentation for `contour_labeled` for details on how to '
            'transition to the new filter.'
            '\nSee https://github.com/pyvista/pyvista/issues/5981 for details.',
            PyVistaDeprecationWarning,
            stacklevel=2,
        )

        if not hasattr(_vtk, 'vtkSurfaceNets3D'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError  # noqa: PLC0415

            msg = 'Surface nets 3D require VTK 9.3.0 or newer.'
            raise VTKVersionError(msg)

        alg = _vtk.vtkSurfaceNets3D()
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
            if field != FieldAssociation.POINT:
                msg = 'If `scalars` not given, active scalars must be point array.'
                raise ValueError(msg)
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
            if field != FieldAssociation.POINT:
                msg = (
                    f'Can only process point data, given `scalars` are {field.name.lower()} data.'
                )
                raise ValueError(msg)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetInputData(self)
        if n_labels is not None:
            alg.GenerateLabels(n_labels, 1, n_labels)
        if output_mesh_type == 'quads':
            alg.SetOutputMeshTypeToQuads()
        elif output_mesh_type == 'triangles':
            alg.SetOutputMeshTypeToTriangles()
        else:
            msg = f'Invalid output mesh type "{output_mesh_type}", use "quads" or "triangles"'  # type: ignore[unreachable]
            raise ValueError(msg)
        if output_style == 'default':
            alg.SetOutputStyleToDefault()
        elif output_style == 'boundary':
            alg.SetOutputStyleToBoundary()
        elif output_style == 'selected':  # type: ignore[unreachable]
            msg = f'Output style "{output_style}" is not implemented'
            raise NotImplementedError(msg)
        else:
            msg = f'Invalid output style "{output_style}", use "default" or "boundary"'
            raise ValueError(msg)
        if smoothing:
            alg.SmoothingOn()
            alg.GetSmoother().SetNumberOfIterations(smoothing_num_iterations)
            alg.GetSmoother().SetRelaxationFactor(smoothing_relaxation_factor)
            alg.GetSmoother().SetConstraintDistance(smoothing_constraint_distance)
        else:
            alg.SmoothingOff()
        # Suppress improperly used INFO for debugging messages in vtkSurfaceNets3D
        with pyvista.vtk_verbosity('off'):
            _update_alg(
                alg, progress_bar=progress_bar, message='Performing Labeled Surface Extraction'
            )
        return wrap(alg.GetOutput())

    def contour_labels(  # type: ignore[misc]
        self: ImageData,
        boundary_style: Literal['external', 'internal', 'all', 'strict_external'] = 'external',
        *,
        background_value: int = 0,
        select_inputs: int | VectorLike[int] | None = None,
        select_outputs: int | VectorLike[int] | None = None,
        pad_background: bool = True,
        output_mesh_type: Literal['quads', 'triangles'] | None = None,
        scalars: str | None = None,
        orient_faces: bool = True,
        simplify_output: bool | None = None,
        smoothing: bool = True,
        smoothing_iterations: int = 16,
        smoothing_relaxation: float = 0.5,
        smoothing_distance: float | None = None,
        smoothing_scale: float = 1.0,
        progress_bar: bool = False,
    ) -> PolyData:
        """Generate surface contours from 3D image label maps.

        This filter uses :vtk:`vtkSurfaceNets3D`
        to extract polygonal surface contours from non-continuous label maps, which
        corresponds to discrete regions in an input 3D image (i.e., volume). It is
        designed to generate surfaces from image point data, e.g. voxel point
        samples from 3D medical images, though images with cell data are also supported.

        The generated surface is smoothed using a constrained smoothing filter, which
        may be fine-tuned to control the smoothing process. Optionally, smoothing may
        be disabled to generate a staircase-like surface.

        The output surface includes a two-component cell data array ``'boundary_labels'``.
        The array indicates the labels/regions on either side of the polygons composing
        the output. The array's values are structured as follows:

        External boundary values

            Polygons between a foreground region and the background have the
            form ``[foreground, background]``.

            E.g. ``[1, 0]`` for the boundary between region ``1`` and background ``0``.

        Internal boundary values

            Polygons between two connected foreground regions are sorted in ascending order.

            E.g. ``[1, 2]`` for the boundary between regions ``1`` and ``2``.

        By default, this filter returns ``'external'`` contours only. Optionally,
        only the ``'internal'`` contours or ``'all'`` contours (i.e. internal and
        external) may be returned.

        .. note::

            This filter requires VTK version ``9.3.0`` or greater.

        .. versionadded:: 0.45

        Parameters
        ----------
        boundary_style : 'external' | 'internal' | 'all' | 'strict_external', default: 'external'
            Style of boundary polygons to generate. ``'internal'`` polygons are generated
            between two connected foreground regions. ``'external'`` polygons are
            generated between foreground and background regions. ``'all'``  includes
            both internal and external boundary polygons.

            These styles are generated such that ``internal + external = all``.
            Internally, the filter computes all boundary polygons by default and
            then removes any undesired polygons in post-processing.
            This improves the quality of the output, but can negatively affect the
            filter's performance since all boundaries are always initially computed.

            The ``'strict_external'`` style can be used as a fast alternative to
            ``'external'``. This style `strictly` generates external polygons and does
            not compute or consider internal boundaries. This computation is fast, but
            also results in jagged, non-smooth boundaries between regions. The
            ``select_inputs`` and ``select_outputs`` options cannot be used with this
            style.

        background_value : int, default: 0
            Background value of the input image. All other values are considered
            as foreground.

        select_inputs : int | VectorLike[int], default: None
            Specify label ids to include as inputs to the filter. Labels that are not
            selected are removed from the input *before* generating the surface. By
            default, all label ids are used.

            Since the smoothing operation occurs across selected input regions, using
            this option to filter the input can result in smoother and more visually
            pleasing surfaces since non-selected inputs are not considered during
            smoothing. However, this also means that the generated surface will change
            shape depending on which inputs are selected.

            .. note::

                Selecting inputs can affect whether a boundary polygon is considered to
                be ``internal`` or ``external``. That is, an internal boundary becomes an
                external boundary when only one of the two foreground regions on the
                boundary is selected.

        select_outputs : int | VectorLike[int], default: None
            Specify label ids to include in the output of the filter. Labels that are
            not selected are removed from the output *after* generating the surface. By
            default, all label ids are used.

            Since the smoothing operation occurs across all input regions, using this
            option to filter the output means that the selected output regions will have
            the same shape (i.e. smoothed in the same manner), regardless of the outputs
            that are selected. This is useful for generating a surface for specific
            labels while also preserving sharp boundaries with non-selected outputs.

            .. note::

                Selecting outputs does not affect whether a boundary polygon is
                considered to be ``internal`` or ``external``. That is, an internal
                boundary remains internal even if only one of the two foreground regions
                on the boundary is selected.

        pad_background : bool, default: True
            :meth:`Pad <pyvista.ImageDataFilters.pad_image>` the image
            with ``background_value`` prior to contouring. This will
            generate polygons to "close" the surface at the boundaries of the image.
            This option is only relevant when there are foreground regions on the border
            of the image. Setting this value to ``False`` is useful if processing multiple
            volumes separately so that the generated surfaces fit together without
            creating surface overlap.

        output_mesh_type : str, default: None
            Type of the output mesh. Can be either ``'quads'``, or ``'triangles'``. By
            default, the output mesh has :attr:`~pyvista.CellType.TRIANGLE` cells when
            ``smoothing`` is enabled and :attr:`~pyvista.CellType.QUAD` cells (quads)
            otherwise. The mesh type can be forced to be triangles or quads; however,
            if smoothing is enabled and the type is ``'quads'``, the generated quads
            may not be planar.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars. If cell
            scalars are specified, the input image is first re-meshed with
            :meth:`~pyvista.ImageDataFilters.cells_to_points` to transform the cell
            data into point data.

        orient_faces : bool, default: True
            Orient the faces of the generated contours so that they have consistent
            ordering and face outward. If ``False``, the generated polygons may have
            inconsistent ordering and orientation, which can negatively impact
            downstream calculations and the shading used for rendering.

            .. note::

                Orienting the faces can be computationally expensive for large meshes.
                Consider disabling this option to improve this filter's performance.

            .. warning::

                Enabling this option is `likely` to generate surfaces with normals
                pointing outward when ``pad_background`` is ``True`` and
                ``boundary_style`` is ``'external'`` (the default). However, this is
                not guaranteed if the generated surface is not closed or if internal
                boundaries are generated. Do not assume the normals will point outward
                in all cases.

        simplify_output : bool, optional
            Simplify the ``'boundary_labels'`` array as a single-component 1D array.
            If ``False``, the returned ``'boundary_labels'`` array is a two-component
            2D array. This simplification is useful when only external boundaries
            are generated and/or when visualizing internal boundaries. The
            simplification is as follows:

            - External boundaries are simplified by keeping the first component and
              removing the second. Since external polygons may only share a boundary
              with the background, the second component is always ``background_value``
              and therefore can be dropped without loss of information. The values
              of external boundaries always match the foreground values of the input.

            - Internal boundaries are simplified by assigning them unique negative
              values sequentially. E.g. the boundary label ``[1, 2]`` is replaced with
              ``-1``, ``[1, 3]`` is replaced with ``-2``, etc. The mapping to negative
              values is not fixed, and can change depending on the input.

              This simplification is particularly useful for unsigned integer labels
              (e.g. scalars with ``'uint8'`` dtype) since external boundaries
              will be positive and internal boundaries will be negative in this case.

            By default, the output is simplified when ``boundary_type`` is
            ``'external'`` or ``'strict_external'``, and is not simplified otherwise.

        smoothing : bool, default: True
            Smooth the generated surface using a constrained smoothing filter. Each
            point in the surface is smoothed as follows:

                For a point ``pi`` connected to a list of points ``pj`` via an edge, ``pi``
                is moved towards the average position of ``pj`` multiplied by the
                ``smoothing_relaxation`` factor, and limited by the ``smoothing_distance``
                constraint. This process is repeated either until convergence occurs, or
                the maximum number of ``smoothing_iterations`` is reached.

        smoothing_iterations : int, default: 16
            Maximum number of smoothing iterations to use.

        smoothing_relaxation : float, default: 0.5
            Relaxation factor used at each smoothing iteration.

        smoothing_distance : float, default: None
            Maximum distance each point is allowed to move (in any direction) during
            smoothing. This distance may be scaled with ``smoothing_scale``. By default,
            the distance is computed dynamically from the image spacing as:

                ``distance = norm(image_spacing) * smoothing_scale``.

        smoothing_scale : float, default: 1.0
            Relative scaling factor applied to ``smoothing_distance``. See that
            parameter for details.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Surface mesh of labeled regions.

        See Also
        --------
        :meth:`~pyvista.DataSetFilters.voxelize_binary_mask`
            Filter that generates binary labeled :class:`~pyvista.ImageData` from
            :class:`~pyvista.PolyData` surface contours. Can beloosely considered as
            an inverse of this filter.

        :meth:`~pyvista.ImageDataFilters.cells_to_points`
            Re-mesh :class:`~pyvista.ImageData` to a points-based representation.

        :meth:`~pyvista.DataSetFilters.extract_values`
            Threshold-like filter which can used to process the multi-component
            scalars generated by this filter.

        :meth:`~pyvista.DataSetFilters.contour`
            Generalized contouring method which uses MarchingCubes or FlyingEdges.

        :meth:`~pyvista.DataSetFilters.pack_labels`
            Function used internally by SurfaceNets to generate contiguous label data.

        :meth:`~pyvista.DataSetFilters.color_labels`
            Color labeled data, e.g. labeled volumes or contours.

        :ref:`contouring_example`, :ref:`anatomical_groups_example`
            Additional examples using this filter.

        References
        ----------
        S. Frisken, “SurfaceNets for Multi-Label Segmentations with Preservation of
        Sharp Boundaries”, J. Computer Graphics Techniques, 2022. Available online:
        http://jcgt.org/published/0011/01/03/

        W. Schroeder, S. Tsalikis, M. Halle, S. Frisken. A High-Performance SurfaceNets
        Discrete Isocontouring Algorithm. arXiv:2401.14906. 2024. Available online:
        `http://arxiv.org/abs/2401.14906 <http://arxiv.org/abs/2401.14906>`__

        Examples
        --------
        Load labeled image data with a background region ``0`` and four foreground
        regions.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> from pyvista import examples
        >>> image = examples.load_channels()
        >>> label_ids = np.unique(image.active_scalars)
        >>> label_ids
        pyvista_ndarray([0, 1, 2, 3, 4])
        >>> image.dimensions
        (251, 251, 101)

        Crop the image to simplify the data.

        >>> image = image.extract_subset(voi=(75, 109, 75, 109, 85, 100))
        >>> image.dimensions
        (35, 35, 16)

        Plot the cropped image for context. Use :meth:`~pyvista.DataSetFilters.color_labels`
        to generate consistent coloring of the regions for all plots. Negative indexing
        is used for plotting internal boundaries.

        >>> def labels_plotter(mesh, zoom=None):
        ...     colored_mesh = mesh.color_labels(negative_indexing=True)
        ...     plotter = pv.Plotter()
        ...     plotter.add_mesh(colored_mesh, show_edges=True)
        ...     if zoom:
        ...         plotter.camera.zoom(zoom)
        ...     return plotter
        >>>
        >>> labels_plotter(image).show()

        Generate surface contours of the foreground regions and plot it. Note that
        the ``background_value`` is ``0`` by default.

        >>> contours = image.contour_labels()
        >>> labels_plotter(contours, zoom=1.5).show()

        By default, only external boundary polygons are generated and the returned
        ``'boundary_labels'`` array is a single-component array. The output values
        match the input label values.

        >>> contours['boundary_labels'].ndim
        1
        >>> np.unique(contours['boundary_labels'])
        pyvista_ndarray([1, 2, 3, 4])

        Set ``simplify_output`` to ``False`` to generate a two-component
        array instead showing the two boundary regions associated with each polygon.

        >>> contours = image.contour_labels(simplify_output=False)
        >>> contours['boundary_labels'].ndim
        2

        Show the unique values. Since only ``'external'`` boundaries are generated
        by default, the second component is always ``0`` (i.e. the ``background_value``).
        Note that all four foreground regions share a boundary with the background.

        >>> np.unique(contours['boundary_labels'], axis=0)
        array([[1, 0],
               [2, 0],
               [3, 0],
               [4, 0]])

        Repeat the example but this time generate internal contours only. The generated
        array is 2D by default.

        >>> contours = image.contour_labels('internal')
        >>> contours['boundary_labels'].ndim
        2

        Show the unique two-component boundary labels again. From these values we can
        determine that all foreground regions share an internal boundary with each
        other `except`  for regions ``1`` and ``3`` since the boundary value ``[1, 3]``
        is missing.

        >>> np.unique(contours['boundary_labels'], axis=0)
        array([[1, 2],
               [1, 4],
               [2, 3],
               [2, 4],
               [3, 4]])

        Simplify the output so that each internal multi-component boundary value is
        assigned a unique negative integer value instead. This makes it easier to
        visualize the result with :meth:`~pyvista.DataSetFilters.color_labels` using
        the ``negative_indexing`` option.

        >>> contours = image.contour_labels('internal', simplify_output=True)
        >>> contours['boundary_labels'].ndim
        1
        >>> np.unique(contours['boundary_labels'])
        pyvista_ndarray([-5, -4, -3, -2, -1])

        >>> labels_plotter(contours, zoom=1.5).show()

        Generate contours for all boundaries, and use ``select_outputs`` to filter
        the output to only include polygons which share a boundary with region ``3``.

        >>> region_3 = image.contour_labels(
        ...     'all', select_outputs=3, simplify_output=True
        ... )
        >>> labels_plotter(region_3, zoom=3).show()

        Note how using ``select_outputs`` preserves the sharp features and boundary
        labels for non-selected regions. If desired, use ``select_inputs`` instead to
        completely "ignore" non-selected regions.

        >>> region_3 = image.contour_labels(select_inputs=3)
        >>> labels_plotter(region_3, zoom=3).show()

        The sharp features are now smoothed and the internal boundaries are now labeled
        as external boundaries. Note that using ``'all'`` here is optional since
        using ``select_inputs`` converts previously-internal boundaries into external
        ones.

        Do not pad the image with background values before contouring. Since the input image
        has foreground regions visible at the edges of the image (e.g. the ``+Z`` bound),
        setting ``pad_background=False`` in this example causes the top and sides of
        the mesh to be "open".

        >>> surf = image.contour_labels(pad_background=False)
        >>> labels_plotter(surf, zoom=1.5).show()

        Disable smoothing to generate staircase-like surface. Without smoothing, the
        surface has quadrilateral cells by default.

        >>> surf = image.contour_labels(smoothing=False)
        >>> labels_plotter(surf, zoom=1.5).show()

        Keep smoothing enabled but reduce the smoothing scale. A smoothing scale
        less than one may help preserve sharp features (e.g. corners).

        >>> surf = image.contour_labels(smoothing_scale=0.5)
        >>> labels_plotter(surf, zoom=1.5).show()

        Use the ``'strict_external'`` style to compute external contours quickly. Note
        that this produces jagged and non-smooth boundaries between regions, which may
        not be desirable. Also note how the top of the surface is perfectly flat compared
        to the default ``'external'`` style (see first example above) since the strict
        style ignores the smoothing effects of all internal boundaries.

        >>> surf = image.contour_labels('strict_external')
        >>> labels_plotter(surf, zoom=1.5).show()

        """
        temp_scalars_name = '_PYVISTA_TEMP'

        def _get_unique_labels_no_background(
            array: NumpyArray[int], background: int
        ) -> NumpyArray[int]:
            unique = np.unique(array)
            return unique[unique != background]

        def _get_alg_input(image: ImageData, scalars_: str | None) -> ImageData:
            if scalars_ is None:
                set_default_active_scalars(image)
                field, scalars_ = image.active_scalars_info
            else:
                field = image.get_array_association(scalars_, preference='point')

            return (
                image
                if field == FieldAssociation.POINT
                else image.cells_to_points(scalars=scalars_, copy=False)
            )

        def _process_select_inputs(
            image: ImageData,
            select_inputs_: int | VectorLike[int],
            scalars_: pyvista_ndarray,
        ) -> NumpyArray[int]:
            select_inputs = np.atleast_1d(select_inputs_)
            # Remove non-selected label ids from the input. We do this by setting
            # non-selected ids to the background value to remove them from the input
            temp_scalars = scalars_.copy()
            input_ids = _get_unique_labels_no_background(temp_scalars, background_value)
            keep_labels = [*select_inputs, background_value]
            for label in input_ids:
                if label not in keep_labels:
                    temp_scalars[temp_scalars == label] = background_value

            image.point_data[temp_scalars_name] = temp_scalars
            image.set_active_scalars(temp_scalars_name, preference='point')

            return input_ids

        def _set_output_mesh_type(alg_: _vtk.vtkSurfaceNets3D):
            if output_mesh_type is None:
                alg_.SetOutputMeshTypeToDefault()
            elif output_mesh_type == 'quads':
                alg_.SetOutputMeshTypeToQuads()
            else:  # output_mesh_type == 'triangles':
                alg_.SetOutputMeshTypeToTriangles()

        def _configure_boundaries(
            alg_: _vtk.vtkSurfaceNets3D,
            *,
            array_: pyvista_ndarray,
            select_inputs_: int | VectorLike[int] | None,
            select_outputs_: int | VectorLike[int] | None,
        ):
            # WARNING: Setting the output style to default or boundary does not really work
            # as expected. Specifically, `SetOutputStyleToDefault` by itself will not actually
            # produce meshes with interior faces at the boundaries between foreground regions
            # (even though this is what is suggested by the docs). Instead, simply calling
            # `SetLabels` below will enable internal boundaries, regardless of the value of
            # `OutputStyle`. Also, using `SetOutputStyleToBoundary` generates jagged/rough
            # 'lines' between two exterior regions; enabling internal boundaries fixes this.
            input_ids = (
                _process_select_inputs(alg_input, select_inputs_, array_)
                if select_inputs_ is not None
                else None
            )
            alg_.SetOutputStyleToSelected()
            if select_outputs_ is not None:
                # Use selected outputs
                output_ids = _get_unique_labels_no_background(
                    np.atleast_1d(select_outputs_),
                    background_value,
                )
            elif input_ids is not None:
                # Set outputs to be same as inputs
                output_ids = input_ids
            else:
                # Output all labels
                output_ids = _get_unique_labels_no_background(
                    array_,
                    background_value,
                )
            output_ids = output_ids.astype(float)

            # Add selected outputs
            [alg.AddSelectedLabel(label_id) for label_id in output_ids]  # type: ignore[func-returns-value]

            # The following logic enables the generation of internal boundaries
            if input_ids is not None:
                # Generate internal boundaries for selected inputs only
                internal_ids: NumpyArray[int] = input_ids
            elif select_outputs is None:
                # No inputs or outputs selected, so generate internal
                # boundaries for all labels in input array
                internal_ids = output_ids
            else:
                internal_ids = _get_unique_labels_no_background(
                    array_,
                    background_value,
                )

            [alg.SetLabel(int(val), val) for val in internal_ids]  # type: ignore[func-returns-value]

        def _configure_smoothing(
            alg_: _vtk.vtkSurfaceNets3D,
            *,
            spacing_: tuple[float, float, float],
            iterations_: int,
            relaxation_: float,
            scale_: float,
            distance_: float | None,
        ):
            def _is_small_number(num) -> bool | np.bool_:
                return isinstance(num, (float, int, np.floating, np.integer)) and num < 1e-8

            if smoothing and not _is_small_number(scale_) and not _is_small_number(distance_):
                # Only enable smoothing if distance is not very small, since a small
                # distance will actually result in large smoothing (suspected division
                # by zero error in vtk code)
                alg_.SmoothingOn()
                alg_.GetSmoother().SetNumberOfIterations(iterations_)
                alg_.GetSmoother().SetRelaxationFactor(relaxation_)

                # Auto-constraints are On by default which only allows you to scale
                # relative distance (with SetConstraintScale) but not set its value
                # directly. Here, we turn this off so that we can both set its value
                # and/or scale it independently
                alg_.AutomaticSmoothingConstraintsOff()

                # Dynamically calculate distance if not specified.
                # This emulates the auto-constraint calc from vtkSurfaceNets3D
                distance_ = distance_ or np.linalg.norm(spacing_)
                alg_.GetSmoother().SetConstraintDistance(distance_ * scale_)
            else:
                alg_.SmoothingOff()

        if not hasattr(_vtk, 'vtkSurfaceNets3D'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError  # noqa: PLC0415

            msg = 'Surface nets 3D require VTK 9.3.0 or newer.'
            raise VTKVersionError(msg)

        _validation.check_contains(
            ['all', 'internal', 'external', 'strict_external'],
            must_contain=boundary_style,
            name='boundary_style',
        )
        _validation.check_contains(
            [None, 'quads', 'triangles'],
            must_contain=output_mesh_type,
            name='output_mesh_type',
        )

        alg_input = _get_alg_input(self, scalars)
        active_scalars = cast('pyvista.pyvista_ndarray', alg_input.active_scalars)
        if np.allclose(active_scalars, background_value):
            # Empty input, no contour will be generated
            return pyvista.PolyData()

        # Pad with background values to close surfaces at image boundaries
        alg_input = alg_input.pad_image(background_value) if pad_background else alg_input

        alg = _vtk.vtkSurfaceNets3D()
        alg.SetBackgroundLabel(background_value)
        alg.SetInputData(alg_input)

        _set_output_mesh_type(alg)
        if boundary_style == 'strict_external':
            # Use default alg parameters
            if select_inputs is not None or select_outputs is not None:
                msg = 'Selecting inputs and/or outputs is not supported by `strict_external`.'
                raise TypeError(msg)
        else:
            _configure_boundaries(
                alg,
                array_=cast('pyvista.pyvista_ndarray', alg_input.active_scalars),
                select_inputs_=select_inputs,
                select_outputs_=select_outputs,
            )
        _configure_smoothing(
            alg,
            spacing_=alg_input.spacing,
            iterations_=smoothing_iterations,
            relaxation_=smoothing_relaxation,
            scale_=smoothing_scale,
            distance_=smoothing_distance,
        )

        # Get output
        # Suppress improperly used INFO for debugging messages in vtkSurfaceNets3D
        with pyvista.vtk_verbosity('off'):
            _update_alg(alg, progress_bar=progress_bar, message='Generating label contours')

        output: pyvista.PolyData = _get_output(alg)

        (  # Clear temp scalars from input
            alg_input.point_data.remove(temp_scalars_name)
            if temp_scalars_name in alg_input.point_data
            else None
        )

        VTK_NAME = 'BoundaryLabels'
        PV_NAME = 'boundary_labels'
        if VTK_NAME in output.cell_data.keys():
            labels_array = output.cell_data[VTK_NAME]
            if not all(labels_array.shape):
                # Array is empty but has non-zero shape, fix it here
                # Mesh may also have non-zero points but this is cleaned later
                output.cell_data[VTK_NAME] = np.empty((0, 0))
            output.rename_array(VTK_NAME, PV_NAME)
            if boundary_style in ['external', 'internal']:
                # Output contains all boundary cells, need to remove cells we don't want
                is_external = np.any(labels_array == background_value, axis=1)
                remove = is_external if boundary_style == 'internal' else ~is_external
                output.remove_cells(remove, inplace=True)

        is_external = 'external' in boundary_style
        if simplify_output is None:
            simplify_output = is_external
        if simplify_output:
            # Simplify scalars to a single component
            if not is_external:
                # Replace internal boundary values with negative integers
                labels_array = output.cell_data[PV_NAME]
                is_internal = (
                    np.full((output.n_cells,), True)
                    if boundary_style == 'internal'
                    else np.all(labels_array != background_value, axis=1)
                )
                internal_values = labels_array[is_internal, :]
                unique_values = np.unique(internal_values, axis=0)
                for i, value in enumerate(unique_values):
                    is_value = np.all(labels_array == value, axis=1)
                    labels_array[is_value, 0] = -(i + 1)  # type: ignore[index]

            # Keep first component only
            output.cell_data[PV_NAME] = output.cell_data[PV_NAME][:, 0]

        if select_outputs is not None:
            # This option generates unused points
            # Use clean to remove these points (without merging points)
            output.clean(
                point_merging=False,
                lines_to_points=False,
                polys_to_lines=False,
                strips_to_polys=False,
                inplace=True,
            )

        if orient_faces and output.n_cells > 0:
            if pyvista.vtk_version_info >= (9, 4):
                filter_ = _vtk.vtkOrientPolyData()
                filter_.SetInputData(output)
                filter_.ConsistencyOn()
                filter_.AutoOrientNormalsOn()
                filter_.NonManifoldTraversalOn()
                filter_.Update()
                oriented = wrap(filter_.GetOutput())
                output.points = oriented.points
                output.faces = oriented.faces
            else:
                # Orient the faces but discard the normals array
                output.compute_normals(
                    cell_normals=True,
                    point_normals=False,
                    consistent_normals=True,
                    auto_orient_normals=True,
                    non_manifold_traversal=True,
                    inplace=True,
                )
                del output.cell_data['Normals']
        return output

    def points_to_cells(  # type: ignore[misc]
        self: ImageData,
        scalars: str | None = None,
        *,
        dimensionality: VectorLike[bool]
        | Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D', 'preserve'] = 'preserve',
        copy: bool = True,
    ):
        """Re-mesh image data from a point-based to a cell-based representation.

        This filter changes how image data is represented. Data represented as points
        at the input is re-meshed into an alternative representation as cells at the
        output. Only the :class:`~pyvista.ImageData` container is modified so that
        the number of input points equals the number of output cells. The re-meshing is
        otherwise lossless in the sense that point data at the input is passed through
        unmodified and stored as cell data at the output. Any cell data at the input is
        ignored and is not used by this filter.

        To change the image data's representation, the input points are used to
        represent the centers of the output cells. This has the effect of "growing" the
        input image dimensions by one along each axis (i.e. half the cell width on each
        side). For example, an image with 100 points and 99 cells along an axis at the
        input will have 101 points and 100 cells at the output. If the input has 1mm
        spacing, the axis size will also increase from 99mm to 100mm. By default,
        only non-singleton dimensions are increased such that 1D or 2D inputs remain
        1D or 2D at the output.

        Since filters may be inherently cell-based (e.g. some :class:`~pyvista.DataSetFilters`)
        or may operate on point data exclusively (e.g. most :class:`~pyvista.ImageDataFilters`),
        re-meshing enables the same data to be used with either kind of filter while
        ensuring the input data to those filters has the appropriate representation.
        This filter is also useful when plotting image data to achieve a desired visual
        effect, such as plotting images as voxel cells instead of as points.

        .. note::
            Only the input's :attr:`~pyvista.ImageData.dimensions`, and
            :attr:`~pyvista.ImageData.origin` are modified by this filter. Other spatial
            properties such as :attr:`~pyvista.ImageData.spacing` and
            :attr:`~pyvista.ImageData.direction_matrix` are not affected.

        .. versionadded:: 0.44.0

        See Also
        --------
        cells_to_points
            Inverse of this filter to represent cells as points.
        :meth:`~pyvista.DataObjectFilters.cell_data_to_point_data`
            Resample point data as cell data without modifying the container.
        :meth:`~pyvista.DataObjectFilters.cell_data_to_point_data`
            Resample cell data as point data without modifying the container.

        Parameters
        ----------
        scalars : str, optional
            Name of point data scalars to pass through to the output as cell data. Use
            this parameter to restrict the output to only include the specified array.
            By default, all point data arrays at the input are passed through as cell
            data at the output.

        dimensionality : VectorLike[bool], Literal[0, 1, 2, 3, "0D", "1D", "2D", "3D", "preserve"]
            Control which dimensions will be modified by the filter.
            ``'preserve'`` is used by default.

            - Can be specified as a sequence of 3 boolean to allow modification on a per
                dimension basis.
            - ``0`` or ``'0D'``: convenience alias to output a 0D ImageData with
              dimensions ``(1, 1, 1)``. Only valid for 0D inputs.
            - ``1`` or ``'1D'``: convenience alias to output a 1D ImageData where
              exactly one dimension is greater than one, e.g. ``(>1, 1, 1)``. Only valid
              for 0D or 1D inputs.
            - ``2`` or ``'2D'``: convenience alias to output a 2D ImageData where
              exactly two dimensions are greater than one, e.g. ``(>1, >1, 1)``. Only
              valid for 0D, 1D, or 2D inputs.
            - ``3`` or ``'3D'``: convenience alias to output a 3D ImageData, where all
              three dimensions are greater than one, e.g. ``(>1, >1, >1)``. Valid for
              any 0D, 1D, 2D, or 3D inputs.
            - ``'preserve'`` (default): convenience alias to not modify singleton
              dimensions.

        copy : bool, default: True
            Copy the input point data before associating it with the output cell data.
            If ``False``, the input and output will both refer to the same data array(s).

        Returns
        -------
        pyvista.ImageData
            Image with a cell-based representation.

        Examples
        --------
        Load an image with point data.

        >>> from pyvista import examples
        >>> image = examples.load_uniform()

        Show the current properties and point arrays of the image.

        >>> image
        ImageData (...)
          N Cells:      729
          N Points:     1000
          X Bounds:     0.000e+00, 9.000e+00
          Y Bounds:     0.000e+00, 9.000e+00
          Z Bounds:     0.000e+00, 9.000e+00
          Dimensions:   10, 10, 10
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     2

        >>> image.point_data.keys()
        ['Spatial Point Data']

        Re-mesh the points and point data as cells and cell data.

        >>> cells_image = image.points_to_cells()

        Show the properties and cell arrays of the re-meshed image.

        >>> cells_image
        ImageData (...)
          N Cells:      1000
          N Points:     1331
          X Bounds:     -5.000e-01, 9.500e+00
          Y Bounds:     -5.000e-01, 9.500e+00
          Z Bounds:     -5.000e-01, 9.500e+00
          Dimensions:   11, 11, 11
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        >>> cells_image.cell_data.keys()
        ['Spatial Point Data']

        Observe that:

        - The input point array is now a cell array
        - The output has one less array (the input cell data is ignored)
        - The dimensions have increased by one
        - The bounds have increased by half the spacing
        - The output ``N Cells`` equals the input ``N Points``

        Since the input points are 3D (i.e. there are no singleton dimensions), the
        output cells are 3D :attr:`~pyvista.CellType.VOXEL` cells.

        >>> cells_image.get_cell(0).type
        <CellType.VOXEL: 11>

        If the input points are 2D (i.e. one dimension is singleton), the
        output cells are 2D :attr:`~pyvista.CellType.PIXEL` cells when ``dimensions`` is
        set to ``'preserve'``.

        >>> image2D = examples.download_beach()
        >>> image2D.dimensions
        (100, 100, 1)

        >>> pixel_cells_image = image2D.points_to_cells(dimensionality='preserve')
        >>> pixel_cells_image.dimensions
        (101, 101, 1)
        >>> pixel_cells_image.get_cell(0).type
        <CellType.PIXEL: 8>

        This is equivalent as requesting a 2D output.

        >>> pixel_cells_image = image2D.points_to_cells(dimensionality='2D')
        >>> pixel_cells_image.dimensions
        (101, 101, 1)
        >>> pixel_cells_image.get_cell(0).type
        <CellType.PIXEL: 8>

        Use ``(True, True, True)`` to re-mesh 2D points as 3D cells.

        >>> voxel_cells_image = image2D.points_to_cells(
        ...     dimensionality=(True, True, True)
        ... )
        >>> voxel_cells_image.dimensions
        (101, 101, 2)
        >>> voxel_cells_image.get_cell(0).type
        <CellType.VOXEL: 11>

        Or request a 3D output.

        >>> voxel_cells_image = image2D.points_to_cells(dimensionality='3D')
        >>> voxel_cells_image.dimensions
        (101, 101, 2)
        >>> voxel_cells_image.get_cell(0).type
        <CellType.VOXEL: 11>

        See :ref:`image_representations_example` for more examples using this filter.

        """
        if scalars is not None:
            field = self.get_array_association(scalars, preference='point')
            if field != FieldAssociation.POINT:
                msg = (
                    f"Scalars '{scalars}' must be associated with point data. "
                    f'Got {field.name.lower()} data instead.'
                )
                raise ValueError(msg)
        return self._remesh_points_cells(
            points_to_cells=True,
            scalars=scalars,
            dimensionality=dimensionality,
            copy=copy,
        )

    def cells_to_points(  # type: ignore[misc]
        self: ImageData,
        scalars: str | None = None,
        *,
        dimensionality: VectorLike[bool]
        | Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D', 'preserve'] = 'preserve',
        copy: bool = True,
    ):
        """Re-mesh image data from a cell-based to a point-based representation.

        This filter changes how image data is represented. Data represented as cells
        at the input is re-meshed into an alternative representation as points at the
        output. Only the :class:`~pyvista.ImageData` container is modified so that
        the number of input cells equals the number of output points. The re-meshing is
        otherwise lossless in the sense that cell data at the input is passed through
        unmodified and stored as point data at the output. Any point data at the input is
        ignored and is not used by this filter.

        To change the image data's representation, the input cell centers are used to
        represent the output points. This has the effect of "shrinking" the
        input image dimensions by one along each axis (i.e. half the cell width on each
        side). For example, an image with 101 points and 100 cells along an axis at the
        input will have 100 points and 99 cells at the output. If the input has 1mm
        spacing, the axis size will also decrease from 100mm to 99mm.

        Since filters may be inherently cell-based (e.g. some :class:`~pyvista.DataSetFilters`)
        or may operate on point data exclusively (e.g. most :class:`~pyvista.ImageDataFilters`),
        re-meshing enables the same data to be used with either kind of filter while
        ensuring the input data to those filters has the appropriate representation.
        This filter is also useful when plotting image data to achieve a desired visual
        effect, such as plotting images as points instead of as voxel cells.

        .. note::
            Only the input's :attr:`~pyvista.ImageData.dimensions`, and
            :attr:`~pyvista.ImageData.origin` are modified by this filter. Other spatial
            properties such as :attr:`~pyvista.ImageData.spacing` and
            :attr:`~pyvista.ImageData.direction_matrix` are not affected.

        .. versionadded:: 0.44.0

        See Also
        --------
        points_to_cells
            Inverse of this filter to represent points as cells.
        :meth:`~pyvista.DataObjectFilters.cell_data_to_point_data`
            Resample cell data as point data without modifying the container.
        :meth:`~pyvista.DataObjectFilters.cell_data_to_point_data`
            Resample point data as cell data without modifying the container.

        Parameters
        ----------
        scalars : str, optional
            Name of cell data scalars to pass through to the output as point data. Use
            this parameter to restrict the output to only include the specified array.
            By default, all cell data arrays at the input are passed through as point
            data at the output.

        dimensionality : VectorLike[bool], Literal[0, 1, 2, 3, "0D", "1D", "2D", "3D", "preserve"]
            Control which dimensions will be modified by the filter.
            ``'preserve'`` is used by default.

            - Can be specified as a sequence of 3 boolean to allow modification on a per
                dimension basis.
            - ``0`` or ``'0D'``: convenience alias to output a 0D ImageData with
              dimensions ``(1, 1, 1)``. Only valid for 0D inputs.
            - ``1`` or ``'1D'``: convenience alias to output a 1D ImageData where
              exactly one dimension is greater than one, e.g. ``(>1, 1, 1)``. Only valid
              for 0D or 1D inputs.
            - ``2`` or ``'2D'``: convenience alias to output a 2D ImageData where
              exactly two dimensions are greater than one, e.g. ``(>1, >1, 1)``. Only
              valid for 0D, 1D, or 2D inputs.
            - ``3`` or ``'3D'``: convenience alias to output a 3D ImageData, where all
              three dimensions are greater than one, e.g. ``(>1, >1, >1)``. Valid for
              any 0D, 1D, 2D, or 3D inputs.
            - ``'preserve'`` (default): convenience alias to not modify singleton
              dimensions.

            .. note::
                This filter does not modify singleton dimensions with ``dimensionality``
                set as ``'preserve'`` by default.

        copy : bool, default: True
            Copy the input cell data before associating it with the output point data.
            If ``False``, the input and output will both refer to the same data array(s).

        Returns
        -------
        pyvista.ImageData
            Image with a point-based representation.

        Examples
        --------
        Load an image with cell data.

        >>> from pyvista import examples
        >>> image = examples.load_uniform()

        Show the current properties and cell arrays of the image.

        >>> image
        ImageData (...)
          N Cells:      729
          N Points:     1000
          X Bounds:     0.000e+00, 9.000e+00
          Y Bounds:     0.000e+00, 9.000e+00
          Z Bounds:     0.000e+00, 9.000e+00
          Dimensions:   10, 10, 10
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     2

        >>> image.cell_data.keys()
        ['Spatial Cell Data']

        Re-mesh the cells and cell data as points and point data.

        >>> points_image = image.cells_to_points()

        Show the properties and point arrays of the re-meshed image.

        >>> points_image
        ImageData (...)
          N Cells:      512
          N Points:     729
          X Bounds:     5.000e-01, 8.500e+00
          Y Bounds:     5.000e-01, 8.500e+00
          Z Bounds:     5.000e-01, 8.500e+00
          Dimensions:   9, 9, 9
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        >>> points_image.point_data.keys()
        ['Spatial Cell Data']

        Observe that:

        - The input cell array is now a point array
        - The output has one less array (the input point data is ignored)
        - The dimensions have decreased by one
        - The bounds have decreased by half the spacing
        - The output ``N Points`` equals the input ``N Cells``

        See :ref:`image_representations_example` for more examples using this filter.

        """
        if scalars is not None:
            field = self.get_array_association(scalars, preference='cell')
            if field != FieldAssociation.CELL:
                msg = (
                    f"Scalars '{scalars}' must be associated with cell data. "
                    f'Got {field.name.lower()} data instead.'
                )
                raise ValueError(msg)
        return self._remesh_points_cells(
            points_to_cells=False,
            scalars=scalars,
            dimensionality=dimensionality,
            copy=copy,
        )

    def _remesh_points_cells(  # type: ignore[misc]
        self: ImageData,
        *,
        points_to_cells: bool,
        scalars: str | None,
        dimensionality: VectorLike[bool] | Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D', 'preserve'],
        copy: bool,
    ):
        """Re-mesh points to cells or vice-versa.

        The active cell or point scalars at the input will be set as active point or
        cell scalars at the output, respectively.

        Parameters
        ----------
        points_to_cells : bool
            Set to ``True`` to re-mesh points to cells.
            Set to ``False`` to re-mesh cells to points.

        scalars : str
            If set, only these scalars are passed through.

        dimensionality : VectorLike[bool], Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D', 'preserve']
            Control which dimensions will be modified by the filter.

            - Can be specified as a sequence of 3 boolean to allow modification on a per
                dimension basis.
            - ``0`` or ``'0D'``: convenience alias to output a 0D ImageData with
              dimensions ``(1, 1, 1)``. Only valid for 0D inputs.
            - ``1`` or ``'1D'``: convenience alias to output a 1D ImageData where
              exactly one dimension is greater than one, e.g. ``(>1, 1, 1)``. Only valid
              for 0D or 1D inputs.
            - ``2`` or ``'2D'``: convenience alias to output a 2D ImageData where
              exactly two dimensions are greater than one, e.g. ``(>1, >1, 1)``. Only
              valid for 0D, 1D, or 2D inputs.
            - ``3`` or ``'3D'``: convenience alias to output a 3D ImageData, where all
              three dimensions are greater than one, e.g. ``(>1, >1, >1)``. Valid for
              any 0D, 1D, 2D, or 3D inputs.
            - ``'preserve'``: convenience alias to not modify singleton
              dimensions.

        copy : bool
            Copy the input data before associating it with the output data.

        Returns
        -------
        pyvista.ImageData
            Re-meshed image.

        """

        def _get_output_scalars(preference):
            active_scalars = self.active_scalars_name
            if active_scalars:
                field = self.get_array_association(
                    active_scalars,
                    preference=preference,
                )
                active_scalars = active_scalars if field.name.lower() == preference else None
            return active_scalars

        point_data = self.point_data
        cell_data = self.cell_data

        # Get data to use and operations to perform for the conversion
        new_image = pyvista.ImageData()

        if points_to_cells:
            output_scalars = scalars or _get_output_scalars('point')
            # Enlarge image so points become cell centers
            origin_operator = operator.sub
            dims_operator = operator.add  # Increase dimensions
            old_data = point_data
            new_data = new_image.cell_data
        else:  # cells_to_points
            output_scalars = scalars or _get_output_scalars('cell')
            # Shrink image so cell centers become points
            origin_operator = operator.add
            dims_operator = operator.sub  # Decrease dimensions
            old_data = cell_data
            new_data = new_image.point_data

        dims_mask, dims_result = self._validate_dimensional_operation(
            operation_mask=dimensionality, operator=dims_operator, operation_size=1
        )

        # Prepare the new image
        new_image.origin = origin_operator(
            self.origin,
            (np.array(self.spacing) / 2) * dims_mask,
        )
        extent_min = self.extent[::2]
        new_image.extent = (
            extent_min[0],
            extent_min[0] + dims_result[0] - 1,
            extent_min[1],
            extent_min[1] + dims_result[1] - 1,
            extent_min[2],
            extent_min[2] + dims_result[2] - 1,
        )
        new_image.spacing = self.spacing
        new_image.direction_matrix = self.direction_matrix

        # Check the validity of the operation
        if points_to_cells:
            if new_image.n_cells != self.n_points:
                msg = (
                    'Cannot re-mesh points to cells. The dimensions of the input'
                    f' {self.dimensions} is not compatible with the dimensions of the'
                    f' output {new_image.dimensions} and would require to map'
                    f' {self.n_points} points on {new_image.n_cells} cells.'
                )
                raise ValueError(msg)
        elif new_image.n_points != self.n_cells:
            msg = (
                'Cannot re-mesh cells to points. The dimensions of the input'
                f' {self.dimensions} is not compatible with the dimensions of the'
                f' output {new_image.dimensions} and would require to map'
                f' {self.n_cells} cells on {new_image.n_points} points.'
            )
            raise ValueError(msg)

        # Copy field data
        new_image.field_data.update(self.field_data)

        # Copy old data (point or cell) to new data (cell or point)
        array_names = [scalars] if scalars else old_data.keys()
        for array_name in array_names:
            new_data[array_name] = old_data[array_name].copy() if copy else old_data[array_name]

        new_image.set_active_scalars(output_scalars)
        return new_image

    def pad_image(
        self,
        pad_value: float | VectorLike[float] | Literal['wrap', 'mirror'] = 0.0,
        *,
        pad_size: int | VectorLike[int] = 1,
        dimensionality: VectorLike[bool]
        | Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D', 'preserve'] = 'preserve',
        scalars: str | None = None,
        pad_all_scalars: bool = False,
        progress_bar: bool = False,
        pad_singleton_dims: bool | None = None,
    ) -> pyvista.ImageData:
        """Enlarge an image by padding its boundaries with new points.

        .. versionadded:: 0.44.0

        Padded points may be mirrored, wrapped, or filled with a constant value. By
        default, all boundaries of the image are padded with a single constant value.

        This filter is designed to work with 1D, 2D, or 3D image data and will only pad
        non-singleton dimensions unless otherwise specified.

        Parameters
        ----------
        pad_value : float | sequence[float] | 'mirror' | 'wrap', default: 0.0
            Padding value(s) given to new points outside the original image extent.
            Specify:

            - a number: New points are filled with the specified constant value.
            - a vector: New points are filled with the specified multi-component vector.
            - ``'wrap'``: New points are filled by wrapping around the padding axis.
            - ``'mirror'``: New points are filled by mirroring the padding axis.

        pad_size : int | sequence[int], default: 1
            Number of points to add to the image boundaries. Specify:

            - A single value to pad all boundaries equally.
            - Two values, one for each ``(X, Y)`` axis, to apply symmetric padding to
              each axis independently.
            - Three values, one for each ``(X, Y, Z)`` axis, to apply symmetric padding
              to each axis independently.
            - Four values, one for each ``(-X, +X, -Y, +Y)`` boundary, to apply
              padding to each boundary independently.
            - Six values, one for each ``(-X, +X, -Y, +Y, -Z, +Z)`` boundary, to apply
              padding to each boundary independently.

        dimensionality : VectorLike[bool], Literal[1, 2, 3, "1D", "2D", "3D", "preserve"]
            Control which dimensions will be padded by the filter.
            ``'preserve'`` is used by default.

            - Can be specified as a sequence of 3 boolean to apply padding on a per
                dimension basis.
            - ``1`` or ``'1D'``: apply padding such that the output is a 1D ImageData
              where exactly one dimension is greater than one, e.g. ``(>1, 1, 1)``.
              Only valid for 0D or 1D inputs.
            - ``2`` or ``'2D'``: apply padding such that the output is a 2D ImageData
              where exactly two dimensions are greater than one, e.g. ``(>1, >1, 1)``.
              Only valid for 0D, 1D, or 2D inputs.
            - ``3`` or ``'3D'``: apply padding such that the output is a 3D ImageData,
              where all three dimensions are greater than one, e.g. ``(>1, >1, >1)``.
              Valid for any 0D, 1D, 2D, or 3D inputs.

            .. note::
                The ``pad_size`` for singleton dimensions is set to ``0`` by default, even
                if non-zero pad sizes are specified for these axes with this parameter.
                Set ``dimensionality`` to a value different than ``'preserve'`` to
                override this behavior and enable padding any or all dimensions.

            .. versionadded:: 0.45.0

        scalars : str, optional
            Name of scalars to pad. Defaults to currently active scalars. Unless
            ``pad_all_scalars`` is ``True``, only the specified ``scalars`` are included
            in the output.

        pad_all_scalars : bool, default: False
            Pad all point data scalars and include them in the output. This is useful
            for padding images with multiple scalars. If ``False``, only the specified
            ``scalars`` are padded.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        pad_singleton_dims : bool, optional
            Control whether to pad singleton dimensions.

            .. deprecated:: 0.45.0
                Deprecated, use ``dimensionality='preserve'`` instead of
                ``pad_singleton_dims=True`` and ``dimensionality='3D'`` instead of
                ``pad_singleton_dims=False``.

                Estimated removal on v0.48.0.

        Returns
        -------
        pyvista.ImageData
            Padded image.

        See Also
        --------
        crop, resample, contour_labels

        Examples
        --------
        Pad a grayscale image with a 100-pixel wide border. The padding is black
        (i.e. has a value of ``0``) by default.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>>
        >>> gray_image = examples.download_moonlanding_image()
        >>> gray_image.dimensions
        (630, 474, 1)
        >>> padded = gray_image.pad_image(pad_size=100)
        >>> padded.dimensions
        (830, 674, 1)

        Plot the image. To show grayscale images correctly, we define a custom plotting
        method.

        >>> def grayscale_image_plotter(image):
        ...     import vtk
        ...
        ...     actor = vtk.vtkImageActor()
        ...     actor.GetMapper().SetInputData(image)
        ...     actor.GetProperty().SetInterpolationTypeToNearest()
        ...     plot = pv.Plotter()
        ...     plot.add_actor(actor)
        ...     plot.view_xy()
        ...     plot.camera.tight()
        ...     return plot
        >>>
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad only the x-axis with a white border.

        >>> padded = gray_image.pad_image(pad_value=255, pad_size=(200, 0))
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad with wrapping.

        >>> padded = gray_image.pad_image('wrap', pad_size=100)
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad with mirroring.

        >>> padded = gray_image.pad_image('mirror', pad_size=100)
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad a color image using multi-component color vectors. Here, RGBA values are
        used.

        >>> color_image = examples.download_beach()
        >>> red = (255, 0, 0)  # RGB
        >>> padded = color_image.pad_image(pad_value=red, pad_size=50)
        >>>
        >>> plot_kwargs = dict(cpos='xy', zoom='tight', rgb=True, show_axes=False)
        >>> padded.plot(**plot_kwargs)

        Pad each edge of the image separately with a different color.

        >>> orange = pv.Color('orange').int_rgb
        >>> purple = pv.Color('purple').int_rgb
        >>> blue = pv.Color('blue').int_rgb
        >>> green = pv.Color('green').int_rgb
        >>>
        >>> padded = color_image.pad_image(orange, pad_size=(25, 0, 0, 0))
        >>> padded = padded.pad_image(purple, pad_size=(0, 25, 0, 0))
        >>> padded = padded.pad_image(blue, pad_size=(0, 0, 25, 0))
        >>> padded = padded.pad_image(green, pad_size=(0, 0, 0, 25))
        >>>
        >>> padded.plot(**plot_kwargs)

        """
        # Deprecated on v0.45.0, estimated removal on v0.48.0
        if pad_singleton_dims is not None:
            if pad_singleton_dims:
                warnings.warn(
                    'Use of `pad_singleton_dims=True` is deprecated. '
                    'Use `dimensionality="3D"` instead',
                    PyVistaDeprecationWarning,
                    stacklevel=2,
                )
                dimensionality = '3D'
            else:
                warnings.warn(
                    'Use of `pad_singleton_dims=False` is deprecated. '
                    'Use `dimensionality="preserve"` instead',
                    PyVistaDeprecationWarning,
                    stacklevel=2,
                )
                dimensionality = 'preserve'

        def _get_num_components(array_):
            return 1 if array_.ndim == 1 else array_.shape[1]

        # Validate scalars
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
        if field != FieldAssociation.POINT:
            msg = (
                f"Scalars '{scalars}' must be associated with point data. "
                f'Got {field.name.lower()} data instead.'
            )
            raise ValueError(msg)

        all_pad_sizes = _validate_padding(pad_size)

        # Combine size 2 by 2 to get a (3, ) shaped array
        dims_mask, _ = self._validate_dimensional_operation(
            operation_mask=dimensionality,
            operator=operator.add,
            operation_size=all_pad_sizes[::2] + all_pad_sizes[1::2],
        )
        all_pad_sizes = all_pad_sizes * np.repeat(dims_mask, 2)
        padded_extents = _pad_extent(self.GetExtent(), all_pad_sizes)  # type: ignore[attr-defined]

        # Validate pad value
        pad_multi_component = None  # Flag for multi-component constants
        error_msg = (
            f"Invalid pad value {pad_value}. Must be 'mirror' or 'wrap', or a "
            f'number/component vector for constant padding.'
        )
        if isinstance(pad_value, str):
            if pad_value == 'mirror':
                alg = _vtk.vtkImageMirrorPad()
            elif pad_value == 'wrap':
                alg = _vtk.vtkImageWrapPad()  # type: ignore[assignment]
            else:
                raise ValueError(error_msg)
        else:
            val = np.atleast_1d(pad_value)
            num_input_components = _get_num_components(self.active_scalars)  # type: ignore[attr-defined]
            if not (
                val.ndim == 1
                and (np.issubdtype(val.dtype, np.floating) or np.issubdtype(val.dtype, np.integer))
            ):
                raise ValueError(error_msg)
            if (num_value_components := len(val)) not in [1, num_input_components]:
                msg = (
                    f'Number of components ({num_value_components}) in pad value {pad_value} must '
                    f"match the number components ({num_input_components}) in array '{scalars}'."
                )
                raise ValueError(msg)
            val = np.broadcast_to(val, (num_input_components,))
            if num_input_components > 1:
                pad_multi_component = True
                data = self.point_data  # type: ignore[attr-defined]
                array_names = data.keys() if pad_all_scalars else [scalars]
                for array_name in array_names:
                    array = data[array_name]
                    if not np.array_equal(val, val.astype(array.dtype)):
                        msg = (
                            f"Pad value {pad_value} with dtype '{val.dtype.name}' is not "
                            f"compatible with dtype '{array.dtype}' of array {array_name}."
                        )
                        raise TypeError(msg)
                    if (n_comp := _get_num_components(data[array_name])) != num_input_components:
                        msg = (
                            f"Cannot pad array '{array_name}' with value {pad_value}. "
                            f"Number of components ({n_comp}) in '{array_name}' must match "
                            f'the number of components ({num_value_components}) in value.'
                            f'\nTry setting `pad_all_scalars=False` or update the array.'
                        )
                        raise ValueError(msg)
            else:
                pad_multi_component = False
            alg = _vtk.vtkImageConstantPad()  # type: ignore[assignment]

        alg.SetInputDataObject(self)
        alg.SetOutputWholeExtent(*padded_extents)

        def _get_padded_output(scalars_):
            """Update the active scalars and get the output.

            Includes special handling for padding with multi-component values.
            """

            def _update_and_get_output():
                _update_alg(alg, progress_bar=progress_bar, message='Padding image')
                return _get_output(alg)

            # Set scalars since the filter only operates on the active scalars
            self.set_active_scalars(scalars_, preference='point')  # type: ignore[attr-defined]
            if pad_multi_component is None:
                return _update_and_get_output()
            else:
                # Constant padding
                alg.SetConstant(val[0])  # type: ignore[attr-defined]
                output = _update_and_get_output()
                if pad_multi_component is False:
                    # Single component padding
                    return output
                else:  # Mulit-component padding
                    # The constant pad filter only pads with a single value.
                    # We need to apply the filter multiple times for each component.
                    output_scalars = output.active_scalars
                    num_output_components = _get_num_components(output_scalars)
                    for component in range(1, num_output_components):
                        alg.SetConstant(val[component])  # type: ignore[attr-defined]
                        output_scalars[:, component] = _update_and_get_output()[scalars_][
                            :,
                            component,
                        ]
                    output.point_data[scalars_] = output_scalars
                    return output

        output = _get_padded_output(scalars)

        # This filter pads only the active scalars, other arrays are returned empty.
        # We need to pad those other arrays or remove them from the output.
        for point_array in self.point_data:  # type: ignore[attr-defined]
            if point_array != scalars:
                if pad_all_scalars:
                    output[point_array] = _get_padded_output(point_array)[point_array]
                else:
                    output.point_data.remove(point_array)
        for cell_array in (data := output.cell_data):
            data.remove(cell_array)

        # Restore active scalars
        self.set_active_scalars(scalars, preference='point')  # type: ignore[attr-defined]
        return output

    def label_connectivity(
        self,
        *,
        scalars: str | None = None,
        scalar_range: (Literal['auto', 'foreground', 'vtk_default'] | VectorLike[float]) = 'auto',
        extraction_mode: Literal['all', 'largest', 'seeded'] = 'all',
        point_seeds: (MatrixLike[float] | VectorLike[float] | _vtk.vtkDataSet | None) = None,
        label_mode: Literal['size', 'constant', 'seeds'] = 'size',
        constant_value: int | None = None,
        inplace: bool = False,
        progress_bar: bool = False,
    ) -> tuple[pyvista.ImageData, NDArray[int], NDArray[int]]:
        """Find and label connected regions in a :class:`~pyvista.ImageData`.

        Only points whose `scalar` value is within the `scalar_range` are considered for
        connectivity. A 4-connectivity is used for 2D images or a 6-connectivity for 3D
        images. This filter operates on point-based data. If cell-based data are provided,
        they are re-meshed to a point-based representation using
        :func:`~pyvista.ImageDataFilters.cells_to_points` and the output is meshed back
        to a cell-based representation with :func:`~pyvista.ImageDataFilters.points_to_cells`,
        effectively filtering based on face connectivity. The connected regions are
        extracted and labelled according to the strategy defined by ``extraction_mode``
        and ``label_mode``, respectively. Unconnected regions are labelled with ``0`` value.

        .. versionadded:: 0.45.0

        Notes
        -----
        This filter implements :vtk:`vtkImageConnectivityFilter`.

        Parameters
        ----------
        scalars : str, optional
            Scalars to use to filter points. If ``None`` is provided, the scalars is
            automatically set, if possible.

        scalar_range : str, Literal['auto', 'foreground', 'vtk_default'], VectorLike[float]
            Points whose scalars value is within ``'scalar_range'`` are considered for
            connectivity. The bounds are inclusive.

            - ``'auto'``: (default) includes the full data range, similarly to
              :meth:`~pyvista.DataSetFilters.connectivity`.
            - ``'foreground'``: includes the full data range except the smallest value.
            - ``'vtk_default'``: default to [``0.5``, :const:`~vtk.VTK_DOUBLE_MAX`].
            - ``VectorLike[float]``: explicitly set the range.

            The bounds are always cast to floats since vtk expects doubles. The scalars
            data are also cast to floats to avoid unexpected behavior arising from implicit
            type conversion. The only exceptions is if both bounds are whole numbers, in
            which case the implicit conversion is safe. It will optimize resources consumption
            if the data are integers.

        extraction_mode : Literal['all', 'largest', 'seeded'], default: 'all'
            Determine how the connected regions are extracted. If ``'all'``, all connected
            regions are extracted. If ``'largest'``, only the largest region is extracted.
            If ``'seeded'``, only the regions that include the points defined with
            ``point_seeds`` are extracted.

        point_seeds : MatrixLike[float], VectorLike[float], :vtk:`vtkDataSet`, optional
            The point coordinates to use as seeds, specified as a (N, 3) array like or
            as a :vtk:`vtkDataSet`. Has no effect if ``extraction_mode`` is not
            ``'seeded'``.

        label_mode : Literal['size', 'constant', 'seeds'], default: 'size'
            Determine how the extracted regions are labelled. If ``'size'``, label regions
            by decreasing size (i.e., count of cells), starting at ``1``. If ``'constant'``,
            label with the provided ``constant_value``. If ``'seeds'``, label according to
            the seed order, starting at ``1``.

        constant_value : int, optional
            The constant label value to use. Has no effect if ``label_mode`` is not ``'seeds'``.

        inplace : bool, default: False
            If ``True``, perform an inplace labelling of the ImageData. Else, returns a
            new ImageData.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Either the input ImageData or a generated one where connected regions are
            labelled with a ``'RegionId'`` point-based or cell-based data.

        NDArray[int]
            The labels of each extracted regions.

        NDArray[int]
            The size (i.e., number of cells) of each extracted regions.

        See Also
        --------
        pyvista.DataSetFilters.connectivity
            Similar general-purpose filter that performs 1-connectivity.

        Examples
        --------
        Prepare a segmented grid.

        >>> import pyvista as pv
        >>> segmented_grid = pv.ImageData(dimensions=(4, 3, 3))
        >>> segmented_grid.cell_data['Data'] = [
        ...     0,
        ...     0,
        ...     0,
        ...     1,
        ...     0,
        ...     1,
        ...     1,
        ...     2,
        ...     0,
        ...     0,
        ...     0,
        ...     0,
        ... ]
        >>> segmented_grid.plot(show_edges=True)

        Label the connected regions. The cells with a ``0`` value are excluded from the
        connected regions and labelled with ``0``. The remaining cells define 3 different
        regions that are labelled by decreasing size.

        >>> connected, labels, sizes = segmented_grid.label_connectivity(
        ...     scalar_range='foreground'
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(connected.threshold(0.5), show_edges=True)
        >>> _ = pl.add_mesh(
        ...     connected.threshold(0.5, invert=True),
        ...     show_edges=True,
        ...     opacity=0.5,
        ... )
        >>> pl.show()

        Exclude the cell with a ``2`` value.

        >>> connected, labels, sizes = segmented_grid.label_connectivity(
        ...     scalar_range=[1, 1]
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(connected.threshold(0.5), show_edges=True)
        >>> _ = pl.add_mesh(
        ...     connected.threshold(0.5, invert=True),
        ...     show_edges=True,
        ...     opacity=0.5,
        ... )
        >>> pl.show()

        Label all connected regions with a constant value.

        >>> connected, labels, sizes = segmented_grid.label_connectivity(
        ...     scalar_range='foreground',
        ...     label_mode='constant',
        ...     constant_value=10,
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(connected.threshold(0.5), show_edges=True)
        >>> _ = pl.add_mesh(
        ...     connected.threshold(0.5, invert=True),
        ...     show_edges=True,
        ...     opacity=0.5,
        ... )
        >>> pl.show()

        Label only the regions that include seed points, by seed order.

        >>> points = [(2.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        >>> connected, labels, sizes = segmented_grid.label_connectivity(
        ...     scalar_range='foreground',
        ...     extraction_mode='seeded',
        ...     point_seeds=points,
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(connected.threshold(0.5), show_edges=True)
        >>> _ = pl.add_mesh(
        ...     connected.threshold(0.5, invert=True),
        ...     show_edges=True,
        ...     opacity=0.5,
        ... )
        >>> pl.show()

        """
        # Get a copy of input to not overwrite data
        input_mesh = self.copy()  # type: ignore[attr-defined]

        if scalars is None:
            set_default_active_scalars(input_mesh)
        else:
            input_mesh.set_active_scalars(scalars)

        # Make sure we have point data (required by the filter)
        field, scalars = input_mesh.active_scalars_info
        if field == FieldAssociation.CELL:
            # Convert to point data
            input_mesh = input_mesh.cells_to_points(
                scalars=scalars, dimensionality=(True, True, True), copy=False
            )

        # Set vtk algorithm
        alg = _vtk.vtkImageConnectivityFilter()
        alg.SetInputDataObject(input_mesh)

        # Set the scalar range considered for connectivity
        # vtk default is 0.5 to VTK_DOUBLE_MAX
        # See https://vtk.org/doc/nightly/html/classvtkImageConnectivityFilter.html
        if scalar_range != 'vtk_default':
            if scalar_range == 'auto':
                scalar_range = input_mesh.get_data_range(scalars, preference='point')
            elif scalar_range == 'foreground':
                unique_scalars = np.unique(input_mesh.point_data[scalars])
                scalar_range = (unique_scalars[1], unique_scalars[-1])
            else:
                scalar_range = _validation.validate_data_range(scalar_range)  # type: ignore[arg-type]
            alg.SetScalarRange(*scalar_range)

        scalars_casted_to_float = False
        if (
            scalar_range == 'vtk_default'
            or not float(scalar_range[0]).is_integer()
            or not float(scalar_range[1]).is_integer()
        ) and np.issubdtype(input_mesh.point_data[scalars].dtype, np.integer):
            input_mesh.point_data[scalars] = input_mesh.point_data[scalars].astype(float)
            # Keep track of the operation to cast back to int when the operation is inplace
            scalars_casted_to_float = True

        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)

        if extraction_mode == 'all':
            alg.SetExtractionModeToAllRegions()
        elif extraction_mode == 'largest':
            alg.SetExtractionModeToLargestRegion()
        elif extraction_mode == 'seeded':
            if point_seeds is None:
                msg = '`point_seeds` must be specified when `extraction_mode="seeded"`.'
                raise ValueError(msg)
            elif not isinstance(point_seeds, _vtk.vtkDataSet):
                point_seeds = pyvista.PointSet(point_seeds)

            alg.SetExtractionModeToSeededRegions()
            alg.SetSeedData(point_seeds)
        else:
            msg = (  # type: ignore[unreachable]
                f'Invalid `extraction_mode` "{extraction_mode}", '
                f'use "all", "largest", or "seeded".'
            )
            raise ValueError(msg)

        if label_mode == 'size':
            alg.SetLabelModeToSizeRank()
        elif label_mode == 'constant':
            alg.SetLabelModeToConstantValue()
            if constant_value is None:
                msg = f'`constant_value` must be provided when `extraction_mode`is "{label_mode}".'
                raise ValueError(msg)
            alg.SetLabelConstantValue(int(constant_value))
        elif label_mode == 'seeds':
            if point_seeds is None:
                msg = '`point_seeds` must be specified when `label_mode="seeds"`.'
                raise ValueError(msg)
            alg.SetLabelModeToSeedScalar()
        else:
            msg = f'Invalid `label_mode` "{label_mode}", use "size", "constant", or "seeds".'  # type: ignore[unreachable]
            raise ValueError(msg)

        _update_alg(
            alg, progress_bar=progress_bar, message='Identifying and Labelling Connected Regions'
        )

        output = _get_output(alg)

        labels: NDArray[int] = _vtk.vtk_to_numpy(alg.GetExtractedRegionLabels())

        sizes: NDArray[int] = _vtk.vtk_to_numpy(alg.GetExtractedRegionSizes())

        if field == FieldAssociation.CELL:
            # Convert back to cell data
            output = output.points_to_cells(dimensionality=(True, True, True), copy=False)
            # Add label `RegionId` to original dataset as cell data if required
            if inplace:
                self.cell_data['RegionId'] = output.cell_data['RegionId']  # type: ignore[attr-defined]
                self.set_active_scalars(name='RegionId', preference='cell')  # type: ignore[attr-defined]

        elif inplace:
            # Add label `RegionId` to original dataset as point data if required
            self.point_data['RegionId'] = output.point_data['RegionId']  # type: ignore[attr-defined]
            self.set_active_scalars(name='RegionId', preference='point')  # type: ignore[attr-defined]
            if scalars_casted_to_float:
                input_mesh.point_data[scalars] = input_mesh.point_data[scalars].astype(int)

        if inplace:
            return self, labels, sizes  # type: ignore[return-value]
        return output, labels, sizes

    def _validate_dimensional_operation(
        self,
        operation_mask: VectorLike[bool] | Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D', 'preserve'],
        operator: Callable,  # type: ignore[type-arg]
        operation_size: int | VectorLike[int],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Validate dimensional operations (internal helper).

        Return a dimensional mask to apply the operation on the source ImageData as well
        as the resulting dimensions.

        Provide convenience aliases ``0``, ``1``, ``2``, ``3`` ``'0D'``, ``'1D'``,
        ``'2D'``, ``'3D'``, and ``'preserve'`` to automatically provide a result with
        the proper dimensions. Raise errors if the desired output cannot be obtained.

        .. versionadded:: 0.45.0

        Parameters
        ----------
        operation_mask : VectorLike[bool], Literal['0D', '1D', '2D', '3D', 'preserve']
            The desired mask to control whether to resize dimensions.

            - Can be specified as a sequence of 3 boolean to allow modification on a per
                dimension basis.
            - ``0`` or ``'0D'``: convenience alias to output a 0D ImageData with
              dimensions ``(1, 1, 1)``. Only valid for 0D inputs.
            - ``1`` or ``'1D'``: convenience alias to output a 1D ImageData where
              exactly one dimension is greater than one, e.g. ``(>1, 1, 1)``. Only valid
              for 0D or 1D inputs.
            - ``2`` or ``'2D'``: convenience alias to output a 2D ImageData where
              exactly two dimensions are greater than one, e.g. ``(>1, >1, 1)``. Only
              valid for 0D, 1D, or 2D inputs.
            - ``3`` or ``'3D'``: convenience alias to output a 3D ImageData, where all
              three dimensions are greater than one, e.g. ``(>1, >1, >1)``. Valid for
              any 0D, 1D, 2D, or 3D inputs.
            - ``'preserve'``: convenience alias to not modify singleton
              dimensions.

        operator: Callable
            The operation that will be perform on the dimensions. Must be a :module:`~operator`.

        operation_size : int, VectorLike[int]
            The size of the operation, applied to all dimensions if specified as a ``int``
            or applied on a per dimension basis.

        Returns
        -------
        NDArray[bool]
            A (3, ) shaped mask array that indicates which dimensions will be modified.

        NDArray[int]
            A (3, ) shaped array that with the new ImageData dimensions after applying
            the operation.

        Examples
        --------
        Get the dimensions on which to operate to obtain a 2D output while adding ``2``.

        >>> import pyvista as pv
        >>> import operator
        >>> image = pv.ImageData(dimensions=(4, 1, 4))
        >>> image._validate_dimensional_operation(
        ...     operation_mask='2D',
        ...     operator=operator.add,
        ...     operation_size=2,
        ... )
        (array([ True, False,  True]), array([6, 1, 6]))

        """
        dimensions = np.asarray(self.dimensions)  # type: ignore[attr-defined]
        # Build an array of the operation size
        operation_size = _validation.validate_array3(operation_size, reshape=True, broadcast=True)

        if not isinstance(operation_mask, str) and operation_mask not in [0, 1, 2, 3]:
            # Build a bool array of the mask
            dimensions_mask = _validation.validate_array3(
                operation_mask,
                reshape=True,
                broadcast=False,
                must_have_dtype=bool,
                must_be_real=False,
            )

        elif operation_mask == 'preserve':
            # Ensure that singleton dims remain unmodified
            dimensions_mask = dimensions > 1

        else:
            # Validate that the target dimensionality is valid
            try:
                target_dimensionality = _validation.validate_dimensionality(operation_mask)  # type: ignore[arg-type]
            except ValueError:
                msg = (
                    f'`{operation_mask}` is not a valid `operation_mask`.'
                    ' Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D", "preserve"].'
                )
                raise ValueError(msg)

            # Brute force all possible combinations: only 8 combinations to test
            # dimensions_masks is ordered such as the behavior is predictable
            dimensions_masks = np.array(
                [
                    [True, True, True],
                    [True, True, False],
                    [True, False, True],
                    [False, True, True],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                ]
            )

            # Predict the resulting dimensions for all possible masks
            result_dims = operator(dimensions, operation_size * dimensions_masks)

            # Required number of singleton dimensions to satisfy the desired dimensionality
            nof_non_singleton = target_dimensionality
            nof_singleton = 3 - nof_non_singleton

            # Select the first admissible mask that produces the desired dimensionality
            try:
                dimensions_mask = dimensions_masks[
                    ((result_dims == 1).sum(axis=1) == nof_singleton)
                    & ((result_dims > 1).sum(axis=1) == nof_non_singleton)
                ][0]

            except IndexError:
                desired_dimensions = {
                    0: '(1, 1, 1)',
                    1: '(>1, 1, 1)',
                    2: '(>1, >1, 1)',
                    3: '(>1, >1, >1)',
                }[target_dimensionality]
                msg = (
                    f'The operation requires to {operator.__name__} at least {operation_size} '
                    f'dimension(s) to {self.dimensions}. A {operation_mask} ImageData with dims '  # type: ignore[attr-defined]
                    f'{desired_dimensions} cannot be obtained.'
                )
                raise ValueError(msg)

        # Check that the resulting dimensions are admissible
        dimensions_result = operator(dimensions, operation_size * dimensions_mask)

        if not (dimensions_result >= 1).all():
            msg = (
                f'The mask {operation_mask}, size {operation_size}, and '
                f'operation {operator.__name__} would result in {dimensions_result} '
                f'which contains <= 0 dimensions.'
            )
            raise ValueError(msg)

        return dimensions_mask, dimensions_result

    def resample(  # type: ignore[misc]
        self: ImageData,
        sample_rate: float | VectorLike[float] | None = None,
        interpolation: _InterpolationOptions = 'nearest',
        *,
        border_mode: Literal['clamp', 'wrap', 'mirror'] = 'clamp',
        reference_image: ImageData | None = None,
        dimensions: VectorLike[int] | None = None,
        anti_aliasing: bool = False,
        extend_border: bool | None = None,
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'point',
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        """Resample the image to modify its dimensions and spacing.

        The resampling can be controlled in several ways:

        #. Specify the output geometry using a ``reference_image``.

        #. Specify the ``dimensions`` explicitly.

        #. Specify the ``sample_rate`` explicitly.

        Use ``reference_image`` for full control of the resampled geometry. For
        all other options, the geometry is implicitly defined such that the resampled
        image fits the bounds of the input.

        This filter may be used to resample either point or cell data. For point data,
        this filter assumes the data is from discrete samples in space which represent
        pixels or voxels; the resampled bounds are therefore extended by 1/2 voxel
        spacing by default though this may be disabled.

        .. note::

            Singleton dimensions are not resampled by this filter, e.g. 2D images
            will remain 2D.

        .. versionadded:: 0.45

        Parameters
        ----------
        sample_rate : float | VectorLike[float], optional
            Sampling rate(s) to use. Can be a single value or vector of three values
            for each axis. Values greater than ``1.0`` will up-sample the axis and
            values less than ``1.0`` will down-sample it. Values must be greater than ``0``.

        interpolation : 'nearest', 'linear', 'cubic', 'lanczos', 'hamming', 'blackman', 'bspline'
            Interpolation mode to use.

            - ``'nearest'`` (default) duplicates (if upsampling) or removes (if downsampling)
              values but does not modify them.
            - ``'linear'`` and ``'cubic'`` use linear and cubic interpolation, respectively.
            - ``'lanczos'``, ``'hamming'``, and ``'blackman'`` use a windowed sinc filter
              and may be used to preserve sharp details and/or reduce image artifacts.
            - ``'bspline'`` uses an n-degree basis spline to smoothly interpolate across points.
              The default degree is ``3``, but can range from ``0`` to ``9``. Append the desired
              degree to the string to set it, e.g. ``'bspline5'`` for a 5th-degree B-spline.

            .. versionadded:: 0.47
                Added ``'bspline'`` interpolation.

            .. note::

                - use ``'nearest'`` for pixel art or categorical data such as segmentation masks
                - use ``'linear'`` for speed-critical tasks
                - use ``'cubic'`` for upscaling or general-purpose resampling
                - use ``'lanczos'`` for high-detail downsampling (at the cost of some ringing)
                - use ``'blackman'`` for minimizing ringing artifacts (at the cost of some detail)
                - use ``'hamming'`` for a balance between detail-preservation and reducing ringing

        border_mode : 'clamp' | 'wrap' | 'mirror', default: 'clamp'
            Controls the interpolation at the image's borders.

            - ``'clamp'`` - values outside the image are clamped to the nearest edge.
            - ``'wrap'`` - values outside the image are wrapped periodically along the axis.
            - ``'mirror'`` - values outside the image are mirrored at the boundary.

            .. versionadded:: 0.47

        reference_image : ImageData, optional
            Reference image to use. If specified, the input is resampled
            to match the geometry of the reference. The :attr:`~pyvista.ImageData.dimensions`,
            :attr:`~pyvista.ImageData.spacing`, :attr:`~pyvista.ImageData.origin`,
            :attr:`~pyvista.ImageData.offset`, and :attr:`~pyvista.ImageData.direction_matrix`
            of the resampled image will all match the reference image.

        dimensions : VectorLike[int], optional
            Set the output :attr:`~pyvista.ImageData.dimensions` of the resampled image.

            .. note::

                Dimensions is the number of `points` along each axis. If resampling
                `cell` data, each dimension should be one more than the number of
                desired output cells (since there are ``N`` cells and ``N+1`` points
                along each axis). See examples.

        anti_aliasing : bool, default: False
            Enable antialiasing. This will blur the image as part of the resampling
            to reduce image artifacts when down-sampling. Has no effect when up-sampling.

        extend_border : bool, optional
            Extend the apparent input border by approximately half the
            :attr:`~pyvista.ImageData.spacing`. If enabled, the bounds of the
            resampled points will be larger than the input image bounds.
            Enabling this option also has the effect that the re-sampled spacing
            will directly correlate with the resampled dimensions, e.g. if
            the dimensions are doubled the spacing will be halved. See examples.

            This option is enabled by default when resampling point data. Has no effect
            when resampling cell data or when a ``reference_image`` is provided.

        scalars : str, optional
            Name of scalars to resample. Defaults to currently active scalars.

        preference : str, default: 'point'
            When scalars is specified, this is the preferred array type to search
            for in the dataset.  Must be either ``'point'`` or ``'cell'``.

        inplace : bool, default: False
            If ``True``, resample the image inplace. By default, a new
            :class:`~pyvista.ImageData` instance is returned.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        ImageData
            Resampled image.

        See Also
        --------
        crop
            Crop image to remove points at the image's boundaries.

        :meth:`~pyvista.DataObjectFilters.sample`
            Resample array data from one mesh onto another.

        :meth:`~pyvista.DataSetFilters.interpolate`
            Interpolate values from one mesh onto another.

        :ref:`image_representations_example`
            Compare images represented as points vs. cells.

        Examples
        --------
        Create a small 2D grayscale image with dimensions ``3 x 2`` for demonstration.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> from pyvista import examples
        >>> image = pv.ImageData(dimensions=(3, 2, 1))
        >>> image.point_data['data'] = np.linspace(0, 255, 6, dtype=np.uint8)

        Define a custom plotter to show the image. Although the image data is defined
        as point data, we use :meth:`points_to_cells` to display the image as
        :attr:`~pyvista.CellType.PIXEL` (or :attr:`~pyvista.CellType.VOXEL`) cells
        instead. Grayscale coloring is used and the camera is adjusted to fit the image.

        >>> def image_plotter(image: pv.ImageData, clim=(0, 255)) -> pv.Plotter:
        ...     pl = pv.Plotter()
        ...     image = image.points_to_cells()
        ...     pl.add_mesh(
        ...         image,
        ...         lighting=False,
        ...         show_edges=True,
        ...         cmap='grey',
        ...         clim=clim,
        ...         show_scalar_bar=False,
        ...         line_width=3,
        ...     )
        ...     pl.view_xy()
        ...     pl.camera.tight()
        ...     pl.enable_anti_aliasing()
        ...     return pl

        Show the image.

        >>> plot = image_plotter(image)
        >>> plot.show()

        Use ``sample_rate`` to up-sample the image. ``'nearest'`` interpolation is
        used by default.

        >>> upsampled = image.resample(sample_rate=2.0)
        >>> plot = image_plotter(upsampled)
        >>> plot.show()

        Use ``'linear'`` interpolation. Note that the argument names ``sample_rate``
        and ``interpolation`` may be omitted.

        >>> upsampled = image.resample(2.0, 'linear')
        >>> plot = image_plotter(upsampled)
        >>> plot.show()

        Use ``'cubic'`` interpolation. Here we also specify the output
        ``dimensions`` explicitly instead of using ``sample_rate``.

        >>> upsampled = image.resample(dimensions=(6, 4, 1), interpolation='cubic')
        >>> plot = image_plotter(upsampled)
        >>> plot.show()

        Compare the relative physical size of the image before and after resampling.

        >>> image
        ImageData (...)
          N Cells:      2
          N Points:     6
          X Bounds:     0.000e+00, 2.000e+00
          Y Bounds:     0.000e+00, 1.000e+00
          Z Bounds:     0.000e+00, 0.000e+00
          Dimensions:   3, 2, 1
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        >>> upsampled
        ImageData (...)
          N Cells:      15
          N Points:     24
          X Bounds:     -2.500e-01, 2.250e+00
          Y Bounds:     -2.500e-01, 1.250e+00
          Z Bounds:     0.000e+00, 0.000e+00
          Dimensions:   6, 4, 1
          Spacing:      5.000e-01, 5.000e-01, 1.000e+00
          N Arrays:     1

        Note that the upsampled :attr:`~pyvista.ImageData.dimensions` are doubled and
        the :attr:`~pyvista.ImageData.spacing` is halved (as expected). Also note,
        however, that the physical bounds of the input differ from the output.
        The upsampled :attr:`~pyvista.ImageData.origin` also differs:

        >>> image.origin
        (0.0, 0.0, 0.0)
        >>> upsampled.origin
        (-0.25, -0.25, 0.0)

        This is because the resampling is done with ``extend_border`` enabled by default
        which adds a half cell-width border to the image and adjusts the origin and
        spacing such that the bounds match when the image is represented as cells.

        Apply :meth:`points_to_cells` to the input and resampled images and show that
        the bounds match.

        >>> image_as_cells = image.points_to_cells()
        >>> image_as_cells.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  2.5,
                    y_min = -0.5,
                    y_max =  1.5,
                    z_min =  0.0,
                    z_max =  0.0)

        >>> upsampled_as_cells = upsampled.points_to_cells()
        >>> upsampled_as_cells.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  2.5,
                    y_min = -0.5,
                    y_max =  1.5,
                    z_min =  0.0,
                    z_max =  0.0)

        Plot the two images together as wireframe to visualize them. The original is in
        red, and the resampled image is in black.

        >>> plt = pv.Plotter()
        >>> _ = plt.add_mesh(
        ...     image_as_cells, style='wireframe', color='red', line_width=10
        ... )
        >>> _ = plt.add_mesh(
        ...     upsampled_as_cells, style='wireframe', color='black', line_width=2
        ... )
        >>> plt.view_xy()
        >>> plt.camera.tight()
        >>> plt.show()

        Disable ``extend_border`` to force the input and output bounds of the points
        to be the same instead.

        >>> upsampled = image.resample(sample_rate=2, extend_border=False)

        Compare the two images again.

        >>> image
        ImageData (...)
          N Cells:      2
          N Points:     6
          X Bounds:     0.000e+00, 2.000e+00
          Y Bounds:     0.000e+00, 1.000e+00
          Z Bounds:     0.000e+00, 0.000e+00
          Dimensions:   3, 2, 1
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        >>> upsampled
        ImageData (...)
          N Cells:      15
          N Points:     24
          X Bounds:     0.000e+00, 2.000e+00
          Y Bounds:     0.000e+00, 1.000e+00
          Z Bounds:     0.000e+00, 0.000e+00
          Dimensions:   6, 4, 1
          Spacing:      4.000e-01, 3.333e-01, 1.000e+00
          N Arrays:     1

        This time the input and output bounds match without any further processing.
        Like before, the dimensions have doubled; unlike before, however, the spacing is
        not halved, but is instead smaller than half which is necessary to ensure the
        bounds remain the same. Also unlike before, the origin is unaffected:

        >>> image.origin
        (0.0, 0.0, 0.0)
        >>> upsampled.origin
        (0.0, 0.0, 0.0)

        All the above examples are with 2D images with point data. However, the filter
        also works with 3D volumes and will also work with cell data.

        Convert the 2D image with point data into a 3D volume with cell data and plot
        it for context.

        >>> volume = image.points_to_cells(dimensionality='3D')
        >>> volume.plot(show_edges=True, cmap='grey')

        Up-sample the volume. Set the sampling rate for each axis separately.

        >>> resampled = volume.resample(sample_rate=(3.0, 2.0, 1.0))
        >>> resampled.plot(show_edges=True, cmap='grey')

        Alternatively, we could have set the dimensions explicitly. Since we want
        ``9 x 4 x 1`` cells along the x-y-z axes (respectively), we set the dimensions
        to ``(10, 5, 2)``, i.e. one more than the desired number of cells.

        >>> resampled = volume.resample(dimensions=(10, 5, 2))
        >>> resampled.plot(show_edges=True, cmap='grey')

        Compare the bounds before and after resampling. Unlike with point data, the
        bounds are not (and cannot be) extended.

        >>> volume.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  2.5,
                    y_min = -0.5,
                    y_max =  1.5,
                    z_min = -0.5,
                    z_max =  0.5)
        >>> resampled.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  2.5,
                    y_min = -0.5,
                    y_max =  1.5,
                    z_min = -0.5,
                    z_max =  0.5)

        Use a reference image to control the resampling instead. Here we load two
        images with different dimensions:
        :func:`~pyvista.examples.downloads.download_puppy` and
        :func:`~pyvista.examples.downloads.download_gourds`.

        >>> bird = examples.download_bird()
        >>> bird.dimensions
        (458, 342, 1)

        >>> gourds = examples.download_gourds()
        >>> gourds.dimensions
        (640, 480, 1)

        Use ``reference_image`` to resample the bird to match the gourds geometry or
        vice-versa.

        >>> bird_resampled = bird.resample(reference_image=gourds)
        >>> bird_resampled.dimensions
        (640, 480, 1)

        >>> gourds_resampled = gourds.resample(reference_image=bird)
        >>> gourds_resampled.dimensions
        (458, 342, 1)

        Downsample the gourds image to 1/10th its original resolution using ``'lanczos'``
        interpolation.

        >>> downsampled = gourds.resample(1 / 8, 'lanczos')
        >>> downsampled.dimensions
        (80, 60, 1)

        Compare the downsampled image to the original and zoom in to show detail.

        >>> def compare_images_plotter(image1, image2):
        ...     plt = pv.Plotter(shape=(1, 2))
        ...     _ = plt.add_mesh(image1, rgba=True, show_edges=False, lighting=False)
        ...     plt.subplot(0, 1)
        ...     _ = plt.add_mesh(image2, rgba=True, show_edges=False, lighting=False)
        ...     plt.link_views()
        ...     plt.view_xy()
        ...     plt.camera.zoom(3.0)
        ...     return plt

        >>> plt = compare_images_plotter(gourds, downsampled)
        >>> plt.show()

        Note that downsampling can create image artifacts caused by aliasing. Enable
        anti-aliasing to smooth the image before resampling.

        >>> downsampled2 = gourds.resample(1 / 8, 'lanczos', anti_aliasing=True)

        Compare down-sampling with aliasing (left) to without aliasing (right).

        >>> plt = compare_images_plotter(downsampled, downsampled2)
        >>> plt.show()

        Load an MRI of a knee and downsample it.

        >>> knee = pv.examples.download_knee().resample(
        ...     0.1, 'linear', anti_aliasing=True
        ... )

        Crop and plot it.

        >>> knee = knee.crop(normalized_bounds=[0.2, 0.8, 0.2, 0.8, 0.0, 1.0])
        >>> vmin = knee.active_scalars.min()
        >>> vmax = knee.active_scalars.max()
        >>> plt = image_plotter(knee, clim=[vmin, vmax])
        >>> plt.show()

        Upsample it with B-spline interpolation. The interpolation is very smooth.

        >>> upsampled = knee.resample(2.0, 'bspline', border_mode='clamp')
        >>> plt = image_plotter(upsampled, clim=[vmin, vmax])
        >>> plt.show()

        Use the ``'wrap'`` border mode. Note how points at the border are brighter than previously,
        since the bright pixels from the opposite edge are now included in the interpolation.

        >>> upsampled = knee.resample(2.0, 'bspline', border_mode='wrap')
        >>> plt = image_plotter(upsampled, clim=[vmin, vmax])
        >>> plt.show()

        Compare B-spline interpolation to ``'hamming'``.

        >>> upsampled = knee.resample(2.0, 'hamming')
        >>> plt = image_plotter(upsampled, clim=[vmin, vmax])
        >>> plt.show()

        """

        def set_border_mode(
            obj: _vtk.vtkImageBSplineCoefficients | _vtk.vtkAbstractImageInterpolator,
        ):
            if border_mode == 'clamp':
                obj.SetBorderModeToClamp()
            elif border_mode == 'mirror':
                obj.SetBorderModeToMirror()
            elif border_mode == 'wrap':
                obj.SetBorderModeToRepeat()
            else:  # pragma: no cover
                msg = f"Unexpected border mode '{border_mode}'."  # type: ignore[unreachable]
                raise RuntimeError(msg)

        # Process scalars
        if scalars is None:
            field, name = set_default_active_scalars(self)
        else:
            name = scalars
            field = self.get_array_association(scalars, preference=preference)

        active_scalars = self.get_array(name, preference=field.name.lower())  # type: ignore[arg-type]

        # Validate interpolation and modify scalars as needed
        input_dtype = active_scalars.dtype
        has_int_scalars = input_dtype == np.int64
        _validation.check_contains(
            get_args(_InterpolationOptions),
            must_contain=interpolation,
            name='interpolation',
        )
        _validation.check_contains(
            ['clamp', 'wrap', 'mirror'],
            must_contain=border_mode,
            name='border_mode',
        )
        if has_int_scalars:
            # int (long long) is not supported by the filter so we cast to float
            input_image = self.copy(deep=False)
            input_image[name] = active_scalars.astype(float)
        else:
            input_image = self

        # Make sure we have point scalars
        processing_cell_scalars = field == FieldAssociation.CELL
        if processing_cell_scalars:
            if extend_border:
                msg = '`extend_border` cannot be set when resampling cell data.'
                raise ValueError(msg)
            dimensionality = input_image.dimensionality
            input_image = input_image.cells_to_points(scalars=scalars, copy=False)

        # Set default extend_border value
        if extend_border is None:
            # Only extend border with point data
            extend_border = not processing_cell_scalars
        elif extend_border and reference_image is not None:
            msg = '`extend_border` cannot be set when a `image_reference` is provided.'
            raise ValueError(msg)

        # Setup reference image
        if reference_image is None:
            # Use the input as a reference
            reference_image = pyvista.ImageData()
            reference_image.copy_structure(input_image)
            reference_image_provided = False
        else:
            if dimensions is not None or sample_rate is not None:
                msg = (
                    'Cannot specify a reference image along with `dimensions` or `sample_rate` '
                    'parameters.\n`reference_image` must define the geometry exclusively.'
                )
                raise ValueError(msg)
            _validation.check_instance(reference_image, pyvista.ImageData, name='reference_image')
            reference_image_provided = True

        # Use SetMagnificationFactors to indirectly set the dimensions.
        # To compute the magnification factors we first define input (old) and output
        # (new) dimensions.
        old_dimensions = np.array(input_image.dimensions)
        if sample_rate is not None:
            if reference_image_provided or dimensions is not None:
                msg = (
                    'Cannot specify a sample rate along with `reference_image` or `sample_rate` '
                    'parameters.\n`sample_rate` must define the sampling geometry exclusively.'
                )
                raise ValueError(msg)
            # Set reference dimensions from the sample rate
            sample_rate_ = _validation.validate_array3(
                sample_rate,
                broadcast=True,
                must_be_in_range=[0, np.inf],
                strict_lower_bound=True,
                name='sample_rate',
            )
            new_dimensions = old_dimensions * sample_rate_
        else:
            if dimensions is not None:
                dimensions_ = np.array(dimensions)
                dimensions_ = dimensions_ - 1 if processing_cell_scalars else dimensions_
                reference_image.dimensions = dimensions_
            new_dimensions = np.array(reference_image.dimensions)

        # Compute the magnification factors to use with the filter
        # Note that SetMagnificationFactors will multiply the factors by the extent
        # but we want to multiply the dimensions. These values are off by one.
        singleton_dims = old_dimensions == 1
        with np.errstate(divide='ignore', invalid='ignore'):
            # Ignore division by zero, this is fixed with singleton_dims on the next line
            magnification_factors = (new_dimensions - 1) / (old_dimensions - 1)
        magnification_factors[singleton_dims] = 1

        resize_filter = _vtk.vtkImageResize()
        resize_filter.SetInputData(input_image)
        resize_filter.SetResizeMethodToMagnificationFactors()
        resize_filter.SetMagnificationFactors(*magnification_factors)

        # Set interpolation mode
        interpolator: _vtk.vtkAbstractImageInterpolator
        if interpolation == 'nearest':
            interpolator = _vtk.vtkImageInterpolator()
            interpolator.SetInterpolationModeToNearest()
        elif interpolation == 'linear':
            interpolator = _vtk.vtkImageInterpolator()
            interpolator.SetInterpolationModeToLinear()
        elif interpolation == 'cubic':
            interpolator = _vtk.vtkImageInterpolator()
            interpolator.SetInterpolationModeToCubic()
        elif interpolation == 'lanczos':
            interpolator = _vtk.vtkImageSincInterpolator()
            interpolator.SetWindowFunctionToLanczos()
        elif interpolation == 'hamming':
            interpolator = _vtk.vtkImageSincInterpolator()
            interpolator.SetWindowFunctionToHamming()
        elif interpolation == 'blackman':
            interpolator = _vtk.vtkImageSincInterpolator()
            interpolator.SetWindowFunctionToBlackman()
        elif interpolation.startswith('bspline'):
            interpolator = _vtk.vtkImageBSplineInterpolator()
            # Set degree
            degree = 3 if interpolation.endswith('bspline') else int(interpolation[-1])
            interpolator.SetSplineDegree(degree)
            # Need to pre-compute coefficients
            coefficients = _vtk.vtkImageBSplineCoefficients()
            coefficients.SetInputData(input_image)
            set_border_mode(coefficients)
            _update_alg(
                coefficients, progress_bar=progress_bar, message='Computing spline coefficients.'
            )
            input_image = _get_output(coefficients)
        else:  # pragma: no cover
            msg = f"Unexpected interpolation mode '{interpolation}'."
            raise RuntimeError(msg)

        set_border_mode(interpolator)
        if anti_aliasing and np.any(magnification_factors < 1.0):
            if isinstance(interpolator, _vtk.vtkImageSincInterpolator):
                interpolator.AntialiasingOn()
            else:
                resize_filter.SetInputData(input_image.gaussian_smooth())

        resize_filter.SetInterpolator(interpolator)

        # Get output
        _update_alg(resize_filter, progress_bar=progress_bar, message='Resampling image.')
        output_image = _get_output(resize_filter).copy(deep=False)

        # Set geometry from the reference
        output_image.direction_matrix = reference_image.direction_matrix
        output_image.origin = reference_image.origin
        output_image.offset = reference_image.offset

        if reference_image_provided:
            output_image.spacing = reference_image.spacing
        else:
            # Need to fixup the spacing
            old_spacing = np.array(input_image.spacing)
            output_dimensions = np.array(output_image.dimensions)

            if extend_border and not processing_cell_scalars:
                # Compute spacing to have the same effective sample rate as the dimensions
                actual_sample_rate = output_dimensions / old_dimensions
                new_spacing = old_spacing / actual_sample_rate

                # This will enlarge the image, so we need to shift the origin accordingly
                # Shift the origin by 1/2 of the old and new spacing, but keep the spacing
                # unchanged for singleton dimensions.
                shift_old = old_spacing[~singleton_dims] / 2
                shift_new = new_spacing[~singleton_dims] / 2
                new_origin = np.array(input_image.origin)
                new_origin[~singleton_dims] += shift_new - shift_old

                output_image.origin = new_origin
            else:
                # Compute spacing to match bounds of input and dimensions of output
                size = np.array(input_image.bounds_size)
                if processing_cell_scalars:
                    new_spacing = (size + input_image.spacing) / output_dimensions
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # Ignore division by zero, this is fixed with
                        # singleton_dims on the next line
                        new_spacing = size / (output_dimensions - 1)

            # For singleton dimensions, keep the original spacing value
            new_spacing[singleton_dims] = old_spacing[singleton_dims]
            output_image.spacing = new_spacing

        if output_image.active_scalars_name == 'ImageScalars':
            output_image.rename_array('ImageScalars', name)

        if has_int_scalars:
            # Can safely cast to int to match input
            output_image.point_data[name] = output_image.point_data[name].astype(input_dtype)

        if processing_cell_scalars:
            # Convert back to cells. This modifies origin so we need to reset it.
            output_image = output_image.points_to_cells(
                scalars=name, copy=False, dimensionality=dimensionality
            )
            output_image.origin = (
                reference_image.origin if reference_image_provided else self.origin
            )
            output_image.point_data.clear()
        else:
            output_image.cell_data.clear()

        if inplace:
            self.copy_from(output_image)
            return self
        return output_image

    def select_values(  # type: ignore[misc]
        self: ImageData,
        values: (
            float | VectorLike[float] | MatrixLike[float] | dict[str, float] | dict[float, str]
        )
        | None = None,
        *,
        ranges: (
            VectorLike[float]
            | MatrixLike[float]
            | dict[str, VectorLike[float]]
            | dict[tuple[float, float], str]
        )
        | None = None,
        fill_value: float | VectorLike[float] = 0,
        replacement_value: float | VectorLike[float] | None = None,
        scalars: str | None = None,
        preference: Literal['point', 'cell'] = 'point',
        component_mode: Literal['any', 'all', 'multi'] | int = 'all',
        invert: bool = False,
        split: bool = False,
    ):
        """Select values of interest and fill the rest with a constant.

        Point or cell data may be selected with a single value, multiple values, a range
        of values, or any mix of values and ranges. This enables threshold-like
        filtering of data in a discontinuous manner to select a single label or groups
        of labels from categorical data, or to select multiple regions from continuous
        data. Selected values may optionally be split into separate meshes.

        The selected values are stored in an array with the same name as the input.

        .. versionadded:: 0.45

        Parameters
        ----------
        values : float | ArrayLike[float] | dict, optional
            Value(s) to select. Can be a number, an iterable of numbers, or a dictionary
            with numeric entries. For ``dict`` inputs, either its keys or values may be
            numeric, and the other field must be strings. The numeric field is used as
            the input for this parameter, and if ``split`` is ``True``, the string field
            is used to set the block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                When selecting multi-component values with ``component_mode=multi``,
                each value is specified as a multi-component scalar. In this case,
                ``values`` can be a single vector or an array of row vectors.

        ranges : ArrayLike[float] | dict, optional
            Range(s) of values to select. Can be a single range (i.e. a sequence of
            two numbers in the form ``[lower, upper]``), a sequence of ranges, or a
            dictionary with range entries. Any combination of ``values`` and ``ranges``
            may be specified together. The endpoints of the ranges are included in the
            selection. Ranges cannot be set when ``component_mode=multi``.

            For ``dict`` inputs, either its keys or values may be numeric, and the other
            field must be strings. The numeric field is used as the input for this
            parameter, and if ``split`` is ``True``, the string field is used to set the
            block names of the returned :class:`~pyvista.MultiBlock`.

            .. note::
                Use ``+/-`` infinity to specify an unlimited bound, e.g.:

                - ``[0, float('inf')]`` to select values greater than or equal to zero.
                - ``[float('-inf'), 0]`` to select values less than or equal to zero.

        fill_value : float | VectorLike[float], default: 0
            Value used to fill the image. Can be a single value or a multi-component
            vector. Non-selected parts of the image will have this value.

        replacement_value : float | VectorLike[float], optional
            Replacement value for the output array. Can be a single value or a
            multi-component vector. If provided, selected values will be replaced with
            the given value. If no value is given, the selected values are retained and
            returned as-is. Setting this value is useful for generating a binarized
            output array.

        scalars : str, optional
            Name of scalars to select from. Defaults to currently active scalars.

        preference : str, default: 'point'
            When ``scalars`` is specified, this is the preferred array type to search
            for in the dataset.  Must be either ``'point'`` or ``'cell'``.

        component_mode : int | 'any' | 'all' | 'multi', default: 'all'
            Specify the component(s) to use when ``scalars`` is a multi-component array.
            Has no effect when the scalars have a single component. Must be one of:

            - number: specify the component number as a 0-indexed integer. The selected
              component must have the specified value(s).
            - ``'any'``: any single component can have the specified value(s).
            - ``'all'``: all individual components must have the specified values(s).
            - ``'multi'``: the entire multi-component item must have the specified value.

        invert : bool, default: False
            Invert the selection. If ``True`` values are selected which do *not* have
            the specified values.

        split : bool, default: False
            If ``True``, each value in ``values`` and each range in ``range`` is
            selected independently and returned as a :class:`~pyvista.MultiBlock`.
            The number of blocks returned equals the number of input values and ranges.
            The blocks may be named if a dictionary is used as input. See ``values``
            and ``ranges`` for details.

            .. note::
                Output blocks may contain meshes with only the ``fill_value`` if no
                values meet the selection criteria.

        See Also
        --------
        image_threshold
            Similar filter for thresholding :class:`~pyvista.ImageData`.
        :meth:`~pyvista.DataSetFilters.extract_values`
            Similar threshold-like filter for extracting values from any dataset.
        :meth:`~pyvista.DataSetFilters.split_values`
            Split a mesh by value into separate meshes.
        :meth:`~pyvista.DataSetFilters.threshold`
            Generalized thresholding filter which returns a :class:`~pyvista.UnstructuredGrid`.

        Returns
        -------
        pyvista.ImageData or pyvista.MultiBlock
            Image with selected values or a composite of meshes with selected
            values, depending on ``split``.

        Examples
        --------
        Load a CT image. Here we load
        :func:`~pyvista.examples.downloads.download_whole_body_ct_male`.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> dataset = examples.download_whole_body_ct_male()
        >>> ct_image = dataset['ct']

        Show the initial data range.

        >>> ct_image.get_data_range()
        (np.int16(-1348), np.int16(3409))

        Select intensity values above ``150`` to select the bones.

        >>> bone_range = [150, float('inf')]
        >>> fill_value = -1000  # fill with intensity values corresponding to air
        >>> bone_image = ct_image.select_values(
        ...     ranges=bone_range, fill_value=fill_value
        ... )

        Show the new data range.

        >>> bone_image.get_data_range()
        (np.int16(-1000), np.int16(3409))

        Plot the selected values. Use ``'foreground'`` opacity to make the fill value
        transparent and the selected values opaque.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_volume(
        ...     bone_image,
        ...     opacity='foreground',
        ...     cmap='bone',
        ... )
        >>> pl.view_zx()
        >>> pl.camera.up = (0, 0, 1)
        >>> pl.show()

        Use ``'replacement_value'`` to binarize the selected values instead. The fill
        value, or background, is ``0`` by default.

        >>> bone_mask = ct_image.select_values(ranges=bone_range, replacement_value=1)
        >>> bone_mask.get_data_range()
        (np.int16(0), np.int16(1))

        Generate a surface contour of the mask and plot it.

        >>> surf = bone_mask.contour_labels()

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(surf, color=True)
        >>> pl.view_zx()
        >>> pl.camera.up = (0, 0, 1)
        >>> pl.show()

        Load a color image. Here we load :func:`~pyvista.examples.downloads.download_beach`.

        >>> image = examples.download_beach()
        >>> plot_kwargs = dict(
        ...     cpos='xy', rgb=True, lighting=False, zoom='tight', show_axes=False
        ... )
        >>> image.plot(**plot_kwargs)

        Select components from the image which have a strong red component.
        Use ``replacement_value`` to replace these pixels with a pure red color
        and ``fill_value`` to fill the rest of the image with white pixels.

        >>> white = [255, 255, 255]
        >>> red = [255, 0, 0]
        >>> red_range = [200, 255]
        >>> red_component = 0
        >>> selected = image.select_values(
        ...     ranges=red_range,
        ...     component_mode=red_component,
        ...     replacement_value=red,
        ...     fill_value=white,
        ... )

        >>> selected.plot(**plot_kwargs)

        """
        validated = self._validate_extract_values(
            values=values,
            ranges=ranges,
            scalars=scalars,
            preference=preference,
            component_mode=component_mode,
            split=split,
            mesh_type=pyvista.ImageData,
        )
        if isinstance(validated, tuple):
            (
                valid_values,
                valid_ranges,
                value_names,
                range_names,
                array,
                array_name,
                association,
                component_logic,
            ) = validated
        else:
            # Return empty dataset
            return validated

        kwargs = dict(
            values=valid_values,
            ranges=valid_ranges,
            array=array,
            association=association,
            component_logic=component_logic,
            invert=invert,
            array_name=array_name,
            fill_value=fill_value,
            replacement_value=replacement_value,
        )

        if split:
            return self._split_values(
                method=self._select_values,
                value_names=value_names,
                range_names=range_names,
                **kwargs,
            )

        return self._select_values(**kwargs)

    def _select_values(  # type: ignore[misc]
        self: ImageData,
        *,
        values,
        ranges,
        array,
        component_logic,
        invert,
        association,
        array_name,
        fill_value,
        replacement_value,
    ):
        id_mask = self._apply_component_logic_to_array(
            values=values,
            ranges=ranges,
            array=array,
            component_logic=component_logic,
            invert=invert,
        )

        # Generate output array
        input_array = cast(
            'pyvista.pyvista_ndarray',
            get_array(self, name=array_name, preference=association),
        )
        array_out = np.full_like(input_array, fill_value=fill_value)
        replacement_values = (
            input_array[id_mask] if replacement_value is None else replacement_value
        )
        array_out[id_mask] = replacement_values

        output = pyvista.ImageData()
        output.copy_structure(self)
        output[array_name] = array_out
        return output

    def concatenate(  # type: ignore[misc]
        self: ImageData,
        images: ImageData | Sequence[ImageData],
        axis: _AxisOptions | None = None,
        *,
        mode: _ConcatenateModeOptions | None = None,
        resample_kwargs: dict[str, Any] | None = None,
        dtype_policy: _ConcatenateDTypePolicyOptions | None = None,
        component_policy: _ConcatenateComponentPolicyOptions | None = None,
    ):
        """Combine multiple images into one.

        This filter uses :vtk:`vtkImageAppend` to combine multiple images. By default, images are
        concatenated along the specified ``axis``, and all images must have:

        #. identical dimensions except along the specified ``axis``,
        #. the same scalar dtype, and
        #. the same number of scalar components.

        Use ``mode`` for cases with mismatched dimensions, ``dtype_policy`` for cases with
        mismatched dtypes, and/or ``component_policy`` for cases with mismatched scalar components.

        The output has the same :attr:`~pyvista.ImageData.origin` and
        :attr:`~pyvista.ImageData.spacing` as the first input. The origin and spacing of all other
        inputs are ignored.

        .. versionadded:: 0.47

        Parameters
        ----------
        images : ImageData | Sequence[ImageData]
            The input image(s) to concatenate. The default active scalars are used for all images.

        axis : int | str, default: 'x'
            Axis along which the images are concatenated:

            - ``0`` or ``'x'``: x-axis
            - ``1`` or ``'y'``: y-axis
            - ``2`` or ``'z'``: z-axis

        mode : str, default: 'strict'
            Concatenation mode to use. This determines how images are placed in the output. All
            modes operate along the specified ``axis`` except for ``'preserve-extents'``.
            Specify one of:

            - ``'strict'``: all images must have identical dimensions except along the specified
              ``axis``.
            - ``'resample-off-axis'``: :meth:`resample` off-axis dimensions of concatenated images
              to match the input. The on-axis dimension is `not` resampled.
            - ``'resample-proportional'``: :meth:`resample` concatenated images proportionally to
              preserve their aspect ratio(s). For 3D cases, this may not be possible, and a
              ``ValueError`` may be raised.
            - ``'crop-off-axis'``: :meth:`crop` off-axis dimensions of concatenated images
              to match the input. The on-axis dimension is `not` cropped.
            - ``'crop-match'``: Use :meth:`crop` to center-crop concatenated images such that
              their dimensions match the input dimensions exactly.
            - ``'preserve-extents'``: the extents of all images are preserved and used to place the
              images in the output. The whole extent of the output is the union of the input whole
              extents. The origin and spacing is taken from the first input. ``axis`` is not used
              by this mode.

            .. note::
                For the ``crop`` and ``preserve-extents`` modes, any portion of the output not
                covered by the inputs is set to zero.

        dtype_policy : 'strict' | 'promote' | 'match', default: 'strict'
            - ``'strict'``: Do not cast any scalar array dtypes. All images being concatenated must
              have the same dtype, else a ``TypeError`` is raised.
            - ``'promote'``: Use :func:`numpy.result_type` to compute the dtype of the output
              image scalars. This option safely casts all input arrays to a common dtype before
              concatenating.
            - ``'match'``: Cast all array dtypes to match the input's dtype. This casting is
              unsafe as it may downcast values and lose precision.

        component_policy : 'strict' | 'promote_rgba', default: 'strict'
            - ``'strict'``: Do not modify the number of components of any scalars. All images being
              concatenated must have the number of components, else a ``ValueError`` is raised.
            - ``'promote_rgba'``: Increase the number of components if necessary. Grayscale scalars
              with one component may be promoted to RGB or RGBA scalars by duplicating values,
              and RGB scalars may be promoted to RGBA scalars by including an opacity component.

        resample_kwargs : dict, optional
            Keyword arguments passed to :meth:`resample` when using ``'resample-off-axis'`` or
            ``'reample-proportional'`` modes. Specify ``interpolation``, ``border_mode`` or
            ``anti_aliasing`` options.

        Returns
        -------
        ImageData
            The concatenated image.

        Examples
        --------
        .. pyvista-plot::
            :force_static:

            Load a 2D image: :func:`~pyvista.examples.downloads.download_beach`.

            >>> import pyvista as pv
            >>> from pyvista import examples
            >>> beach = examples.download_beach()

            Use :meth:`select_values` to make a second version with white values converted to black
            to distinguish it from the original.

            >>> white = [255, 255, 255]
            >>> black = [0, 0, 0]
            >>> beach_black = beach.select_values(white, fill_value=black, invert=True)

            Concatenate them along the x-axis.

            >>> concatenated = beach.concatenate(beach_black, axis='x')
            >>> plot_kwargs = dict(
            ...     rgb=True,
            ...     lighting=False,
            ...     cpos='xy',
            ...     zoom='tight',
            ...     show_axes=False,
            ...     show_scalar_bar=False,
            ... )
            >>> concatenated.plot(**plot_kwargs)

            Concatenate them along the y-axis.

            >>> concatenated = beach.concatenate(beach_black, axis='y')
            >>> concatenated.plot(**plot_kwargs)

            By default, concatenation requires that all off-axis dimensions match the input. Use
            the ``mode`` keyword to enable concatenation with mismatched dimensions.

            Load a second 2D image with different dimensions:
            :func:`~pyvista.examples.downloads.download_bird`.

            >>> bird = examples.download_bird()
            >>> bird.dimensions
            (458, 342, 1)
            >>> beach.dimensions
            (100, 100, 1)

            Concatenate using ``'resample-proportional'`` mode to preserve the aspect ratio of the
            concatenated image. Linear interpolation with antialiasing is used to avoid sampling
            artifacts.

            >>> resample_kwargs = {'interpolation': 'linear', 'anti_aliasing': True}
            >>> concatenated = beach.concatenate(
            ...     bird, mode='resample-proportional', resample_kwargs=resample_kwargs
            ... )
            >>> concatenated.dimensions
            (233, 100, 1)
            >>> concatenated.plot(**plot_kwargs)

            Use ``'resample-off-axis'`` to only resample off-axis dimensions. This option may
            distort the image.

            >>> concatenated = beach.concatenate(
            ...     bird, mode='resample-off-axis', resample_kwargs=resample_kwargs
            ... )
            >>> concatenated.dimensions
            (558, 100, 1)
            >>> concatenated.plot(**plot_kwargs)

            Use the ``'preserve-extents'`` mode. Using this mode naively may not produce the
            desired result, e.g. if we concatenate ``beach`` with ``bird``, the ``beach`` image is
            completely overwritten since their :attr:`~pyvista.ImageData.extent` fully overlap.

            >>> beach.extent
            (0, 99, 0, 99, 0, 0)
            >>> bird.extent
            (0, 457, 0, 341, 0, 0)

            >>> concatenated = beach.concatenate(bird, mode='preserve-extents')
            >>> concatenated.extent
            (0, 457, 0, 341, 0, 0)
            >>> concatenated.plot(**plot_kwargs)

            Set the ``beach`` :attr:`~pyvista.ImageData.offset` so that there is only partial
            overlap instead.

            >>> beach.offset = (-50, -50, 0)
            >>> beach.extent
            (-50, 49, -50, 49, 0, 0)

            >>> concatenated = beach.concatenate(bird, mode='preserve-extents')
            >>> concatenated.extent
            (-50, 457, -50, 341, 0, 0)
            >>> concatenated.plot(**plot_kwargs)

            Reverse the concatenation order.

            >>> concatenated = bird.concatenate(beach, mode='preserve-extents')
            >>> concatenated.plot(**plot_kwargs)

            Use ``'crop-off-axis'`` to only crop off-axis dimensions.

            >>> concatenated = beach.concatenate(bird, mode='crop-off-axis')
            >>> concatenated.plot(**plot_kwargs)

            Reverse the concatenation order.

            >>> concatenated = bird.concatenate(beach, mode='crop-off-axis')
            >>> concatenated.plot(**plot_kwargs)

            Use ``'crop-match'`` to center-crop the images to match the input's
            dimensions.

            >>> concatenated = beach.concatenate(bird, mode='crop-match')
            >>> concatenated.plot(**plot_kwargs)

            Reverse the concatenation order.

            >>> concatenated = bird.concatenate(beach, mode='crop-match')
            >>> concatenated.plot(**plot_kwargs)

            Load a binary image: :func:`~pyvista.examples.downloads.download_yinyang()`.

            >>> yinyang = examples.download_yinyang()

            Use ``component_policy`` to concatenate grayscale images with RGB(A) images.

            >>> concatenated = yinyang.concatenate(
            ...     beach, mode='resample-proportional', component_policy='promote_rgba'
            ... )
            >>> concatenated.plot(**plot_kwargs)

        """

        def _compute_dimensions(
            reference_dimensions: tuple[int, int, int], image_dimensions: tuple[int, int, int]
        ) -> tuple[int, int, int]:
            # Image must match the reference's non-concatenating axes exactly,
            # but we leave the image's concatenating axis unchanged
            new_dims = list(reference_dimensions)
            new_dims[axis_num] = image_dimensions[axis_num]
            return cast('tuple[int, int, int]', tuple(new_dims))

        def _compute_sample_rate(reference_image: ImageData, image: ImageData) -> float:
            ref_dims = np.array(reference_image.dimensions)
            img_dims = np.array(image.dimensions)
            # Try to preserve image aspect ratio
            off_axis = np.arange(3) != axis_num
            not_singleton = ref_dims != 1
            fixed_axes = off_axis & not_singleton

            # Resample the image proportionally to match the axes
            sample_rate_array = ref_dims[fixed_axes] / img_dims[fixed_axes]
            if sample_rate_array.size == 0:
                return 1.0
            sample_rate = sample_rate_array.tolist()[0]
            n_fixed_axes = np.count_nonzero(fixed_axes)

            if n_fixed_axes == 1:
                # No issues resampling to match the single axis
                return sample_rate
            elif n_fixed_axes == 2:
                # We must check that both axes have the same proportion for both images
                ref_ratio = ref_dims[fixed_axes][0] / ref_dims[fixed_axes][1]
                img_ratio = img_dims[fixed_axes][0] / img_dims[fixed_axes][1]
                if np.isclose(ref_ratio, img_ratio):
                    return sample_rate

            msg = (
                f'Unable to proportionally resample image with dimensions {image.dimensions} to '
                f'match\ninput dimensions {self.dimensions} for concatenation along axis '
                f'{axis_num}.'
            )
            raise ValueError(msg)

        # Validate mode
        if mode is not None:
            options = get_args(_ConcatenateModeOptions)
            _validation.check_contains(options, must_contain=mode, name='mode')
        else:
            mode = 'strict'

        # Validate axis
        mapping_axis_to_num = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
        mapping_axis_to_str = {'x': 'x', 'y': 'y', 'z': 'z', 0: 'x', 1: 'y', 2: 'z'}
        if axis is not None:
            if mode == 'preserve-extents':
                msg = "The axis keyword cannot be used with 'preserve-extents' mode."
                raise ValueError(msg)
            options = get_args(_AxisOptions)
            _validation.check_contains(options, must_contain=axis, name='axis')
            axis_num = mapping_axis_to_num[axis]
        else:
            axis_num = 0

        # Validate dtype policy
        if dtype_policy is not None:
            options = get_args(_ConcatenateDTypePolicyOptions)
            _validation.check_contains(options, must_contain=dtype_policy, name='dtype_policy')
        else:
            dtype_policy = 'strict'

        # Validate component policy
        if component_policy is not None:
            options = get_args(_ConcatenateComponentPolicyOptions)
            _validation.check_contains(
                options, must_contain=component_policy, name='component_policy'
            )
        else:
            component_policy = 'strict'

        all_images = [self, *images] if isinstance(images, Sequence) else [self, images]

        self_dimensions = self.dimensions
        all_dtypes: list[np.dtype] = []
        all_n_components: list[int] = []
        all_scalars: list[str] = []
        for i, img in enumerate(all_images):
            if i > 0:
                _validation.check_instance(img, pyvista.ImageData)

            # Create shallow copies so we can safely modify if needed
            img_shallow_copy = img.copy(deep=False)
            _, scalars = img_shallow_copy._validate_point_scalars()
            all_scalars.append(scalars)
            array = img.point_data[scalars]
            all_dtypes.append(array.dtype)
            n_components = array.shape[1] if array.ndim == 2 else array.ndim
            all_n_components.append(n_components)

            if i > 0 and mode != 'preserve-extents':
                if (dims := img.dimensions) != self_dimensions:
                    # Need to deal with the dimensions mismatch
                    if mode == 'strict':
                        # Allow mismatch only along concatenating axis
                        for ax in range(3):
                            if ax != axis_num and dims[ax] != self_dimensions[ax]:
                                image_num = f'{i - 1} ' if len(all_images) > 2 else ''
                                this_axis = mapping_axis_to_str[ax]
                                msg = (
                                    f'Image {image_num}dimensions {img.dimensions} must '
                                    f'match off-axis dimensions {self.dimensions} for axis '
                                    f'{axis_num}.\nGot {this_axis} dimension {dims[ax]}, expected '
                                    f'{self_dimensions[ax]}. Use the `mode` keyword to allow '
                                    f'concatenation with\nmismatched dimensions.'
                                )
                                raise ValueError(msg)
                    elif mode.startswith('resample'):
                        kwargs = {}
                        if resample_kwargs:
                            _validation.check_instance(resample_kwargs, dict)
                            allowed_kwargs = ('anti_aliasing', 'interpolation', 'border_mode')
                            for kwarg in resample_kwargs.keys():
                                _validation.check_contains(
                                    allowed_kwargs, must_contain=kwarg, name='resample_kwargs'
                                )
                            kwargs.update(resample_kwargs)

                        if mode == 'resample-off-axis':
                            kwargs['dimensions'] = _compute_dimensions(
                                self.dimensions, img.dimensions
                            )
                        else:  # mode == 'resample-proportional
                            kwargs['sample_rate'] = _compute_sample_rate(self, img)

                        img_shallow_copy = img_shallow_copy.resample(**kwargs)

                    elif mode == 'crop-off-axis':
                        dimensions = _compute_dimensions(
                            self.dimensions, img_shallow_copy.dimensions
                        )
                        img_shallow_copy = img_shallow_copy.crop(dimensions=dimensions)
                    elif mode == 'crop-match':
                        img_shallow_copy = img_shallow_copy.crop(dimensions=self_dimensions)

            if mode in [
                'resample-off-axis',
                'resample-proportional',
                'crop-off-axis',
                'crop-match',
            ]:
                # These modes should not be affected by offset, so we zero it
                img_shallow_copy.offset = (0, 0, 0)

            # Replace input with shallow copy
            all_images[i] = img_shallow_copy

        if len(set(all_dtypes)) > 1:
            # Need to cast all scalars to the same dtype
            if dtype_policy == 'strict':
                msg = (
                    f'The dtypes of the scalar arrays do not match. Got multiple '
                    f"dtypes: {set(all_dtypes)}.\nSet the dtype policy to 'promote' or "
                    f"'match' to cast the inputs to a single dtype."
                )
                raise TypeError(msg)
            elif dtype_policy == 'promote':
                dtype_out = np.result_type(*all_dtypes)
            else:  # dtype_policy == 'match'
                dtype_out = all_dtypes[0]

            for img, scalars in zip(all_images, all_scalars):
                array = img.point_data[scalars]
                img.point_data[scalars] = array.astype(dtype_out, copy=False)
        else:
            dtype_out = all_images[0].point_data[all_scalars[0]].dtype

        if len(set(all_n_components)) > 1:
            # Need to ensure all scalars have the same number of components
            if component_policy == 'strict':
                msg = (
                    f'The number of components in the scalar arrays do not match. Got n '
                    f'components: {set(all_n_components)}.\nSet the component policy to '
                    f"'promote_rgba' to automatically increase the number of components as needed."
                )
                raise ValueError(msg)
            else:  # component_policy == 'promote_rgba'
                if not set(all_n_components) < {1, 3, 4}:
                    msg = (
                        'Unable to promote scalar components. Only promotion for grayscale (1 '
                        'component), RGB (3 components),\nand RGBA (4 components) is supported. '
                        f'Got: {set(all_n_components)}'
                    )
                    raise ValueError(msg)
                target_n_components = max(all_n_components)
                for img, n_components, scalars in zip(all_images, all_n_components, all_scalars):
                    if n_components < target_n_components:
                        array = img.point_data[scalars]
                        if n_components < 3:
                            array = np.vstack((array, array, array)).T  # type: ignore[assignment]
                        if target_n_components == 4:
                            fill_value = (
                                np.iinfo(dtype_out).max
                                if np.issubdtype(dtype_out, np.integer)
                                else 1.0
                            )
                            new_array = np.full((len(array), 4), fill_value, dtype=dtype_out)
                            new_array[:, :3] = array
                            array = new_array  # type: ignore[assignment]

                        img.point_data[scalars] = array

        alg = _vtk.vtkImageAppend()
        alg.SetAppendAxis(axis_num)
        alg.SetPreserveExtents(mode == 'preserve-extents')

        for img in all_images:
            alg.AddInputData(img)

        _update_alg(alg)
        output = _get_output(alg)
        output.offset = self.offset
        return output


def _validate_padding(pad_size):
    # Process pad size to create a length-6 tuple (-X,+X,-Y,+Y,-Z,+Z)
    padding = np.atleast_1d(pad_size)
    if padding.ndim != 1:
        msg = f'Pad size must be one dimensional. Got {padding.ndim} dimensions.'
        raise ValueError(msg)
    if not np.issubdtype(padding.dtype, np.integer):
        msg = f'Pad size must be integers. Got dtype {padding.dtype.name}.'
        raise TypeError(msg)
    if np.any(padding < 0):
        msg = f'Pad size cannot be negative. Got {pad_size}.'
        raise ValueError(msg)

    length = len(padding)
    if length == 1:
        all_pad_sizes = np.broadcast_to(padding, (6,)).copy()
    elif length == 2:
        all_pad_sizes = np.array(
            (padding[0], padding[0], padding[1], padding[1], 0, 0),
        )
    elif length == 3:
        all_pad_sizes = np.array(
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
    elif length == 4:
        all_pad_sizes = np.array(
            (padding[0], padding[1], padding[2], padding[3], 0, 0),
        )
    elif length == 6:
        all_pad_sizes = padding
    else:
        msg = f'Pad size must have 1, 2, 3, 4, or 6 values, got {length} instead.'
        raise ValueError(msg)
    return all_pad_sizes


def _pad_extent(extent, padding):
    pad_xn, pad_xp, pad_yn, pad_yp, pad_zn, pad_zp = padding
    ext_xn, ext_xp, ext_yn, ext_yp, ext_zn, ext_zp = extent

    return (
        ext_xn - pad_xn,  # minX
        ext_xp + pad_xp,  # maxX
        ext_yn - pad_yn,  # minY
        ext_yp + pad_yp,  # maxY
        ext_zn - pad_zn,  # minZ
        ext_zp + pad_zp,  # maxZ
    )
