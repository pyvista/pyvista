"""Module containing geometry helper functions."""

from __future__ import annotations

from collections.abc import Sequence
import os
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.core.filters import _update_alg
from pyvista.core.utilities.helpers import wrap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyvista import DataSet
    from pyvista import ImageData
    from pyvista import MultiBlock
    from pyvista import StructuredGrid
    from pyvista.core._typing_core import ArrayLikeFloat
    from pyvista.core._typing_core import VectorLikeFloat
    from pyvista.core._typing_core import VectorLikeInt


def create_grid(dataset: DataSet, dimensions: VectorLikeInt | None = (101, 101, 101)) -> ImageData:
    """Create a uniform grid surrounding the given dataset.

    The output grid will have the specified dimensions and is commonly used
    for interpolating the input dataset.

    Parameters
    ----------
    dataset : DataSet
        Input dataset used as a reference for the grid creation.
    dimensions : VectorLikeInt, default: (101, 101, 101)
        The dimensions of the grid to be created. Each value in the tuple
        represents the number of grid points along the corresponding axis.

    Raises
    ------
    NotImplementedError
        If the dimensions parameter is set to None. Currently, the function
        does not support automatically determining the "optimal" grid size
        based on the sparsity of the points in the input dataset.

    Returns
    -------
    ImageData
        A uniform grid with the specified dimensions that surrounds the input
        dataset.

    See Also
    --------
    pyvista.DataObjectFilters.resample_to_image
        Build a grid and resample the dataset onto it in a single call. Its voxels fit
        the dataset's bounds, whereas this grid's points lie on them.

    """
    bounds = np.array(dataset.bounds)
    if dimensions is None:
        # TODO: we should implement an algorithm to automatically determine an
        # "optimal" grid size by looking at the sparsity of the points in the
        # input dataset - I actually think VTK might have this implemented
        # somewhere
        msg = 'Please specify dimensions.'
        raise NotImplementedError(msg)
    dimensions = np.array(dimensions, dtype=int)
    image = pv.ImageData()
    image.dimensions = dimensions
    dims = dimensions - 1
    dims[dims == 0] = 1
    image.spacing = (bounds[1::2] - bounds[:-1:2]) / dims
    image.origin = bounds[::2]
    return image


def grid_from_sph_coords(
    theta: VectorLikeFloat, phi: VectorLikeFloat, r: VectorLikeFloat
) -> StructuredGrid:
    """Create a structured grid from arrays of spherical coordinates.

    Parameters
    ----------
    theta : VectorLikeFloat
        Azimuthal angle in degrees ``[0, 360]``.
    phi : VectorLikeFloat
        Polar (zenith) angle in degrees ``[0, 180]``.
    r : VectorLikeFloat
        Distance (radius) from the point of origin.

    Returns
    -------
    pyvista.StructuredGrid
        Structured grid.

    Notes
    -----
    The returned grid has no point normals. Warping it with
    :func:`~pyvista.DataSetFilters.warp_by_scalar` therefore moves every point
    along a single fixed direction rather than radially outward -- see that
    filter's notes. For a radial warp, use
    :func:`~pyvista.DataSetFilters.warp_by_vector` with the (normalized) point
    coordinates as the vector array instead.

    """
    x, y, z = np.meshgrid(np.radians(theta), np.radians(phi), r)
    # Transform grid to cartesian coordinates
    x_cart = z * np.sin(y) * np.cos(x)
    y_cart = z * np.sin(y) * np.sin(x)
    z_cart = z * np.cos(y)
    # Make a grid object
    return pv.StructuredGrid(x_cart, y_cart, z_cart)


def transform_vectors_sph_to_cart(  # numpydoc ignore=RT02
    *,
    theta: VectorLikeFloat,
    phi: VectorLikeFloat,
    r: VectorLikeFloat,
    u: ArrayLikeFloat,
    v: ArrayLikeFloat,
    w: ArrayLikeFloat,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Transform vectors from spherical (r, phi, theta) to Cartesian coordinates (z, y, x).

    Note the "reverse" order of arrays's axes, commonly used in geosciences.

    Parameters
    ----------
    theta : VectorLikeFloat
        Azimuthal angle in degrees ``[0, 360]`` of shape ``(M,)``.
    phi : VectorLikeFloat
        Polar (zenith) angle in degrees ``[0, 180]`` of shape ``(N,)``.
    r : VectorLikeFloat
        Distance (radius) from the point of origin of shape ``(P,)``.
    u : ArrayLikeFloat
        X-component of the vector of shape ``(M, N, P)`` with length-one axes dropped.
    v : ArrayLikeFloat
        Y-component of the vector of shape ``(M, N, P)`` with length-one axes dropped.
    w : ArrayLikeFloat
        Z-component of the vector of shape ``(M, N, P)`` with length-one axes dropped.

    Returns
    -------
    u_t, v_t, w_t : :class:`numpy.ndarray`
        Arrays of transformed x-, y-, z-components, respectively.

    """
    xx, yy, _ = np.meshgrid(np.radians(theta), np.radians(phi), r, indexing='ij')
    th, ph = xx.squeeze(), yy.squeeze()

    # Transform wind components from spherical to cartesian coordinates
    # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    u_t = np.sin(ph) * np.cos(th) * w + np.cos(ph) * np.cos(th) * v - np.sin(th) * u
    v_t = np.sin(ph) * np.sin(th) * w + np.cos(ph) * np.sin(th) * v + np.cos(th) * u
    w_t = np.cos(ph) * w - np.sin(ph) * v

    return u_t, v_t, w_t


def cartesian_to_spherical(
    x: NDArray[np.integer | np.floating],
    y: NDArray[np.integer | np.floating],
    z: NDArray[np.integer | np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert 3D Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : numpy.ndarray
        Cartesian coordinates.

    Returns
    -------
    r : numpy.ndarray
        Radial distance.

    phi : numpy.ndarray
        Angle (radians) with respect to the polar axis. Also known
        as polar angle.

    theta : numpy.ndarray
        Angle (radians) of rotation from the initial meridian plane.
        Also known as azimuthal angle.

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>> grid = pv.ImageData(dimensions=(3, 3, 3))
    >>> x, y, z = grid.points.T
    >>> r, phi, theta = pv.cartesian_to_spherical(x, y, z)

    """
    xy2 = x**2 + y**2
    r = np.sqrt(xy2 + z**2)
    phi = np.arctan2(np.sqrt(xy2), z)  # the polar angle in radian angles
    theta = np.arctan2(y, x)  # the azimuth angle in radian angles

    return r, phi, theta


def spherical_to_cartesian(
    r: ArrayLikeFloat, phi: ArrayLikeFloat, theta: ArrayLikeFloat
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert Spherical coordinates to 3D Cartesian coordinates.

    Parameters
    ----------
    r : ArrayLikeFloat
        Radial distance.

    phi : ArrayLikeFloat
        Angle (radians) with respect to the polar axis. Also known
        as polar angle.

    theta : ArrayLikeFloat
        Angle (radians) of rotation from the initial meridian plane.
        Also known as azimuthal angle.

    Returns
    -------
    output : numpy.ndarray, numpy.ndarray, numpy.ndarray
        Cartesian coordinates.

    """
    s = np.sin(phi)
    x = s * r * np.cos(theta)
    y = s * r * np.sin(theta)
    z = np.cos(phi) * r
    return x, y, z


def merge(
    datasets: Sequence[DataSet] | MultiBlock[Any],
    *,
    merge_points: bool = True,
    main_has_priority: bool | None = None,
    progress_bar: bool = False,
) -> DataSet:
    """Merge several datasets.

    .. note::
       The behavior of this filter varies from the
       :func:`PolyDataFilters.boolean_union` filter. This filter
       does not attempt to create a manifold mesh and will include
       internal surfaces when two meshes overlap.

    .. warning::

        The merge order of this filter depends on the installed version
        of VTK. For example, if merging meshes ``a``, ``b``, and ``c``,
        the merged order is ``bca`` for VTK<9.5 and ``abc`` for VTK>=9.5.
        This may be a breaking change for some applications. If only
        merging two meshes, it may be possible to maintain `some` backwards
        compatibility by swapping the input order of the two meshes,
        though this may also affect the merged arrays and is therefore
        not fully backwards-compatible.

    Parameters
    ----------
    datasets : sequence[:class:`pyvista.DataSet`] | :class:`pyvista.MultiBlock`
        Sequence of datasets. Can be of any :class:`pyvista.DataSet`. A
        :class:`pyvista.MultiBlock` is accepted, and raises ``TypeError`` if any
        of its blocks is not a dataset.

    merge_points : bool, default: True
        Merge equivalent points when ``True``.

    main_has_priority : bool, optional
        When this parameter is ``True`` and ``merge_points=True``, the arrays
        of the merging grids will be overwritten by the original main mesh.

        .. deprecated:: 0.46

            Omit this keyword; the main mesh already has priority. ``False`` raises
            :class:`ValueError` with VTK 9.5.0 or later and still selects the other
            mesh with older VTK. It will be removed in a future version.

    progress_bar : bool, default: False
        Display a progress bar to indicate progress.

    Returns
    -------
    pyvista.DataSet
        :class:`pyvista.PolyData` if all items in datasets are
        :class:`pyvista.PolyData`, otherwise returns a
        :class:`pyvista.UnstructuredGrid`.

    Examples
    --------
    Merge two polydata datasets.

    >>> import pyvista as pv
    >>> sphere = pv.Sphere(center=(0, 0, 1))
    >>> cube = pv.Cube()
    >>> mesh = pv.merge([cube, sphere])
    >>> mesh.plot()

    """
    if not isinstance(datasets, Sequence):
        msg = f'Expected a sequence, got {type(datasets).__name__}'  # type: ignore[unreachable]
        raise TypeError(msg)

    if len(datasets) < 1:
        msg = 'Expected at least one dataset.'
        raise ValueError(msg)

    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, pv.DataSet):
            msg = f'Expected pyvista.DataSet, not {type(dataset).__name__} at index {i}'
            raise TypeError(msg)

    return datasets[0].merge(
        datasets[1:],
        merge_points=merge_points,
        main_has_priority=main_has_priority,
        progress_bar=progress_bar,
    )


def perlin_noise(
    amplitude: float, freq: Sequence[float], phase: Sequence[float]
) -> _vtk.vtkPerlinNoise:
    """Return the implicit function that implements Perlin noise.

    Uses :vtk:`vtkPerlinNoise` and computes a Perlin noise field as
    an implicit function. :vtk:`vtkPerlinNoise` is a concrete
    implementation of :vtk:`vtkImplicitFunction`. Perlin noise,
    originally described by Ken Perlin, is a non-periodic and
    continuous noise function useful for modeling real-world objects.

    The amplitude and frequency of the noise pattern are
    adjustable. This implementation of Perlin noise is derived closely
    from Greg Ward's version in Graphics Gems II.

    Parameters
    ----------
    amplitude : float
        Amplitude of the noise function.

        ``amplitude`` can be negative. The noise function varies
        randomly between ``-|Amplitude|`` and
        ``|Amplitude|``. Therefore the range of values is
        ``2*|Amplitude|`` large. The initial amplitude is 1.

    freq : sequence[float]
        The frequency, or physical scale, of the noise function
        (higher is finer scale).

        The frequency can be adjusted per axis, or the same for all axes.

    phase : sequence[float]
        Set/get the phase of the noise function.

        This parameter can be used to shift the noise function within
        space (perhaps to avoid a beat with a noise pattern at another
        scale). Phase tends to repeat about every unit, so a phase of
        0.5 is a half-cycle shift.

    Returns
    -------
    :vtk:`vtkPerlinNoise`
        Instance of :vtk:`vtkPerlinNoise` to a Perlin noise field as an
        implicit function. Use with :func:`~pyvista.sample_function`.

    Examples
    --------
    Create a Perlin noise function with an amplitude of 0.1, frequency
    for all axes of 1, and a phase of 0 for all axes.

    >>> import pyvista as pv
    >>> noise = pv.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))

    Sample Perlin noise over a structured grid and plot it.

    >>> grid = pv.sample_function(noise, bounds=[0, 5, 0, 5, 0, 5])
    >>> grid.plot()

    """
    noise = _vtk.vtkPerlinNoise()
    noise.SetAmplitude(amplitude)
    noise.SetFrequency(freq)
    noise.SetPhase(phase)
    return noise


def sample_function(
    function: _vtk.vtkImplicitFunction,
    *,
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    dim: Sequence[int] = (50, 50, 50),
    compute_normals: bool = False,
    output_type: np.dtype = np.double,  # type: ignore[assignment]
    capping: bool = False,
    cap_value: float = sys.float_info.max,
    scalar_arr_name: str = 'scalars',
    normal_arr_name: str = 'normals',
    progress_bar: bool = False,
) -> ImageData:
    """Sample an implicit function over a structured point set.

    Uses :vtk:`vtkSampleFunction`

    This method evaluates an implicit function and normals at each
    point in a :vtk:`vtkStructuredPoints`. The user can specify the
    sample dimensions and location in space to perform the sampling.

    To create closed surfaces (in conjunction with the
    :vtk:`vtkContourFilter`), capping can be turned on to set a particular
    value on the boundaries of the sample space.

    Parameters
    ----------
    function : :vtk:`vtkImplicitFunction`
        Implicit function to evaluate.  For example, the function
        generated from :func:`~pyvista.perlin_noise`.

    bounds : sequence[float], default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        Specify the bounds in the format of:

        - ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

    dim : sequence[float], default: (50, 50, 50)
        Dimensions of the data on which to sample in the format of
        ``(xdim, ydim, zdim)``.

    compute_normals : bool, default: False
        Enable or disable the computation of normals.

    output_type : numpy.dtype, default: numpy.double
        Set the output scalar type.  One of the following:

        - ``np.float64``
        - ``np.float32``
        - ``np.int64``
        - ``np.uint64``
        - ``np.int32``
        - ``np.uint32``
        - ``np.int16``
        - ``np.uint16``
        - ``np.int8``
        - ``np.uint8``

    capping : bool, default: False
        Enable or disable capping. If capping is enabled, then the outer
        boundaries of the structured point set are set to cap value. This can
        be used to ensure surfaces are closed.

    cap_value : float, default: sys.float_info.max
        Capping value used with the ``capping`` parameter.

    scalar_arr_name : str, default: "scalars"
        Set the scalar array name for this data set.

    normal_arr_name : str, default: "normals"
        Set the normal array name for this data set.

    progress_bar : bool, default: False
        Display a progress bar to indicate progress.

    Returns
    -------
    pyvista.ImageData
        Uniform grid with sampled data.

    Examples
    --------
    Sample Perlin noise over a structured grid in 3D.

    >>> import pyvista as pv
    >>> noise = pv.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))
    >>> grid = pv.sample_function(
    ...     noise, bounds=[0, 3.0, -0, 1.0, 0, 1.0], dim=(60, 20, 20)
    ... )
    >>> grid.plot(cmap='gist_earth_r', show_scalar_bar=False, show_edges=True)

    Sample Perlin noise in 2D and plot it.

    >>> noise = pv.perlin_noise(0.1, (5, 5, 5), (0, 0, 0))
    >>> surf = pv.sample_function(noise, dim=(200, 200, 1))
    >>> surf.plot()

    """
    samp = _vtk.vtkSampleFunction()
    samp.SetImplicitFunction(function)
    samp.SetSampleDimensions(dim)  # type: ignore[call-overload]
    samp.SetModelBounds(bounds)
    samp.SetComputeNormals(compute_normals)
    samp.SetCapping(capping)
    samp.SetCapValue(cap_value)
    samp.SetNormalArrayName(normal_arr_name)
    samp.SetScalarArrayName(scalar_arr_name)

    if output_type == np.float64:
        samp.SetOutputScalarTypeToDouble()
    elif output_type == np.float32:
        samp.SetOutputScalarTypeToFloat()
    elif output_type == np.int64:
        if os.name == 'nt':
            msg = 'This function on Windows only supports int32 or smaller'
            raise ValueError(msg)
        samp.SetOutputScalarTypeToLong()
    elif output_type == np.uint64:
        if os.name == 'nt':
            msg = 'This function on Windows only supports int32 or smaller'
            raise ValueError(msg)
        samp.SetOutputScalarTypeToUnsignedLong()
    elif output_type == np.int32:
        samp.SetOutputScalarTypeToInt()
    elif output_type == np.uint32:
        samp.SetOutputScalarTypeToUnsignedInt()
    elif output_type == np.int16:
        samp.SetOutputScalarTypeToShort()
    elif output_type == np.uint16:
        samp.SetOutputScalarTypeToUnsignedShort()
    elif output_type == np.int8:
        samp.SetOutputScalarTypeToChar()
    elif output_type == np.uint8:
        samp.SetOutputScalarTypeToUnsignedChar()
    else:
        msg = f'Invalid output_type {output_type}'
        raise ValueError(msg)

    _update_alg(samp, progress_bar=progress_bar, message='Sampling')
    return cast('ImageData', wrap(samp.GetOutput()))
