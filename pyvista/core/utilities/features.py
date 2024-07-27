"""Module containing geometry helper functions."""

from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk

from .helpers import wrap


def voxelize(mesh, density=None, check_surface=True):
    """Voxelize mesh to UnstructuredGrid.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to voxelize.

    density : float | array_like[float]
        The uniform size of the voxels when single float passed.
        A list of densities along x,y,z directions.
        Defaults to 1/100th of the mesh length.

    check_surface : bool, default: True
        Specify whether to check the surface for closure. If on, then the
        algorithm first checks to see if the surface is closed and
        manifold. If the surface is not closed and manifold, a runtime
        error is raised.

    Returns
    -------
    pyvista.UnstructuredGrid
        Voxelized unstructured grid of the original mesh.

    Notes
    -----
    Prior to version 0.39.0, this method improperly handled the order of
    structured coordinates.

    Examples
    --------
    Create an equal density voxelized mesh.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.download_bunny_coarse().clean()
    >>> vox = pv.voxelize(mesh, density=0.01)
    >>> vox.plot(show_edges=True)

    Create a voxelized mesh using unequal density dimensions.

    >>> vox = pv.voxelize(mesh, density=[0.01, 0.005, 0.002])
    >>> vox.plot(show_edges=True)

    """
    if not pyvista.is_pyvista_dataset(mesh):
        mesh = wrap(mesh)
    if density is None:
        density = mesh.length / 100
    if isinstance(density, (int, float, np.number)):
        density_x, density_y, density_z = [density] * 3
    elif isinstance(density, (Sequence, np.ndarray)):
        density_x, density_y, density_z = density
    else:
        raise TypeError(f'Invalid density {density!r}, expected number or array-like.')

    # check and pre-process input mesh
    surface = mesh.extract_geometry()  # filter preserves topology
    if not surface.faces.size:
        # we have a point cloud or an empty mesh
        raise ValueError('Input mesh must have faces for voxelization.')
    if not surface.is_all_triangles:
        # reduce chance for artifacts, see gh-1743
        surface.triangulate(inplace=True)

    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(x_min, x_max, density_x)
    y = np.arange(y_min, y_max, density_y)
    z = np.arange(z_min, z_max, density_z)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    # indexing='ij' is used here in order to make grid and ugrid with x-y-z ordering, not y-x-z ordering
    # see https://github.com/pyvista/pyvista/pull/4365

    # Create unstructured grid from the structured grid
    grid = pyvista.StructuredGrid(x, y, z)
    ugrid = pyvista.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(surface, tolerance=0.0, check_surface=check_surface)
    mask = selection.point_data['SelectedPoints'].view(np.bool_)

    # extract cells from point indices
    return ugrid.extract_points(mask)


def voxelize_volume(mesh, density=None, check_surface=True):
    """Voxelize mesh to create a RectilinearGrid voxel volume.

    Creates a voxel volume that encloses the input mesh and discretizes the cells
    within the volume that intersect or are contained within the input mesh.
    ``InsideMesh``, an array in ``cell_data``, is ``1`` for cells inside and ``0`` outside.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to voxelize.

    density : float | array_like[float]
        The uniform size of the voxels when single float passed.
        Nonuniform voxel size if a list of values are passed along x,y,z directions.
        Defaults to 1/100th of the mesh length.

    check_surface : bool, default: True
        Specify whether to check the surface for closure. If on, then the
        algorithm first checks to see if the surface is closed and
        manifold. If the surface is not closed and manifold, a runtime
        error is raised.

    Returns
    -------
    pyvista.RectilinearGrid
        RectilinearGrid as voxelized volume with discretized cells.

    See Also
    --------
    pyvista.voxelize
    pyvista.DataSetFilters.select_enclosed_points

    Examples
    --------
    Create an equal density voxel volume from input mesh.

    >>> import pyvista as pv
    >>> import numpy as np

    Load file from PyVista examples.

    >>> from pyvista import examples
    >>> mesh = examples.download_cow()

    Create an equal density voxel volume and plot the result.

    >>> vox = pv.voxelize_volume(mesh, density=0.15)
    >>> cpos = [(15, 3, 15), (0, 0, 0), (0, 0, 0)]
    >>> vox.plot(scalars='InsideMesh', show_edges=True, cpos=cpos)

    Slice the voxel volume to view ``InsideMesh``.

    >>> slices = vox.slice_orthogonal()
    >>> slices.plot(scalars='InsideMesh', show_edges=True)

    Create a voxel volume from unequal density dimensions and plot result.

    >>> vox = pv.voxelize_volume(mesh, density=[0.15, 0.15, 0.5])
    >>> vox.plot(scalars='InsideMesh', show_edges=True, cpos=cpos)

    Slice the unequal density voxel volume to view ``InsideMesh``.

    >>> slices = vox.slice_orthogonal()
    >>> slices.plot(scalars='InsideMesh', show_edges=True, cpos=cpos)

    """
    mesh = wrap(mesh)
    if density is None:
        density = mesh.length / 100
    if isinstance(density, (int, float, np.number)):
        density_x, density_y, density_z = [density] * 3
    elif isinstance(density, (Sequence, np.ndarray)):
        density_x, density_y, density_z = density
    else:
        raise TypeError(f'Invalid density {density!r}, expected number or array-like.')

    # check and pre-process input mesh
    surface = mesh.extract_geometry()  # filter preserves topology
    if not surface.faces.size:
        # we have a point cloud or an empty mesh
        raise ValueError('Input mesh must have faces for voxelization.')
    if not surface.is_all_triangles:
        # reduce chance for artifacts, see gh-1743
        surface.triangulate(inplace=True)

    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(x_min, x_max, density_x)
    y = np.arange(y_min, y_max, density_y)
    z = np.arange(z_min, z_max, density_z)

    # Create a RectilinearGrid
    voi = pyvista.RectilinearGrid(x, y, z)

    # get part of the mesh within the mesh's bounding surface.
    selection = voi.select_enclosed_points(surface, tolerance=0.0, check_surface=check_surface)
    mask_vol = selection.point_data['SelectedPoints'].view(np.bool_)

    # Get voxels that fall within input mesh boundaries
    cell_ids = np.unique(voi.extract_points(np.argwhere(mask_vol))["vtkOriginalCellIds"])

    # Create new element of grid where all cells _within_ mesh boundary are
    # given new name 'MeshCells' and a discrete value of 1
    voi['InsideMesh'] = np.zeros(voi.n_cells)
    voi['InsideMesh'][cell_ids] = 1

    return voi


def create_grid(dataset, dimensions=(101, 101, 101)):
    """Create a uniform grid surrounding the given dataset.

    The output grid will have the specified dimensions and is commonly used
    for interpolating the input dataset.

    Parameters
    ----------
    dataset : DataSet
        Input dataset used as a reference for the grid creation.
    dimensions : tuple of int, default: (101, 101, 101)
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

    """
    bounds = np.array(dataset.bounds)
    if dimensions is None:
        # TODO: we should implement an algorithm to automatically determine an
        # "optimal" grid size by looking at the sparsity of the points in the
        # input dataset - I actually think VTK might have this implemented
        # somewhere
        raise NotImplementedError('Please specify dimensions.')
    dimensions = np.array(dimensions, dtype=int)
    image = pyvista.ImageData()
    image.dimensions = dimensions
    dims = dimensions - 1
    dims[dims == 0] = 1
    image.spacing = (bounds[1::2] - bounds[:-1:2]) / dims
    image.origin = bounds[::2]
    return image


def grid_from_sph_coords(theta, phi, r):
    """Create a structured grid from arrays of spherical coordinates.

    Parameters
    ----------
    theta : array_like[float]
        Azimuthal angle in degrees ``[0, 360]``.
    phi : array_like[float]
        Polar (zenith) angle in degrees ``[0, 180]``.
    r : array_like[float]
        Distance (radius) from the point of origin.

    Returns
    -------
    pyvista.StructuredGrid
        Structured grid.

    """
    x, y, z = np.meshgrid(np.radians(theta), np.radians(phi), r)
    # Transform grid to cartesian coordinates
    x_cart = z * np.sin(y) * np.cos(x)
    y_cart = z * np.sin(y) * np.sin(x)
    z_cart = z * np.cos(y)
    # Make a grid object
    return pyvista.StructuredGrid(x_cart, y_cart, z_cart)


def transform_vectors_sph_to_cart(theta, phi, r, u, v, w):  # numpydoc ignore=RT02
    """Transform vectors from spherical (r, phi, theta) to cartesian coordinates (z, y, x).

    Note the "reverse" order of arrays's axes, commonly used in geosciences.

    Parameters
    ----------
    theta : array_like[float]
        Azimuthal angle in degrees ``[0, 360]`` of shape ``(M,)``.
    phi : array_like[float]
        Polar (zenith) angle in degrees ``[0, 180]`` of shape ``(N,)``.
    r : array_like[float]
        Distance (radius) from the point of origin of shape ``(P,)``.
    u : array_like[float]
        X-component of the vector of shape ``(P, N, M)``.
    v : array_like[float]
        Y-component of the vector of shape ``(P, N, M)``.
    w : array_like[float]
        Z-component of the vector of shape ``(P, N, M)``.

    Returns
    -------
    u_t, v_t, w_t : :class:`numpy.ndarray`
        Arrays of transformed x-, y-, z-components, respectively.

    """
    xx, yy, _ = np.meshgrid(np.radians(theta), np.radians(phi), r, indexing="ij")
    th, ph = xx.squeeze(), yy.squeeze()

    # Transform wind components from spherical to cartesian coordinates
    # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    u_t = np.sin(ph) * np.cos(th) * w + np.cos(ph) * np.cos(th) * v - np.sin(th) * u
    v_t = np.sin(ph) * np.sin(th) * w + np.cos(ph) * np.sin(th) * v + np.cos(th) * u
    w_t = np.cos(ph) * w - np.sin(ph) * v

    return u_t, v_t, w_t


def cartesian_to_spherical(x, y, z):
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


def spherical_to_cartesian(r, phi, theta):
    """Convert Spherical coordinates to 3D Cartesian coordinates.

    Parameters
    ----------
    r : numpy.ndarray
        Radial distance.

    phi : numpy.ndarray
        Angle (radians) with respect to the polar axis. Also known
        as polar angle.

    theta : numpy.ndarray
        Angle (radians) of rotation from the initial meridian plane.
        Also known as azimuthal angle.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Cartesian coordinates.
    """
    s = np.sin(phi)
    x = r * s * np.cos(theta)
    y = r * s * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def merge(
    datasets,
    merge_points=True,
    main_has_priority=True,
    progress_bar=False,
):
    """Merge several datasets.

    .. note::
       The behavior of this filter varies from the
       :func:`PolyDataFilters.boolean_union` filter. This filter
       does not attempt to create a manifold mesh and will include
       internal surfaces when two meshes overlap.

    Parameters
    ----------
    datasets : sequence[:class:`pyvista.Dataset`]
        Sequence of datasets. Can be of any :class:`pyvista.Dataset`.

    merge_points : bool, default: True
        Merge equivalent points when ``True``.

    main_has_priority : bool, default: True
        When this parameter is ``True`` and ``merge_points=True``, the arrays
        of the merging grids will be overwritten by the original main mesh.

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
        raise TypeError(f"Expected a sequence, got {type(datasets).__name__}")

    if len(datasets) < 1:
        raise ValueError("Expected at least one dataset.")

    first = datasets[0]
    if not isinstance(first, pyvista.DataSet):
        raise TypeError(f"Expected pyvista.DataSet, not {type(first).__name__}")

    return datasets[0].merge(
        datasets[1:],
        merge_points=merge_points,
        main_has_priority=main_has_priority,
        progress_bar=progress_bar,
    )


def perlin_noise(amplitude, freq: Sequence[float], phase: Sequence[float]):
    """Return the implicit function that implements Perlin noise.

    Uses ``vtk.vtkPerlinNoise`` and computes a Perlin noise field as
    an implicit function. ``vtk.vtkPerlinNoise`` is a concrete
    implementation of ``vtk.vtkImplicitFunction``. Perlin noise,
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
    vtk.vtkPerlinNoise
        Instance of ``vtk.vtkPerlinNoise`` to a Perlin noise field as an
        implicit function. Use with :func:`pyvista.sample_function()
        <pyvista.core.utilities.features.sample_function>`.

    Examples
    --------
    Create a Perlin noise function with an amplitude of 0.1, frequency
    for all axes of 1, and a phase of 0 for all axes.

    >>> import pyvista as pv
    >>> noise = pv.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))

    Sample Perlin noise over a structured grid and plot it.

    >>> grid = pv.sample_function(noise, [0, 5, 0, 5, 0, 5])
    >>> grid.plot()

    """
    noise = _vtk.vtkPerlinNoise()
    noise.SetAmplitude(amplitude)
    noise.SetFrequency(freq)
    noise.SetPhase(phase)
    return noise


def sample_function(
    function: _vtk.vtkImplicitFunction,
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    dim: Sequence[int] = (50, 50, 50),
    compute_normals: bool = False,
    output_type: np.dtype = np.double,  # type: ignore[assignment, type-arg]
    capping: bool = False,
    cap_value: float = sys.float_info.max,
    scalar_arr_name: str = "scalars",
    normal_arr_name: str = "normals",
    progress_bar: bool = False,
):
    """Sample an implicit function over a structured point set.

    Uses ``vtk.vtkSampleFunction``

    This method evaluates an implicit function and normals at each
    point in a ``vtk.vtkStructuredPoints``. The user can specify the
    sample dimensions and location in space to perform the sampling.

    To create closed surfaces (in conjunction with the
    vtkContourFilter), capping can be turned on to set a particular
    value on the boundaries of the sample space.

    Parameters
    ----------
    function : vtk.vtkImplicitFunction
        Implicit function to evaluate.  For example, the function
        generated from :func:`perlin_noise() <pyvista.core.utilities.features.perlin_noise>`.

    bounds : sequence[float], default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        Specify the bounds in the format of:

        - ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

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
    ...     noise, [0, 3.0, -0, 1.0, 0, 1.0], dim=(60, 20, 20)
    ... )
    >>> grid.plot(
    ...     cmap='gist_earth_r', show_scalar_bar=False, show_edges=True
    ... )

    Sample Perlin noise in 2D and plot it.

    >>> noise = pv.perlin_noise(0.1, (5, 5, 5), (0, 0, 0))
    >>> surf = pv.sample_function(noise, dim=(200, 200, 1))
    >>> surf.plot()

    See :ref:`perlin_noise_2d_example` for a full example using this function.

    """
    # internal import to avoide circular dependency
    from pyvista.core.filters import _update_alg

    samp = _vtk.vtkSampleFunction()
    samp.SetImplicitFunction(function)
    samp.SetSampleDimensions(dim)
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
            raise ValueError('This function on Windows only supports int32 or smaller')
        samp.SetOutputScalarTypeToLong()
    elif output_type == np.uint64:
        if os.name == 'nt':
            raise ValueError('This function on Windows only supports int32 or smaller')
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
        raise ValueError(f'Invalid output_type {output_type}')

    _update_alg(samp, progress_bar=progress_bar, message='Sampling')
    return wrap(samp.GetOutput())
