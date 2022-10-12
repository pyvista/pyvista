"""Module containing geometry helper functions."""

import collections.abc
import warnings

import numpy as np

import pyvista
from pyvista.utilities.misc import PyVistaDeprecationWarning


def voxelize(mesh, density=None, check_surface=True):
    """Voxelize mesh to UnstructuredGrid.

    Parameters
    ----------
    density : float, np.ndarray or collections.abc.Sequence
        The uniform size of the voxels when single float passed.
        A list of densities along x,y,z directions.
        Defaults to 1/100th of the mesh length.

    check_surface : bool
        Specify whether to check the surface for closure. If on, then the
        algorithm first checks to see if the surface is closed and
        manifold. If the surface is not closed and manifold, a runtime
        error is raised.

    Returns
    -------
    pyvista.UnstructuredGrid
        Voxelized unstructured grid of the original mesh.

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
        mesh = pyvista.wrap(mesh)
    if density is None:
        density = mesh.length / 100
    if isinstance(density, (int, float, np.number)):
        density_x, density_y, density_z = [density] * 3
    elif isinstance(density, (collections.abc.Sequence, np.ndarray)):
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
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pyvista.StructuredGrid(x, y, z)
    ugrid = pyvista.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(surface, tolerance=0.0, check_surface=check_surface)
    mask = selection.point_data['SelectedPoints'].view(np.bool_)

    # extract cells from point indices
    vox = ugrid.extract_points(mask)
    return vox


def create_grid(dataset, dimensions=(101, 101, 101)):
    """Create a uniform grid surrounding the given dataset.

    The output grid will have the specified dimensions and is commonly used
    for interpolating the input dataset.

    """
    bounds = np.array(dataset.bounds)
    if dimensions is None:
        # TODO: we should implement an algorithm to automatically determine an
        # "optimal" grid size by looking at the sparsity of the points in the
        # input dataset - I actually think VTK might have this implemented
        # somewhere
        raise NotImplementedError('Please specify dimensions.')
    dimensions = np.array(dimensions, dtype=int)
    image = pyvista.UniformGrid()
    image.dimensions = dimensions
    dims = dimensions - 1
    dims[dims == 0] = 1
    image.spacing = (bounds[1::2] - bounds[:-1:2]) / dims
    image.origin = bounds[::2]
    return image


def single_triangle():  # pragma: no cover
    """Create a single PolyData triangle."""
    warnings.warn(
        "Use of `single_triangle` is deprecated. Use `pyvista.Triangle` instead.",
        PyVistaDeprecationWarning,
    )
    points = np.zeros((3, 3))
    points[1] = [1, 0, 0]
    points[2] = [0.5, 0.707, 0]
    cells = np.array([[3, 0, 1, 2]])
    return pyvista.PolyData(points, cells)


def grid_from_sph_coords(theta, phi, r):
    """Create a structured grid from arrays of spherical coordinates.

    Parameters
    ----------
    theta: array-like
        Azimuthal angle in degrees ``[0, 360]``.
    phi: array-like
        Polar (zenith) angle in degrees ``[0, 180]``.
    r: array-like
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


def transform_vectors_sph_to_cart(theta, phi, r, u, v, w):
    """Transform vectors from spherical (r, phi, theta) to cartesian coordinates (z, y, x).

    Note the "reverse" order of arrays's axes, commonly used in geosciences.

    Parameters
    ----------
    theta : sequence
        Azimuthal angle in degrees ``[0, 360]`` of shape (M,)
    phi : sequence
        Polar (zenith) angle in degrees ``[0, 180]`` of shape (N,)
    r : sequence
        Distance (radius) from the point of origin of shape (P,)
    u : sequence
        X-component of the vector of shape (P, N, M)
    v : sequence
        Y-component of the vector of shape (P, N, M)
    w : sequence
        Z-component of the vector of shape (P, N, M)

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

    datasets : sequence of :class:`pyvista.Dataset`
        Sequence of datasets. Can be of any :class:`pyvista.Dataset`

    merge_points : bool, optional
        Merge equivalent points when ``True``. Defaults to ``True``.

    main_has_priority : bool, optional
        When this parameter is ``True`` and ``merge_points=True``,
        the arrays of the merging grids will be overwritten
        by the original main mesh.

    progress_bar : bool, optional
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

    >>> import pyvista
    >>> sphere = pyvista.Sphere(center=(0, 0, 1))
    >>> cube = pyvista.Cube()
    >>> mesh = pyvista.merge([cube, sphere])
    >>> mesh.plot()

    """
    if not isinstance(datasets, collections.abc.Sequence):
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
