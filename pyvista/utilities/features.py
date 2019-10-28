import ctypes
import numpy as np

import pyvista


def voxelize(mesh, density):
    """voxelize mesh to UnstructuredGrid"""
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pyvista.StructuredGrid(x, y, z)
    ugrid = pyvista.UnstructuredGrid(grid)

    # get part of the mesh within the mesh
    selection = ugrid.select_enclosed_points(mesh, tolerance=0.0)
    mask = selection.point_arrays['SelectedPoints'].view(np.bool)

    # extract cells from point indices
    return ugrid.extract_points(mask)


def create_grid(dataset, dimensions=(101, 101, 101)):
    """Creates a uniform grid surrounding the given dataset with the specified
    dimensions. This grid is commonly used for interpolating the input dataset.
    """
    bounds = np.array(dataset.bounds)
    if dimensions is None:
        # TODO: we should implement an algorithm to automatically determine an
        # "optimal" grid size by looking at the sparsity of the points in the
        # input dataset - I actaully think VTK might have this implemented
        # somewhere
        raise NotImplementedError('Please specifiy dimensions.')
    dimensions = np.array(dimensions, dtype=int)
    image = pyvista.UniformGrid()
    image.dimensions = dimensions
    dims = (dimensions - 1)
    dims[dims == 0] = 1
    image.spacing = (bounds[1::2] - bounds[:-1:2]) / dims
    image.origin = bounds[::2]
    return image


def single_triangle():
    """ A single PolyData triangle """
    points = np.zeros((3, 3))
    points[1] = [1, 0, 0]
    points[2] = [0.5, 0.707, 0]
    cells = np.array([[3, 0, 1, 2]], ctypes.c_long)
    return pyvista.PolyData(points, cells)


def grid_from_sph_coords(theta, phi, r):
    """
    Create a structured grid from arrays of spherical coordinates.

    Parameters
    ----------
    theta: array-like
        Azimuthal angle in degrees [0, 360)
    phi: array-like
        Polar (zenith) angle in degrees [0, 180]
    r: array-like
        Distance (radius) from the point of origin

    Returns
    -------
    pyvista.StructuredGrid
    """
    x, y, z = np.meshgrid(np.radians(theta), np.radians(phi), r)
    # Transform grid to cartesian coordinates
    x_cart = z * np.sin(y) * np.cos(x)
    y_cart = z * np.sin(y) * np.sin(x)
    z_cart = z * np.cos(y)
    # Make a grid object
    return pyvista.StructuredGrid(x_cart, y_cart, z_cart)


def transform_vectors_sph_to_cart(theta, phi, r, u, v, w):
    """
    Transform vectors from spherical (r, phi, theta) to cartesian coordinates (z, y, x).

    Note the "reverse" order of arrays's axes, commonly used in geosciences.

    Parameters
    ----------
    theta: array-like
        Azimuthal angle in degrees [0, 360) of shape (M,)
    phi: array-like
        Polar (zenith) angle in degrees [0, 180] of shape (N,)
    r: array-like
        Distance (radius) from the point of origin of shape (P,)
    u: array-like
        X-component of the vector of shape (P, N, M)
    v: array-like
        Y-component of the vector of shape (P, N, M)
    w: array-like
        Z-component of the vector of shape (P, N, M)

    Returns
    -------
    u_t, v_t, w_t: array-like
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
