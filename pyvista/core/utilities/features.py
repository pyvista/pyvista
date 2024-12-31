"""Module containing geometry helper functions."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import os
import sys
from typing import TYPE_CHECKING
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.helpers import wrap

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import UnstructuredGrid
    from pyvista.core._typing_core import VectorLike


def _padded_bins(mesh, density):
    """Construct bin edges for voxelization.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to voxelize.

    density : array_like[float]
        A list of densities along x,y,z directions.

    Returns
    -------
    list[np.ndarray]
        List of bin edges for each axis.

    Notes
    -----
    Ensures limits of voxelization are padded to ensure the mesh is fully enclosed.

    """
    bounds = np.array(mesh.bounds).reshape(3, 2)
    bin_count = np.ceil(1e-10 + (bounds[:, 1] - bounds[:, 0]) / density)
    pad = (bin_count * density - (bounds[:, 1] - bounds[:, 0])) / 2

    return [
        np.arange(bounds[i, 0] - pad[i], bounds[i, 1] + pad[i] + density[i] / 2, density[i])
        for i in range(3)
    ]


def voxelize_unstructured_grid(
    mesh: DataSet | MultiBlock | _vtk.vtkDataSet | _vtk.vtkMultiBlockDataSet,
    *,
    dimensions: VectorLike[int] | None = None,
    spacing: float | VectorLike[float] | None = None,
    rounding_func: Callable[[VectorLike[float]], VectorLike[int]] | None = None,
    cell_length_percentile: float | None = None,
    cell_length_sample_size: int | None = None,
    mesh_length_fraction: float | None = None,
    progress_bar: bool = False,
) -> UnstructuredGrid:
    """Voxelize :class:`~pyvista.PolyData` as a binary :class:`~pyvista.ImageData` mask.

    The binary mask is a point data array where points inside and outside of the
    input surface are labelled with ``foreground_value`` and ``background_value``,
    respectively.

    This filter implements `vtkPolyDataToImageStencil
    <https://vtk.org/doc/nightly/html/classvtkPolyDataToImageStencil.html>`_. This
    algorithm operates as follows:

    * The algorithm iterates through the z-slice of the ``reference_volume``.
    * For each slice, it cuts the input :class:`~pyvista.PolyData` surface to create
      2D polylines at that z position. It attempts to close any open polylines.
    * For each x position along the polylines, the corresponding y positions are
      determined.
    * For each slice, the grid points are labelled as foreground or background based
      on their xy coordinates.

    The voxelization can be controlled in several ways:

    #. Specify the output geometry using a ``reference_volume``.

    #. Specify the ``spacing`` explicitly.

    #. Specify the ``dimensions`` explicitly.

    #. Specify the ``cell_length_percentile``. The spacing is estimated from the
       surface's cells using the specified percentile.

    #. Specify ``mesh_length_fraction``. The spacing is computed as a fraction of
       the mesh's diagonal length.

    Use ``reference_volume`` for full control of the output mask's geometry. For
    all other options, the geometry is implicitly defined such that the generated
    mask fits the bounds of the input surface.

    If no inputs are provided, ``cell_length_percentile=0.1`` (10th percentile) is
    used by default to estimate the spacing. On systems with VTK < 9.2, the default
    spacing is set to ``mesh_length_fraction=1/100``.

    .. versionadded:: 0.45.0

    .. note::
        For best results, ensure the input surface is a closed surface. The
        surface is considered closed if it has zero :attr:`~pyvista.PolyData.n_open_edges`.

    .. note::
        This filter returns voxels represented as point data, not :attr:`~pyvista.CellType.VOXEL` cells.
        This differs from :func:`~pyvista.voxelize` and :func:`~pyvista.voxelize_volume`
        which return meshes with voxel cells. See :ref:`image_representations_example`
        for examples demonstrating the difference.

    .. note::
        This filter does not discard internal surfaces, due, for instance, to
        intersecting meshes. Instead, the intersection will be considered as
        background which may produce unexpected results. See `Examples`.

    Parameters
    ----------
    mesh : DataSet | MultiBlock
        Mesh to voxelize.

    dimensions : VectorLike[int], optional
        Dimensions of the generated mask image. Set this value to control the
        dimensions explicitly. If unset, the dimensions are defined implicitly
        through other parameter. See summary and examples for details.

    spacing : VectorLike[float], optional
        Approximate spacing to use for the generated mask image. Set this value
        to control the spacing explicitly. If unset, the spacing is defined
        implicitly through other parameters. See summary and examples for details.

    rounding_func : Callable[VectorLike[float], VectorLike[int]], optional
        Control how the dimensions are rounded to integers based on the provided or
        calculated ``spacing``. Should accept a length-3 vector containing the
        dimension values along the three directions and return a length-3 vector.
        :func:`numpy.round` is used by default.

        Rounding the dimensions implies rounding the actual spacing.

        Has no effect if ``reference_volume`` or ``dimensions`` are specified.

    cell_length_percentile : float, optional
        Cell length percentage ``p`` to use for computing the default ``spacing``.
        Default is ``0.1`` (10th percentile) and must be between ``0`` and ``1``.
        The ``p``-th percentile is computed from the cumulative distribution function
        (CDF) of lengths which are representative of the cell length scales present
        in the input. The CDF is computed by:

        #. Triangulating the input cells.
        #. Sampling a subset of up to ``cell_length_sample_size`` cells.
        #. Computing the distance between two random points in each cell.
        #. Inserting the distance into an ordered set to create the CDF.

        Has no effect if ``dimension`` or ``reference_volume`` are specified.

        .. note::
            This option is only available for VTK 9.2 or greater.

    cell_length_sample_size : int, optional
        Number of samples to use for the cumulative distribution function (CDF)
        when using the ``cell_length_percentile`` option. ``100 000`` samples are
        used by default.

    mesh_length_fraction : float, optional
        Fraction of the surface mesh's length to use for computing the default
        ``spacing``. Set this to any fractional value (e.g. ``1/100``) to enable
        this option. This is used as an alternative to using ``cell_length_percentile``.

    progress_bar : bool, default: False
        Display a progress bar to indicate progress.

    Returns
    -------
    pyvista.ImageData
        Generated binary mask with a ``'mask'``  point data array. The data array
        has dtype :class:`numpy.uint8` if the foreground and background values are
        unsigned and less than 256.

    See Also
    --------
    pyvista.voxelize
        Similar function that returns a :class:`~pyvista.UnstructuredGrid` of
        :attr:`~pyvista.CellType.VOXEL` cells.

    pyvista.voxelize_volume
        Similar function that returns a :class:`~pyvista.RectilinearGrid` with cell data.

    pyvista.ImageDataFilters.contour_labeled
        Filter that generates surface contours from labeled image data. Can be
        loosely considered as an inverse of this filter.

    pyvista.ImageDataFilters.points_to_cells
        Convert voxels represented as points to :attr:`~pyvista.CellType.VOXEL`
        cells.

    pyvista.ImageData
        Class used to build custom ``reference_volume``.

    Examples
    --------
    Generate a binary mask from a coarse mesh.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> poly = examples.download_bunny_coarse()
    >>> mask = poly.voxelize_binary_mask()

    The mask is stored as :class:`~pyvista.ImageData` with point data scalars
    (zeros for background, ones for foreground).

    >>> mask
    ImageData (...)
      N Cells:      7056
      N Points:     8228
      X Bounds:     -1.245e-01, 1.731e-01
      Y Bounds:     -1.135e-01, 1.807e-01
      Z Bounds:     -1.359e-01, 9.140e-02
      Dimensions:   22, 22, 17
      Spacing:      1.417e-02, 1.401e-02, 1.421e-02
      N Arrays:     1

    >>> np.unique(mask.point_data['mask'])
    pyvista_ndarray([0, 1], dtype=uint8)

    To visualize it as voxel cells, use :meth:`~pyvista.ImageDataFilters.points_to_cells`,
    then use :meth:`~pyvista.DataSetFilters.threshold` to extract the foreground.

    We also plot the voxel cells in blue and the input poly data in green for
    comparison.

    >>> def mask_and_polydata_plotter(mask, poly):
    ...     voxel_cells = mask.points_to_cells().threshold(0.5)
    ...
    ...     plot = pv.Plotter()
    ...     _ = plot.add_mesh(voxel_cells, color='blue')
    ...     _ = plot.add_mesh(poly, color='lime')
    ...     plot.camera_position = 'xy'
    ...     return plot

    >>> plot = mask_and_polydata_plotter(mask, poly)
    >>> plot.show()

    The spacing of the mask image is automatically adjusted to match the
    density of the input.

    Repeat the previous example with a finer mesh.

    >>> poly = examples.download_bunny()
    >>> mask = poly.voxelize_binary_mask()
    >>> plot = mask_and_polydata_plotter(mask, poly)
    >>> plot.show()

    Control the spacing manually instead. Here, a very coarse spacing is used.

    >>> mask = poly.voxelize_binary_mask(spacing=(0.01, 0.04, 0.02))
    >>> plot = mask_and_polydata_plotter(mask, poly)
    >>> plot.show()

    Note that the spacing is only approximate. Check the mask's actual spacing.

    >>> mask.spacing
    (0.009731187485158443, 0.03858340159058571, 0.020112216472625732)

    The actual values may be greater or less than the specified values. Use
    ``rounding_func=np.floor`` to force all values to be greater.

    >>> mask = poly.voxelize_binary_mask(
    ...     spacing=(0.01, 0.04, 0.02), rounding_func=np.floor
    ... )
    >>> mask.spacing
    (0.01037993331750234, 0.05144453545411428, 0.020112216472625732)

    Set the dimensions instead of the spacing.

    >>> mask = poly.voxelize_binary_mask(dimensions=(10, 20, 30))
    >>> plot = mask_and_polydata_plotter(mask, poly)
    >>> plot.show()

    >>> mask.dimensions
    (10, 20, 30)

    Create a mask using a reference volume. First generate polydata from
    an existing mask.

    >>> volume = examples.load_frog_tissues()
    >>> poly = volume.contour_labeled(smoothing=True)

    Now create the mask from the polydata using the volume as a reference.

    >>> mask = poly.voxelize_binary_mask(reference_volume=volume)
    >>> plot = mask_and_polydata_plotter(mask, poly)
    >>> plot.show()

    Visualize the effect of internal surfaces.

    >>> mesh = pv.Cylinder() + pv.Cylinder((0, 0.75, 0))
    >>> binary_mask = mesh.voxelize_binary_mask(
    ...     dimensions=(1, 100, 50)
    ... ).points_to_cells()
    >>> plot = pv.Plotter()
    >>> _ = plot.add_mesh(binary_mask)
    >>> _ = plot.add_mesh(mesh.slice(), color='red')
    >>> plot.show(cpos='yz')

    Note how the intersection is excluded from the mask.
    To include the voxels delimited by internal surfaces in the foreground, the internal
    surfaces should be removed, for instance by applying a boolean union. Note that
    this operation in unreliable in VTK but may be performed with external tools such
    as `vtkbool <https://github.com/zippy84/vtkbool>`_.

    Alternatively, the intersecting parts of the mesh can be processed sequentially.

    >>> cylinder_1 = pv.Cylinder()
    >>> cylinder_2 = pv.Cylinder((0, 0.75, 0))

    >>> reference_volume = pv.ImageData(
    ...     dimensions=(1, 100, 50),
    ...     spacing=(1, 0.0175, 0.02),
    ...     origin=(0, -0.5 + 0.0175 / 2, -0.5 + 0.02 / 2),
    ... )

    >>> binary_mask_1 = cylinder_1.voxelize_binary_mask(
    ...     reference_volume=reference_volume
    ... ).points_to_cells()
    >>> binary_mask_2 = cylinder_2.voxelize_binary_mask(
    ...     reference_volume=reference_volume
    ... ).points_to_cells()

    >>> binary_mask_1['mask'] = binary_mask_1['mask'] | binary_mask_2['mask']

    >>> plot = pv.Plotter()
    >>> _ = plot.add_mesh(binary_mask_1)
    >>> _ = plot.add_mesh(cylinder_1.slice(), color='red')
    >>> _ = plot.add_mesh(cylinder_2.slice(), color='red')
    >>> plot.show(cpos='yz')

    When multiple internal surfaces are nested, they are successively treated as
    interfaces between background and foreground.

    >>> mesh = pv.Tube(radius=2) + pv.Tube(radius=3) + pv.Tube(radius=4)
    >>> binary_mask = mesh.voxelize_binary_mask(
    ...     dimensions=(1, 50, 50)
    ... ).points_to_cells()
    >>> plot = pv.Plotter()
    >>> _ = plot.add_mesh(binary_mask)
    >>> _ = plot.add_mesh(mesh.slice(), color='red')
    >>> plot.show(cpos='yz')

    """
    surface = wrap(mesh).extract_geometry()
    if not surface.faces.size:
        # we have a point cloud or an empty mesh
        raise ValueError('Input mesh must have faces for voxelization.')

    binary_mask = surface.voxelize_binary_mask(
        dimensions=dimensions,
        spacing=spacing,
        rounding_func=rounding_func,
        cell_length_percentile=cell_length_percentile,
        cell_length_sample_size=cell_length_sample_size,
        mesh_length_fraction=mesh_length_fraction,
        progress_bar=progress_bar,
    )
    voxel_cells = binary_mask.points_to_cells(dimensionality='3D', copy=False).threshold(0.5)
    del voxel_cells.cell_data['mask']
    return voxel_cells


def voxelize(
    mesh, density=None, check_surface: bool = True, enclosed: bool = False, fit_bounds: bool = False
):
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

    enclosed : bool, default: False
        If True, the voxel bounds will be outside the mesh.
        If False, the voxel bounds will be at or inside the mesh bounds.

    fit_bounds : bool, default: False
        If enabled, the end bound of the input mesh is used as the end bound of the
        voxel grid and the density is updated to the closest compatible one. Otherwise,
        the end bound is excluded. Has no effect if `enclosed` is enabled.

    Returns
    -------
    pyvista.UnstructuredGrid
        Voxelized unstructured grid of the original mesh.

    Notes
    -----
    Prior to version 0.39.0, this method improperly handled the order of
    structured coordinates.

    See Also
    --------
    pyvista.voxelize_volume
        Similar function that returns a :class:`pyvista.RectilinearGrid` with cell data.

    pyvista.PolyDataFilters.voxelize_binary_mask
        Similar function that returns a :class:`pyvista.ImageData` with point data.

    """
    warnings.warn('deprecated', PyVistaDeprecationWarning)
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

    if enclosed:
        # Get x, y, z bin edges
        x, y, z = _padded_bins(mesh, [density_x, density_y, density_z])
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        if fit_bounds:
            # Calculate an integer number of voxels, floor to ensure that the voxels
            # don't exceed the input mesh
            nof_voxels_x = int(np.round((x_max - x_min) / density_x))
            nof_voxels_y = int(np.round((y_max - y_min) / density_y))
            nof_voxels_z = int(np.round((z_max - z_min) / density_z))

            # One additional point is required to ensure the proper number of voxels
            x = np.linspace(x_min, x_max, nof_voxels_x + 1)
            y = np.linspace(y_min, y_max, nof_voxels_y + 1)
            z = np.linspace(z_min, z_max, nof_voxels_z + 1)
        else:
            x = np.arange(x_min, x_max, density_x)
            y = np.arange(y_min, y_max, density_y)
            z = np.arange(z_min, z_max, density_z)

    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    # indexing='ij' is used here in order to make grid and ugrid with x-y-z ordering, not y-x-z ordering
    # see https://github.com/pyvista/pyvista/pull/4365

    # Create unstructured grid from the structured grid
    grid = pyvista.StructuredGrid(x, y, z)
    ugrid = pyvista.UnstructuredGrid(grid)

    if enclosed:
        # Normalise cells to unit size
        ugrid_norm = ugrid.copy()
        surface_norm = surface.copy()
        ugrid_norm.points /= np.array(density)
        surface_norm.points /= np.array(density)
        # Select cells if they're within one unit of the surface
        ugrid_norm = ugrid_norm.compute_implicit_distance(surface_norm)
        mask = ugrid_norm['implicit_distance'] < 1
        del ugrid_norm, surface_norm
    else:
        # get part of the mesh within the mesh's bounding surface.
        selection = ugrid.select_enclosed_points(
            surface, tolerance=0.0, check_surface=check_surface
        )
        mask = selection.point_data['SelectedPoints'].view(np.bool_)
        del selection

    # extract cells from point indices
    return ugrid.extract_points(mask)


def voxelize_volume(
    mesh, density=None, check_surface: bool = True, enclosed: bool = False, fit_bounds: bool = False
):
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

    enclosed : bool, default: False
        If True, the voxel bounds will be outside the mesh.
        If False, the voxel bounds will be at or inside the mesh bounds.

    fit_bounds : bool, default: False
        If enabled, the end bound of the input mesh is used as the end bound of the
        voxel grid and the density is updated to the closest compatible one. Otherwise,
        the end bound is excluded. Has no effect if `enclosed` is enabled.

    Returns
    -------
    pyvista.RectilinearGrid
        RectilinearGrid as voxelized volume with discretized cells.

    See Also
    --------
    pyvista.voxelize
        Similar function that returns a :class:`pyvista.UnstructuredGrid` of
        :attr:`~pyvista.CellType.VOXEL` cells.

    pyvista.PolyDataFilters.voxelize_binary_mask
        Similar function that returns a :class:`pyvista.ImageData` with point data.

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

    Create an equal density voxel volume without enclosing input mesh.

    >>> vox = pv.voxelize_volume(mesh, density=0.15)
    >>> vox = vox.select_enclosed_points(mesh, tolerance=0.0)
    >>> vox.plot(scalars='SelectedPoints', show_edges=True, cpos=cpos)

    Create an equal density voxel volume enclosing input mesh.

    >>> vox = pv.voxelize_volume(mesh, density=0.15, enclosed=True)
    >>> vox = vox.select_enclosed_points(mesh, tolerance=0.0)
    >>> vox.plot(scalars='SelectedPoints', show_edges=True, cpos=cpos)

    Create an equal density voxel volume that does not fit the input mesh's bounds.

    >>> mesh = pv.examples.load_nut()
    >>> vox = pv.voxelize_volume(mesh=mesh, density=2.5)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh=vox, show_edges=True)
    >>> _ = pl.add_mesh(mesh=mesh, show_edges=True, opacity=1)
    >>> pl.show()

    Create an equal density voxel volume that fits the input mesh's bounds.

    >>> vox = pv.voxelize_volume(mesh=mesh, density=2.5, fit_bounds=True)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh=vox, show_edges=True)
    >>> _ = pl.add_mesh(mesh=mesh, show_edges=True, opacity=1)
    >>> pl.show()

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

    if enclosed:
        # Get x, y, z bin edges
        x, y, z = _padded_bins(mesh, [density_x, density_y, density_z])
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        if fit_bounds:
            # Calculate an integer number of voxels, floor to ensure that the voxels
            # don't exceed the input mesh
            nof_voxels_x = int(np.round((x_max - x_min) / density_x))
            nof_voxels_y = int(np.round((y_max - y_min) / density_y))
            nof_voxels_z = int(np.round((z_max - z_min) / density_z))

            # One additional point is required to ensure the proper number of voxels
            x = np.linspace(x_min, x_max, nof_voxels_x + 1)
            y = np.linspace(y_min, y_max, nof_voxels_y + 1)
            z = np.linspace(z_min, z_max, nof_voxels_z + 1)
        else:
            x = np.arange(x_min, x_max, density_x)
            y = np.arange(y_min, y_max, density_y)
            z = np.arange(z_min, z_max, density_z)

    # Create a RectilinearGrid
    voi = pyvista.RectilinearGrid(x, y, z)

    # get part of the mesh within the mesh's bounding surface.
    selection = voi.select_enclosed_points(surface, tolerance=0.0, check_surface=check_surface)
    mask_vol = selection.point_data['SelectedPoints'].view(np.bool_)

    # Get voxels that fall within input mesh boundaries
    cell_ids = np.unique(voi.extract_points(np.argwhere(mask_vol))['vtkOriginalCellIds'])

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
    dimensions : tuple[int, int, int], default: (101, 101, 101)
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
    image.origin = bounds[::2]  # type: ignore[assignment]
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
    xx, yy, _ = np.meshgrid(np.radians(theta), np.radians(phi), r, indexing='ij')
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
    merge_points: bool = True,
    main_has_priority: bool = True,
    progress_bar: bool = False,
):
    """Merge several datasets.

    .. note::
       The behavior of this filter varies from the
       :func:`PolyDataFilters.boolean_union` filter. This filter
       does not attempt to create a manifold mesh and will include
       internal surfaces when two meshes overlap.

    Parameters
    ----------
    datasets : sequence[:class:`pyvista.DataSet`]
        Sequence of datasets. Can be of any :class:`pyvista.DataSet`.

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
        raise TypeError(f'Expected a sequence, got {type(datasets).__name__}')

    if len(datasets) < 1:
        raise ValueError('Expected at least one dataset.')

    first = datasets[0]
    if not isinstance(first, pyvista.DataSet):
        raise TypeError(f'Expected pyvista.DataSet, not {type(first).__name__}')

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
    scalar_arr_name: str = 'scalars',
    normal_arr_name: str = 'normals',
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
    ...     noise, [0, 3.0, -0, 1.0, 0, 1.0], dim=(60, 20, 20)
    ... )
    >>> grid.plot(cmap='gist_earth_r', show_scalar_bar=False, show_edges=True)

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
