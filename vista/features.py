import numpy as np

import vista


def voxelize(mesh, density):
    """voxelize mesh to UnstructuredGrid"""
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = vista.StructuredGrid(x, y, z)
    ugrid = vista.UnstructuredGrid(grid)

    # get part of the mesh within the mesh
    selection = ugrid.select_enclosed_points(mesh, tolerance=0.0)
    mask = selection.point_arrays['SelectedPoints'].view(np.bool)

    # extract cells from point indices
    return ugrid.extract_selection_points(mask)
