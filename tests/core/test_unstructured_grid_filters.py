"""Tests for UnstructuredGridFilters."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples


def test_clean_points():
    """Test on a set of points."""
    n_unique_points = 100
    u_points = np.random.default_rng().random((n_unique_points, 3))
    points = np.vstack([u_points] * 2)

    grid = pv.PolyData(points).cast_to_unstructured_grid()
    grid.point_data['data'] = np.hstack((np.zeros(n_unique_points), np.ones(n_unique_points)))
    grid_orig = grid.copy()  # ensure grid is not modified in-place

    cleaned = grid.clean(produce_merge_map=True)
    assert grid_orig == grid
    assert cleaned.field_data['PointMergeMap'].size == points.shape[0]
    assert cleaned.n_points == n_unique_points
    assert np.allclose(cleaned.point_data['data'], 0.5)
    assert cleaned.point_data['data'].size == n_unique_points

    # verify not averaging works
    cleaned = grid.clean(average_point_data=False)
    assert cleaned.n_points == n_unique_points
    assert not (cleaned.point_data['data'] == 0.5).any()

    # verify tolerance
    u_points = np.random.default_rng().random((n_unique_points, 3))
    u_points_shift = u_points.copy()
    u_points_shift[:, 2] += 1e-4
    points = np.vstack((u_points, u_points_shift))

    grid = pv.PolyData(points).cast_to_unstructured_grid()
    grid_orig = grid.copy()
    cleaned = grid.clean(tolerance=1e-5)
    assert cleaned.n_points == points.shape[0]
    assert grid_orig == grid


def test_clean_grid(hexbeam):
    hexbeam_shifted = hexbeam.translate([1, 0, 0])

    hexbeam.point_data['data'] = np.zeros(hexbeam.n_points)
    hexbeam_shifted.point_data['data'] = np.ones(hexbeam.n_points)
    hexbeam_orig = hexbeam.copy()

    merged = hexbeam.merge(hexbeam_shifted, merge_points=False)
    cleaned = merged.clean(average_point_data=True, produce_merge_map=False)
    assert 'PointMergeMap' not in cleaned.field_data
    assert hexbeam_orig == hexbeam

    # expect averaging for all merged nodes
    n_merged = merged.n_points - cleaned.n_points
    assert (cleaned['data'] == 0.5).sum() == n_merged

    cleaned = merged.clean(average_point_data=False)
    assert not (cleaned['data'] == 0.5).any()

    # test merging_array
    cleaned = merged.clean(average_point_data=True, merging_array_name='data')
    assert cleaned.n_points == hexbeam.n_points * 2

    hexbeam.point_data['data_2'] = np.ones(hexbeam.n_points)
    hexbeam_shifted.point_data['data_2'] = np.ones(hexbeam.n_points)
    merged = hexbeam.merge(hexbeam_shifted, merge_points=False)

    # test merging_array
    cleaned = merged.clean(average_point_data=True, merging_array_name='data_2')
    assert cleaned.n_points == 165


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('mesh_type', [pv.PolyData, pv.UnstructuredGrid])
def test_remove_unused_points(mesh_type, inplace):
    cube_poly = pv.Cube(clean=False)
    array_names = cube_poly.array_names
    assert array_names == ['Normals', 'TCoords', 'FaceIndex']
    n_points_expected = 24  # cube has duplicate points at the vertices
    assert cube_poly.n_points == n_points_expected

    cube_ug = cube_poly.cast_to_unstructured_grid()

    # Insert an unused point at the start of the mesh's points array
    cube_ug.points = np.insert(cube_ug.points, 0, [[0, 0, 0]], axis=0)
    cube_ug['Normals'] = np.insert(cube_ug['Normals'], 0, [[0, 0, 0]], axis=0)
    cube_ug['TCoords'] = np.insert(cube_ug['TCoords'], 0, [[0, 0]], axis=0)
    # Need to offset cell connectivity by one to account for new point
    connectivity = cube_ug.GetCells().GetConnectivityArray()
    for i in range(connectivity.GetNumberOfTuples()):
        connectivity.SetValue(i, connectivity.GetValue(i) + 1)
    assert cube_ug.n_points == n_points_expected + 1

    # Append an unused point at the end of the mesh's points array
    cube_ug.points = np.append(cube_ug.points, [[0, 0, 0]], axis=0)
    cube_ug['Normals'] = np.append(cube_ug['Normals'], [[0, 0, 0]], axis=0)
    cube_ug['TCoords'] = np.append(cube_ug['TCoords'], [[0, 0]], axis=0)
    assert cube_ug.n_points == n_points_expected + 2

    mesh_in = (
        cube_ug
        if mesh_type is pv.UnstructuredGrid
        else cube_poly.extract_surface(algorithm=None, pass_cellid=False, pass_pointid=False)
    )
    mesh_out = mesh_in.remove_unused_points(inplace=inplace)

    # Test unused point is removed and no points are merged
    assert mesh_out.n_points == n_points_expected
    # Mesh geometry (point and cell order) and arrays should also match cube input exactly
    assert mesh_out.cast_to_unstructured_grid() == cube_poly.cast_to_unstructured_grid()

    # Test mesh and array copies
    assert (mesh_in is mesh_out) == inplace
    for name in array_names:
        assert np.shares_memory(mesh_in[name], mesh_out[name]) == inplace
    assert mesh_out.array_names == array_names

    # Test empty
    key = 'data'
    empty = mesh_type()
    empty.field_data[key] = [1, 2, 3]
    out = empty.remove_unused_points(inplace=inplace)
    assert (out is empty) == inplace
    assert out.array_names == [key]


def _with_an_unused_point(mesh_type):
    """Return a cube of the given type with one point no cell refers to."""
    mesh = pv.Cube(clean=False)
    mesh.clear_point_data()
    if mesh_type is pv.UnstructuredGrid:
        mesh = mesh.cast_to_unstructured_grid()
    mesh.points = np.append(mesh.points, [[9.0, 9.0, 9.0]], axis=0)
    return mesh


@pytest.mark.parametrize('mesh_type', [pv.PolyData, pv.UnstructuredGrid])
def test_remove_unused_points_leaves_input_alone(mesh_type):
    """A copy is only removed from, never the input it was made from."""
    mesh = _with_an_unused_point(mesh_type)
    mesh.field_data['meta'] = np.array([1.0, 2.0, 3.0])
    before = mesh.copy()

    out = mesh.remove_unused_points()

    assert out.n_points == mesh.n_points - 1
    assert mesh.cast_to_unstructured_grid() == before.cast_to_unstructured_grid()
    # Field data is the one array vtkExtractCells passes straight through
    assert not np.shares_memory(mesh.field_data['meta'], out.field_data['meta'])


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('mesh_type', [pv.PolyData, pv.UnstructuredGrid])
def test_remove_unused_points_leaves_a_shallow_copy_alone(mesh_type, inplace):
    """The cells the filter grows must not be cells another mesh shares."""
    mesh = _with_an_unused_point(mesh_type)
    sibling = mesh.copy(deep=False)
    before = mesh.copy()

    out = mesh.remove_unused_points(inplace=inplace)

    assert out.n_points == before.n_points - 1
    assert sibling.n_cells == before.n_cells
    assert sibling.cast_to_unstructured_grid() == before.cast_to_unstructured_grid()
    sibling.validate_mesh('cells', action='error')


@pytest.mark.parametrize('inplace', [True, False])
def test_remove_unused_points_keeps_subclass(inplace):
    class _Grid(pv.UnstructuredGrid):
        pass

    mesh = _Grid(pv.Cube(clean=False).cast_to_unstructured_grid())
    assert type(mesh.remove_unused_points(inplace=inplace)) is _Grid
    assert type(_Grid().remove_unused_points(inplace=inplace)) is _Grid


@pytest.mark.parametrize(
    ('dimensions', 'to_general'),
    [((2, 3, 4), 'to_hexahedra'), ((2, 3, 1), 'to_quads'), ((1, 3, 4), 'to_quads')],
)
def test_generalize_cells_matches_grid_cast(dimensions, to_general):
    image = pv.ImageData(dimensions=dimensions)
    image.cell_data['ids'] = np.arange(image.n_cells)
    grid = image.cast_to_unstructured_grid()
    general = grid.generalize_cells()
    expected = getattr(image, to_general)()
    assert np.array_equal(general.celltypes, expected.celltypes)
    assert np.array_equal(general.cell_connectivity, expected.cell_connectivity)
    assert np.array_equal(general.cell_data['ids'], image.cell_data['ids'])
    general.validate_mesh(action='error')


@pytest.mark.parametrize('inplace', [True, False])
def test_generalize_cells_leaves_other_cells_alone(inplace):
    cells = [examples.cells.Voxel(), examples.cells.Tetrahedron(), examples.cells.Pixel()]
    grid = pv.merge(cells, merge_points=False)
    if pv.vtk_version_info >= (9, 5):
        grid = pv.merge([grid, examples.cells.Polyhedron()], merge_points=False)
    sibling = grid.copy(deep=False)
    before = grid.copy()

    general = grid.generalize_cells(inplace=inplace)

    assert (general is grid) == inplace
    general_types = {
        pv.CellType.VOXEL: pv.CellType.HEXAHEDRON,
        pv.CellType.PIXEL: pv.CellType.QUAD,
    }
    expected = [general_types.get(cell_type, cell_type) for cell_type in before.celltypes]
    assert np.array_equal(general.celltypes, expected)
    for cell_id in np.flatnonzero(~np.isin(before.celltypes, list(general_types))):
        assert general.get_cell(cell_id) == before.get_cell(cell_id)
    assert np.array_equal(general.points, before.points)
    assert sibling == before
    if not inplace:
        assert grid == before


def test_generalize_cells_without_voxels_or_pixels():
    grid = examples.load_tetbeam()
    assert grid.generalize_cells() == grid
