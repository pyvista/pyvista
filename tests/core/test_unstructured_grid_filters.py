"""Tests for UnstructuredGridFilters."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv

skip_lesser_9_2_2 = pytest.mark.needs_vtk_version(9, 2, 2, reason='Requires VTK>=9.2.2')


@skip_lesser_9_2_2
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


@skip_lesser_9_2_2
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


@pytest.mark.needs_vtk_version(9, 2)
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

    mesh_in = cube_ug if mesh_type is pv.UnstructuredGrid else cube_poly.extract_geometry()
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
