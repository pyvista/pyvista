"""Tests for pyvista.plotting.utilities.algorithms."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.utilities.algorithms import ActiveScalarsAlgorithm
from pyvista.plotting.utilities.algorithms import AddIDsAlgorithm
from pyvista.plotting.utilities.algorithms import PointSetToPolyDataAlgorithm
from pyvista.plotting.utilities.algorithms import active_scalars_algorithm
from pyvista.plotting.utilities.algorithms import add_ids_algorithm
from pyvista.plotting.utilities.algorithms import algorithm_to_mesh_handler
from pyvista.plotting.utilities.algorithms import cell_data_to_point_data_algorithm
from pyvista.plotting.utilities.algorithms import crinkle_algorithm
from pyvista.plotting.utilities.algorithms import decimation_algorithm
from pyvista.plotting.utilities.algorithms import extract_surface_algorithm
from pyvista.plotting.utilities.algorithms import outline_algorithm
from pyvista.plotting.utilities.algorithms import point_data_to_cell_data_algorithm
from pyvista.plotting.utilities.algorithms import pointset_to_polydata_algorithm
from pyvista.plotting.utilities.algorithms import set_algorithm_input
from pyvista.plotting.utilities.algorithms import triangulate_algorithm


@pytest.fixture
def sphere_with_scalars():
    mesh = pv.Sphere()
    mesh.point_data['z'] = mesh.points[:, 2]
    return mesh


@pytest.fixture
def cube_with_cell_data():
    mesh = pv.Cube()
    mesh.cell_data['cell_id'] = np.arange(mesh.n_cells)
    return mesh


def test_algorithm_to_mesh_handler_passthrough(sphere_with_scalars):
    mesh, algo = algorithm_to_mesh_handler(sphere_with_scalars)
    assert mesh is sphere_with_scalars
    assert algo is None


def test_algorithm_to_mesh_handler_with_algorithm(sphere_with_scalars):
    alg = outline_algorithm(sphere_with_scalars)
    mesh, algo = algorithm_to_mesh_handler(alg)
    assert isinstance(mesh, pv.DataSet)
    assert algo is alg


def test_set_algorithm_input_dataset(sphere_with_scalars):
    alg = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(alg, sphere_with_scalars)
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))
    assert 'z' in out.point_data


def test_set_algorithm_input_algorithm(sphere_with_scalars):
    alg1 = outline_algorithm(sphere_with_scalars)
    alg2 = extract_surface_algorithm(alg1)
    alg2.Update()
    out = pv.wrap(alg2.GetOutputDataObject(0))
    assert out.n_points > 0


def test_active_scalars_algo_point(sphere_with_scalars):
    algo = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = algo.GetOutputDataObject(0)
    scalars = out.GetPointData().GetScalars()
    assert scalars is not None
    assert scalars.GetName() == 'z'


def test_active_scalars_algo_cell(cube_with_cell_data):
    algo = ActiveScalarsAlgorithm(name='cell_id', preference='cell')
    set_algorithm_input(algo, cube_with_cell_data)
    algo.Update()
    out = algo.GetOutputDataObject(0)
    scalars = out.GetCellData().GetScalars()
    assert scalars is not None
    assert scalars.GetName() == 'cell_id'


def test_active_scalars_algo_shallow_copy(sphere_with_scalars):
    algo = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert np.shares_memory(sphere_with_scalars['z'], out['z'])


def test_active_scalars_algo_propagates_changes(sphere_with_scalars):
    algo = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()

    # Modify the original mesh in place
    sphere_with_scalars['z'][:] = 42.0
    algo.Modified()
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert np.allclose(out['z'], 42.0)


def test_active_scalars_algo_wrapper(sphere_with_scalars):
    algo = active_scalars_algorithm(sphere_with_scalars, 'z', preference='point')
    algo.Update()
    out = algo.GetOutputDataObject(0)
    assert out.GetPointData().GetScalars().GetName() == 'z'


def test_pointset_to_polydata_conversion():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    pointset = pv.PointSet(points)
    algo = PointSetToPolyDataAlgorithm()
    set_algorithm_input(algo, pointset)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert isinstance(out, pv.PolyData)
    assert out.n_points == 3


def test_pointset_to_polydata_wrapper():
    pointset = pv.PointSet(np.random.default_rng().random((10, 3)))
    algo = pointset_to_polydata_algorithm(pointset)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert isinstance(out, pv.PolyData)


def test_add_ids_point_and_cell(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=True, cell_ids=True)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' in out.point_data
    assert 'cell_ids' in out.cell_data
    assert np.array_equal(out['point_ids'], np.arange(sphere_with_scalars.n_points))
    assert np.array_equal(out['cell_ids'], np.arange(sphere_with_scalars.n_cells))


def test_add_ids_point_only(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=True, cell_ids=False)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' in out.point_data
    assert 'cell_ids' not in out.cell_data


def test_add_ids_cell_only(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=False, cell_ids=True)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' not in out.point_data
    assert 'cell_ids' in out.cell_data


def test_add_ids_shallow_copy(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=True, cell_ids=False)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert np.shares_memory(sphere_with_scalars['z'], out['z'])


def test_add_ids_wrapper(sphere_with_scalars):
    algo = add_ids_algorithm(sphere_with_scalars, point_ids=True, cell_ids=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' in out.point_data
    assert 'cell_ids' in out.cell_data


def test_crinkle():
    source = pv.Sphere()
    source.cell_data['data'] = np.arange(source.n_cells)
    clipped = source.clip()
    # Add cell_ids needed by crinkle
    clipped.cell_data['cell_ids'] = np.arange(clipped.n_cells)
    algo = crinkle_algorithm(clipped, source)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_cells > 0


def test_outline(sphere_with_scalars):
    algo = outline_algorithm(sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_points == 8  # Bounding box corners
    assert out.n_cells > 0


def test_outline_with_faces(sphere_with_scalars):
    algo = outline_algorithm(sphere_with_scalars, generate_faces=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_points == 8


def test_extract_surface():
    grid = pv.ImageData(dimensions=(5, 5, 5))
    algo = extract_surface_algorithm(grid)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert isinstance(out, pv.PolyData)
    assert out.n_points > 0


def test_extract_surface_pass_ids():
    grid = pv.ImageData(dimensions=(5, 5, 5))
    algo = extract_surface_algorithm(grid, pass_pointid=True, pass_cellid=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'vtkOriginalPointIds' in out.point_data
    assert 'vtkOriginalCellIds' in out.cell_data


def test_cell_data_to_point_data(cube_with_cell_data):
    algo = cell_data_to_point_data_algorithm(cube_with_cell_data)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'cell_id' in out.point_data


def test_cell_data_to_point_data_pass_cell(cube_with_cell_data):
    algo = cell_data_to_point_data_algorithm(cube_with_cell_data, pass_cell_data=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'cell_id' in out.point_data
    assert 'cell_id' in out.cell_data


def test_point_data_to_cell_data(sphere_with_scalars):
    algo = point_data_to_cell_data_algorithm(sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'z' in out.cell_data


def test_point_data_to_cell_data_pass_point(sphere_with_scalars):
    algo = point_data_to_cell_data_algorithm(sphere_with_scalars, pass_point_data=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'z' in out.cell_data
    assert 'z' in out.point_data


def test_triangulate():
    mesh = pv.Plane()
    algo = triangulate_algorithm(mesh)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_cells >= mesh.n_cells


def test_decimation(sphere_with_scalars):
    algo = decimation_algorithm(sphere_with_scalars, target_reduction=0.5)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_points < sphere_with_scalars.n_points


def test_active_scalars_algo_modified_propagates(sphere_with_scalars):
    """Re-running after Modified() must reflect the new array name.

    Targets the rewritten ``RequestData`` — the Modified() pathway is what
    lets ``DataSetMapper.set_active_scalars`` swap arrays in place without
    re-splicing the pipeline.
    """
    sphere_with_scalars.point_data['x'] = sphere_with_scalars.points[:, 0]

    algo = ActiveScalarsAlgorithm(name='x', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    assert algo.GetOutputDataObject(0).GetPointData().GetScalars().GetName() == 'x'

    algo.scalars_name = 'z'  # setter calls Modified()
    algo.Update()
    assert algo.GetOutputDataObject(0).GetPointData().GetScalars().GetName() == 'z'


def test_active_scalars_algo_does_not_mutate_input(sphere_with_scalars):
    """The original mesh's active scalars must not change when the algo runs."""
    sphere_with_scalars.point_data['x'] = sphere_with_scalars.points[:, 0]
    sphere_with_scalars.set_active_scalars('x')

    algo = active_scalars_algorithm(sphere_with_scalars, 'z', preference='point')
    algo.Update()

    assert sphere_with_scalars.active_scalars_name == 'x'


def test_add_ids_preserves_active_scalars(sphere_with_scalars):
    """Adding ID arrays must not steal the active-scalars slot."""
    sphere_with_scalars.set_active_scalars('z')

    algo = add_ids_algorithm(sphere_with_scalars, point_ids=True, cell_ids=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))

    assert out.point_data.active_scalars_name == 'z'
    # Original is also untouched
    assert sphere_with_scalars.active_scalars_name == 'z'
