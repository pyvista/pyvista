from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting import _vtk
from pyvista.plotting.mapper import DataSetMapper
from pyvista.plotting.utilities.algorithms import ActiveScalarsAlgorithm


@pytest.fixture
def dataset_mapper(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    return pl.mapper


def test_init(sphere):
    mapper = DataSetMapper()
    mapper.dataset = sphere
    assert mapper.dataset is sphere


def test_scalar_range(dataset_mapper):
    assert isinstance(dataset_mapper.scalar_range, tuple)
    rng = (0, 2)
    dataset_mapper.scalar_range = rng
    assert dataset_mapper.scalar_range == rng


def test_bounds(dataset_mapper):
    assert isinstance(dataset_mapper.bounds, tuple)
    assert dataset_mapper.bounds == (-126.0, 125.0, -127.0, 126.0, -127.0, 127.0)


def test_lookup_table(dataset_mapper):
    assert isinstance(dataset_mapper.lookup_table, _vtk.vtkLookupTable)

    table = _vtk.vtkLookupTable()

    dataset_mapper.lookup_table = table
    assert dataset_mapper.lookup_table is table


def test_interpolate_before_map(dataset_mapper):
    assert isinstance(dataset_mapper.interpolate_before_map, bool)
    dataset_mapper.interpolate_before_map = True
    assert dataset_mapper.interpolate_before_map is True


def test_color_mode(dataset_mapper):
    assert isinstance(dataset_mapper.color_mode, str)

    dataset_mapper.color_mode = 'direct'
    assert dataset_mapper.color_mode == 'direct'

    dataset_mapper.color_mode = 'map'
    assert dataset_mapper.color_mode == 'map'

    with pytest.raises(ValueError, match='Color mode must be either'):
        dataset_mapper.color_mode = 'invalid'


def test_set_scalars(dataset_mapper):
    scalars = dataset_mapper.dataset.points[:, 2]
    n_colors = 128
    dataset_mapper.set_scalars(scalars, 'z', n_colors=n_colors)
    assert dataset_mapper.lookup_table.GetNumberOfTableValues() == n_colors


def test_array_name(dataset_mapper):
    name = 'scalars'
    dataset_mapper.array_name = name
    assert dataset_mapper.array_name == name


def test_copy(dataset_mapper, sphere):
    dataset_mapper.dataset = sphere
    dataset_mapper.interpolate_before_map = False
    dataset_mapper.scalar_range = (2, 5)
    map_cp = dataset_mapper.copy()
    assert isinstance(map_cp, DataSetMapper)
    assert map_cp is not dataset_mapper
    assert map_cp.scalar_range == dataset_mapper.scalar_range
    assert map_cp.dataset is dataset_mapper.dataset

    map_cp.scalar_range = (5, 10)
    assert map_cp.scalar_range != dataset_mapper.scalar_range


@pytest.mark.parametrize('resolve', ['polygon_offset', 'shift_zbuffer', 'off'])
def test_resolve(dataset_mapper, resolve):
    dataset_mapper.resolve = resolve
    assert dataset_mapper.resolve == resolve


def test_invalid_resolve(dataset_mapper):
    match = 'Resolve must be either "off", "polygon_offset" or "shift_zbuffer"'
    with pytest.raises(ValueError, match=match):
        dataset_mapper.resolve = 'invalid'


def test_mapper_input_dataset_stored(sphere):
    mapper = DataSetMapper(dataset=sphere)
    assert mapper._input_dataset is sphere


def test_mapper_active_scalars_algo_created(sphere):
    sphere['data_a'] = sphere.points[:, 0]
    mapper = DataSetMapper(dataset=sphere)
    assert mapper._active_scalars_algo is None

    mapper.set_scalars(sphere['data_a'], 'data_a')
    assert mapper._active_scalars_algo is not None
    assert mapper._active_scalars_algo.scalars_name == 'data_a'


def test_mapper_active_scalars_algo_updated(sphere):
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]
    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data_a'], 'data_a')
    first_algo = mapper._active_scalars_algo

    mapper.set_scalars(sphere['data_b'], 'data_b')
    # Same algo instance should be reused, not recreated
    assert mapper._active_scalars_algo is first_algo
    assert mapper._active_scalars_algo.scalars_name == 'data_b'


def test_mapper_array_name_syncs_algo(sphere):
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]
    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data_a'], 'data_a')
    assert mapper._active_scalars_algo.scalars_name == 'data_a'

    mapper.array_name = 'data_b'
    assert mapper._active_scalars_algo.scalars_name == 'data_b'


def test_mapper_does_not_mutate_mesh_active_scalars(sphere):
    """Verify that setting scalars on the mapper does not modify the original mesh."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]
    sphere.set_active_scalars('data_a')

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data_b'], 'data_b')

    # The original mesh's active scalars should NOT be changed
    assert sphere.active_scalars_name == 'data_a'


def test_active_scalars_algorithm_shallow_copy():
    """Verify ActiveScalarsAlgorithm output shares memory with input."""
    mesh = pv.Sphere()
    mesh['data'] = np.arange(mesh.n_points, dtype=float)

    algo = ActiveScalarsAlgorithm(name='data', preference='point')
    algo.SetInputDataObject(mesh)
    algo.Update()
    output = pv.wrap(algo.GetOutputDataObject(0))

    # Verify the output shares data arrays with the input (shallow copy)
    assert np.shares_memory(mesh['data'], output['data'])
