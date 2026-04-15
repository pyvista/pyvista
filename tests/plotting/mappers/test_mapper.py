from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting import _vtk
from pyvista.plotting.mapper import DataSetMapper


@pytest.fixture
def dataset_mapper(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    return pl.mapper


def test_init(sphere):
    mapper = DataSetMapper()
    mapper.dataset = sphere
    assert mapper.dataset is sphere


def test_dataset_reassign_dataset_then_algorithm(sphere):
    """Reassigning to a vtkAlgorithm must not return the previously cached DataSet."""
    mapper = DataSetMapper()
    mapper.dataset = sphere
    assert mapper.dataset is sphere

    source = _vtk.vtkSphereSource()
    source.SetRadius(2.0)
    mapper.dataset = source
    mapper.update()

    result = mapper.dataset
    assert result is not sphere
    assert isinstance(result, pv.DataSet)
    assert np.isclose(np.max(np.abs(result.bounds)), 2.0)


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


def test_mapper_dataset_property_returns_original(sphere):
    """Verify dataset property returns the original mesh, not the algo output."""
    mapper = DataSetMapper(dataset=sphere)
    assert mapper.dataset is sphere


def test_mapper_set_scalars_does_not_mutate_mesh(sphere):
    """Verify that setting scalars on the mapper does not modify the original mesh."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]
    sphere.set_active_scalars('data_a')

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data_b'], 'data_b')

    # The original mesh's active scalars must NOT be changed
    assert sphere.active_scalars_name == 'data_a'


def test_mapper_pipeline_output_active_scalars(sphere):
    """Verify the mapper's pipeline produces the correct active scalars."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data_a'], 'data_a')
    assert np.array_equal(mapper._mapped_scalars, sphere['data_a'])

    # Changing scalars should update the pipeline output
    mapper.set_scalars(sphere['data_b'], 'data_b')
    assert np.array_equal(mapper._mapped_scalars, sphere['data_b'])


def test_mapper_array_name_setter_updates_pipeline(sphere):
    """Setting ``array_name`` must redirect the internal pipeline too."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_active_scalars('data_a', preference='point')
    assert mapper.array_name == 'data_a'
    assert np.array_equal(mapper._mapped_scalars, sphere['data_a'])

    mapper.array_name = 'data_b'
    assert mapper.array_name == 'data_b'
    assert np.array_equal(mapper._mapped_scalars, sphere['data_b'])


def test_mapper_array_name_initializes_point_scalars_pipeline(sphere):
    """Setting ``array_name`` directly must activate point scalars."""
    sphere['keep_active'] = sphere.points[:, 0]
    sphere['data'] = sphere.points[:, 2]
    sphere.set_active_scalars('keep_active')

    mapper = DataSetMapper(dataset=sphere)
    mapper.array_name = 'data'

    assert mapper._active_scalars_algo is not None
    assert mapper._active_scalars_algo.preference == 'point'
    assert mapper.array_name == 'data'
    assert np.array_equal(mapper._mapped_scalars, sphere['data'])
    assert sphere.active_scalars_name == 'keep_active'


def test_mapper_array_name_initializes_cell_scalars_pipeline():
    """Setting ``array_name`` directly must activate cell scalars."""
    mesh = pv.Cube()
    mesh.point_data['keep_active'] = mesh.points[:, 0]
    mesh.cell_data['cell_data'] = np.arange(mesh.n_cells, dtype=float)
    mesh.set_active_scalars('keep_active')

    mapper = DataSetMapper(dataset=mesh)
    mapper.array_name = 'cell_data'

    assert mapper._active_scalars_algo is not None
    assert mapper._active_scalars_algo.preference == 'cell'
    assert mapper.array_name == 'cell_data'
    assert np.array_equal(mapper._mapped_scalars, mesh.cell_data['cell_data'])
    assert mesh.active_scalars_name == 'keep_active'


def test_mapper_copy_preserves_scalars_config(sphere):
    """copy() must reproduce the active-scalars pipeline."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_active_scalars('data_a', preference='point')

    mapper_copy = mapper.copy()
    assert mapper_copy is not mapper
    assert mapper_copy.dataset is sphere
    assert mapper_copy.array_name == 'data_a'
    assert np.array_equal(mapper_copy._mapped_scalars, sphere['data_a'])

    # The copy's pipeline must be independent of the original
    mapper_copy.set_active_scalars('data_b', preference='point')
    assert mapper.array_name == 'data_a'
    assert mapper_copy.array_name == 'data_b'


def test_mapper_dataset_setter_reconnects_pipeline(sphere):
    """Verify setting a new dataset reconnects the active scalars pipeline."""
    sphere['data'] = sphere.points[:, 0]

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data'], 'data')

    # Create a new mesh with the same array
    new_sphere = pv.Sphere()
    new_sphere['data'] = new_sphere.points[:, 2]
    mapper.dataset = new_sphere

    # Mapper should now reference the new dataset
    assert mapper.dataset is new_sphere
    assert np.array_equal(mapper._mapped_scalars, new_sphere['data'])


def test_mapped_scalars_cell_data():
    """Verify _mapped_scalars returns the correct cell array after set_scalars.

    Exercises the cell-data branch: when scalars are cell-sized, the
    mapper's internal algorithm should use preference='cell' and
    _mapped_scalars must return the cell array, not point data.
    """
    mesh = pv.Cube()
    expected = np.arange(mesh.n_cells, dtype=float)
    mesh.cell_data['cell_val'] = expected

    mapper = DataSetMapper(dataset=mesh)
    mapper.set_scalars(expected, 'cell_val')

    result = mapper._mapped_scalars
    assert result is not None
    assert np.array_equal(result, expected)
    # Verify it actually came from cell_data, not point_data
    assert 'cell_val' not in mesh.point_data


def test_mapped_scalars_fallback_without_algo(sphere):
    """Verify _mapped_scalars falls back to active_scalars when no algo.

    When set_scalars has never been called, the mapper has no internal
    ActiveScalarsAlgorithm and _mapped_scalars should delegate to the
    dataset's own active_scalars.
    """
    sphere['data'] = sphere.points[:, 0]
    sphere.set_active_scalars('data')

    mapper = DataSetMapper(dataset=sphere)
    # No set_scalars call — _active_scalars_algo remains None
    assert mapper._mapped_scalars is not None
    assert np.array_equal(mapper._mapped_scalars, sphere['data'])
    # Also verify the None-dataset edge case
    empty_mapper = DataSetMapper()
    assert empty_mapper._mapped_scalars is None


def test_set_active_scalars_does_not_mutate_dataset(sphere):
    """Public API: set_active_scalars must leave the dataset unchanged."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]
    sphere.set_active_scalars('data_a')

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_active_scalars('data_b', preference='point')

    # Mapper points at data_b
    assert mapper.array_name == 'data_b'
    assert np.array_equal(mapper._mapped_scalars, sphere['data_b'])
    # Dataset is untouched
    assert sphere.active_scalars_name == 'data_a'


def test_set_active_scalars_in_place_update(sphere):
    """Calling set_active_scalars twice updates the existing algorithm."""
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_active_scalars('data_a', preference='point')
    algo = mapper._active_scalars_algo
    assert algo is not None

    mapper.set_active_scalars('data_b', preference='point')
    # The same algorithm instance must be reused — not a new pipeline
    assert mapper._active_scalars_algo is algo
    assert algo.scalars_name == 'data_b'
    assert np.array_equal(mapper._mapped_scalars, sphere['data_b'])


def test_set_active_scalars_cell_preference():
    """Cell preference must wire the cell-data array, not point data."""
    mesh = pv.Cube()
    cell_array = np.arange(mesh.n_cells, dtype=float)
    mesh.cell_data['c'] = cell_array

    mapper = DataSetMapper(dataset=mesh)
    mapper.set_active_scalars('c', preference='cell')

    assert mapper._active_scalars_algo.preference == 'cell'
    assert np.array_equal(mapper._mapped_scalars, cell_array)
    # Verify the array is *not* on point data — exercises the cell branch
    assert 'c' not in mesh.point_data


def test_clear_active_scalars_detaches_pipeline(sphere):
    """clear_active_scalars must remove the algorithm and reset the input."""
    sphere['data'] = sphere.points[:, 0]

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_active_scalars('data', preference='point')
    assert mapper._active_scalars_algo is not None

    mapper.clear_active_scalars()
    assert mapper._active_scalars_algo is None
    # The mapper must still see its dataset
    assert mapper.dataset is sphere
    # And the VTK pipeline must produce valid output
    mapper.update()
    out = pv.wrap(mapper.GetInputDataObject(0, 0))
    assert out.n_points == sphere.n_points


def test_clear_active_scalars_noop_when_unset(sphere):
    """Calling clear_active_scalars on a fresh mapper is a safe no-op."""
    mapper = DataSetMapper(dataset=sphere)
    assert mapper._active_scalars_algo is None
    mapper.clear_active_scalars()
    assert mapper._active_scalars_algo is None
    assert mapper.dataset is sphere


def test_clear_active_scalars_after_algorithm_input():
    """clear_active_scalars must restore the upstream algorithm input.

    When the mapper was originally fed by a vtkAlgorithm (not a dataset),
    detaching the active-scalars algo must reconnect the mapper to that
    upstream algorithm — not leave it dangling.
    """
    source = _vtk.vtkSphereSource()
    source.SetRadius(2.0)

    mapper = DataSetMapper()
    mapper.dataset = source
    mapper.set_active_scalars('Normals', preference='point')
    assert mapper._active_scalars_algo is not None

    mapper.clear_active_scalars()
    mapper.update()
    out = pv.wrap(mapper.GetInputDataObject(0, 0))
    # Output bounds should still come from the upstream sphere source
    assert np.isclose(np.max(np.abs(out.bounds)), 2.0)


def test_as_rgba_uses_mapped_scalars(sphere):
    """Verify as_rgba produces RGBA from the correct mapped scalars.

    When an ActiveScalarsAlgorithm manages the active array, as_rgba
    must use that array (via _mapped_scalars), not the dataset's own
    active_scalars which may point elsewhere.
    """
    sphere['data_a'] = sphere.points[:, 0]
    sphere['data_b'] = sphere.points[:, 2]
    sphere.set_active_scalars('data_a')

    mapper = DataSetMapper(dataset=sphere)
    mapper.set_scalars(sphere['data_b'], 'data_b')
    mapper.as_rgba()

    assert mapper.color_mode == 'direct'
    assert '__rgba__' in sphere.point_data
    # The mesh's own active scalars must still be data_a
    assert sphere.active_scalars_name == 'data_a'
    # Calling as_rgba again should be a no-op (already direct)
    mapper.as_rgba()
    assert mapper.color_mode == 'direct'
