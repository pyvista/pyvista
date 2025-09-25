from __future__ import annotations

import pytest
import vtk

import pyvista as pv
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


def test_scalar_range(dataset_mapper):
    assert isinstance(dataset_mapper.scalar_range, tuple)
    rng = (0, 2)
    dataset_mapper.scalar_range = rng
    assert dataset_mapper.scalar_range == rng


def test_bounds(dataset_mapper):
    assert isinstance(dataset_mapper.bounds, tuple)
    assert dataset_mapper.bounds == (-126.0, 125.0, -127.0, 126.0, -127.0, 127.0)


def test_lookup_table(dataset_mapper):
    assert isinstance(dataset_mapper.lookup_table, vtk.vtkLookupTable)

    table = vtk.vtkLookupTable()

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
