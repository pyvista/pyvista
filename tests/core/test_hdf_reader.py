"""Tests for temporal VTKHDF reading without downloaded data."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

import pyvista as pv


@pytest.fixture
def hdf_file(tmp_path):
    """Write a tetrahedron with shared geometry and nonuniform physical times."""
    path = tmp_path / 'temporal.vtkhdf'
    with h5py.File(path, 'w') as file:
        group = file.create_group('VTKHDF', track_order=True)
        group.attrs['Version'] = np.array([2, 0], dtype=np.int64)
        group.attrs['Type'] = np.bytes_('UnstructuredGrid')
        datasets = {
            'NumberOfPoints': [4],
            'NumberOfCells': [1],
            'NumberOfConnectivityIds': [4],
            'Points': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'Offsets': [0, 4],
            'Connectivity': [0, 1, 2, 3],
            'Types': np.array([10], dtype=np.uint8),
            'PointData/temperature': np.arange(12, dtype=float),
            'PointData/velocity': np.arange(36, dtype=float).reshape(12, 3),
            'CellData/pressure': [10.0, 11.0, 12.0],
        }
        for name, values in datasets.items():
            group[name] = values
        steps = group.create_group('Steps')
        steps.attrs['NSteps'] = 3
        for name, values in {
            'Values': [0.0, 0.25, 2.5],
            'NumberOfParts': [1, 1, 1],
            'PartOffsets': [0, 0, 0],
            'PointOffsets': [0, 0, 0],
            'CellOffsets': [0, 0, 0],
            'ConnectivityIdOffsets': [0, 0, 0],
            'PointDataOffsets/temperature': [0, 4, 8],
            'PointDataOffsets/velocity': [0, 4, 8],
            'CellDataOffsets/pressure': [0, 1, 2],
        }.items():
            steps[name] = values
    return path


@pytest.fixture
def hdf_image_file(tmp_path):
    """Write a temporal image whose point values equal the step index."""
    path = tmp_path / 'image.vtkhdf'
    with h5py.File(path, 'w') as file:
        group = file.create_group('VTKHDF', track_order=True)
        group.attrs['Version'] = np.array([2, 0], dtype=np.int64)
        group.attrs['Type'] = np.bytes_('ImageData')
        group.attrs['WholeExtent'] = np.array([0, 4, 0, 3, 0, 2], dtype=np.int64)
        group.attrs['Origin'] = np.zeros(3)
        group.attrs['Spacing'] = np.ones(3)
        group.attrs['Direction'] = np.eye(3).ravel()
        # Arrays are stored as [time, z, y, x]
        group['PointData/step'] = np.repeat(np.arange(4.0), 60).reshape(4, 3, 4, 5)
        steps = group.create_group('Steps')
        steps.attrs['NSteps'] = 4
        steps['Values'] = [0.0, 10.0, 20.0, 30.0]
    return path


def test_hdf_temporal_metadata(hdf_file):
    """Discover physical times without executing a data request."""
    reader = pv.get_reader(hdf_file)
    events = []
    reader.reader.AddObserver('StartEvent', lambda *_: events.append('read'))
    assert isinstance(reader, pv.TimeReader)
    assert reader.time_values == [0.0, 0.25, 2.5]
    assert reader.number_time_points == 3
    assert reader.active_time_value == 0.0
    assert reader.time_point_value(1) == 0.25
    reader.set_active_time_point(2)
    assert reader.active_time_value == 2.5
    reader.set_active_time_value(0.25)
    assert reader.active_time_value == 0.25
    assert events == []


@pytest.mark.parametrize('by_value', [False, True])
def test_hdf_temporal_navigation(hdf_file, by_value):
    """Read forward, backward, and repeated frames with scalar and vector data."""
    reader = pv.HDFReader(hdf_file)
    initial = reader.read().copy()
    for index in [0, 2, 2, 1, 0, 2]:
        time = reader.time_values[index]
        if by_value:
            reader.set_active_time_value(time)
        else:
            reader.set_active_time_point(index)
        assert reader.active_time_value == time
        mesh = reader.read()
        assert reader.active_time_value == time
        assert np.array_equal(mesh['temperature'], np.arange(4) + index * 4)
        assert np.array_equal(mesh['velocity'], np.arange(12).reshape(4, 3) + index * 12)
        assert np.array_equal(mesh['pressure'], [10.0 + index])
        assert np.array_equal(mesh.points, initial.points)
        assert np.array_equal(mesh.cells, initial.cells)
    assert np.array_equal(initial['temperature'], np.arange(4))


def test_hdf_temporal_pipeline_request(hdf_file):
    """Override an earlier pipeline time request when selecting an index."""
    reader = pv.HDFReader(hdf_file)
    reader.reader.UpdateTimeStep(2.5)
    reader.set_active_time_point(1)
    assert np.array_equal(reader.read()['temperature'], np.arange(4) + 4)


@pytest.mark.parametrize('index', [-1, 3])
def test_hdf_invalid_time_point(hdf_file, index):
    """Reject invalid indices without changing the selection."""
    reader = pv.HDFReader(hdf_file)
    reader.set_active_time_point(1)
    with pytest.raises(ValueError, match='out of range'):
        reader.set_active_time_point(index)
    with pytest.raises(ValueError, match='out of range'):
        reader.time_point_value(index)
    assert reader.active_time_value == 0.25


@pytest.mark.parametrize('index', [1.0, 1.5, True, '1'])
def test_hdf_non_integer_time_point(hdf_file, index):
    """Reject non-integer indices without changing the selection."""
    reader = pv.HDFReader(hdf_file)
    reader.set_active_time_point(np.int64(2))
    with pytest.raises(TypeError, match='must be an integer'):
        reader.set_active_time_point(index)
    with pytest.raises(TypeError, match='must be an integer'):
        reader.time_point_value(index)
    assert reader.active_time_value == 2.5


@pytest.mark.parametrize('time', [-1.0, 1.0, float('nan'), float('inf')])
def test_hdf_invalid_time_value(hdf_file, time):
    """Require a time that exists in the file."""
    reader = pv.HDFReader(hdf_file)
    with pytest.raises(ValueError, match='Not a valid time'):
        reader.set_active_time_value(time)
    assert reader.active_time_value == 0.0


def test_hdf_copied_frame_is_independent(hdf_file):
    """Keep a copied frame unchanged when later frames are read and modified."""
    reader = pv.HDFReader(hdf_file)
    display = reader.read().copy()
    reader.set_active_time_point(1)
    frame = reader.read()
    frame.points[0] = 9.0
    frame['temperature'][:] = -1.0
    assert np.array_equal(display.points[0], [0.0, 0.0, 0.0])
    assert np.array_equal(display['temperature'], np.arange(4))


def _make_static(path):
    with h5py.File(path, 'a') as file:
        del file['VTKHDF/Steps']
        for name in ['temperature', 'velocity']:
            values = file[f'VTKHDF/PointData/{name}'][:4]
            del file[f'VTKHDF/PointData/{name}']
            file[f'VTKHDF/PointData/{name}'] = values
        del file['VTKHDF/CellData/pressure']
        file['VTKHDF/CellData/pressure'] = [10.0]


def test_hdf_static(hdf_file):
    """Preserve static reads and expose no invented time values."""
    _make_static(hdf_file)
    reader = pv.HDFReader(hdf_file)
    assert reader.time_values == []
    assert reader.number_time_points == 0
    with pytest.raises(ValueError, match='No time points'):
        _ = reader.active_time_value
    with pytest.raises(ValueError, match='out of range'):
        reader.set_active_time_point(0)
    with pytest.raises(ValueError, match='Not a valid time'):
        reader.set_active_time_value(0.0)
    assert np.array_equal(reader.read()['temperature'], np.arange(4))
    assert reader.read().n_cells == 1


def test_hdf_change_path(hdf_file, tmp_path):
    """Reset both the selected index and pipeline request when changing files."""
    other = tmp_path / 'other.vtkhdf'
    shutil.copyfile(hdf_file, other)
    with h5py.File(other, 'a') as file:
        steps = file['VTKHDF/Steps']
        steps.attrs['NSteps'] = 2
        del steps['Values']
        steps['Values'] = [10.0, 20.0]
    reader = pv.HDFReader(hdf_file)
    reader.set_active_time_point(2)
    reader.read()
    reader.path = other
    assert reader.time_values == [10.0, 20.0]
    assert reader.active_time_value == 10.0
    assert np.array_equal(reader.read()['temperature'], np.arange(4))
    reader.set_active_time_value(20.0)
    assert np.array_equal(reader.read()['temperature'], np.arange(4) + 4)
    reader.path = hdf_file
    assert reader.active_time_value == 0.0
    assert np.array_equal(reader.read()['temperature'], np.arange(4))


def test_hdf_change_path_to_static(hdf_file, tmp_path):
    """Drop the time selection when switching from a temporal to a static file."""
    static = tmp_path / 'static.vtkhdf'
    shutil.copyfile(hdf_file, static)
    _make_static(static)
    reader = pv.HDFReader(hdf_file)
    reader.set_active_time_point(2)
    reader.read()
    reader.path = static
    assert reader.time_values == []
    with pytest.raises(ValueError, match='No time points'):
        _ = reader.active_time_value
    assert np.array_equal(reader.read()['temperature'], np.arange(4))
    reader.path = hdf_file
    assert reader.active_time_value == 0.0
    assert np.array_equal(reader.read()['temperature'], np.arange(4))


def test_hdf_image_revisit(hdf_image_file):
    """Return the selected image frame when revisiting earlier steps."""
    reader = pv.HDFReader(hdf_image_file)
    assert reader.time_values == [0.0, 10.0, 20.0, 30.0]
    for index in [0, 2, 0, 1, 2, 1, 3, 0]:
        reader.set_active_time_point(index)
        mesh = reader.read()
        assert isinstance(mesh, pv.ImageData)
        assert mesh.dimensions == (5, 4, 3)
        assert np.array_equal(mesh['step'], np.full(60, float(index)))


@pytest.mark.needs_vtk_version(at_least=(9, 4))
@pytest.mark.skipif(
    pv.vtk_version_info[:2] == (9, 6),
    reason='VTK 9.6 logs an invalid initial extent for temporal images inside composites.',
)
def test_hdf_composite_image_revisit(hdf_image_file, tmp_path):
    """Return the selected image frame inside a composite output."""
    path = tmp_path / 'composite.vtkhdf'
    with h5py.File(hdf_image_file, 'r') as source, h5py.File(path, 'w') as target:
        group = target.create_group('VTKHDF', track_order=True)
        group.attrs['Version'] = source['VTKHDF'].attrs['Version']
        group.attrs['Type'] = np.bytes_('MultiBlockDataSet')
        source.copy('VTKHDF', group, name='image')
        group.create_group('Assembly', track_order=True)['image'] = h5py.SoftLink('/VTKHDF/image')
    reader = pv.HDFReader(path)
    assert reader.time_values == [0.0, 10.0, 20.0, 30.0]
    for index in [0, 1, 2, 3, 0, 2]:
        reader.set_active_time_point(index)
        mesh = reader.read()
        assert isinstance(mesh, pv.MultiBlock)
        assert len(mesh) == 1
        assert isinstance(mesh[0], pv.ImageData)
        assert np.array_equal(mesh[0]['step'], np.full(60, float(index)))


def test_hdf_array_selection(hdf_file):
    """Read only enabled arrays and keep the selection across time steps."""
    reader = pv.HDFReader(hdf_file)
    assert isinstance(reader, pv.PointCellDataSelection)
    assert reader.point_array_names == ['temperature', 'velocity']
    assert reader.cell_array_names == ['pressure']
    assert reader.all_point_arrays_status == {'temperature': True, 'velocity': True}
    reader.disable_point_array('velocity')
    reader.disable_all_cell_arrays()
    assert reader.all_point_arrays_status == {'temperature': True, 'velocity': False}
    assert not reader.cell_array_status('pressure')
    for index in [2, 0, 1]:
        reader.set_active_time_point(index)
        mesh = reader.read()
        assert mesh.point_data.keys() == ['temperature']
        assert mesh.cell_data.keys() == []
        assert np.array_equal(mesh['temperature'], np.arange(4) + index * 4)


@pytest.mark.parametrize('fixture', ['hdf_file', 'hdf_image_file'])
def test_hdf_reenable_array(fixture, request):
    """Restore an array enabled again after a read without it."""
    path = request.getfixturevalue(fixture)
    reader = pv.HDFReader(path)
    name = reader.point_array_names[0]
    reader.set_active_time_point(1)
    expected = reader.read()[name].copy()
    reader.disable_all_point_arrays()
    assert reader.read().point_data.keys() == []
    reader.enable_all_point_arrays()
    mesh = reader.read()
    assert mesh.point_data.keys() == reader.point_array_names
    assert np.array_equal(mesh[name], expected)


@pytest.mark.parametrize(('association', 'name'), [('point', 'temperature'), ('cell', 'pressure')])
@pytest.mark.parametrize('read_first', [False, True])
@pytest.mark.parametrize('disable_first', [False, True])
def test_hdf_array_configuration_preserves_cache(
    hdf_file, association, name, read_first, disable_first, mocker
):
    """Preserve geometry caching for no-op enables and selection before reading."""
    reader = pv.HDFReader(hdf_file)
    disable_cache = mocker.spy(reader, '_disable_cache')
    if read_first:
        reader.read()
    if disable_first:
        getattr(reader, f'disable_{association}_array')(name)
    getattr(reader, f'enable_{association}_array')(name)
    needs_workaround = read_first and disable_first and (9, 7) <= pv.vtk_version_info < (9, 8)
    assert disable_cache.call_count == int(needs_workaround)
    first = reader.read()
    reader.set_active_time_point(1)
    second = reader.read()
    assert name in getattr(second, f'{association}_data')
    if (9, 7) <= pv.vtk_version_info < (9, 8) and not needs_workaround:
        assert np.shares_memory(first.points, second.points)
