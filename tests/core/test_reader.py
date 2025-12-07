from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import TYPE_CHECKING

from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities.fileio import _try_imageio_imread
from pyvista.core.utilities.reader import _CLASS_READER_RETURN_TYPE
from pyvista.core.utilities.reader import CLASS_READERS
from pyvista.examples.downloads import download_file

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

HAS_IMAGEIO = True
try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False


def assert_output_type(mesh: pv.DataObject, reader: pv.BaseReader):
    mesh_type = _CLASS_READER_RETURN_TYPE[type(reader)]
    allowed_types = (mesh_type,) if isinstance(mesh_type, str) else mesh_type
    actual_type = type(mesh).__name__
    assert actual_type in allowed_types


def test_reader_output_type_defined():
    expected = set(CLASS_READERS.values())
    actual = set(_CLASS_READER_RETURN_TYPE.keys())
    assert actual == expected, 'Return type must be defined for every reader'


def test_read_raises():
    with pytest.raises(
        ValueError, match=r'Only one of `file_format` and `force_ext` may be specified.'
    ):
        pv.read(Path('foo.vtp'), force_ext='foo', file_format='foo')


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(npoints=st.integers().filter(lambda x: x < 2))
def test_read_texture_raises(mocker: MockerFixture, npoints):
    from pyvista.core.utilities import fileio

    m = mocker.patch.object(fileio, 'read')
    m().n_points = npoints

    m = mocker.patch.object(fileio, '_try_imageio_imread')
    m.return_value = None

    pv.read_texture(file := Path('foo.vtp'))
    m.assert_called_once_with(file.expanduser().resolve())


@pytest.mark.parametrize('sideset', [1.0, None, object(), np.array([])])
def test_read_exodus_raises(sideset):
    with pytest.raises(
        TypeError,
        match=re.escape(f'Could not parse sideset ID/name: {sideset}'),
    ):
        pv.read_exodus(examples.download_mug(load=False), enabled_sidesets=[sideset])


def test_get_reader_fail(tmp_path):
    with pytest.raises(ValueError):  # noqa: PT011
        pv.get_reader('not_a_supported_file.no_data')
    match = '`pyvista.get_reader` does not support reading from directory:\n\t'
    with pytest.raises(ValueError, match=match):
        pv.get_reader(str(tmp_path))


def test_reader_invalid_file():
    # cannot use the BaseReader
    with pytest.raises(FileNotFoundError, match='does not exist'):
        pv.DICOMReader('dummy/')


def test_xmlimagedatareader(tmpdir):
    tmpfile = tmpdir.join('temp.vti')
    mesh = pv.ImageData()
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    new_mesh = reader.read()
    assert_output_type(new_mesh, reader)
    assert isinstance(new_mesh, pv.ImageData)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlrectilineargridreader(tmpdir):
    tmpfile = tmpdir.join('temp.vtr')
    mesh = pv.RectilinearGrid()
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    new_mesh = reader.read()
    assert_output_type(new_mesh, reader)
    assert isinstance(new_mesh, pv.RectilinearGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlunstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join('temp.vtu')
    mesh = pv.UnstructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    vtk_version = pv.vtk_version_info
    if (9, 1, 0) <= vtk_version < (9, 3, 0):
        match = 'No Points element available in first piece found in file'
        with pytest.warns(pv.VTKExecutionWarning, match=match):
            new_mesh = reader.read()
    else:
        new_mesh = reader.read()
    assert_output_type(new_mesh, reader)
    assert isinstance(new_mesh, pv.UnstructuredGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlpolydatareader(tmpdir):
    tmpfile = tmpdir.join('temp.vtp')
    mesh = pv.Sphere()
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    new_mesh = reader.read()
    assert_output_type(new_mesh, reader)
    assert isinstance(new_mesh, pv.PolyData)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join('temp.vts')
    mesh = pv.StructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    new_mesh = reader.read()
    assert_output_type(new_mesh, reader)
    assert isinstance(new_mesh, pv.StructuredGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlmultiblockreader(tmpdir):
    tmpfile = tmpdir.join('temp.vtm')
    mesh = pv.MultiBlock([pv.Sphere() for i in range(5)])
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pv.MultiBlock)
    assert new_mesh.n_blocks == mesh.n_blocks
    for i in range(new_mesh.n_blocks):
        assert new_mesh[i].n_points == mesh[i].n_points
        assert new_mesh[i].n_cells == mesh[i].n_cells


def test_reader_cell_point_data(tmpdir):
    tmpfile = tmpdir.join('temp.vtp')
    mesh = pv.Sphere()
    mesh['height'] = mesh.points[:, 1]
    mesh['id'] = np.arange(mesh.n_cells)
    mesh.save(tmpfile.strpath)
    # mesh has an additional 'Normals' point data array

    reader = pv.get_reader(tmpfile.strpath)

    assert reader.number_cell_arrays == 1
    assert reader.number_point_arrays == 2

    assert reader.cell_array_names == ['id']
    assert reader.point_array_names == ['Normals', 'height']

    assert reader.all_cell_arrays_status == {'id': True}
    assert reader.all_point_arrays_status == {'Normals': True, 'height': True}

    assert reader.cell_array_status('id') is True
    assert reader.point_array_status('Normals') is True

    reader.disable_all_cell_arrays()
    assert reader.cell_array_status('id') is False

    reader.disable_all_point_arrays()
    assert reader.all_point_arrays_status == {'Normals': False, 'height': False}

    reader.enable_all_cell_arrays()
    assert reader.cell_array_status('id') is True

    reader.enable_all_point_arrays()
    assert reader.all_point_arrays_status == {'Normals': True, 'height': True}

    reader.disable_cell_array('id')
    assert reader.cell_array_status('id') is False

    reader.disable_point_array('Normals')
    assert reader.point_array_status('Normals') is False

    reader.enable_cell_array('id')
    assert reader.cell_array_status('id') is True

    reader.enable_point_array('Normals')
    assert reader.point_array_status('Normals') is True


def test_ensightreader_arrays():
    filename = examples.download_backward_facing_step(load=False)

    reader = pv.get_reader(filename)
    assert_output_type(reader.read(), reader)
    assert reader.path == filename
    assert reader.number_cell_arrays == 9
    assert reader.number_point_arrays == 0

    assert reader.cell_array_names == [
        'v2',
        'nut',
        'k',
        'nuTilda',
        'p',
        'omega',
        'f',
        'epsilon',
        'U',
    ]
    assert reader.point_array_names == []

    reader.disable_all_cell_arrays()
    reader.enable_cell_array('k')

    assert reader.all_cell_arrays_status == {
        'v2': False,
        'nut': False,
        'k': True,
        'nuTilda': False,
        'p': False,
        'omega': False,
        'f': False,
        'epsilon': False,
        'U': False,
    }

    mesh = reader.read()
    assert isinstance(mesh, pv.MultiBlock)

    for i in range(mesh.n_blocks):
        assert all([mesh[i].n_points, mesh[i].n_cells])
        assert mesh[i].array_names == ['k']

    # re-enable all cell arrays and read again
    reader.enable_all_cell_arrays()
    all_mesh = reader.read()
    assert isinstance(all_mesh, pv.MultiBlock)

    for i in range(all_mesh.n_blocks):
        assert all([all_mesh[i].n_points, all_mesh[i].n_cells])
        assert all_mesh[i].array_names == [
            'v2',
            'nut',
            'k',
            'nuTilda',
            'p',
            'omega',
            'f',
            'epsilon',
            'U',
        ]


def test_ensightreader_timepoints():
    filename = examples.download_naca(load=False)

    reader = pv.get_reader(filename)
    assert reader.path == filename

    assert reader.number_time_points == 2
    assert reader.time_values == [1.0, 3.0]
    assert reader.time_point_value(0) == 1.0
    assert reader.time_point_value(1) == 3.0

    assert reader.active_time_value == 1.0
    mesh_1 = reader.read()

    reader.set_active_time_value(3.0)
    assert reader.active_time_value == 3.0
    mesh_3 = reader.read()

    # assert all the data is different
    for m_1, m_3 in zip(mesh_1, mesh_3, strict=True):
        assert not all(m_1['DENS'] == m_3['DENS'])

    reader.set_active_time_point(0)
    assert reader.active_time_value == 1.0

    with pytest.raises(ValueError, match='Not a valid time'):
        reader.set_active_time_value(1000.0)


def test_ensightreader_time_sets():
    filename = examples.download_lshape(load=False)

    reader = pv.get_reader(filename)
    assert reader.active_time_set == 0

    reader.set_active_time_set(1)
    assert reader.number_time_points == 11

    mesh = reader.read()['all']
    assert reader.number_time_points == 11
    assert np.isclose(mesh['displacement'][1, 1], 0.0, 1e-10)

    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()['all']
    assert np.isclose(mesh['displacement'][1, 1], 0.0028727285, 1e-10)

    reader.set_active_time_set(0)
    assert reader.number_time_points == 1

    with pytest.raises(IndexError, match='Time set index'):
        reader.set_active_time_set(2)


def test_ensightreader_non_time_series_data():
    filename = examples.download_file('simple_tetra/example.case')
    examples.download_file('simple_tetra/example.geo')

    reader = pv.get_reader(filename)
    assert reader.number_time_points == 0
    assert reader.time_values == []


def test_dcmreader():
    # Test reading directory (image stack)
    directory = examples.download_dicom_stack(load=False)
    reader = pv.get_reader(directory)
    assert directory in str(reader)
    assert isinstance(reader, pv.DICOMReader)
    assert reader.path == directory

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert isinstance(mesh, pv.ImageData)
    assert all([mesh.n_points, mesh.n_cells])

    # Test reading single file (*.dcm)
    filename = str(Path(directory) / '1-1.dcm')
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.DICOMReader)
    assert reader.path == filename

    mesh = reader.read()
    assert isinstance(mesh, pv.ImageData)
    assert all([mesh.n_points, mesh.n_cells])


def test_plyreader():
    filename = examples.spherefile
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PLYReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_objreader():
    filename = examples.download_doorman(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.OBJReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_stlreader():
    filename = examples.download_gears(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.STLReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_tecplotreader():
    filename = examples.download_tecplot_ascii(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.TecplotReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh[0].n_points, mesh[0].n_cells])


def test_vtkreader():
    filename = examples.hexbeamfile
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.VTKDataSetReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_byureader():
    filename = examples.download_teapot(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.BYUReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_facetreader():
    filename = examples.download_clown(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.FacetReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_plot3dmetareader():
    filename = download_file('multi.p3d')
    download_file('multi-bin.xyz')
    download_file('multi-bin.q')
    download_file('multi-bin.f')
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.Plot3DMetaReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    for m in mesh:
        assert all([m.n_points, m.n_cells])


def test_multiblockplot3dreader():
    filename = download_file('multi-bin.xyz')
    q_filename = download_file('multi-bin.q')
    reader = pv.MultiBlockPlot3DReader(filename)
    assert reader.path == filename

    mesh = reader.read()
    for m in mesh:
        assert all([m.n_points, m.n_cells])
        assert len(m.array_names) == 0

    # Reader doesn't yet support reusability
    reader = pv.MultiBlockPlot3DReader(filename)
    reader.add_q_files(q_filename)

    reader.add_function(112)  # add by int
    reader.add_function(pv.reader.Plot3DFunctionEnum.PRESSURE_GRADIENT)  # add by enum
    reader.add_function(reader.KINETIC_ENERGY)  # add by class variable (alias to enum value)
    reader.add_function(reader.ENTROPY)  # add ENTROPY by class variable
    reader.remove_function(170)  # remove ENTROPY by int

    reader.add_function(reader.ENTROPY)
    reader.remove_function(reader.ENTROPY)  # remove by class variable

    mesh = reader.read()
    for m in mesh:
        assert len(m.array_names) > 0

    assert 'MachNumber' in mesh[0].point_data
    assert 'PressureGradient' in mesh[0].point_data
    assert 'KineticEnergy' in mesh[0].point_data
    assert 'Entropy' not in mesh[0].point_data

    reader = pv.MultiBlockPlot3DReader(filename)
    reader.add_q_files([q_filename])
    mesh = reader.read()
    for m in mesh:
        assert len(m.array_names) > 0

    reader = pv.MultiBlockPlot3DReader(filename)

    # get/set of `auto_detect_format`
    reader.auto_detect_format = False
    assert reader.auto_detect_format is False
    reader.auto_detect_format = True
    assert reader.auto_detect_format is True

    # get/set of `preserve_intermediate_functions`
    reader.preserve_intermediate_functions = False
    assert reader.preserve_intermediate_functions is False
    reader.preserve_intermediate_functions = True
    assert reader.preserve_intermediate_functions is True

    # get/set of `gamma`
    reader.gamma = 1.5
    assert reader.gamma == 1.5
    reader.gamma = 99
    assert reader.gamma == 99

    # get/set of `r_gas_constant`
    reader.r_gas_constant = 5
    assert reader.r_gas_constant == 5
    reader.r_gas_constant = 10
    assert reader.r_gas_constant == 10

    # check removing all functions
    reader = pv.MultiBlockPlot3DReader(filename)
    reader.add_q_files(q_filename)
    reader.add_function(reader.ENTROPY)
    reader.remove_all_functions()
    mesh_no_functions = reader.read()
    assert 'ENTROPY' not in mesh_no_functions[0].point_data


def test_binarymarchingcubesreader():
    filename = examples.download_pine_roots(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.BinaryMarchingCubesReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])
    read_mesh = pv.read(filename)
    assert mesh == read_mesh


def test_pvdreader():
    filename = examples.download_wavy(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PVDReader)
    assert isinstance(reader.reader, pv.core.utilities.reader._PVDReader)
    assert reader.path == filename

    assert reader.number_time_points == 15
    assert reader.time_point_value(1) == 1.0
    assert np.array_equal(reader.time_values, np.arange(0, 15, dtype=float))

    assert reader.active_time_value == reader.time_values[0]

    active_datasets = reader.active_datasets
    assert len(active_datasets) == 1
    active_dataset0 = active_datasets[0]
    assert active_dataset0.time == 0.0
    assert active_dataset0.path == 'wavy/wavy00.vts'
    assert active_dataset0.group == ''
    assert active_dataset0.part == 0

    assert len(reader.datasets) == len(reader.time_values)

    active_readers = reader.active_readers
    assert len(active_readers) == 1
    active_reader = active_readers[0]
    assert isinstance(active_reader, pv.XMLStructuredGridReader)

    reader.set_active_time_value(1.0)
    assert reader.active_time_value == 1.0

    reader.set_active_time_point(2)
    assert reader.active_time_value == 2.0

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert len(mesh) == 1
    assert isinstance(mesh[0], pv.StructuredGrid)


def test_pvdreader_no_time_group():
    filename = examples.download_dual_sphere_animation(load=False)  # download all the files
    # Use a pvd file that has no timestep or group and two parts.
    filename = str(Path(filename).parent / 'dualSphereNoTime.pvd')

    reader = pv.PVDReader(filename)
    assert reader.time_values == [0.0]
    assert reader.active_time_value == 0.0

    assert len(reader.active_datasets) == 2
    for i, dataset in enumerate(reader.active_datasets):
        assert dataset.time == 0.0
        assert dataset.group is None
        assert dataset.part == i


@pytest.mark.skip_windows
def test_pvdreader_no_part_group():
    filename = examples.download_dual_sphere_animation(load=False)  # download all the files
    # Use a pvd file that has no parts and with timesteps.
    filename = str(Path(filename).parent / 'dualSphereAnimation4NoPart.pvd')

    reader = pv.PVDReader(filename)
    assert reader.active_time_value == 0.0
    assert len(reader.active_datasets) == 1

    reader.set_active_time_value(1.0)
    assert len(reader.active_datasets) == 2
    for dataset in reader.active_datasets:
        assert dataset.time == 1.0
        assert dataset.group == ''
        assert dataset.part == 0


def get_cavity_reader():
    filename = examples.download_cavity(load=False)
    return pv.get_reader(filename)


def test_openfoamreader_arrays_time():
    reader = get_cavity_reader()
    assert isinstance(reader, pv.OpenFOAMReader)

    assert reader.number_point_arrays == 0
    assert reader.number_cell_arrays == 2

    assert reader.number_time_points == 6
    assert reader.time_values == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def test_openfoamreader_active_time():
    reader = get_cavity_reader()
    assert reader.active_time_value == 0.0
    reader.set_active_time_point(1)
    assert reader.active_time_value == 0.5
    reader.set_active_time_value(1.0)
    assert reader.active_time_value == 1.0

    with pytest.raises(
        ValueError,
        match=r'Not a valid .* time values: \[0.0, 0.5, 1.0, 1.5, 2.0, 2.5\]',
    ):
        reader.set_active_time_value(1000)


def test_openfoamreader_read_data_time_value():
    reader = get_cavity_reader()

    reader.set_active_time_value(0.0)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 0.0, 0.0, 1e-10)

    reader.set_active_time_value(0.5)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.524879113887437e-05, 0.0, 1e-10)

    reader.set_active_time_value(1.0)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.5253094867803156e-05, 0.0, 1e-10)

    reader.set_active_time_value(1.5)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.525657641352154e-05, 0.0, 1e-10)

    reader.set_active_time_value(2.0)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.5258551836013794e-05, 0.0, 1e-10)

    reader.set_active_time_value(2.5)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.525951953837648e-05, 0.0, 1e-10)


def test_openfoamreader_read_data_time_point():
    reader = get_cavity_reader()

    reader.set_active_time_point(0)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 0.0, 0.0, 1e-10)

    reader.set_active_time_point(1)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.524879113887437e-05, 0.0, 1e-10)

    reader.set_active_time_point(2)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.5253094867803156e-05, 0.0, 1e-10)

    reader.set_active_time_point(3)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.525657641352154e-05, 0.0, 1e-10)

    reader.set_active_time_point(4)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.5258551836013794e-05, 0.0, 1e-10)

    reader.set_active_time_point(5)
    data = reader.read()['internalMesh']
    assert np.isclose(data.cell_data['U'][:, 1].mean(), 4.525951953837648e-05, 0.0, 1e-10)


@pytest.mark.needs_vtk_version(
    less_than=(9, 3),
    reason='polyhedra decomposition was removed after 9.3',
)
def test_openfoam_decompose_polyhedra():
    reader = get_cavity_reader()
    reader.decompose_polyhedra = False
    assert reader.decompose_polyhedra is False
    reader.decompose_polyhedra = True
    assert reader.decompose_polyhedra is True


def test_openfoam_skip_zero_time():
    reader = get_cavity_reader()

    reader.skip_zero_time = True
    assert reader.skip_zero_time is True
    assert 0.0 not in reader.time_values

    # Test from 'True' to 'False'
    reader.skip_zero_time = False
    assert reader.skip_zero_time is False
    assert 0.0 in reader.time_values

    # Test from 'False' to 'True'
    reader.skip_zero_time = True
    assert reader.skip_zero_time is True
    assert 0.0 not in reader.time_values


def test_openfoam_cell_to_point_default():
    reader = get_cavity_reader()
    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert reader.cell_to_point_creation is True
    assert mesh[0].n_arrays == 4

    reader = get_cavity_reader()
    reader.cell_to_point_creation = False
    assert reader.cell_to_point_creation is False
    mesh = reader.read()
    assert mesh[0].n_arrays == 2

    reader = get_cavity_reader()
    mesh = reader.read()
    reader.cell_to_point_creation = True
    assert reader.cell_to_point_creation is True
    assert mesh[0].n_arrays == 4


def test_openfoam_patch_arrays():
    patch_array_key = 'boundary'
    reader_patch_prefix = 'patch/'

    reader = get_cavity_reader()
    assert reader.number_patch_arrays == 4
    assert reader.patch_array_names == [
        'internalMesh',
        f'{reader_patch_prefix}movingWall',
        f'{reader_patch_prefix}fixedWalls',
        f'{reader_patch_prefix}frontAndBack',
    ]
    assert reader.all_patch_arrays_status == {
        'internalMesh': True,
        f'{reader_patch_prefix}movingWall': True,
        f'{reader_patch_prefix}fixedWalls': True,
        f'{reader_patch_prefix}frontAndBack': True,
    }

    # first only read in 'internalMesh'
    for patch_array in reader.patch_array_names[1:]:
        reader.disable_patch_array(patch_array)
    assert reader.all_patch_arrays_status == {
        'internalMesh': True,
        f'{reader_patch_prefix}movingWall': False,
        f'{reader_patch_prefix}fixedWalls': False,
        f'{reader_patch_prefix}frontAndBack': False,
    }
    mesh = reader.read()
    assert mesh.n_blocks == 1
    assert patch_array_key not in mesh.keys()

    # now read in one more patch
    reader = get_cavity_reader()
    reader.disable_all_patch_arrays()
    reader.enable_patch_array('internalMesh')
    reader.enable_patch_array(f'{reader_patch_prefix}fixedWalls')
    mesh = reader.read()
    assert mesh.n_blocks == 2
    assert patch_array_key in mesh.keys()
    assert mesh[patch_array_key].keys() == ['fixedWalls']

    # check multiple patch arrays without 'internalMesh'
    reader = get_cavity_reader()
    reader.disable_patch_array('internalMesh')
    mesh = reader.read()
    assert mesh.n_blocks == 1
    assert patch_array_key in mesh.keys()
    assert mesh[patch_array_key].keys() == ['movingWall', 'fixedWalls', 'frontAndBack']


def test_openfoam_case_type():
    reader = get_cavity_reader()
    reader.case_type = 'decomposed'
    assert reader.case_type == 'decomposed'
    reader.case_type = 'reconstructed'
    assert reader.case_type == 'reconstructed'
    with pytest.raises(ValueError, match=r"Unknown case type 'wrong_value'."):
        reader.case_type = 'wrong_value'


def test_read_cgns():
    filename = examples.download_cgns_structured(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.CGNSReader)
    assert 'CGNS' in str(reader)
    reader.show_progress()
    assert reader._progress_bar is True
    assert_output_type(reader.read(), reader)

    assert reader.distribute_blocks in [True, False]  # don't insist on a VTK default
    reader.distribute_blocks = True
    assert reader.distribute_blocks
    reader.disable_all_bases()
    reader.disable_all_families()
    reader.disable_all_cell_arrays()
    reader.disable_all_point_arrays()
    reader.load_boundary_patch = False
    empty_block = reader.read()
    assert len(empty_block) == 0

    reader.enable_all_bases()
    reader.enable_all_families()
    reader.enable_all_cell_arrays()
    reader.enable_all_point_arrays()

    # check can modify unsteady_pattern
    orig = reader.unsteady_pattern
    reader.unsteady_pattern = not orig
    assert reader.unsteady_pattern is not orig

    # check can modify vector_3d
    orig = reader.vector_3d
    reader.vector_3d = not orig
    assert reader.vector_3d is not orig

    # check can boundary_patch accessible
    assert reader.load_boundary_patch in [True, False]

    reader.hide_progress()
    assert reader._progress_bar is False

    block = reader.read()
    assert len(block[0]) == 12

    # actual block
    assert len(block[0][0].cell_data) == 3

    assert reader.base_array_names == ['SQNZ']
    assert reader.base_array_status('SQNZ') is True

    assert reader.family_array_names == ['inflow', 'outflow', 'sym', 'wall']
    assert reader.family_array_status('inflow') is True


def test_bmp_reader_writer(tmp_path):
    filename = examples.download_masonry_texture(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.BMPReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    new_filename = tmp_path / 'new.bmp'
    mesh.save(new_filename)
    assert pv.read(filename) == pv.read(new_filename)


def test_demreader():
    filename = examples.download_st_helens(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.DEMReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_jpeg_reader_writer(tmp_path):
    filename = examples.download_bird(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.JPEGReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    new_filename = tmp_path / 'new.jpg'
    mesh.save(new_filename)
    assert pv.compare_images(filename, new_filename) < 5

    new_filename = tmp_path / 'new.jpeg'
    mesh.save(new_filename)
    assert pv.compare_images(filename, new_filename) < 5


def test_meta_image_reader():
    filename = examples.download_chest(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.MetaImageReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_nifti_reader_writer(tmp_path):
    filename = examples.download_brain_atlas_with_sides(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.NIFTIReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    new_filename = tmp_path / 'new.nii'
    mesh.save(new_filename)
    assert pv.read(filename) == pv.read(new_filename)

    new_filename = tmp_path / 'new.nii.gz'
    mesh.save(new_filename)
    assert pv.read(filename) == pv.read(new_filename)


def test_nrrd_reader():
    filename = examples.download_beach(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.NRRDReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_png_reader_writer(tmp_path):
    filename = examples.download_vtk_logo(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PNGReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    new_filename = tmp_path / 'new.png'
    mesh.save(new_filename)
    assert pv.read(filename) == pv.read(new_filename)


def test_pnm_reader_writer(tmp_path):
    filename = examples.download_gourds_pnm(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PNMReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    new_filename = tmp_path / 'new.pnm'
    mesh.save(new_filename)
    assert pv.read(filename) == pv.read(new_filename)


def test_slc_reader():
    filename = examples.download_knee_full(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.SLCReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_tiff_reader_writer(tmp_path):
    filename = examples.download_crater_imagery(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.TIFFReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    new_filename = tmp_path / 'new.tif'
    mesh.save(new_filename)
    old = pv.read(filename)
    new = pv.read(new_filename)
    # We should be able to do `assert new == old` but the equality check is too strict since
    # there is floating point error associated with the spacing, so use `np.allclose` instead
    assert np.allclose(old.active_scalars, new.active_scalars)
    assert np.allclose(old.index_to_physical_matrix, new.index_to_physical_matrix)

    new_filename = tmp_path / 'new.tiff'
    mesh.save(new_filename)
    old = pv.read(filename)
    new = pv.read(new_filename)
    assert np.allclose(old.active_scalars, new.active_scalars)
    assert np.allclose(old.index_to_physical_matrix, new.index_to_physical_matrix)


def test_hdr_reader():
    filename = examples.download_parched_canal_4k(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.HDRReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_avsucd_reader():
    filename = examples.download_cells_nd(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.AVSucdReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_hdf_reader():
    filename = examples.download_can_crushed_hdf(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.HDFReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])
    assert mesh.n_points == 6724
    assert 'VEL' in mesh.point_data
    assert mesh.n_cells == 4800


def test_xdmf_reader():
    filename = examples.download_meshio_xdmf(load=False)

    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.XdmfReader)
    assert reader.path == filename

    assert reader.number_time_points == 5
    assert reader.time_values == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert reader.time_point_value(0) == 0.0
    assert reader.time_point_value(1) == 0.25

    assert reader.number_grids == 6
    assert reader.number_point_arrays == 2

    assert reader.point_array_names == ['phi', 'u']
    assert reader.cell_array_names == ['a']

    blocks = reader.read()
    assert_output_type(blocks, reader)
    assert reader.active_time_value == 0.0
    assert np.array_equal(blocks['TimeSeries_meshio']['phi'], np.array([0.0, 0.0, 0.0, 0.0]))
    reader.set_active_time_value(0.25)
    assert reader.active_time_value == 0.25
    blocks = reader.read()
    assert np.array_equal(blocks['TimeSeries_meshio']['phi'], np.array([0.25, 0.25, 0.25, 0.25]))
    reader.set_active_time_value(0.5)
    assert reader.active_time_value == 0.5
    blocks = reader.read()
    assert np.array_equal(blocks['TimeSeries_meshio']['phi'], np.array([0.5, 0.5, 0.5, 0.5]))
    reader.set_active_time_value(0.75)
    assert reader.active_time_value == 0.75
    blocks = reader.read()
    assert np.array_equal(blocks['TimeSeries_meshio']['phi'], np.array([0.75, 0.75, 0.75, 0.75]))
    reader.set_active_time_value(1.0)
    assert reader.active_time_value == 1.0
    blocks = reader.read()
    assert np.array_equal(blocks['TimeSeries_meshio']['phi'], np.array([1.0, 1.0, 1.0, 1.0]))

    reader.set_active_time_point(0)
    assert reader.active_time_value == 0.0

    with pytest.raises(ValueError, match='Not a valid time'):
        reader.set_active_time_value(1000.0)


@pytest.mark.skipif(not HAS_IMAGEIO, reason='Requires imageio')
def test_try_imageio_imread():
    img = _try_imageio_imread(examples.mapfile)
    assert isinstance(img, (imageio.core.util.Array, np.ndarray))


def test_xmlpartitioneddatasetreader(tmpdir):
    tmpfile = tmpdir.join('temp.vtpd')
    partitions = pv.PartitionedDataSet(
        [pv.Wavelet(extent=(0, 10, 0, 10, 0, 5)), pv.Wavelet(extent=(0, 10, 0, 10, 5, 10))],
    )
    partitions.save(tmpfile.strpath)
    reader = pv.get_reader(tmpfile)
    new_partitions = pv.read(tmpfile.strpath)
    assert_output_type(new_partitions, reader)
    assert len(new_partitions) == len(partitions)
    for i, new_partition in enumerate(new_partitions):
        assert isinstance(new_partition, pv.ImageData)
        assert new_partition.n_cells == partitions[i].n_cells


@pytest.mark.needs_vtk_version(
    9, 3, 0, reason='Requires VTK>=9.3.0 for a concrete FLUENTCFFReader class.'
)
def test_fluentcffreader():
    filename = examples.download_room_cff(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.FLUENTCFFReader)
    assert reader.path == filename

    blocks = reader.read()
    assert_output_type(blocks, reader)
    assert blocks.n_blocks == 1
    assert isinstance(blocks[0], pv.UnstructuredGrid)
    assert blocks.bounds == (0.0, 4.0, 0.0, 4.0, 0.0, 0.0)


def test_gambitreader():
    filename = examples.download_prism(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.GambitReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_gaussian_cubes_reader():
    filename = examples.download_m4_total_density(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.GaussianCubeReader)
    assert reader.path == filename

    hb_scale = 1.1
    b_scale = 10.0
    reader.hb_scale = hb_scale
    reader.b_scale = b_scale
    assert reader.hb_scale == hb_scale
    assert reader.b_scale == b_scale

    grid = reader.read(grid=True)
    assert isinstance(grid, pv.ImageData)
    assert_output_type(grid, reader)
    assert all([grid.n_points, grid.n_cells])

    poly = reader.read(grid=False)
    assert isinstance(poly, pv.PolyData)
    assert_output_type(poly, reader)
    assert all([poly.n_points, poly.n_cells])


def test_gesignareader():
    filename = examples.download_e07733s002i009(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.GESignaReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_pdbreader():
    filename = examples.download_caffeine(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PDBReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_particle_reader():
    filename = examples.download_particles(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.ParticleReader)
    assert reader.path == filename

    reader.endian = 'BigEndian'
    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])

    reader.endian = 'LittleEndian'

    with pytest.raises(ValueError, match='Invalid endian:'):
        reader.endian = 'InvalidEndian'


def test_prostar_reader():
    filename = examples.download_prostar(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.ProStarReader)
    assert reader.path == filename

    mesh = reader.read()
    assert_output_type(mesh, reader)
    assert all([mesh.n_points, mesh.n_cells])


def test_grdecl_reader(tmp_path):
    def read(content, include_content, **kwargs):
        path = tmp_path

        with Path.open(path / '3x3x3.grdecl', 'w') as f:
            f.write(''.join(content))

        with Path.open(path / '3x3x3_include.grdecl', 'w') as f:
            f.write(''.join(include_content))

        return pv.core.utilities.fileio.read_grdecl(path / '3x3x3.grdecl', **kwargs)

    path = Path(__file__).parent.parent / 'example_files'

    with Path.open(path / '3x3x3.grdecl') as f:
        content = list(f)

    with Path.open(path / '3x3x3_include.grdecl') as f:
        include_content = list(f)

    # Test base sample file
    grid = read(content, include_content, other_keywords=['KEYWORD'])

    assert grid.n_cells == 27
    assert grid.n_points == 216
    assert np.allclose(grid.cell_data['PORO'].sum(), 0.1 * 27)
    assert grid.user_dict['MAPAXES'] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert grid.user_dict['MAPUNITS'] == 'METRES'
    assert grid.user_dict['GRIDUNIT'] == 'METRES MAP'
    assert ''.join(grid.user_dict['KEYWORD']) == '1234'
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 5
    assert np.allclose(grid.points[:, 1].sum(), 108.0)

    # Test fails
    match = 'Cylindric grids are not supported'
    content_copy = list(content)
    content_copy[1] = content_copy[1].replace('F', 'T')
    with pytest.raises(TypeError, match=match):
        _ = read(content_copy, include_content)

    match = "Unable to generate grid without keyword 'SPECGRID'"
    content_copy = list(content)
    content_copy[0] = content_copy[0].replace('SPECGRID', 'PLACEHOLDER')
    with pytest.raises(ValueError, match=match):
        _ = read(content_copy, include_content)

    # Test relative coordinates
    include_content[5] = include_content[5].replace('METRES MAP', 'METRES')
    grid = read(content, include_content)
    assert np.allclose(grid.points[:, 1].sum(), 324.0)

    # Test relative warnings
    match = 'Unable to convert relative coordinates with different grid and map units'
    include_content_copy = list(include_content)
    include_content_copy[3] = include_content_copy[3].replace('METRES', 'FEET')
    with pytest.warns(UserWarning, match=match):
        _ = read(content, include_content_copy)

    match = "Unable to convert relative coordinates without keyword 'MAPUNITS'"
    include_content_copy = list(include_content)
    include_content_copy[2] = include_content_copy[2].replace('MAPUNITS', 'PLACEHOLDER')
    with pytest.warns(UserWarning, match=match):
        _ = read(content, include_content_copy)

    match = "Unable to convert relative coordinates without keyword 'MAPAXES'"
    include_content_copy = list(include_content)
    include_content_copy[0] = include_content_copy[0].replace('MAPAXES', 'PLACEHOLDER')
    with pytest.warns(UserWarning, match=match):
        _ = read(content, include_content_copy)


def test_nek5000_reader():
    # load nek5000 file
    filename = examples.download_nek5000(load=False)

    # this class only available for vtk versions >= 9.3
    if pv.vtk_version_info < (9, 3):
        with pytest.raises(pv.VTKVersionError):
            _ = pv.get_reader(filename)
        return

    # test get_reader
    nek_reader = pv.get_reader(filename)
    assert_output_type(nek_reader.read(), nek_reader)

    # test time routines
    # Check correct number of time points
    ntimes = 11
    dt = 0.01
    assert nek_reader.number_time_points == ntimes, 'Checks number of time points'
    assert nek_reader.active_time_point == 0, 'Checks the first time set'

    assert nek_reader.time_values == [dt * i for i in range(ntimes)]

    # check setting and getting of time points and times
    for i in range(ntimes):
        nek_reader.set_active_time_point(i)
        assert nek_reader.active_time_point == i, 'check time point set'

        t = i * dt
        assert nek_reader.time_point_value(i) == t, 'Check correct times'

        assert nek_reader.active_time_value == t, 'Check correct time set'

    # check time setting based on time
    for i in range(ntimes):
        nek_reader.set_active_time_value(i * dt)
        assert nek_reader.active_time_point == i, 'check time point set'

    assert all(
        array in nek_reader.point_array_names
        for array in ['Pressure', 'Velocity', 'Velocity Magnitude']
    )

    match = re.escape('Time point (-1) out of range [0, 10]')
    with pytest.raises(ValueError, match=match):
        nek_reader.set_active_time_point(-1)

    match = re.escape('Time point (11) out of range [0, 10]')
    with pytest.raises(ValueError, match=match):
        nek_reader.set_active_time_point(11)

    # check deactivation of cell array routines
    with pytest.raises(AttributeError):
        _ = nek_reader.number_cell_arrays

    with pytest.raises(AttributeError):
        _ = nek_reader.cell_array_names

    name = 'test'
    with pytest.raises(AttributeError):
        nek_reader.enable_cell_array(name)

    with pytest.raises(AttributeError):
        nek_reader.disable_cell_array(name)

    with pytest.raises(AttributeError):
        nek_reader.cell_array_status(name)

    with pytest.raises(AttributeError):
        nek_reader.enable_all_cell_arrays()

    with pytest.raises(AttributeError):
        nek_reader.disable_all_cell_arrays()

    # check enabling and disabling of point arrays
    for name in nek_reader.point_array_names:
        # Should be enabled by default
        assert nek_reader.point_array_status(name)

        nek_reader.disable_point_array(name)
        assert not nek_reader.point_array_status(name)

        nek_reader.enable_point_array(name)
        assert nek_reader.point_array_status(name)

    # check default clean grid option
    assert nek_reader.reader.GetCleanGrid() == 0

    # check default spectral element IDs
    assert nek_reader.reader.GetSpectralElementIds() == 0

    # check read() method produces the correct dataset
    nek_reader.set_active_time_point(0)
    nek_data = nek_reader.read()
    assert isinstance(nek_data, pv.UnstructuredGrid), 'Check read type is valid'
    assert all(
        key in nek_data.point_data.keys() for key in ['Pressure', 'Velocity', 'Velocity Magnitude']
    )

    # test merge points routines
    assert nek_data.n_points == 8 * 8 * 16 * 16, 'Check n_points without merging points'
    assert 'spectral element id' not in nek_data.cell_data

    # check that different arrays are returned when the time is changed
    # after an initial read() call
    nek_reader.set_active_time_point(1)
    nek_data1 = nek_reader.read()
    for scalar in nek_data.point_data.keys():
        assert not np.array_equal(nek_data.point_data[scalar], nek_data1.point_data[scalar])

    # Note that for some reason merging points after an initial read()
    # has no effect so re-creating reader
    nek_reader = pv.get_reader(filename)

    # check enable merge points
    nek_reader.enable_merge_points()
    assert nek_reader.reader.GetCleanGrid() == 1

    # positively check disable merge points
    nek_reader.disable_merge_points()
    assert nek_reader.reader.GetCleanGrid() == 0

    # re-enable
    nek_reader.enable_merge_points()

    # check enabling of spectral element IDs
    nek_reader.enable_spectral_element_ids()
    assert nek_reader.reader.GetSpectralElementIds() == 1

    # positively check disable spectral element IDs
    nek_reader.disable_spectral_element_ids()
    assert nek_reader.reader.GetSpectralElementIds() == 0
    nek_reader.enable_spectral_element_ids()

    nek_data = nek_reader.read()
    assert nek_data.n_points == (7 * 16 + 1) * (7 * 16 + 1), 'Check n_points with merging points'
    assert 'spectral element id' in nek_data.cell_data


@pytest.mark.parametrize(
    ('data_object', 'ext'),
    [(pv.MultiBlock([examples.load_ant()]), '.pkl'), (examples.load_ant(), '.pickle')],
)
@pytest.mark.needs_vtk_version(9, 3, reason='VTK version not supported.')
def test_read_write_pickle(tmp_path, data_object, ext):
    filepath = tmp_path / ('data_object' + ext)
    data_object.save(filepath)
    new_data_object = pv.read(filepath)
    assert data_object == new_data_object

    # Test raises
    with open(str(filepath), 'wb') as f:  # noqa: PTH123
        # Create non-mesh pickle file
        pickle.dump([1, 2, 3], f)
    match = (
        "Pickled object must be an instance of <class 'pyvista.core.dataobject.DataObject'>. "
        "Got <class 'list'> instead."
    )
    with pytest.raises(TypeError, match=match):
        pv.read(filepath)

    match = "Filename must be a file path with extension ('.pkl', '.pickle'). Got {} instead."
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.read_pickle({})

    match = (
        "Only <class 'pyvista.core.dataobject.DataObject'> are supported for pickling. "
        "Got <class 'dict'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.save_pickle('filename', {})


def test_exodus_reader_ext():
    # test against mug and exodus to check different valid file
    # extensions: .e and .exo
    reader = pv.ExodusIIReader
    assert reader.extensions == (
        '.e',
        '.ex2',
        '.exii',
        '.exo',
    )

    fname_e = examples.download_mug(load=False)
    fname_exo = examples.download_exodus(load=False)

    e_reader = pv.get_reader(fname_e)
    exo_reader = pv.get_reader(fname_exo)

    assert isinstance(e_reader, pv.ExodusIIReader)
    assert isinstance(exo_reader, pv.ExodusIIReader)

    assert_output_type(e_reader.read(), e_reader)
    assert_output_type(exo_reader.read(), exo_reader)


def test_exodus_reader_core():
    # check internals
    fname_e = examples.download_mug(load=False)
    e_reader = pv.get_reader(fname_e)

    # check enabling of displacements (To match functionality
    # from read_exodus)
    e_reader.enable_displacements()
    assert e_reader.reader.GetApplyDisplacements() == 1
    assert e_reader.reader.GetDisplacementMagnitude() == 1.0

    e_reader.disable_displacements()
    assert e_reader.reader.GetApplyDisplacements() == 0

    # check number of cell and point arrays and their names
    assert e_reader.number_point_arrays == 2
    assert e_reader.number_cell_arrays == 1

    assert all(array in e_reader.point_array_names for array in ['convected', 'diffused'])

    assert 'aux_elem' in e_reader.cell_array_names

    # check enabling and disabling of point arrays
    for name in e_reader.point_array_names:
        # Should be enabled by default
        assert e_reader.point_array_status(name)

        e_reader.disable_point_array(name)
        assert not e_reader.point_array_status(name)

        e_reader.enable_point_array(name)
        assert e_reader.point_array_status(name)

    # check enabling and disabling of cell arrays
    for name in e_reader.cell_array_names:
        # Should be enabled by default
        assert e_reader.cell_array_status(name)

        e_reader.disable_cell_array(name)
        assert not e_reader.cell_array_status(name)

        e_reader.enable_cell_array(name)
        assert e_reader.cell_array_status(name)

    # check global arrays handling
    assert e_reader.number_global_arrays == 1

    for name in e_reader.global_array_names:
        # Should not be enabled by default
        assert not e_reader.global_array_status(name)

        e_reader.enable_global_array(name)
        assert e_reader.global_array_status(name)

        e_reader.disable_global_array(name)
        assert not e_reader.cell_array_status(name)

    e_reader.enable_all_global_arrays()
    e_reader.disable_all_global_arrays()
    e_reader.enable_all_global_arrays()

    table = e_reader.read_global()
    assert table.keys() == ['func_pp', 'Time']
    assert table.n_rows == 21

    # test time routines
    ntimes = 21
    dt = 0.1

    # check correct number of time points
    assert e_reader.number_time_points == ntimes, 'Checks number of time points'
    assert e_reader.reader.GetTimeStep() == 0, 'Checks the first time set'

    assert np.allclose(e_reader.time_values, [dt * i for i in range(ntimes)], atol=1e-8, rtol=1e-8)

    # check setting and getting of time points and times
    for i in range(ntimes):
        e_reader.set_active_time_point(i)
        assert e_reader.reader.GetTimeStep() == i, 'check time point set'

        tp = e_reader.reader.GetTimeStep()
        assert tp == i, 'Check underlying reader time step setting'

        t = i * dt
        assert np.isclose(e_reader.time_point_value(i), t, atol=1e-8, rtol=1e-8), (
            'Check correct times'
        )

        assert np.isclose(e_reader.active_time_value, t, atol=1e-8, rtol=1e-8), (
            'Check correct time set'
        )

    # check time setting based on time
    for i, t in enumerate(e_reader.time_values):
        e_reader.set_active_time_value(t)
        assert e_reader.reader.GetTimeStep() == i, 'check time point set'

    # check for error if time not present
    bad_time = 1.25
    err_msg = re.escape(f'Time {bad_time} not present. Available times are {e_reader.time_values}')
    with pytest.raises(ValueError, match=err_msg):
        e_reader.set_active_time_value(1.25)

    # check read with point and cell arrays present
    multiblock = e_reader.read()

    unstruct = multiblock[0][0]

    assert isinstance(unstruct, pv.UnstructuredGrid)
    for key in e_reader.point_array_names:
        assert key in unstruct.point_data.keys()

    for key in e_reader.cell_array_names:
        assert key in unstruct.cell_data.keys()


def _test_block_names(block, names):
    assert block.number == len(names)
    assert block.names == names

    block.enable_all()

    for name in names:
        assert block.status(name)

    block.disable_all()

    for name in names:
        assert not block.status(name)

    for name in names:
        block.enable(name)
        assert block.status(name)

    for name in names:
        block.disable(name)
        assert not block.status(name)


def _test_block_arrays(block, array_names):
    assert block.number_arrays == len(array_names)
    assert block.array_names == array_names
    block.enable_all_arrays()

    for array_name in array_names:
        assert block.array_status(array_name)

    block.disable_all_arrays()
    for array_name in array_names:
        assert not block.array_status(array_name)

    for array_name in array_names:
        block.enable_array(array_name)
        assert block.array_status(array_name)

    for array_name in array_names:
        block.disable_array(array_name)
        assert not block.array_status(array_name)


def test_exodus_blocks():
    fname_e = examples.download_mug(load=False)
    e_reader = pv.get_reader(fname_e)

    # test instantiation with invalid object type
    match = re.escape('object_type is invalid')
    with pytest.raises(ValueError, match=match):
        # 15 is not associated with ObjectType enum in the
        # vtkExodusIIReader
        pv.ExodusIIBlockSet(e_reader, 15)

    # tests all core routines for each block and set that contains
    # blocks and sets in the examples
    _test_block_names(e_reader.element_blocks, ['Unnamed block ID: 1', 'Unnamed block ID: 76'])

    _test_block_names(e_reader.side_sets, ['bottom', 'top'])

    _test_block_names(e_reader.node_sets, ['Unnamed set ID: 1', 'Unnamed set ID: 2'])

    _test_block_arrays(e_reader.element_blocks, ['aux_elem'])

    # Test example with set arrays
    fname_e = examples.download_biplane(load=False)
    e_reader = pv.get_reader(fname_e)
    _test_block_arrays(e_reader.side_sets, ['PressureRMS'])

    # check construct_result_array for those that do not have blocks in the example
    number_method = e_reader.face_blocks._construct_result_method('GetNumberOf', 's')
    assert number_method == e_reader._reader.GetNumberOfFaceResultArrays

    number_method = e_reader.edge_blocks._construct_result_method('GetNumberOf', 's')
    assert number_method == e_reader._reader.GetNumberOfEdgeResultArrays

    number_method = e_reader.element_sets._construct_result_method('GetNumberOf', 's')
    assert number_method == e_reader._reader.GetNumberOfElementSetResultArrays

    number_method = e_reader.face_sets._construct_result_method('GetNumberOf', 's')
    assert number_method == e_reader._reader.GetNumberOfFaceSetResultArrays
