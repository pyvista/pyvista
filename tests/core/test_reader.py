from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities.fileio import _try_imageio_imread
from pyvista.examples.downloads import download_file

HAS_IMAGEIO = True
try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False


skip_windows = pytest.mark.skipif(os.name == 'nt', reason='Test fails on Windows')


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
    assert isinstance(new_mesh, pv.RectilinearGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlunstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join('temp.vtu')
    mesh = pv.UnstructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pv.get_reader(tmpfile.strpath)
    assert reader.path == tmpfile.strpath
    new_mesh = reader.read()
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
    for m_1, m_3 in zip(mesh_1, mesh_3):
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


def test_dcmreader(tmpdir):
    # Test reading directory (image stack)
    directory = examples.download_dicom_stack(load=False)
    reader = pv.get_reader(directory)
    assert directory in str(reader)
    assert isinstance(reader, pv.DICOMReader)
    assert reader.path == directory

    mesh = reader.read()
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
    assert all([mesh.n_points, mesh.n_cells])


def test_objreader():
    filename = examples.download_doorman(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.OBJReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_stlreader():
    filename = examples.download_gears(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.STLReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_tecplotreader():
    filename = examples.download_tecplot_ascii(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.TecplotReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh[0].n_points, mesh[0].n_cells])


def test_vtkreader():
    filename = examples.hexbeamfile
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.VTKDataSetReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_byureader():
    filename = examples.download_teapot(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.BYUReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_facetreader():
    filename = examples.download_clown(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.FacetReader)
    assert reader.path == filename

    mesh = reader.read()
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
    assert isinstance(mesh, pv.MultiBlock)
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


@skip_windows
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
    # vtk < 9.1.0 does not support
    if pv.vtk_version_info < (9, 1, 0):
        pytest.xfail('OpenFOAMReader GetTimeValue missing on vtk<9.1.0')

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


@pytest.mark.skipif(
    pv.vtk_version_info > (9, 3),
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
    # vtk version 9.1.0 changed the way patch names are handled.
    vtk_version = pv.vtk_version_info
    if vtk_version >= (9, 1, 0):
        patch_array_key = 'boundary'
        reader_patch_prefix = 'patch/'
    else:
        patch_array_key = 'Patches'
        reader_patch_prefix = ''

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
    with pytest.raises(ValueError, match="Unknown case type 'wrong_value'."):
        reader.case_type = 'wrong_value'


@pytest.mark.needs_vtk_version(9, 1)
def test_read_cgns():
    filename = examples.download_cgns_structured(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.CGNSReader)
    assert 'CGNS' in str(reader)
    reader.show_progress()
    assert reader._progress_bar is True

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


def test_bmpreader():
    filename = examples.download_masonry_texture(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.BMPReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_demreader():
    filename = examples.download_st_helens(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.DEMReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_jpegreader():
    filename = examples.planets.download_mars_surface(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.JPEGReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_meta_image_reader():
    filename = examples.download_chest(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.MetaImageReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_nifti_reader():
    filename = examples.download_brain_atlas_with_sides(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.NIFTIReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_nrrd_reader():
    filename = examples.download_beach(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.NRRDReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_png_reader():
    filename = examples.download_vtk_logo(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PNGReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_pnm_reader():
    filename = examples.download_gourds_pnm(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PNMReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_slc_reader():
    filename = examples.download_knee_full(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.SLCReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_tiff_reader():
    filename = examples.download_crater_imagery(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.TIFFReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_hdr_reader():
    filename = examples.download_parched_canal_4k(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.HDRReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_avsucd_reader():
    filename = examples.download_cells_nd(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.AVSucdReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


@pytest.mark.needs_vtk_version(9, 1)
def test_hdf_reader():
    filename = examples.download_can_crushed_hdf(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.HDFReader)
    assert reader.path == filename

    mesh = reader.read()
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


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 1, 0),
    reason='Requires VTK>=9.1.0 for a concrete PartitionedDataSetWriter class.',
)
def test_xmlpartitioneddatasetreader(tmpdir):
    tmpfile = tmpdir.join('temp.vtpd')
    partitions = pv.PartitionedDataSet(
        [pv.Wavelet(extent=(0, 10, 0, 10, 0, 5)), pv.Wavelet(extent=(0, 10, 0, 10, 5, 10))],
    )
    partitions.save(tmpfile.strpath)
    new_partitions = pv.read(tmpfile.strpath)
    assert isinstance(new_partitions, pv.PartitionedDataSet)
    assert len(new_partitions) == len(partitions)
    for i, new_partition in enumerate(new_partitions):
        assert isinstance(new_partition, pv.ImageData)
        assert new_partitions[i].n_cells == partitions[i].n_cells


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 3, 0),
    reason='Requires VTK>=9.3.0 for a concrete FLUENTCFFReader class.',
)
def test_fluentcffreader():
    filename = examples.download_room_cff(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.FLUENTCFFReader)
    assert reader.path == filename

    blocks = reader.read()
    assert blocks.n_blocks == 1
    assert isinstance(blocks[0], pv.UnstructuredGrid)
    assert blocks.bounds == (0.0, 4.0, 0.0, 4.0, 0.0, 0.0)


def test_gambitreader():
    filename = examples.download_prism(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.GambitReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 1, 0),
    reason='Requires VTK>=9.1.0 for a concrete GaussianCubeReader class.',
)
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
    assert all([grid.n_points, grid.n_cells])

    poly = reader.read(grid=False)
    assert isinstance(poly, pv.PolyData)
    assert all([poly.n_points, poly.n_cells])


def test_gesignareader():
    filename = examples.download_e07733s002i009(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.GESignaReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 1, 0),
    reason='Requires VTK>=9.1.0 for a concrete GaussianCubeReader class.',
)
def test_pdbreader():
    filename = examples.download_caffeine(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.PDBReader)
    assert reader.path == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_particle_reader():
    filename = examples.download_particles(load=False)
    reader = pv.get_reader(filename)
    assert isinstance(reader, pv.ParticleReader)
    assert reader.path == filename

    reader.endian = 'BigEndian'
    mesh = reader.read()
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
