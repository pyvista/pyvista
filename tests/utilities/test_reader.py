import os

import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.examples.downloads import _download_file


def test_get_reader_fail():
    with pytest.raises(ValueError):
        reader = pyvista.get_reader("not_a_supported_file.no_data")

def test_xmlimagedatareader(tmpdir):
    tmpfile = tmpdir.join("temp.vti")
    mesh = pyvista.UniformGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.get_reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.UniformGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlrectilineargridreader(tmpdir):
    tmpfile = tmpdir.join("temp.vtr")
    mesh = pyvista.RectilinearGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.get_reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.RectilinearGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlunstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join("temp.vtu")
    mesh = pyvista.UnstructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.get_reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.UnstructuredGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlpolydatareader(tmpdir):
    tmpfile = tmpdir.join("temp.vtp")
    mesh = pyvista.Sphere()
    mesh.save(tmpfile.strpath)
      
    reader = pyvista.get_reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.PolyData)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join("temp.vts")
    mesh = pyvista.StructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.get_reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.StructuredGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlmultiblockreader(tmpdir):
    tmpfile = tmpdir.join("temp.vtm")
    mesh = pyvista.MultiBlock([pyvista.Sphere() for i in range(5)])
    mesh.save(tmpfile.strpath)

    reader = pyvista.get_reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.MultiBlock)
    assert new_mesh.n_blocks == mesh.n_blocks
    for i in range(new_mesh.n_blocks):
        assert new_mesh[i].n_points == mesh[i].n_points
        assert new_mesh[i].n_cells == mesh[i].n_cells


def test_reader_cell_point_data(tmpdir):
    tmpfile = tmpdir.join("temp.vtp")
    mesh = pyvista.Sphere()
    mesh['height'] = mesh.points[:, 1]
    mesh['id'] = np.arange(mesh.n_cells)
    mesh.save(tmpfile.strpath)
    # mesh has an additional 'Normals' point data array

    reader = pyvista.get_reader(tmpfile.strpath)

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

    reader = pyvista.get_reader(filename)
    assert reader.filename == filename
    assert reader.number_cell_arrays == 9
    assert reader.number_point_arrays == 0
    
    assert reader.cell_array_names == ['v2', 'nut', 'k', 'nuTilda', 'p', 
                                       'omega', 'f', 'epsilon', 'U']
    assert reader.point_array_names == []

    reader.disable_all_cell_arrays()
    reader.enable_cell_array('k')

    assert reader.all_cell_arrays_status == {
        'v2': False, 'nut': False, 'k': True, 'nuTilda': False, 'p': False, 
        'omega':False, 'f':False, 'epsilon':False, 'U':False
    }


    mesh = reader.read()
    assert isinstance(mesh, pyvista.MultiBlock)

    for i in range(mesh.n_blocks):
        assert all([mesh[i].n_points, mesh[i].n_cells])
        assert mesh[i].array_names == ['k']

    # re-enable all cell arrays and read again
    reader.enable_all_cell_arrays()
    all_mesh = reader.read()
    assert isinstance(all_mesh, pyvista.MultiBlock)

    for i in range(all_mesh.n_blocks):
        assert all([all_mesh[i].n_points, all_mesh[i].n_cells])
        assert all_mesh[i].array_names == ['v2', 'nut', 'k', 'nuTilda', 'p', 
                                           'omega', 'f', 'epsilon', 'U']


def test_ensightreader_timepoints():
    filename = examples.download_naca(load=False)

    reader = pyvista.get_reader(filename)
    assert reader.filename == filename

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


def test_plyreader():
    filename = examples.spherefile
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.PLYReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_objreader():
    filename = examples.download_doorman(load=False)
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.OBJReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_stlreader():
    filename = examples.download_gears(load=False)
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.STLReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_vtkreader():
    filename = examples.hexbeamfile
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.VTKDataSetReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_byureader():
    filename = examples.download_teapot(load=False)
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.BYUReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_facetreader():
    filename = examples.download_clown(load=False)
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.FacetReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_plot3dmetareader():
    filename, _ = _download_file('multi.p3d')
    _download_file('multi-bin.xyz')
    _download_file('multi-bin.q')
    _download_file('multi-bin.f')
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.Plot3DMetaReader)
    assert reader.filename == filename

    mesh = reader.read()
    for m in mesh:
        assert all([m.n_points, m.n_cells])


def test_binarymarchingcubesreader():
    filename = examples.download_pine_roots(load=False)
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.BinaryMarchingCubesReader)
    assert reader.filename == filename

    mesh = reader.read()
    assert all([mesh.n_points, mesh.n_cells])


def test_pvdreader():
    filename = examples.download_wavy(load=False)
    reader = pyvista.get_reader(filename)
    assert isinstance(reader, pyvista.PVDReader)
    assert reader.reader == reader  # PVDReader refers to itself
    assert reader.filename == filename

    assert reader.number_time_points == 15
    assert reader.time_point_value(1) == 1.0
    assert np.array_equal(reader.time_values, np.arange(0, 15, dtype=np.float))

    assert reader.active_time_value == reader.time_values[0]

    active_datasets = reader.active_datasets
    assert len(active_datasets) == 1
    active_dataset0 = active_datasets[0]
    assert active_dataset0.time == 0.0
    assert active_dataset0.filename == "wavy/wavy00.vts"
    assert active_dataset0.group == ""
    assert active_dataset0.part == 0

    assert len(reader.datasets) == len(reader.time_values)

    active_readers = reader.active_readers
    assert len(active_readers) == 1
    active_reader = active_readers[0]
    assert isinstance(active_reader, pyvista.XMLStructuredGridReader)

    reader.set_active_time_value(1.0)
    assert reader.active_time_value == 1.0

    reader.set_active_time_point(2)
    assert reader.active_time_value == 2.0

    mesh = reader.read()
    assert isinstance(mesh, pyvista.MultiBlock)
    assert len(mesh) == 1
    assert isinstance(mesh[0], pyvista.StructuredGrid)


def test_pvdreader_no_time_group():
    examples.download_dual_sphere_animation(load=False)  # download all the files
    # Use a pvd file that has no timestep or group and two parts.
    filename, _ = _download_file('PVD/paraview/dualSphereNoTime.pvd')
    reader = pyvista.PVDReader(filename)
    assert reader.time_values == [0.0]
    assert reader.active_time_value == 0.0

    assert len(reader.active_datasets) == 2
    for i, dataset in enumerate(reader.active_datasets):
        assert dataset.time == 0.0
        assert dataset.group == None
        assert dataset.part == i
