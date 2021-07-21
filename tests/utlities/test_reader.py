import os

import numpy as np

import pyvista
from pyvista.examples.downloads import _download_file


def test_xmlimagedatareader(tmpdir):
    tmpfile = tmpdir.join("temp.vti")
    mesh = pyvista.UniformGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.Reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.UniformGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlrectilineargridreader(tmpdir):
    tmpfile = tmpdir.join("temp.vtr")
    mesh = pyvista.RectilinearGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.Reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.RectilinearGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlunstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join("temp.vtu")
    mesh = pyvista.UnstructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.Reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.UnstructuredGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlpolydatareader(tmpdir):
    tmpfile = tmpdir.join("temp.vtp")
    mesh = pyvista.Sphere()
    mesh.save(tmpfile.strpath)
      
    reader = pyvista.Reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.PolyData)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlstructuredgridreader(tmpdir):
    tmpfile = tmpdir.join("temp.vts")
    mesh = pyvista.StructuredGrid()
    mesh.save(tmpfile.strpath)

    reader = pyvista.Reader(tmpfile.strpath)
    assert reader.filename == tmpfile.strpath
    new_mesh = reader.read()
    assert isinstance(new_mesh, pyvista.StructuredGrid)
    assert new_mesh.n_points == mesh.n_points
    assert new_mesh.n_cells == mesh.n_cells


def test_xmlmultiblockreader(tmpdir):
    tmpfile = tmpdir.join("temp.vtm")
    mesh = pyvista.MultiBlock([pyvista.Sphere() for i in range(5)])
    mesh.save(tmpfile.strpath)

    reader = pyvista.Reader(tmpfile.strpath)
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

    reader = pyvista.Reader(tmpfile.strpath)

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


def test_ensightreader():
    folder, _ = _download_file('EnSight.zip')
    filename = os.path.join(folder, "foam_case_0_0_0_0.case")

    reader = pyvista.Reader(filename)
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
