import os
from subprocess import PIPE, Popen

import numpy as np
import pytest
import vtk

import vista
from vista import examples as ex
from vista.plotting import system_supports_plotting


def test_multi_block_init_vtk():
    multi = vtk.vtkMultiBlockDataSet()
    multi.SetBlock(0, vtk.vtkRectilinearGrid())
    multi.SetBlock(1, vtk.vtkTable())
    multi = vista.MultiBlock(multi)
    assert isinstance(multi, vista.MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi[0], vista.RectilinearGrid)
    assert isinstance(multi[1], vtk.vtkTable)
    multi = vtk.vtkMultiBlockDataSet()
    multi.SetBlock(0, vtk.vtkRectilinearGrid())
    multi.SetBlock(1, vtk.vtkTable())
    multi = vista.MultiBlock(multi, deep=True)
    assert isinstance(multi, vista.MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi[0], vista.RectilinearGrid)
    assert isinstance(multi[1], vtk.vtkTable)

def test_multi_block_init_dict():
    data = dict()
    data['grid'] = ex.load_rectilinear()
    data['poly'] = ex.load_airplane()
    multi = vista.MultiBlock(data)
    assert isinstance(multi, vista.MultiBlock)
    assert multi.n_blocks == 2
    # Note that disctionaries do not maintain order
    assert isinstance(multi[0], (vista.RectilinearGrid, vista.PolyData))
    assert multi.get_block_name(0) in ['grid','poly']
    assert isinstance(multi[1], (vista.RectilinearGrid, vista.PolyData))
    assert multi.get_block_name(1) in ['grid','poly']

def test_multi_block_init_list():
    data = [ex.load_rectilinear(), ex.load_airplane()]
    multi = vista.MultiBlock(data)
    assert isinstance(multi, vista.MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi[0], vista.RectilinearGrid)
    assert isinstance(multi[1], vista.PolyData)


def test_multi_block_append():
    """This puts all of the example data objects into a a MultiBlock container"""
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_rectilinear())
    # Now check everything
    assert multi.n_blocks == 5
    assert multi.bounds is not None
    assert isinstance(multi[0], vista.PolyData)
    assert isinstance(multi[1], vista.PolyData)
    assert isinstance(multi[2], vista.UniformGrid)
    assert isinstance(multi[3], vista.PolyData)
    assert isinstance(multi[4], vista.RectilinearGrid)
    # Now overwrite a block
    multi[4] = vista.Sphere()
    assert isinstance(multi[4], vista.PolyData)
    multi[4] = vtk.vtkUnstructuredGrid()
    assert isinstance(multi[4], vista.UnstructuredGrid)


def test_multi_block_set_get_ers():
    """This puts all of the example data objects into a a MultiBlock container"""
    multi = vista.MultiBlock()
    # Set the number of blocks
    multi.n_blocks = 6
    assert multi.GetNumberOfBlocks() == 6 # Check that VTK side registered it
    assert multi.n_blocks == 6 # Check vista side registered it
    # Add data to the MultiBlock
    data = ex.load_rectilinear()
    multi[1, 'rect'] = data
    # Make sure number of blocks is constant
    assert multi.n_blocks == 6
    # Check content
    assert isinstance(multi[1], vista.RectilinearGrid)
    for i in [0,2,3,4,5]:
        assert multi[i] == None
    # Check the bounds
    assert multi.bounds == list(data.bounds)
    multi[5] = ex.load_uniform()
    multi.set_block_name(5, 'uni')
    multi.set_block_name(5, None) # Make sure it doesn't get overwritten
    assert isinstance(multi.get(5), vista.UniformGrid)
    # Test get by name
    assert isinstance(multi['uni'], vista.UniformGrid)
    assert isinstance(multi['rect'], vista.RectilinearGrid)
    # Test the del operator
    del multi[0]
    assert multi.n_blocks == 5
    # Make sure the rect grid was moved up
    assert isinstance(multi[0], vista.RectilinearGrid)
    assert multi.get_block_name(0) == 'rect'
    assert multi.get_block_name(2) == None
    # test del by name
    del multi['uni']
    assert multi.n_blocks == 4
    # test the pop operator
    pop = multi.pop(0)
    assert isinstance(pop, vista.RectilinearGrid)
    assert multi.n_blocks == 3
    assert multi.get_block_name(10) is None
    with pytest.raises(KeyError):
        idx = multi.get_index_by_name('foo')


# def test_mutli_block_clean():
#     # now test a clean of the null values
#     multi = vista.MultiBlock()
#     multi[1, 'rect'] = ex.load_rectilinear()
#     multi[5, 'uni'] = ex.load_uniform()
#     # perfromt he clean to remove all Null elements
#     multi.clean()
#     assert multi.n_blocks == 2
#     assert multi.GetNumberOfBlocks() == 2
#     assert isinstance(multi[0], vista.RectilinearGrid)
#     assert isinstance(multi[1], vista.UniformGrid)
#     assert multi.get_block_name(0) == 'rect'
#     assert multi.get_block_name(1) == 'uni'



def test_multi_block_repr():
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(None)
    # Now check everything
    assert multi.n_blocks == 5
    assert multi._repr_html_() is not None


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vtm', 'vtmb'])
def test_multi_block_io(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_globe())
    # Now check everything
    assert multi.n_blocks == 5
    # Save it out
    multi.save(filename, binary)
    foo = vista.MultiBlock(filename)
    assert foo.n_blocks == multi.n_blocks
    foo = vista.read(filename)
    assert foo.n_blocks == multi.n_blocks


def test_multi_io_erros(tmpdir):
    fdir = tmpdir.mkdir("tmpdir")
    multi = vista.MultiBlock()
    # Check saving with bad extension
    bad_ext_name = str(fdir.join('tmp.%s' % 'npy'))
    with pytest.raises(Exception):
        multi.save(bad_ext_name)
    arr = np.random.rand(10, 10)
    np.save(bad_ext_name, arr)
    # Load non existing file
    with pytest.raises(Exception):
        data = vista.MultiBlock('foo.vtm')
    # Load bad extension
    with pytest.raises(IOError):
        data = vista.MultiBlock(bad_ext_name)



def test_extract_geometry():
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_globe())
    # Now check everything
    assert multi.n_blocks == 5
    # Now apply the geometry filter to combine a plethora of data blocks
    geom = multi.extract_geometry()
    assert isinstance(geom, vista.PolyData)


def test_combine_filter():
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_globe())
    # Now check everything
    assert multi.n_blocks == 5
    # Now apply the geometry filter to combine a plethora of data blocks
    geom = multi.combine()
    assert isinstance(geom, vista.UnstructuredGrid)



def test_multi_block_copy():
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_globe())
    # Now check everything
    newobj = multi.copy()
    assert multi.n_blocks == 5 == newobj.n_blocks
    assert id(multi[0]) != id(newobj[0])
    assert id(multi[-1]) != id(newobj[-1])
    return


def test_multi_block_negative_index():
    multi = vista.MultiBlock()
    # Add examples
    multi.append(ex.load_ant())
    multi.append(ex.load_sphere())
    multi.append(ex.load_uniform())
    multi.append(ex.load_airplane())
    multi.append(ex.load_globe())
    # Now check everything
    assert id(multi[-1]) == id(multi[4])
    assert id(multi[-2]) == id(multi[3])
    assert id(multi[-3]) == id(multi[2])
    assert id(multi[-4]) == id(multi[1])
    assert id(multi[-5]) == id(multi[0])
    with pytest.raises(IndexError):
        foo = multi[-6]
    return
