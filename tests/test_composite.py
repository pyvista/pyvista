import pathlib

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import PolyData, RectilinearGrid, UniformGrid, StructuredGrid, MultiBlock
from pyvista import examples as ex


@pytest.fixture()
def vtk_multi():
    return vtk.vtkMultiBlockDataSet()


@pytest.fixture()
def pyvista_multi():
    return pyvista.MultiBlock


def multi_from_datasets(*datasets):
    """Return pyvista multiblock composed of any number of datasets."""
    return MultiBlock([*datasets])


def test_multi_block_init_vtk():
    multi = vtk.vtkMultiBlockDataSet()
    multi.SetBlock(0, vtk.vtkRectilinearGrid())
    multi.SetBlock(1, vtk.vtkStructuredGrid())
    multi = MultiBlock(multi)
    assert isinstance(multi, MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi.GetBlock(0), RectilinearGrid)
    assert isinstance(multi.GetBlock(1), StructuredGrid)
    multi = vtk.vtkMultiBlockDataSet()
    multi.SetBlock(0, vtk.vtkRectilinearGrid())
    multi.SetBlock(1, vtk.vtkStructuredGrid())
    multi = MultiBlock(multi, deep=True)
    assert isinstance(multi, MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi.GetBlock(0), RectilinearGrid)
    assert isinstance(multi.GetBlock(1), StructuredGrid)
    # Test nested structure
    multi = vtk.vtkMultiBlockDataSet()
    multi.SetBlock(0, vtk.vtkRectilinearGrid())
    multi.SetBlock(1, vtk.vtkImageData())
    nested = vtk.vtkMultiBlockDataSet()
    nested.SetBlock(0, vtk.vtkUnstructuredGrid())
    nested.SetBlock(1, vtk.vtkStructuredGrid())
    multi.SetBlock(2, nested)
    # Wrap the nested structure
    multi = MultiBlock(multi)
    assert isinstance(multi, MultiBlock)
    assert multi.n_blocks == 3
    assert isinstance(multi.GetBlock(0), RectilinearGrid)
    assert isinstance(multi.GetBlock(1), UniformGrid)
    assert isinstance(multi.GetBlock(2), MultiBlock)


def test_multi_block_init_dict(rectilinear, airplane):
    data = {'grid': rectilinear, 'poly': airplane}
    multi = MultiBlock(data)
    assert isinstance(multi, MultiBlock)
    assert multi.n_blocks == 2
    # Note that dictionaries do not maintain order
    assert isinstance(multi.GetBlock(0), (RectilinearGrid, PolyData))
    assert multi.get_block_name(0) in ['grid','poly']
    assert isinstance(multi.GetBlock(1), (RectilinearGrid, PolyData))
    assert multi.get_block_name(1) in ['grid','poly']


def test_multi_block_keys(rectilinear, airplane):
    data = {'grid': rectilinear, 'poly': airplane}
    multi = MultiBlock(data)
    assert len(multi.keys()) == 2
    assert 'grid' in multi.keys()
    assert 'poly' in multi.keys()


def test_multi_block_init_list(rectilinear, airplane):
    data = [rectilinear, airplane]
    multi = MultiBlock(data)
    assert isinstance(multi, MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi.GetBlock(0), RectilinearGrid)
    assert isinstance(multi.GetBlock(1), PolyData)


def test_multi_block_append(ant, sphere, uniform, airplane, rectilinear):
    """This puts all of the example data objects into a a MultiBlock container"""
    multi = MultiBlock()
    # Add and test examples
    datasets = (ant, sphere, uniform, airplane, rectilinear)
    for i, dataset in enumerate(datasets):
        multi.append(dataset)
        assert multi.n_blocks == i + 1
        assert isinstance(multi[i], type(dataset))
    assert multi.bounds is not None
    # Now overwrite a block
    multi[4] = pyvista.Sphere()
    assert isinstance(multi[4], PolyData)
    multi[4] = vtk.vtkUnstructuredGrid()
    assert isinstance(multi[4], pyvista.UnstructuredGrid)


def test_multi_block_set_get_ers():
    """This puts all of the example data objects into a a MultiBlock container"""
    multi = MultiBlock()
    # Set the number of blocks
    multi.n_blocks = 6
    assert multi.GetNumberOfBlocks() == 6 # Check that VTK side registered it
    assert multi.n_blocks == 6 # Check pyvista side registered it
    # Add data to the MultiBlock
    data = ex.load_rectilinear()
    multi[1, 'rect'] = data
    # Make sure number of blocks is constant
    assert multi.n_blocks == 6
    # Check content
    assert isinstance(multi[1], RectilinearGrid)
    for i in [0,2,3,4,5]:
        assert multi[i] is None
    # Check the bounds
    assert multi.bounds == list(data.bounds)
    multi[5] = ex.load_uniform()
    multi.set_block_name(5, 'uni')
    multi.set_block_name(5, None) # Make sure it doesn't get overwritten
    assert isinstance(multi.get(5), UniformGrid)
    # Test get by name
    assert isinstance(multi['uni'], UniformGrid)
    assert isinstance(multi['rect'], RectilinearGrid)
    # Test the del operator
    del multi[0]
    assert multi.n_blocks == 5
    # Make sure the rect grid was moved up
    assert isinstance(multi[0], RectilinearGrid)
    assert multi.get_block_name(0) == 'rect'
    assert multi.get_block_name(2) == None
    # test del by name
    del multi['uni']
    assert multi.n_blocks == 4
    # test the pop operator
    pop = multi.pop(0)
    assert isinstance(pop, RectilinearGrid)
    assert multi.n_blocks == 3
    assert multi.get_block_name(10) is None
    with pytest.raises(KeyError):
        _ = multi.get_index_by_name('foo')
    # allow Sequence but not Iterable in setitem
    with pytest.raises(TypeError):
        multi[{1, 'foo'}] = data


def test_multi_block_clean(rectilinear, uniform, ant):
    # now test a clean of the null values
    multi = MultiBlock()
    multi[1, 'rect'] = rectilinear
    multi[2, 'empty'] = PolyData()
    multi[3, 'mempty'] = MultiBlock()
    multi[5, 'uni'] = uniform
    # perform the clean to remove all Null elements
    multi.clean()
    assert multi.n_blocks == 2
    assert multi.GetNumberOfBlocks() == 2
    assert isinstance(multi[0], RectilinearGrid)
    assert isinstance(multi[1], UniformGrid)
    assert multi.get_block_name(0) == 'rect'
    assert multi.get_block_name(1) == 'uni'
    # Test a nested data struct
    foo = MultiBlock()
    foo[3] = ant
    assert foo.n_blocks == 4
    multi = MultiBlock()
    multi[1, 'rect'] = rectilinear
    multi[5, 'multi'] = foo
    # perform the clean to remove all Null elements
    assert multi.n_blocks == 6
    multi.clean()
    assert multi.n_blocks == 2
    assert multi.GetNumberOfBlocks() == 2
    assert isinstance(multi[0], RectilinearGrid)
    assert isinstance(multi[1], MultiBlock)
    assert multi.get_block_name(0) == 'rect'
    assert multi.get_block_name(1) == 'multi'
    assert foo.n_blocks == 1


def test_multi_block_repr(ant, sphere, uniform, airplane):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, None)
    # Now check everything
    assert multi.n_blocks == 5
    assert multi._repr_html_() is not None
    assert repr(multi) is not None
    assert str(multi) is not None


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pyvista.core.composite.MultiBlock._WRITERS)
@pytest.mark.parametrize('use_pathlib', [True, False])
def test_multi_block_io(extension, binary, tmpdir, use_pathlib, ant,
                        sphere, uniform, airplane, globe):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    if use_pathlib:
        pathlib.Path(filename)
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    # Now check everything
    assert multi.n_blocks == 5
    # Save it out
    multi.save(filename, binary)
    foo = MultiBlock(filename)
    assert foo.n_blocks == multi.n_blocks
    foo = pyvista.read(filename)
    assert foo.n_blocks == multi.n_blocks


def test_invalid_arg():
    with pytest.raises(TypeError):
        pyvista.MultiBlock(np.empty(10))
    with pytest.raises(ValueError):
        pyvista.MultiBlock(np.empty(10), np.empty(10))


def test_multi_io_erros(tmpdir):
    fdir = tmpdir.mkdir("tmpdir")
    multi = MultiBlock()
    # Check saving with bad extension
    bad_ext_name = str(fdir.join('tmp.%s' % 'npy'))
    with pytest.raises(ValueError):
        multi.save(bad_ext_name)
    arr = np.random.rand(10, 10)
    np.save(bad_ext_name, arr)
    # Load non existing file
    with pytest.raises(FileNotFoundError):
        _ = MultiBlock('foo.vtm')
    # Load bad extension
    with pytest.raises(ValueError):
        _ = MultiBlock(bad_ext_name)


def test_extract_geometry(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform)
    nested = multi_from_datasets(airplane, globe)
    multi.append(nested)
    # Now check everything
    assert multi.n_blocks == 4
    # Now apply the geometry filter to combine a plethora of data blocks
    geom = multi.extract_geometry()
    assert isinstance(geom, PolyData)


def test_combine_filter(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform)
    nested = multi_from_datasets(airplane, globe)
    multi.append(nested)
    # Now check everything
    assert multi.n_blocks == 4
    # Now apply the append filter to combine a plethora of data blocks
    geom = multi.combine()
    assert isinstance(geom, pyvista.UnstructuredGrid)


def test_multi_block_copy(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    # Now check everything
    multi_copy = multi.copy()
    assert multi.n_blocks == 5 == multi_copy.n_blocks
    assert id(multi[0]) != id(multi_copy[0])
    assert id(multi[-1]) != id(multi_copy[-1])
    for i in range(multi_copy.n_blocks):
        assert pyvista.is_pyvista_dataset(multi_copy.GetBlock(i))
    # Now check shallow
    multi_copy = multi.copy(deep=False)
    assert multi.n_blocks == 5 == multi_copy.n_blocks
    assert id(multi[0]) == id(multi_copy[0])
    assert id(multi[-1]) == id(multi_copy[-1])
    for i in range(multi_copy.n_blocks):
        assert pyvista.is_pyvista_dataset(multi_copy.GetBlock(i))


def test_multi_block_negative_index(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    # Now check everything
    assert id(multi[-1]) == id(multi[4])
    assert id(multi[-2]) == id(multi[3])
    assert id(multi[-3]) == id(multi[2])
    assert id(multi[-4]) == id(multi[1])
    assert id(multi[-5]) == id(multi[0])
    with pytest.raises(IndexError):
        _ = multi[-6]


def test_multi_slice_index(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    # Now check everything
    sub = multi[0:3]
    assert len(sub) == 3
    for i in range(3):
        assert id(sub[i]) == id(multi[i])
        assert sub.get_block_name(i) == multi.get_block_name(i)
    sub = multi[0:-1]
    assert len(sub) == len(multi) == multi.n_blocks
    for i in range(multi.n_blocks):
        assert id(sub[i]) == id(multi[i])
        assert sub.get_block_name(i) == multi.get_block_name(i)
    sub = multi[0:-1:2]
    assert len(sub) == 3
    for i in range(3):
        j = i*2
        assert id(sub[i]) == id(multi[j])
        assert sub.get_block_name(i) == multi.get_block_name(j)


def test_multi_block_list_index(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    # Now check everything
    indices = [0, 3, 4]
    sub = multi[indices]
    assert len(sub) == len(indices)
    for i, j in enumerate(indices):
        assert id(sub[i]) == id(multi[j])
        assert sub.get_block_name(i) == multi.get_block_name(j)
    # check list of key names
    multi = MultiBlock()
    multi["foo"] = pyvista.Sphere()
    multi["goo"] = pyvista.Box()
    multi["soo"] = pyvista.Cone()
    indices = ["goo", "foo"]
    sub = multi[indices]
    assert len(sub) == len(indices)
    assert isinstance(sub["foo"], PolyData)


def test_multi_block_volume(ant, airplane, sphere, uniform):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, None)
    assert multi.volume


def test_multi_block_length(ant, sphere, uniform, airplane):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, None)
    assert multi.length


def test_multi_block_save_lines(tmpdir):
    radius = 1
    xr = np.random.random(10)
    yr = np.random.random(10)
    x = radius * np.sin(yr) * np.cos(xr)
    y = radius * np.sin(yr) * np.sin(xr)
    z = radius * np.cos(yr)
    xyz = np.stack((x, y, z), axis=1)

    poly = pyvista.lines_from_points(xyz, close=False)
    blocks = pyvista.MultiBlock()
    for _ in range(2):
        blocks.append(poly)

    path = tmpdir.mkdir("tmpdir")
    line_filename = str(path.join('lines.vtk'))
    block_filename = str(path.join('blocks.vtmb'))
    poly.save(line_filename)
    blocks.save(block_filename)

    poly_load = pyvista.read(line_filename)
    assert np.allclose(poly_load.points, poly.points)

    blocks_load = pyvista.read(block_filename)
    assert np.allclose(blocks_load[0].points, blocks[0].points)


def test_multi_block_data_range():
    volume = pyvista.Wavelet()
    a = volume.slice_along_axis(5,'x')
    with pytest.raises(ValueError):
        a.get_data_range('foo')
    mi, ma = a.get_data_range(volume.active_scalars_name)
    assert mi is not None
    assert ma is not None
    # Test on a nested MultiBlock
    b = volume.slice_along_axis(5,'y')
    slices = pyvista.MultiBlock([a,b])
    with pytest.raises(ValueError):
        slices.get_data_range('foo')
    mi, ma = slices.get_data_range(volume.active_scalars_name)
    assert mi is not None
    assert ma is not None
