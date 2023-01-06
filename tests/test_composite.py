import pathlib
import platform
import weakref

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import (
    MultiBlock,
    PolyData,
    RectilinearGrid,
    StructuredGrid,
    UniformGrid,
    examples as ex,
)

skip_mac = pytest.mark.skipif(platform.system() == 'Darwin', reason="Flaky Mac tests")


@pytest.fixture()
def vtk_multi():
    return vtk.vtkMultiBlockDataSet()


@pytest.fixture()
def pyvista_multi():
    return pyvista.MultiBlock


def multi_from_datasets(*datasets):
    """Return pyvista multiblock composed of any number of datasets."""
    return MultiBlock(datasets)


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
    assert multi.get_block_name(0) in ['grid', 'poly']
    assert isinstance(multi.GetBlock(1), (RectilinearGrid, PolyData))
    assert multi.get_block_name(1) in ['grid', 'poly']


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

    with pytest.raises(ValueError, match="Cannot nest a composite dataset in itself."):
        multi.append(multi)


def test_multi_block_set_get_ers():
    """This puts all of the example data objects into a a MultiBlock container"""
    multi = MultiBlock()
    # Set the number of blocks
    multi.n_blocks = 6
    assert multi.GetNumberOfBlocks() == 6  # Check that VTK side registered it
    assert multi.n_blocks == 6  # Check pyvista side registered it
    # Add data to the MultiBlock
    data = ex.load_rectilinear()
    multi[1] = data
    multi.set_block_name(1, 'rect')
    # Make sure number of blocks is constant
    assert multi.n_blocks == 6
    # Check content
    assert isinstance(multi[1], RectilinearGrid)
    for i in [0, 2, 3, 4, 5]:
        assert multi[i] is None
    # Check the bounds
    assert multi.bounds == data.bounds
    multi[5] = ex.load_uniform()
    multi.set_block_name(5, 'uni')
    multi.set_block_name(5, None)  # Make sure it doesn't get overwritten
    assert isinstance(multi.get(5), UniformGrid)
    # Test get by name
    assert isinstance(multi['uni'], UniformGrid)
    assert isinstance(multi['rect'], RectilinearGrid)
    assert isinstance(multi.get('uni'), UniformGrid)
    assert multi.get('no key') is None
    assert multi.get('no key', default=pyvista.Sphere()) == pyvista.Sphere()
    # Test the del operator
    del multi[0]
    assert multi.n_blocks == 5
    # Make sure the rect grid was moved up
    assert isinstance(multi[0], RectilinearGrid)
    assert multi.get_block_name(0) == 'rect'
    assert multi.get_block_name(2) is None
    # test del by name
    del multi['uni']
    assert multi.n_blocks == 4
    # test the pop operator
    pop = multi.pop(0)
    assert isinstance(pop, RectilinearGrid)
    assert multi.n_blocks == 3
    assert all([k is None for k in multi.keys()])

    multi["new key"] = pyvista.Sphere()
    assert multi.n_blocks == 4
    assert multi[3] == pyvista.Sphere()

    multi["new key"] = pyvista.Cube()
    assert multi.n_blocks == 4
    assert multi[3] == pyvista.Cube()

    with pytest.raises(KeyError):
        _ = multi.get_index_by_name('foo')

    with pytest.raises(IndexError):
        multi[4] = UniformGrid()

    with pytest.raises(KeyError):
        multi["not a key"]
    with pytest.raises(TypeError):
        data = multi[[0, 1]]

    with pytest.raises(TypeError):
        multi[1, 'foo'] = data


def test_replace():
    spheres = {f"{i}": pyvista.Sphere(phi_resolution=i + 3) for i in range(10)}
    multi = MultiBlock(spheres)
    cube = pyvista.Cube()
    multi.replace(3, cube)
    assert multi.get_block_name(3) == "3"
    assert multi[3] is cube


def test_pop():
    spheres = {f"{i}": pyvista.Sphere(phi_resolution=i + 3) for i in range(10)}
    multi = MultiBlock(spheres)
    assert multi.pop() == spheres["9"]
    assert spheres["9"] not in multi
    assert multi.pop(0) == spheres["0"]
    assert spheres["0"] not in multi


def test_del_slice(sphere):
    multi = MultiBlock({f"{i}": sphere for i in range(10)})
    del multi[0:10:2]
    assert len(multi) == 5
    assert all([f"{i}" in multi.keys() for i in range(1, 10, 2)])

    multi = MultiBlock({f"{i}": sphere for i in range(10)})
    del multi[5:2:-1]
    assert len(multi) == 7
    assert all([f"{i}" in multi.keys() for i in [0, 1, 2, 6, 7, 8, 9]])


def test_slicing_multiple_in_setitem(sphere):
    # equal length
    multi = MultiBlock({f"{i}": sphere for i in range(10)})
    multi[1:3] = [pyvista.Cube(), pyvista.Cube()]
    assert multi[1] == pyvista.Cube()
    assert multi[2] == pyvista.Cube()
    assert multi.count(pyvista.Cube()) == 2
    assert len(multi) == 10

    # len(slice) < len(data)
    multi = MultiBlock({f"{i}": sphere for i in range(10)})
    multi[1:3] = [pyvista.Cube(), pyvista.Cube(), pyvista.Cube()]
    assert multi[1] == pyvista.Cube()
    assert multi[2] == pyvista.Cube()
    assert multi[3] == pyvista.Cube()
    assert multi.count(pyvista.Cube()) == 3
    assert len(multi) == 11

    # len(slice) > len(data)
    multi = MultiBlock({f"{i}": sphere for i in range(10)})
    multi[1:3] = [pyvista.Cube()]
    assert multi[1] == pyvista.Cube()
    assert multi.count(pyvista.Cube()) == 1
    assert len(multi) == 9


def test_reverse(sphere):
    multi = MultiBlock({f"{i}": sphere for i in range(3)})
    multi.append(pyvista.Cube(), "cube")
    multi.reverse()
    assert multi[0] == pyvista.Cube()
    assert np.array_equal(multi.keys(), ["cube", "2", "1", "0"])


def test_insert(sphere):
    multi = MultiBlock({f"{i}": sphere for i in range(3)})
    cube = pyvista.Cube()
    multi.insert(0, cube)
    assert len(multi) == 4
    assert multi[0] is cube

    # test with negative index and name
    multi.insert(-1, pyvista.UniformGrid(), name="uni")
    assert len(multi) == 5
    # inserted before last element
    assert isinstance(multi[-2], pyvista.UniformGrid)  # inserted before last element
    assert multi.get_block_name(-2) == "uni"


def test_extend(sphere, uniform, ant):
    # test with Iterable
    multi = MultiBlock([sphere, ant])
    new_multi = [uniform, uniform]
    multi.extend(new_multi)
    assert len(multi) == 4
    assert multi.count(uniform) == 2

    # test with a MultiBlock
    multi = MultiBlock([sphere, ant])
    new_multi = MultiBlock({"uniform1": uniform, "uniform2": uniform})
    multi.extend(new_multi)
    assert len(multi) == 4
    assert multi.count(uniform) == 2
    assert multi.keys()[-2] == "uniform1"
    assert multi.keys()[-1] == "uniform2"


def test_multi_block_clean(rectilinear, uniform, ant):
    # now test a clean of the null values
    multi = MultiBlock()
    multi.n_blocks = 6
    multi[1] = rectilinear
    multi.set_block_name(1, 'rect')
    multi[2] = PolyData()
    multi.set_block_name(2, 'empty')
    multi[3] = MultiBlock()
    multi.set_block_name(3, 'mempty')
    multi[5] = uniform
    multi.set_block_name(5, 'uni')
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
    foo.n_blocks = 4
    foo[3] = ant
    assert foo.n_blocks == 4
    multi = MultiBlock()
    multi.n_blocks = 6
    multi[1] = rectilinear
    multi.set_block_name(1, 'rect')
    multi[5] = foo
    multi.set_block_name(5, 'multi')
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


def test_multi_block_eq(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    other = multi.copy()

    assert multi is not other
    assert multi == other

    assert pyvista.MultiBlock() == pyvista.MultiBlock()

    other[0] = pyvista.Sphere()
    assert multi != other

    other = multi.copy()
    other.set_block_name(0, "not matching")
    assert multi != other

    other = multi.copy()
    other.append(pyvista.Sphere())
    assert multi != other


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pyvista.core.composite.MultiBlock._WRITERS)
@pytest.mark.parametrize('use_pathlib', [True, False])
def test_multi_block_io(
    extension, binary, tmpdir, use_pathlib, ant, sphere, uniform, airplane, globe
):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.{extension}'))
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


@skip_mac  # fails due to download examples
@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vtm', 'vtmb'])
def test_ensight_multi_block_io(extension, binary, tmpdir, ant, sphere, uniform, airplane, globe):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    # multi = ex.load_bfs()  # .case file
    multi = ex.download_backward_facing_step()  # .case file
    # Now check everything
    assert multi.n_blocks == 4
    array_names = ['v2', 'nut', 'k', 'nuTilda', 'p', 'omega', 'f', 'epsilon', 'U']
    for block in multi:
        assert block.array_names == array_names
    # Save it out
    multi.save(filename, binary)
    foo = MultiBlock(filename)
    assert foo.n_blocks == multi.n_blocks
    for block in foo:
        assert block.array_names == array_names
    foo = pyvista.read(filename)
    assert foo.n_blocks == multi.n_blocks
    for block in foo:
        assert block.array_names == array_names


def test_invalid_arg():
    with pytest.raises(TypeError):
        pyvista.MultiBlock(np.empty(10))
    with pytest.raises(ValueError):
        pyvista.MultiBlock(np.empty(10), np.empty(10))


def test_multi_io_erros(tmpdir):
    fdir = tmpdir.mkdir("tmpdir")
    multi = MultiBlock()
    # Check saving with bad extension
    bad_ext_name = str(fdir.join('tmp.npy'))
    with pytest.raises(ValueError):
        multi.save(bad_ext_name)
    arr = np.random.rand(10, 10)
    np.save(bad_ext_name, arr)
    # Load non existing file
    with pytest.raises(FileNotFoundError):
        _ = MultiBlock('foo.vtm')
    # Load bad extension
    with pytest.raises(IOError):
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

    multi[-1] = ant
    assert multi[4] == ant
    multi[-5] = globe
    assert multi[0] == globe

    with pytest.raises(IndexError):
        multi[-6] = uniform


def test_multi_slice_index(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    # Now check everything
    sub = multi[0:3]
    assert len(sub) == 3
    for i in range(len(sub)):
        assert sub[i] is multi[i]
        assert sub.get_block_name(i) == multi.get_block_name(i)
    sub = multi[0:-1]
    assert len(sub) + 1 == len(multi)
    for i in range(len(sub)):
        assert sub[i] is multi[i]
        assert sub.get_block_name(i) == multi.get_block_name(i)
    sub = multi[0:-1:2]
    assert len(sub) == 2
    for i in range(len(sub)):
        j = i * 2
        assert sub[i] is multi[j]
        assert sub.get_block_name(i) == multi.get_block_name(j)

    sub = [airplane, globe]
    multi[0:2] = sub
    assert multi[0] is airplane
    assert multi[1] is globe


def test_slice_defaults(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    assert multi[:] == multi[0 : len(multi)]


def test_slice_negatives(ant, sphere, uniform, airplane, globe):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, globe)
    test_multi = pyvista.MultiBlock({key: multi[key] for key in multi.keys()[::-1]})
    assert multi[::-1] == test_multi

    test_multi = pyvista.MultiBlock({key: multi[key] for key in multi.keys()[-2:]})
    assert multi[-2:] == test_multi

    test_multi = pyvista.MultiBlock({key: multi[key] for key in multi.keys()[:-1]})
    assert multi[:-1] == test_multi

    test_multi = pyvista.MultiBlock({key: multi[key] for key in multi.keys()[-1:-4:-2]})
    assert multi[-1:-4:-2] == test_multi


def test_multi_block_volume(ant, airplane, sphere, uniform):
    multi = multi_from_datasets(ant, sphere, uniform, airplane, None)
    vols = ant.volume + sphere.volume + uniform.volume + airplane.volume
    assert multi.volume == pytest.approx(vols)


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
    a = volume.slice_along_axis(5, 'x')
    with pytest.raises(KeyError):
        a.get_data_range('foo')
    mi, ma = a.get_data_range(volume.active_scalars_name)
    assert mi is not None
    assert ma is not None
    # Test on a nested MultiBlock
    b = volume.slice_along_axis(5, 'y')
    slices = pyvista.MultiBlock([a, b])
    with pytest.raises(KeyError):
        slices.get_data_range('foo')
    mi, ma = slices.get_data_range(volume.active_scalars_name)
    assert mi is not None
    assert ma is not None


def test_multiblock_ref():
    # can't use fixtures here as we need to remove all references for
    # garbage collection
    sphere = pyvista.Sphere()
    cube = pyvista.Cube()

    block = MultiBlock([sphere, cube])
    block[0]["a_new_var"] = np.zeros(block[0].n_points)
    assert "a_new_var" in block[0].array_names

    assert sphere is block[0]
    assert cube is block[1]

    wref_sphere = weakref.ref(sphere)
    wref_cube = weakref.ref(cube)

    # verify reference remains
    assert wref_sphere() is sphere
    del sphere
    assert wref_sphere() is not None

    # verify __delitem__ works and removes reference
    del block[0]
    assert wref_sphere() is None

    # verify reference remains
    assert wref_cube() is cube

    # verify the __setitem__(index, None) edge case
    del cube
    block[0] = None
    assert wref_cube() is None


def test_set_active_scalars(multiblock_all):
    for block in multiblock_all:
        block.clear_data()
        block.point_data['data'] = range(block.n_points)
        block.point_data['point_data_a'] = range(block.n_points)
        block.point_data['point_data_b'] = range(block.n_points)

        block.cell_data['data'] = range(block.n_cells)
        block.cell_data['cell_data_a'] = range(block.n_cells)
        block.cell_data['cell_data_b'] = range(block.n_cells)

    # test none
    multiblock_all.set_active_scalars(None)
    for block in multiblock_all:
        assert block.point_data.active_scalars_name is None
        assert block.cell_data.active_scalars_name is None

    # test set point_data
    active_scalars_name = 'point_data_a'
    multiblock_all.set_active_scalars(active_scalars_name)
    for block in multiblock_all:
        assert block.point_data.active_scalars_name == active_scalars_name

    # test set point_data
    active_scalars_name = 'cell_data_a'
    multiblock_all.set_active_scalars(active_scalars_name)
    for block in multiblock_all:
        assert block.cell_data.active_scalars_name == active_scalars_name

    # test set point_data
    multiblock_all.set_active_scalars(None)
    active_scalars_name = 'data'
    multiblock_all.set_active_scalars(active_scalars_name, preference='point')
    for block in multiblock_all:
        assert block.point_data.active_scalars_name == active_scalars_name
        assert block.cell_data.active_scalars_name is None

    multiblock_all.set_active_scalars(None)
    active_scalars_name = 'data'
    multiblock_all.set_active_scalars(active_scalars_name, preference='cell')
    for block in multiblock_all:
        assert block.point_data.active_scalars_name is None
        assert block.cell_data.active_scalars_name == active_scalars_name

    # test partial
    multiblock_all[0].clear_data()
    multiblock_all.set_active_scalars(None)
    with pytest.raises(KeyError, match='does not exist'):
        multiblock_all.set_active_scalars('point_data_a')
    multiblock_all.set_active_scalars('point_data_a', allow_missing=True)
    multiblock_all[1].point_data.active_scalars_name == 'point_data_a'

    with pytest.raises(KeyError, match='is missing from all'):
        multiblock_all.set_active_scalars('does not exist', allow_missing=True)


def test_set_active_scalars_multi(multiblock_poly):
    multiblock_poly.set_active_scalars(None)

    block = multiblock_poly[0]
    block.point_data.set_array(range(block.n_points), 'data')
    block.cell_data.set_array(range(block.n_cells), 'data')

    block = multiblock_poly[1]
    block.point_data.set_array(range(block.n_points), 'data')

    multiblock_poly.set_active_scalars('data', preference='point', allow_missing=True)
    for block in multiblock_poly:
        if 'data' in block.point_data:
            assert block.point_data.active_scalars_name == 'data'
        else:
            assert block.point_data.active_scalars_name is None

    multiblock_poly.set_active_scalars('data', preference='cell', allow_missing=True)
    for block in multiblock_poly:
        if 'data' in block.cell_data:
            assert block.cell_data.active_scalars_name == 'data'
        else:
            assert block.cell_data.active_scalars_name is None


def test_set_active_scalars_components(multiblock_poly):
    multiblock_poly[0].point_data['data'] = range(multiblock_poly[0].n_points)
    multiblock_poly[1].point_data['data'] = range(multiblock_poly[1].n_points)
    multiblock_poly[2].point_data['data'] = range(multiblock_poly[2].n_points)

    multiblock_poly.set_active_scalars(None)
    multiblock_poly.set_active_scalars('data')
    for block in multiblock_poly:
        assert multiblock_poly[0].point_data.active_scalars_name == 'data'

    data = np.zeros((multiblock_poly[2].n_points, 3))
    multiblock_poly[2].point_data['data'] = data
    with pytest.raises(ValueError, match='Inconsistent dimensions'):
        multiblock_poly.set_active_scalars('data')

    data = np.arange(multiblock_poly[2].n_points, dtype=np.complex128)
    multiblock_poly[2].point_data['data'] = data
    with pytest.raises(ValueError, match='Inconsistent complex and real'):
        multiblock_poly.set_active_scalars('data')


def test_set_active_multi_multi(multiblock_poly):
    multi_multi = MultiBlock([multiblock_poly, multiblock_poly])
    with pytest.raises(KeyError, match='missing from all'):
        multi_multi.set_active_scalars('does-not-exist', allow_missing=True)

    multi_multi.set_active_scalars('multi-comp', allow_missing=True)


def test_set_active_scalars_mixed(multiblock_poly):
    for block in multiblock_poly:
        block.clear_data()
        block.point_data.set_array(range(block.n_points), 'data')
        block.cell_data.set_array(range(block.n_cells), 'data')

    # remove data from the last block
    del multiblock_poly[-1].point_data['data']
    del multiblock_poly[-1].cell_data['data']

    multiblock_poly.set_active_scalars('data', preference='cell', allow_missing=True)

    for block in multiblock_poly:
        if 'data' in block.cell_data:
            assert block.cell_data.active_scalars_name == 'data'

    multiblock_poly.set_active_scalars('data', preference='point', allow_missing=True)

    for block in multiblock_poly:
        if 'data' in block.point_data:
            assert block.point_data.active_scalars_name == 'data'


def test_to_polydata(multiblock_all):
    assert not multiblock_all.is_all_polydata

    dataset_a = multiblock_all.as_polydata_blocks()
    assert not multiblock_all.is_all_polydata
    assert dataset_a.is_all_polydata

    # verify nested works
    nested_mblock = pyvista.MultiBlock([multiblock_all, multiblock_all])
    assert not nested_mblock.is_all_polydata
    dataset_b = nested_mblock.as_polydata_blocks()
    assert dataset_b.is_all_polydata


def test_compute_normals(multiblock_poly):
    for block in multiblock_poly:
        block.clear_data()
        block['point_data'] = range(block.n_points)
    mblock = multiblock_poly._compute_normals(
        cell_normals=False, split_vertices=True, track_vertices=True
    )
    for block in mblock:
        assert 'Normals' in block.point_data
        assert 'point_data' in block.point_data
        assert 'pyvistaOriginalPointIds' in block.point_data

    # test non-poly raises
    multiblock_poly.append(pyvista.UnstructuredGrid())
    with pytest.raises(RuntimeError, match='This multiblock contains non-PolyData'):
        multiblock_poly._compute_normals()


def test_activate_plotting_scalars(multiblock_poly):
    for block in multiblock_poly:
        data = np.array(['a'] * block.n_points)
        block.point_data.set_array(data, 'data')
