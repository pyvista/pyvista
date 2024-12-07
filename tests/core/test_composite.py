from __future__ import annotations

import itertools
import pathlib
import platform
import re
import weakref

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import ImageData
from pyvista import MultiBlock
from pyvista import PolyData
from pyvista import RectilinearGrid
from pyvista import StructuredGrid
from pyvista import examples as ex

skip_mac = pytest.mark.skipif(platform.system() == 'Darwin', reason='Flaky Mac tests')


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
    assert isinstance(multi.GetBlock(1), ImageData)
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
    multi[4] = pv.Sphere()
    assert isinstance(multi[4], PolyData)
    multi[4] = vtk.vtkUnstructuredGrid()
    assert isinstance(multi[4], pv.UnstructuredGrid)

    with pytest.raises(ValueError, match='Cannot nest a composite dataset in itself.'):
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
    assert isinstance(multi.get(5), ImageData)
    # Test get by name
    assert isinstance(multi['uni'], ImageData)
    assert isinstance(multi['rect'], RectilinearGrid)
    assert isinstance(multi.get('uni'), ImageData)
    assert multi.get('no key') is None
    assert multi.get('no key', default=pv.Sphere()) == pv.Sphere()
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
    assert all(k is None for k in multi.keys())

    multi['new key'] = pv.Sphere()
    assert multi.n_blocks == 4
    assert multi[3] == pv.Sphere()

    multi['new key'] = pv.Cube()
    assert multi.n_blocks == 4
    assert multi[3] == pv.Cube()

    with pytest.raises(KeyError):
        _ = multi.get_index_by_name('foo')

    with pytest.raises(IndexError):
        multi[4] = ImageData()

    with pytest.raises(KeyError):
        multi['not a key']
    with pytest.raises(TypeError):
        data = multi[[0, 1]]

    with pytest.raises(TypeError):
        multi[1, 'foo'] = data


def test_replace():
    spheres = {f'{i}': pv.Sphere(phi_resolution=i + 3) for i in range(10)}
    multi = MultiBlock(spheres)
    cube = pv.Cube()
    multi.replace(3, cube)
    assert multi.get_block_name(3) == '3'
    assert multi[3] is cube


def test_pop():
    spheres = {f'{i}': pv.Sphere(phi_resolution=i + 3) for i in range(10)}
    multi = MultiBlock(spheres)
    assert multi.pop() == spheres['9']
    assert spheres['9'] not in multi
    assert multi.pop(0) == spheres['0']
    assert spheres['0'] not in multi


def test_del_slice(sphere):
    multi = MultiBlock({f'{i}': sphere for i in range(10)})
    del multi[0:10:2]
    assert len(multi) == 5
    assert all(f'{i}' in multi.keys() for i in range(1, 10, 2))

    multi = MultiBlock({f'{i}': sphere for i in range(10)})
    del multi[5:2:-1]
    assert len(multi) == 7
    assert all(f'{i}' in multi.keys() for i in [0, 1, 2, 6, 7, 8, 9])


def test_slicing_multiple_in_setitem(sphere):
    # equal length
    multi = MultiBlock({f'{i}': sphere for i in range(10)})
    multi[1:3] = [pv.Cube(), pv.Cube()]
    assert multi[1] == pv.Cube()
    assert multi[2] == pv.Cube()
    assert multi.count(pv.Cube()) == 2
    assert len(multi) == 10

    # len(slice) < len(data)
    multi = MultiBlock({f'{i}': sphere for i in range(10)})
    multi[1:3] = [pv.Cube(), pv.Cube(), pv.Cube()]
    assert multi[1] == pv.Cube()
    assert multi[2] == pv.Cube()
    assert multi[3] == pv.Cube()
    assert multi.count(pv.Cube()) == 3
    assert len(multi) == 11

    # len(slice) > len(data)
    multi = MultiBlock({f'{i}': sphere for i in range(10)})
    multi[1:3] = [pv.Cube()]
    assert multi[1] == pv.Cube()
    assert multi.count(pv.Cube()) == 1
    assert len(multi) == 9


def test_reverse(sphere):
    multi = MultiBlock({f'{i}': sphere for i in range(3)})
    multi.append(pv.Cube(), 'cube')
    multi.reverse()
    assert multi[0] == pv.Cube()
    assert np.array_equal(multi.keys(), ['cube', '2', '1', '0'])


def test_insert(sphere):
    multi = MultiBlock({f'{i}': sphere for i in range(3)})
    cube = pv.Cube()
    multi.insert(0, cube)
    assert len(multi) == 4
    assert multi[0] is cube

    # test with negative index and name
    multi.insert(-1, pv.ImageData(), name='uni')
    assert len(multi) == 5
    # inserted before last element
    assert isinstance(multi[-2], pv.ImageData)  # inserted before last element
    assert multi.get_block_name(-2) == 'uni'


def test_extend(sphere, uniform, ant):
    # test with Iterable
    multi = MultiBlock([sphere, ant])
    new_multi = [uniform, uniform]
    multi.extend(new_multi)
    assert len(multi) == 4
    assert multi.count(uniform) == 2

    # test with a MultiBlock
    multi = MultiBlock([sphere, ant])
    new_multi = MultiBlock({'uniform1': uniform, 'uniform2': uniform})
    multi.extend(new_multi)
    assert len(multi) == 4
    assert multi.count(uniform) == 2
    assert multi.keys()[-2] == 'uniform1'
    assert multi.keys()[-1] == 'uniform2'


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
    assert isinstance(multi[1], ImageData)
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


def test_multi_block_repr(multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    assert multi._repr_html_() is not None
    pattern = (
        r'MultiBlock \(0x[0-9a-fA-F]+\)'
        r'\s+N Blocks:\s{3}\d+'
        r'\s+X Bounds:\s{3}[+-]?\d*\.\d{3}e[+-]\d+,\s+[-+]?\d\.\d+e[+-]\d+'
        r'\s+Y Bounds:\s{3}[+-]?\d*\.\d{3}e[+-]\d+,\s+[-+]?\d\.\d+e[+-]\d+'
        r'\s+Z Bounds:\s{3}[+-]?\d*\.\d{3}e[+-]\d+,\s+[-+]?\d\.\d+e[+-]\d+'
    )
    match = re.search(pattern, repr(multi))
    assert repr(multi) == match.string
    assert str(multi) == match.string


def test_multi_block_repr_bounds():
    empty_poly = pv.PolyData().extract_cells(0)
    poly_x_bounds = repr(empty_poly).splitlines()[3]
    poly_y_bounds = repr(empty_poly).splitlines()[4]
    poly_z_bounds = repr(empty_poly).splitlines()[5]

    empty_multiblock = pv.MultiBlock([empty_poly])
    multi_x_bounds = repr(empty_multiblock).splitlines()[2]
    multi_y_bounds = repr(empty_multiblock).splitlines()[3]
    multi_z_bounds = repr(empty_multiblock).splitlines()[4]

    assert multi_x_bounds == poly_x_bounds
    assert multi_y_bounds == poly_y_bounds
    assert multi_z_bounds == poly_z_bounds


def test_multi_block_eq(multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    other = multi.copy()

    assert multi is not other
    assert multi == other

    assert pv.MultiBlock() == pv.MultiBlock()

    other[0] = pv.Sphere()
    assert multi != other

    other = multi.copy()
    other.set_block_name(0, 'not matching')
    assert multi != other

    other = multi.copy()
    other.append(pv.Sphere())
    assert multi != other


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pv.core.composite.MultiBlock._WRITERS)
@pytest.mark.parametrize('use_pathlib', [True, False])
def test_multi_block_io(
    extension, binary, tmpdir, use_pathlib, multiblock_all_with_nested_and_none
):
    filename = str(tmpdir.mkdir('tmpdir').join(f'tmp.{extension}'))
    if use_pathlib:
        pathlib.Path(filename)
    multi = multiblock_all_with_nested_and_none

    # Save it out
    multi.save(filename, binary)
    foo = MultiBlock(filename)
    assert foo.n_blocks == multi.n_blocks
    foo = pv.read(filename)
    assert foo.n_blocks == multi.n_blocks


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vtm', 'vtmb'])
def test_ensight_multi_block_io(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.%s' % extension))  # noqa: UP031
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
    foo = pv.read(filename)
    assert foo.n_blocks == multi.n_blocks
    for block in foo:
        assert block.array_names == array_names


def test_invalid_arg():
    with pytest.raises(TypeError):
        pv.MultiBlock(np.empty(10))
    with pytest.raises(ValueError):  # noqa: PT011
        pv.MultiBlock(np.empty(10), np.empty(10))


def test_multi_io_erros(tmpdir):
    fdir = tmpdir.mkdir('tmpdir')
    multi = MultiBlock()
    # Check saving with bad extension
    bad_ext_name = str(fdir.join('tmp.npy'))
    with pytest.raises(ValueError):  # noqa: PT011
        multi.save(bad_ext_name)
    arr = np.random.default_rng().random((10, 10))
    np.save(bad_ext_name, arr)
    # Load non existing file
    with pytest.raises(FileNotFoundError):
        _ = MultiBlock('foo.vtm')
    # Load bad extension
    with pytest.raises(IOError):  # noqa: PT011
        _ = MultiBlock(bad_ext_name)


def test_extract_geometry(multiblock_all_with_nested_and_none):
    geom = multiblock_all_with_nested_and_none.extract_geometry()
    assert isinstance(geom, PolyData)


def test_combine_filter(multiblock_all_with_nested_and_none):
    geom = multiblock_all_with_nested_and_none.combine()
    assert isinstance(geom, pv.UnstructuredGrid)


@pytest.mark.parametrize('inplace', [True, False])
def test_transform_filter(ant, sphere, airplane, tetbeam, inplace):
    # Set up
    multi = pv.MultiBlock([ant, sphere])
    nested = pv.MultiBlock([airplane, tetbeam])
    nested.append(None)
    multi.append(nested)
    multi.append(None)
    for i, _ in enumerate(multi):
        multi.set_block_name(i, str(i))

    NUMBER = 42
    transform = pv.Transform().translate(NUMBER, NUMBER, NUMBER)
    bounds_before = np.array(multi.bounds)
    n_blocks_before = multi.n_blocks
    keys_before = multi.keys()

    # Do test
    output = multi.transform(
        transform, inplace=inplace, transform_all_input_vectors=False, progress_bar=False
    )
    bounds_after = np.array(output.bounds)
    n_blocks_after = output.n_blocks
    keys_after = output.keys()

    assert (output is multi) == inplace
    for block_in, block_out in zip(multi, output):
        assert (block_in is block_out) == inplace or (block_in is None)
    assert np.allclose(bounds_before + NUMBER, bounds_after)
    assert n_blocks_before == n_blocks_after
    assert keys_before == keys_after


@pytest.mark.parametrize('deep', [True, False])
def test_multi_block_copy(deep, multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    multi_copy = multi.copy(deep=deep)
    assert multi.n_blocks == multi_copy.n_blocks
    for i in range(multi_copy.n_blocks):
        block = multi_copy.GetBlock(i)
        assert pv.is_pyvista_dataset(block) or block is None
        assert (multi[i] is multi_copy[i]) != deep or (multi[i] is None)


@pytest.mark.parametrize('recursive', [True, False])
def test_multi_block_shallow_copy(recursive, multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    multi_copy = MultiBlock()
    multi_copy.shallow_copy(multi, recursive=recursive)
    assert multi.n_blocks == multi_copy.n_blocks
    for i, block in enumerate(multi_copy):
        assert pv.is_pyvista_dataset(block) or block is None
        if isinstance(multi[i], MultiBlock):
            assert (multi[i] is multi_copy[i]) != recursive
        else:
            assert multi_copy[i] is multi[i]


def test_multi_block_negative_index(ant, sphere, uniform, airplane, tetbeam):
    multi = pv.MultiBlock([ant, sphere, uniform, airplane, tetbeam])
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
    multi[-5] = tetbeam
    assert multi[0] == tetbeam

    with pytest.raises(IndexError):
        multi[-6] = uniform


def test_multi_slice_index(ant, sphere, uniform, airplane, tetbeam):
    multi = pv.MultiBlock([ant, sphere, uniform, airplane, tetbeam])
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

    sub = [airplane, tetbeam]
    multi[0:2] = sub
    assert multi[0] is airplane
    assert multi[1] is tetbeam


def test_slice_defaults(multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    assert multi[:] == multi[0 : len(multi)]


def test_slice_negatives(multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    test_multi = pv.MultiBlock({key: multi[key] for key in multi.keys()[::-1]})
    assert multi[::-1] == test_multi

    test_multi = pv.MultiBlock({key: multi[key] for key in multi.keys()[-2:]})
    assert multi[-2:] == test_multi

    test_multi = pv.MultiBlock({key: multi[key] for key in multi.keys()[:-1]})
    assert multi[:-1] == test_multi

    test_multi = pv.MultiBlock({key: multi[key] for key in multi.keys()[-1:-4:-2]})
    assert multi[-1:-4:-2] == test_multi


def test_multi_block_volume(ant, airplane, sphere, uniform):
    multi = pv.MultiBlock([ant, sphere, uniform, airplane, None])
    vols = ant.volume + sphere.volume + uniform.volume + airplane.volume
    assert multi.volume == pytest.approx(vols)


def test_multi_block_length(multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    assert multi.length == pv.Box(bounds=multi.bounds).length


def test_multi_block_save_lines(tmpdir):
    radius = 1
    xr = np.random.default_rng().random(10)
    yr = np.random.default_rng().random(10)
    x = radius * np.sin(yr) * np.cos(xr)
    y = radius * np.sin(yr) * np.sin(xr)
    z = radius * np.cos(yr)
    xyz = np.stack((x, y, z), axis=1)

    poly = pv.lines_from_points(xyz, close=False)
    blocks = pv.MultiBlock()
    for _ in range(2):
        blocks.append(poly)

    path = tmpdir.mkdir('tmpdir')
    line_filename = str(path.join('lines.vtk'))
    block_filename = str(path.join('blocks.vtmb'))
    poly.save(line_filename)
    blocks.save(block_filename)

    poly_load = pv.read(line_filename)
    assert np.allclose(poly_load.points, poly.points)

    blocks_load = pv.read(block_filename)
    assert np.allclose(blocks_load[0].points, blocks[0].points)


def test_multi_block_data_range():
    volume = pv.Wavelet()
    a = volume.slice_along_axis(5, 'x')
    with pytest.raises(KeyError):
        a.get_data_range('foo')
    mi, ma = a.get_data_range(volume.active_scalars_name)
    assert mi is not None
    assert ma is not None
    # Test on a nested MultiBlock
    b = volume.slice_along_axis(5, 'y')
    slices = pv.MultiBlock([a, b])
    with pytest.raises(KeyError):
        slices.get_data_range('foo')
    mi, ma = slices.get_data_range(volume.active_scalars_name)
    assert mi is not None
    assert ma is not None


def test_multiblock_ref():
    # can't use fixtures here as we need to remove all references for
    # garbage collection
    sphere = pv.Sphere()
    cube = pv.Cube()

    block = MultiBlock([sphere, cube])
    block[0]['a_new_var'] = np.zeros(block[0].n_points)
    assert 'a_new_var' in block[0].array_names

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
    assert multiblock_all[1].point_data.active_scalars_name == 'point_data_a'

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
        assert block.point_data.active_scalars_name == 'data'

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
    if pv.vtk_version_info >= (9, 1, 0):
        multiblock_all.append(pv.PointSet([0.0, 0.0, 1.0]))  # missing pointset
    assert not multiblock_all.is_all_polydata

    dataset_a = multiblock_all.as_polydata_blocks()
    if pv.vtk_version_info >= (9, 1, 0):
        assert dataset_a[-1].n_points == 1
    assert not multiblock_all.is_all_polydata
    assert dataset_a.is_all_polydata

    # verify nested works
    nested_mblock = pv.MultiBlock([multiblock_all, multiblock_all])
    assert not nested_mblock.is_all_polydata
    dataset_b = nested_mblock.as_polydata_blocks()
    assert dataset_b.is_all_polydata


def test_compute_normals(multiblock_poly):
    for block in multiblock_poly:
        block.clear_data()
        block['point_data'] = range(block.n_points)
    mblock = multiblock_poly._compute_normals(
        cell_normals=False,
        split_vertices=True,
        track_vertices=True,
    )
    for block in mblock:
        assert 'Normals' in block.point_data
        assert 'point_data' in block.point_data
        assert 'pyvistaOriginalPointIds' in block.point_data

    # test non-poly raises
    multiblock_poly.append(pv.UnstructuredGrid())
    with pytest.raises(RuntimeError, match='This multiblock contains non-PolyData'):
        multiblock_poly._compute_normals()


def test_activate_scalars(multiblock_poly):
    for block in multiblock_poly:
        data = np.array(['a'] * block.n_points)
        block.point_data.set_array(data, 'data')


def test_clear_all_data(multiblock_all):
    for block in multiblock_all:
        block.point_data['data'] = range(block.n_points)
        block.cell_data['data'] = range(block.n_cells)
    multiblock_all.append(multiblock_all.copy())
    multiblock_all.clear_all_data()
    for block in multiblock_all:
        if isinstance(block, MultiBlock):
            for subblock in block:
                assert subblock.point_data.keys() == []
                assert subblock.cell_data.keys() == []
        else:
            assert block.point_data.keys() == []
            assert block.cell_data.keys() == []


def test_clear_all_point_data(multiblock_all):
    for block in multiblock_all:
        block.point_data['data'] = range(block.n_points)
        block.cell_data['data'] = range(block.n_cells)
    multiblock_all.append(multiblock_all.copy())
    multiblock_all.clear_all_point_data()
    for block in multiblock_all:
        if isinstance(block, MultiBlock):
            for subblock in block:
                assert subblock.point_data.keys() == []
                assert subblock.cell_data.keys() != []
        else:
            assert block.point_data.keys() == []
            assert block.cell_data.keys() != []


def test_clear_all_cell_data(multiblock_all):
    for block in multiblock_all:
        block.point_data['data'] = range(block.n_points)
        block.cell_data['data'] = range(block.n_cells)
    multiblock_all.append(multiblock_all.copy())
    multiblock_all.clear_all_cell_data()
    for block in multiblock_all:
        if isinstance(block, MultiBlock):
            for subblock in block:
                assert subblock.point_data.keys() != []
                assert subblock.cell_data.keys() == []
        else:
            assert block.point_data.keys() != []
            assert block.cell_data.keys() == []


@pytest.mark.parametrize('container', [pv.MultiBlock, pv.PartitionedDataSet])
def test_multiblock_partitioned_zip(container):
    # Test `__iter__` and `__next__` inheritance
    list_ = [None, None]
    composite = container(list_)
    zipped_container = list(zip(composite, composite))
    zipped_list = list(zip(list_, list_))

    assert len(zipped_container) == len(zipped_list)
    assert len(zipped_container[0]) == len(zipped_list[0])
    for i, j in itertools.product(range(2), repeat=2):
        assert zipped_container[i][j] is zipped_list[i][j] is None
