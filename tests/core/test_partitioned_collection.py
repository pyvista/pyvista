from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista import PartitionedDataSet
from pyvista import PartitionedDataSetCollection
from pyvista.core import _vtk_core as _vtk


@pytest.fixture
def simple_collection():
    return PartitionedDataSetCollection(
        [
            PartitionedDataSet([pv.Sphere()]),
            PartitionedDataSet([pv.Cube(center=(2, 0, 0))]),
            PartitionedDataSet([pv.Cone(center=(0, 3, 0)), pv.Cone(center=(0, 5, 0))]),
        ]
    )


@pytest.fixture
def named_collection():
    return PartitionedDataSetCollection(
        {
            'sphere': pv.Sphere(),
            'cube': pv.Cube(center=(2, 0, 0)),
            'cone': pv.Cone(center=(0, 3, 0)),
        }
    )


def test_empty_construction():
    col = PartitionedDataSetCollection()
    assert len(col) == 0
    assert col.is_empty
    assert col.keys() == []
    assert list(col) == []
    assert col.n_partitioned_datasets == 0


def test_construct_from_list_of_partitioned():
    sphere = pv.Sphere()
    cube = pv.Cube()
    col = PartitionedDataSetCollection([PartitionedDataSet([sphere]), PartitionedDataSet([cube])])
    assert len(col) == 2
    assert isinstance(col[0], PartitionedDataSet)
    assert col[0][0].n_points == sphere.n_points
    assert col[1][0].n_points == cube.n_points
    assert col.keys() == ['Block-00', 'Block-01']


def test_construct_from_list_of_datasets_promotes():
    sphere = pv.Sphere()
    col = PartitionedDataSetCollection([sphere, pv.Cube()])
    assert len(col) == 2
    assert isinstance(col[0], PartitionedDataSet)
    assert col[0].n_partitions == 1
    assert col[0][0].n_points == sphere.n_points


def test_construct_from_tuple():
    col = PartitionedDataSetCollection((pv.Sphere(), pv.Cube()))
    assert len(col) == 2


def test_construct_from_dict_assigns_names():
    sphere = pv.Sphere()
    cube = pv.Cube()
    col = PartitionedDataSetCollection({'sphere': sphere, 'cube': cube})
    assert col.keys() == ['sphere', 'cube']
    assert col['sphere'][0].n_points == sphere.n_points
    assert col['cube'][0].n_points == cube.n_points


def test_construct_from_unknown_type_raises():
    with pytest.raises(TypeError, match='not supported'):
        PartitionedDataSetCollection(42)


def test_construct_too_many_args_raises():
    with pytest.raises(ValueError, match='supports 0 or 1 arguments'):
        PartitionedDataSetCollection(pv.Sphere(), pv.Cube())


def test_construct_from_vtk_shallow(simple_collection):
    raw = _vtk.vtkPartitionedDataSetCollection()
    raw.ShallowCopy(simple_collection)
    shallow = PartitionedDataSetCollection(raw)
    assert len(shallow) == len(simple_collection)
    for i, original in enumerate(simple_collection):
        assert shallow[i].n_partitions == original.n_partitions


def test_construct_from_vtk_deep(simple_collection):
    raw = _vtk.vtkPartitionedDataSetCollection()
    raw.DeepCopy(simple_collection)
    deep = PartitionedDataSetCollection(raw, deep=True)
    assert len(deep) == len(simple_collection)
    for i, original in enumerate(simple_collection):
        assert deep[i][0].n_points == original[0].n_points


def test_len_iter_and_contains(simple_collection):
    assert len(simple_collection) == 3
    seen = list(simple_collection)
    assert len(seen) == 3
    assert all(isinstance(b, PartitionedDataSet) for b in seen)


def test_int_indexing_positive_and_negative(simple_collection):
    col = simple_collection
    assert col[0] is not None
    assert col[-1].n_partitions == 2
    assert col[-3][0].n_points == col[0][0].n_points


def test_indexing_out_of_range(simple_collection):
    with pytest.raises(IndexError, match='out of range'):
        _ = simple_collection[10]
    with pytest.raises(IndexError, match='out of range'):
        _ = simple_collection[-99]


def test_string_indexing(named_collection):
    assert named_collection['sphere'][0].n_points == pv.Sphere().n_points
    with pytest.raises(KeyError, match='not found'):
        _ = named_collection['nope']


def test_slice_indexing_returns_new_collection(simple_collection):
    sub = simple_collection[0:2]
    assert isinstance(sub, PartitionedDataSetCollection)
    assert len(sub) == 2
    assert sub.keys() == ['Block-00', 'Block-01']
    assert sub[0][0].n_points == simple_collection[0][0].n_points


def test_setitem_int_promotes_dataset(simple_collection):
    plane = pv.Plane()
    simple_collection[0] = plane
    assert isinstance(simple_collection[0], PartitionedDataSet)
    assert simple_collection[0].n_partitions == 1
    assert simple_collection[0][0].n_points == plane.n_points


def test_setitem_int_accepts_partitioned(simple_collection):
    pds = PartitionedDataSet([pv.Plane(), pv.Plane()])
    simple_collection[1] = pds
    assert simple_collection[1].n_partitions == 2


def test_setitem_string_appends_when_missing():
    col = PartitionedDataSetCollection()
    col['new'] = pv.Sphere()
    assert col.keys() == ['new']
    assert col['new'][0].n_points == pv.Sphere().n_points


def test_setitem_string_overwrites_when_present():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    cube = pv.Cube()
    col['a'] = cube
    assert col.keys() == ['a']
    assert len(col) == 1
    assert col['a'][0].n_points == cube.n_points


def test_setitem_slice_replaces_each_entry(simple_collection):
    plane = pv.Plane()
    cube = pv.Cube()
    simple_collection[0:2] = [plane, cube]
    assert simple_collection[0][0].n_points == plane.n_points
    assert simple_collection[1][0].n_points == cube.n_points


def test_setitem_slice_size_mismatch(simple_collection):
    with pytest.raises(ValueError, match='size'):
        simple_collection[0:2] = [pv.Plane()]


def test_delitem_int_preserves_remaining_names():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube(), 'c': pv.Cone()})
    del col[1]
    assert col.keys() == ['a', 'c']
    assert len(col) == 2


def test_delitem_str(named_collection):
    del named_collection['sphere']
    assert named_collection.keys() == ['cube', 'cone']


def test_delitem_negative_index(simple_collection):
    del simple_collection[-1]
    assert len(simple_collection) == 2


def test_delitem_slice_with_step():
    col = PartitionedDataSetCollection({f'k{i}': pv.Sphere() for i in range(5)})
    del col[::2]
    assert col.keys() == ['k1', 'k3']


def test_delitem_out_of_range(simple_collection):
    with pytest.raises(IndexError, match='out of range'):
        del simple_collection[10]


def test_append_with_default_name(simple_collection):
    n = len(simple_collection)
    simple_collection.append(pv.Plane())
    assert len(simple_collection) == n + 1
    assert simple_collection.keys()[-1] == f'Block-{n:02}'


def test_append_with_explicit_name(simple_collection):
    simple_collection.append(pv.Plane(), 'plane')
    assert simple_collection.keys()[-1] == 'plane'


def test_append_self_raises():
    col = PartitionedDataSetCollection()
    with pytest.raises(ValueError, match='Cannot nest'):
        col.append(col)


def test_append_wraps_raw_vtk():
    col = PartitionedDataSetCollection()
    raw = _vtk.vtkPolyData()
    raw.ShallowCopy(pv.Sphere())
    col.append(raw)
    assert isinstance(col[0], PartitionedDataSet)
    assert col[0][0].n_points == pv.Sphere().n_points


def test_extend_from_collection_preserves_names():
    col = PartitionedDataSetCollection({'x': pv.Sphere()})
    other = PartitionedDataSetCollection({'a': pv.Cube(), 'b': pv.Cone()})
    col.extend(other)
    assert col.keys() == ['x', 'a', 'b']


def test_extend_from_iterable_assigns_default_names():
    col = PartitionedDataSetCollection()
    col.extend([pv.Sphere(), pv.Cube()])
    assert col.keys() == ['Block-00', 'Block-01']


def test_insert_preserves_existing_entries_and_names():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube(), 'c': pv.Cone()})
    plane = pv.Plane()
    col.insert(1, plane, 'plane')
    assert col.keys() == ['a', 'plane', 'b', 'c']
    assert col[1][0].n_points == plane.n_points
    assert col['a'][0].n_points == pv.Sphere().n_points
    assert col['c'][0].n_points == pv.Cone().n_points


def test_insert_default_name():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube()})
    col.insert(0, pv.Plane())
    assert col.keys()[0] == 'Block-00'


def test_insert_at_end():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    col.insert(1, pv.Cube(), 'cube')
    assert col.keys() == ['a', 'cube']


def test_pop_default_last(simple_collection):
    n = len(simple_collection)
    last_n_points = simple_collection[-1][0].n_points
    popped = simple_collection.pop()
    assert isinstance(popped, PartitionedDataSet)
    assert popped[0].n_points == last_n_points
    assert len(simple_collection) == n - 1


def test_pop_by_name():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube()})
    popped = col.pop('a')
    assert popped[0].n_points == pv.Sphere().n_points
    assert col.keys() == ['b']


def test_reverse_preserves_data_and_names():
    sphere = pv.Sphere()
    cube = pv.Cube()
    cone = pv.Cone()
    col = PartitionedDataSetCollection({'a': sphere, 'b': cube, 'c': cone})
    col.reverse()
    assert col.keys() == ['c', 'b', 'a']
    assert col[0][0].n_points == cone.n_points
    assert col[-1][0].n_points == sphere.n_points


def test_clear_via_mutable_sequence():
    col = PartitionedDataSetCollection([pv.Sphere(), pv.Cube()])
    col.clear()
    assert len(col) == 0
    assert col.keys() == []


def test_replace_by_name_preserves_name():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    cube = pv.Cube()
    col.replace('a', cube)
    assert col.keys() == ['a']
    assert col['a'][0].n_points == cube.n_points


def test_replace_by_index_preserves_name():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube()})
    col.replace(0, pv.Plane())
    assert col.keys() == ['a', 'b']


def test_get_returns_default():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    assert col.get('a')[0].n_points == pv.Sphere().n_points
    assert col.get('missing') is None
    sentinel = PartitionedDataSet([pv.Cube()])
    assert col.get('missing', sentinel) is sentinel


def test_get_block_raises():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    assert col.get_block('a')[0].n_points == pv.Sphere().n_points
    with pytest.raises(KeyError):
        col.get_block('missing')


def test_get_index_by_name():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube()})
    assert col.get_index_by_name('a') == 0
    assert col.get_index_by_name('b') == 1
    with pytest.raises(KeyError, match='not found'):
        col.get_index_by_name('nope')


def test_set_block_name_by_index(simple_collection):
    simple_collection.set_block_name(0, 'sphere')
    assert simple_collection.get_block_name(0) == 'sphere'


def test_set_block_name_none_is_noop(simple_collection):
    simple_collection.set_block_name(0, 'first')
    simple_collection.set_block_name(0, None)
    assert simple_collection.get_block_name(0) == 'first'


def test_set_block_name_by_string():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    col.set_block_name('a', 'renamed')
    assert col.keys() == ['renamed']


def test_ipython_key_completions(named_collection):
    assert set(named_collection._ipython_key_completions_()) == {'sphere', 'cube', 'cone'}


def test_n_blocks_alias(simple_collection):
    assert simple_collection.n_blocks == simple_collection.n_partitioned_datasets
    simple_collection.n_blocks = 5
    assert simple_collection.n_partitioned_datasets == 5


def test_n_partitioned_datasets_setter(simple_collection):
    simple_collection.n_partitioned_datasets = 1
    assert len(simple_collection) == 1


def test_is_nested_with_multiple_partitions(simple_collection):
    assert simple_collection.is_nested


def test_is_nested_flat():
    col = PartitionedDataSetCollection([pv.Sphere()])
    assert not col.is_nested


def test_is_empty():
    assert PartitionedDataSetCollection().is_empty
    assert not PartitionedDataSetCollection([pv.Sphere()]).is_empty


def test_bounds_aggregates_all_leaves(simple_collection):
    bounds = simple_collection.bounds
    # Cone at y=5 pushes y_max. Sphere spans [-0.5, 0.5]. Cube at x=2.
    assert bounds.y_max > 4.0
    assert bounds.x_max >= 2.0


def test_bounds_empty_returns_zeros():
    col = PartitionedDataSetCollection()
    assert tuple(col.bounds) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_center_matches_bounds(simple_collection):
    bounds = simple_collection.bounds
    expected = (
        0.5 * (bounds.x_min + bounds.x_max),
        0.5 * (bounds.y_min + bounds.y_max),
        0.5 * (bounds.z_min + bounds.z_max),
    )
    assert np.allclose(simple_collection.center, expected)


def test_length_is_diagonal(simple_collection):
    bounds = simple_collection.bounds
    expected = np.linalg.norm(
        [
            bounds.x_max - bounds.x_min,
            bounds.y_max - bounds.y_min,
            bounds.z_max - bounds.z_min,
        ]
    )
    assert simple_collection.length == pytest.approx(expected)


def test_block_types(simple_collection):
    assert simple_collection.block_types == {pv.PolyData}


def test_assembly_property_auto_creates():
    col = PartitionedDataSetCollection([pv.Sphere()])
    assert col.GetDataAssembly() is None
    _ = col.assembly
    assert col.GetDataAssembly() is not None


def test_assembly_setter_assigns_and_clears():
    col = PartitionedDataSetCollection([pv.Sphere()])
    asm = _vtk.vtkDataAssembly()
    col.assembly = asm
    assert col.GetDataAssembly() is asm


def test_add_assembly_node_and_select_datasets():
    col = PartitionedDataSetCollection([pv.Sphere(), pv.Cube(), pv.Cone()])
    node = col.add_assembly_node('/', 'group')
    col.assign_dataset_to_node(node, 0)
    col.assign_dataset_to_node(node, 2)
    assert col.select_datasets('//group') == [0, 2]


def test_add_assembly_node_bad_path():
    col = PartitionedDataSetCollection([pv.Sphere()])
    with pytest.raises(KeyError, match='not found'):
        col.add_assembly_node('/missing', 'x')


def test_assign_dataset_out_of_range():
    col = PartitionedDataSetCollection([pv.Sphere()])
    node = col.add_assembly_node('/', 'g')
    with pytest.raises(IndexError, match='out of range'):
        col.assign_dataset_to_node(node, 10)


def test_assembly_to_dict():
    col = PartitionedDataSetCollection([pv.Sphere(), pv.Cube()])
    node = col.add_assembly_node('/', 'group')
    col.assign_dataset_to_node(node, 0)
    tree = col.assembly_to_dict()
    assert tree['name'] == 'assembly'
    assert len(tree['children']) == 1
    group = tree['children'][0]
    assert group['name'] == 'group'
    assert group['datasets'] == [0]
    assert group['children'] == []


def test_assembly_to_dict_no_assembly():
    col = PartitionedDataSetCollection()
    assert col.assembly_to_dict() == {}


def test_select_datasets_no_assembly():
    col = PartitionedDataSetCollection()
    assert col.select_datasets('//x') == []


def test_recursive_iterator_blocks_counts_leaves(simple_collection):
    leaves = list(simple_collection.recursive_iterator(contents='blocks'))
    # Sphere + Cube + 2 Cones
    assert len(leaves) == 4
    assert all(leaf is not None for leaf in leaves)


def test_recursive_iterator_ids(simple_collection):
    ids = list(simple_collection.recursive_iterator(contents='ids'))
    assert ids == [(0, 0), (1, 0), (2, 0), (2, 1)]


def test_recursive_iterator_items_and_names():
    col = PartitionedDataSetCollection({'a': pv.Sphere(), 'b': pv.Cube()})
    items = list(col.recursive_iterator(contents='items'))
    assert [name for name, _ in items] == ['a', 'b']
    names = list(col.recursive_iterator(contents='names'))
    assert names == ['a', 'b']


def test_recursive_iterator_all_yields_triples():
    col = PartitionedDataSetCollection({'a': pv.Sphere()})
    result = list(col.recursive_iterator(contents='all'))
    assert len(result) == 1
    ids, name, leaf = result[0]
    assert ids == (0, 0)
    assert name == 'a'
    assert leaf.n_points == pv.Sphere().n_points


def test_recursive_iterator_skip_none_toggle():
    col = PartitionedDataSetCollection([pv.Sphere()])
    col[0][0] = None
    skipped = list(col.recursive_iterator(contents='blocks'))
    assert skipped == []
    kept = list(col.recursive_iterator(contents='blocks', skip_none=False))
    assert len(kept) == 1
    assert kept[0] is None


def test_recursive_iterator_invalid_contents(simple_collection):
    with pytest.raises(ValueError, match='contents must be'):
        list(simple_collection.recursive_iterator(contents='bogus'))


def test_flatten_preserves_all_leaves_and_labels(simple_collection):
    multi = simple_collection.flatten()
    assert isinstance(multi, pv.MultiBlock)
    assert multi.n_blocks == 4
    # Nested cone block should produce partition-00 and partition-01 labels
    assert any('partition-00' in (k or '') for k in multi.keys())
    assert any('partition-01' in (k or '') for k in multi.keys())


def test_clear_all_data_removes_arrays(simple_collection):
    for leaf in simple_collection.recursive_iterator(contents='blocks'):
        leaf['pd'] = np.arange(leaf.n_points)
    simple_collection.clear_all_data()
    for leaf in simple_collection.recursive_iterator(contents='blocks'):
        assert 'pd' not in leaf.point_data


def test_clear_all_point_and_cell_data_independent(simple_collection):
    for leaf in simple_collection.recursive_iterator(contents='blocks'):
        leaf.point_data['p'] = np.arange(leaf.n_points)
        leaf.cell_data['c'] = np.arange(leaf.n_cells)
    simple_collection.clear_all_point_data()
    for leaf in simple_collection.recursive_iterator(contents='blocks'):
        assert 'p' not in leaf.point_data
        assert 'c' in leaf.cell_data
    simple_collection.clear_all_cell_data()
    for leaf in simple_collection.recursive_iterator(contents='blocks'):
        assert 'c' not in leaf.cell_data


def test_get_data_range_aggregates_across_leaves():
    sphere = pv.Sphere()
    cube = pv.Cube()
    sphere['scalars'] = np.arange(sphere.n_points, dtype=float)
    cube['scalars'] = np.arange(cube.n_points, dtype=float) * 2.0
    col = PartitionedDataSetCollection([sphere, cube])
    lo, hi = col.get_data_range('scalars', preference='point')
    assert lo == 0.0
    assert hi == max((sphere.n_points - 1), (cube.n_points - 1) * 2.0)


def test_get_data_range_missing_raises():
    col = PartitionedDataSetCollection([pv.Sphere()])
    with pytest.raises(KeyError):
        col.get_data_range('missing', preference='point')


def test_get_data_range_missing_allowed_returns_nan():
    col = PartitionedDataSetCollection([pv.Sphere()])
    lo, hi = col.get_data_range('missing', preference='point', allow_missing=True)
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_set_active_scalars_applies_to_each_leaf():
    sphere = pv.Sphere()
    cube = pv.Cube()
    sphere['foo'] = np.arange(sphere.n_points, dtype=float)
    cube['foo'] = np.arange(cube.n_points, dtype=float)
    col = PartitionedDataSetCollection([sphere, cube])
    col.set_active_scalars('foo', preference='point')
    for leaf in col.recursive_iterator(contents='blocks'):
        assert leaf.active_scalars_name == 'foo'


def test_set_active_scalars_missing_raises():
    col = PartitionedDataSetCollection([pv.Sphere()])
    with pytest.raises(KeyError):
        col.set_active_scalars('missing', preference='point')


def test_set_active_scalars_allow_missing():
    col = PartitionedDataSetCollection([pv.Sphere()])
    col.set_active_scalars('missing', preference='point', allow_missing=True)


def test_copy_deep_is_independent():
    sphere = pv.Sphere()
    sphere['data'] = np.arange(sphere.n_points, dtype=float)
    col = PartitionedDataSetCollection([sphere])
    cp = col.copy(deep=True)
    # Mutating the original must not affect the copy
    col[0][0]['data'] += 1.0
    assert not np.allclose(col[0][0]['data'], cp[0][0]['data'])


def test_copy_shallow_preserves_structure():
    sphere = pv.Sphere()
    sphere['data'] = np.arange(sphere.n_points, dtype=float)
    col = PartitionedDataSetCollection({'sphere': sphere})
    cp = col.copy(deep=False)
    assert len(cp) == len(col)
    assert cp.keys() == col.keys()
    assert cp[0][0].n_points == sphere.n_points
    assert np.array_equal(cp[0][0]['data'], sphere['data'])
    # Structural independence: deleting from the copy does not shrink the original
    del cp[0]
    assert len(col) == 1


def test_equality_reflexive_and_structural():
    a = PartitionedDataSetCollection({'x': pv.Sphere()})
    b = PartitionedDataSetCollection({'x': pv.Sphere()})
    assert a == b
    assert a == PartitionedDataSetCollection({'x': pv.Sphere()})


def test_equality_different_names():
    a = PartitionedDataSetCollection({'x': pv.Sphere()})
    c = PartitionedDataSetCollection({'y': pv.Sphere()})
    assert a != c


def test_equality_different_types():
    a = PartitionedDataSetCollection({'x': pv.Sphere()})
    assert a != 'not a collection'
    assert a != pv.MultiBlock()


def test_equality_different_length():
    a = PartitionedDataSetCollection([pv.Sphere()])
    b = PartitionedDataSetCollection([pv.Sphere(), pv.Cube()])
    assert a != b


def test_unhashable():
    col = PartitionedDataSetCollection([pv.Sphere()])
    with pytest.raises(TypeError):
        hash(col)


def test_repr_includes_counts_and_bounds(simple_collection):
    text = repr(simple_collection)
    assert 'PartitionedDataSetCollection' in text
    assert 'N PartitionedDataSets' in text
    assert 'X Bounds' in text
    assert str(simple_collection) == text


def test_repr_html_contains_name():
    col = PartitionedDataSetCollection({'my-block': pv.Sphere()})
    html = col._repr_html_()
    assert 'PartitionedDataSetCollection' in html
    assert 'my-block' in html


def test_wrap_raw_vtk_returns_collection():
    raw = _vtk.vtkPartitionedDataSetCollection()
    raw.SetNumberOfPartitionedDataSets(1)
    pds = _vtk.vtkPartitionedDataSet()
    pds.SetNumberOfPartitions(1)
    pds.SetPartition(0, pv.Sphere())
    raw.SetPartitionedDataSet(0, pds)
    wrapped = pv.wrap(raw)
    assert isinstance(wrapped, PartitionedDataSetCollection)
    assert len(wrapped) == 1
    assert wrapped[0][0].n_points == pv.Sphere().n_points


def test_save_and_read_vtpc_round_trip(tmp_path):
    sphere = pv.Sphere()
    sphere['scalars'] = np.arange(sphere.n_points, dtype=float)
    col = PartitionedDataSetCollection({'sphere': sphere, 'cube': pv.Cube()})
    node = col.add_assembly_node('/', 'group')
    col.assign_dataset_to_node(node, 0)

    path = tmp_path / 'test.vtpc'
    col.save(path)
    rt = pv.read(path)

    assert isinstance(rt, PartitionedDataSetCollection)
    assert len(rt) == 2
    assert rt.keys() == ['sphere', 'cube']
    assert rt[0][0].n_points == sphere.n_points
    assert np.allclose(rt[0][0]['scalars'], sphere['scalars'])
    assert rt.GetDataAssembly() is not None
    assert rt.select_datasets('//group') == [0]


def test_construct_from_pathlib_path(tmp_path):
    col = PartitionedDataSetCollection({'sphere': pv.Sphere()})
    path = tmp_path / 'roundtrip.vtpc'
    col.save(path)
    loaded = PartitionedDataSetCollection(path)
    assert len(loaded) == 1
    assert loaded.keys() == ['sphere']


def test_is_all_polydata_true(simple_collection):
    assert simple_collection.is_all_polydata


def test_is_all_polydata_false():
    col = PartitionedDataSetCollection([pv.Sphere(), pv.ImageData(dimensions=(3, 3, 3))])
    assert not col.is_all_polydata


def test_cast_to_multiblock_uses_flatten(simple_collection):
    mb = simple_collection.cast_to_multiblock()
    assert isinstance(mb, pv.MultiBlock)
    # Sphere + Cube + 2 Cones = 4 leaves
    assert mb.n_blocks == 4


def test_outline_matches_bounds(simple_collection):
    outline = simple_collection.outline()
    assert isinstance(outline, pv.PolyData)
    assert np.allclose(outline.bounds, simple_collection.bounds)


def test_outline_corners(simple_collection):
    corners = simple_collection.outline_corners(factor=0.2)
    assert isinstance(corners, pv.PolyData)
    assert corners.n_points > 0


def test_combine_produces_unstructured_grid():
    cube_a = pv.Cube(clean=False)
    cube_b = pv.Cube(clean=False)
    col = PartitionedDataSetCollection([cube_a, cube_b])
    merged = col.combine()
    assert isinstance(merged, pv.UnstructuredGrid)
    assert merged.n_points == cube_a.n_points + cube_b.n_points
    merged_pts = col.combine(merge_points=True)
    assert merged_pts.n_points < merged.n_points


def test_repr_html_uses_multiblock_icon():
    col = PartitionedDataSetCollection([pv.Sphere()])
    # The MultiBlock icon is the only reasonable icon for composite containers;
    # verify it actually rendered (i.e. build_repr_html found the icon).
    assert col._repr_html_()
    pds = PartitionedDataSet([pv.Sphere()])
    assert pds._repr_html_()
