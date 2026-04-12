from __future__ import annotations

import re

import pytest

import pyvista as pv
from pyvista import PartitionedDataSet
from pyvista.core.errors import PartitionedDataSetsNotSupported


def partitions_from_datasets(*datasets):
    """Return pyvista partitions of any number of datasets."""
    return PartitionedDataSet(datasets)


def test_reverse(sphere):
    partitions = PartitionedDataSet([sphere for i in range(3)])
    partitions.append(pv.Cube())
    partitions.reverse()
    assert partitions[0] == pv.Cube()


def test_insert(sphere):
    partitions = PartitionedDataSet([sphere for i in range(3)])
    cube = pv.Cube()
    partitions.insert(0, cube)
    assert len(partitions) == 4
    assert partitions[0] is cube


def test_pop():
    spheres = [pv.Sphere(phi_resolution=i + 3) for i in range(10)]
    partitions = PartitionedDataSet(spheres)
    match = 'The requested operation is not supported for PartitionedDataSetss.'
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        partitions.pop()


def test_partitions_slice_index(ant, sphere, uniform, airplane, tetbeam):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, tetbeam)
    sub = partitions[0:3]
    assert len(sub) == 3
    for i in range(len(sub)):
        assert sub[i] is partitions[i]
    sub = partitions[0:-1]
    assert len(sub) + 1 == len(partitions)
    for i in range(len(sub)):
        assert sub[i] is partitions[i]
    sub = partitions[0:-1:2]
    assert len(sub) == 2
    for i in range(len(sub)):
        j = i * 2
        assert sub[i] is partitions[j]
    sub = [airplane, tetbeam]
    partitions[0:2] = sub
    assert partitions[0] is airplane
    assert partitions[1] is tetbeam


def test_slice_defaults(ant, sphere, uniform, airplane, tetbeam):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, tetbeam)
    assert partitions[:] == partitions[0 : len(partitions)]


def test_partitioned_dataset_deep_copy(ant, sphere, uniform, airplane, tetbeam):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, tetbeam)
    partitions_copy = partitions.copy()
    assert partitions.n_partitions == 5 == partitions_copy.n_partitions
    assert id(partitions[0]) != id(partitions_copy[0])
    assert id(partitions[-1]) != id(partitions_copy[-1])
    for i in range(partitions_copy.n_partitions):
        assert pv.is_pyvista_dataset(partitions_copy.GetPartition(i))


def test_partitioned_dataset_shallow_copy(ant, sphere, uniform, airplane, tetbeam):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, tetbeam)
    match = 'The requested operation is not supported for PartitionedDataSetss.'
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        _ = partitions.copy(deep=False)


def test_partitioned_dataset_negative_index(ant, sphere, uniform, airplane, tetbeam):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, tetbeam)
    assert id(partitions[-1]) == id(partitions[4])
    assert id(partitions[-2]) == id(partitions[3])
    assert id(partitions[-3]) == id(partitions[2])
    assert id(partitions[-4]) == id(partitions[1])
    assert id(partitions[-5]) == id(partitions[0])
    with pytest.raises(IndexError):
        _ = partitions[-6]
    partitions[-1] = ant
    assert partitions[4] == ant
    partitions[-5] = tetbeam
    assert partitions[0] == tetbeam
    index = -6
    match = re.escape(f'index ({index}) out of range for this dataset.')
    with pytest.raises(IndexError, match=match):
        partitions[index] = uniform


def test_del_slice(sphere):
    partitions = PartitionedDataSet([sphere for i in range(10)])
    match = 'The requested operation is not supported for PartitionedDataSetss.'
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        del partitions[0:10:2]


def test_partitioned_dataset_repr(ant, sphere, uniform, airplane):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, None)
    assert partitions.n_partitions == 5
    assert partitions._repr_html_() is not None
    assert repr(partitions) is not None
    assert str(partitions) is not None


def test_partitioned_is_empty(sphere):
    assert PartitionedDataSet().is_empty
    assert not PartitionedDataSet([sphere]).is_empty
