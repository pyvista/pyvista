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
    match = "The requested operation is not supported for PartitionedDataSetss."
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        partitions.pop()


def test_slice_defaults(ant, sphere, uniform, airplane, tetbeam):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, tetbeam)
    assert partitions[:] == partitions[0 : len(partitions)]


def test_del_slice(sphere):
    partitions = PartitionedDataSet([sphere for i in range(10)])
    match = "The requested operation is not supported for PartitionedDataSetss."
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        del partitions[0:10:2]


def test_partitioned_dataset_repr(ant, sphere, uniform, airplane):
    partitions = partitions_from_datasets(ant, sphere, uniform, airplane, None)
    assert partitions.n_partitions == 5
    assert partitions._repr_html_() is not None
    assert repr(partitions) is not None
    assert str(partitions) is not None
