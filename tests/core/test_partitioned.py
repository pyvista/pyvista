import pytest

import pyvista as pv
from pyvista import PartitionedDataSet
from pyvista.core.errors import PartitionedDataSetsNotSupported


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
