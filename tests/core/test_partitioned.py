import pytest

import pyvista as pv
from pyvista import PartitionedDataSet
from pyvista.core.errors import PartitionedDataSetsNotSupported


def test_reverse(sphere):
    multi = PartitionedDataSet([sphere for i in range(3)])
    multi.append(pv.Cube())
    multi.reverse()
    assert multi[0] == pv.Cube()


def test_pop():
    spheres = [pv.Sphere(phi_resolution=i + 3) for i in range(10)]
    partitions = PartitionedDataSet(spheres)
    match = "The requested operation is not supported for PartitionedDataSetss."
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        partitions.pop()
