import pytest

import pyvista as pv
from pyvista import PartitionedDataSet
from pyvista.core.errors import PartitionedDataSetsNotSupported


def test_pop():
    spheres = [pv.Sphere(phi_resolution=i + 3) for i in range(10)]
    partitions = PartitionedDataSet(spheres)
    match = "The requested operation is not supported for PartitionedDataSetss."
    with pytest.raises(PartitionedDataSetsNotSupported, match=match):
        partitions.pop()
