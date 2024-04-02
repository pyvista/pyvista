import pyvista as pv
from pyvista import PartitionedDataSet


def test_pop():
    spheres = [pv.Sphere(phi_resolution=i + 3) for i in range(10)]
    partitions = PartitionedDataSet(spheres)
    assert partitions.pop() == spheres[9]
    assert spheres[9] not in partitions
    assert partitions.pop(0) == spheres[0]
    assert spheres[0] not in partitions
