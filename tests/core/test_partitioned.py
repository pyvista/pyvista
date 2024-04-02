import pyvista as pv
from pyvista import PartitionedDataSet


def test_pop():
    spheres = [pv.Sphere(phi_resolution=i + 3) for i in range(10)]
    multi = PartitionedDataSet(spheres)
    assert multi.pop() == spheres[9]
    assert spheres[9] not in multi
    assert multi.pop(0) == spheres[0]
    assert spheres[0] not in multi
