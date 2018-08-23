import numpy as np
import vtkInterface as vtki
from vtkInterface import examples
spherefile = examples.spherefile


class TestPolyData(object):
    sphere = vtki.Sphere()
    
    def test_loadfromfile(self):
        sphere = vtki.PolyData(spherefile)
        assert sphere.GetNumberOfPoints()
        assert sphere.GetNumberOfCells()
        assert hasattr(sphere, 'points')

    def test_raytrace(self):
        points, ind = self.sphere.RayTrace([0, 0, 0], [1, 1, 1])
        assert np.any(points)
        assert np.any(ind)
