import numpy as np
import vtkInterface as vtki
from vtkInterface import examples
spherefile = examples.spherefile


class TestPolyData(object):
    sphere = vtki.PolyData(spherefile)

    def test_loadfromfile(self):
        sphere = vtki.PolyData(spherefile)
        assert sphere.GetNumberOfPoints()
        assert sphere.GetNumberOfCells()
        assert hasattr(sphere, 'points')

