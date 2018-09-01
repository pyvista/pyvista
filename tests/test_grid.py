import numpy as np
import vtkInterface as vtki
from vtkInterface import examples


def test_merge():
    from vtkInterface import examples
    beamA = vtki.UnstructuredGrid(examples.hexbeamfile)
    beamB = beamA.Copy()
    beamB.points[:, 1] += 1
    beamA.Merge(beamB)

