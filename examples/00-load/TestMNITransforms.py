"""Demonstrates how to create a ThinPlateSpline transform, write it to a file, read it back, and convert it to a grid transform."""

#!/usr/bin/env python
from __future__ import annotations

from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonTransforms import vtkGeneralTransform
from vtkmodules.vtkCommonTransforms import vtkThinPlateSplineTransform
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersHybrid import vtkGridTransform
from vtkmodules.vtkFiltersHybrid import vtkTransformToGrid
from vtkmodules.vtkIOMINC import vtkMNITransformReader
from vtkmodules.vtkIOMINC import vtkMNITransformWriter

VTK_DATA_ROOT = vtkGetDataRoot()

# The current directory must be writeable.
#
filename = 'mni-thinplatespline.xfm'

# next, create a ThinPlateSpline transform
p1 = vtkPoints()
p1.SetNumberOfPoints(8)
p1.SetPoint(0, 0, 0, 0)
p1.SetPoint(1, 0, 255, 0)
p1.SetPoint(2, 255, 0, 0)
p1.SetPoint(3, 255, 255, 0)
p1.SetPoint(4, 96, 96, 0)
p1.SetPoint(5, 96, 159, 0)
p1.SetPoint(6, 159, 159, 0)
p1.SetPoint(7, 159, 96, 0)

p2 = vtkPoints()
p2.SetNumberOfPoints(8)
p2.SetPoint(0, 0, 0, 0)
p2.SetPoint(1, 0, 255, 0)
p2.SetPoint(2, 255, 0, 0)
p2.SetPoint(3, 255, 255, 0)
p2.SetPoint(4, 96, 159, 0)
p2.SetPoint(5, 159, 159, 0)
p2.SetPoint(6, 159, 96, 0)
p2.SetPoint(7, 96, 96, 0)

thinPlate0 = vtkThinPlateSplineTransform()
thinPlate0.SetSourceLandmarks(p1)
thinPlate0.SetTargetLandmarks(p2)
thinPlate0.SetBasisToR2LogR()

# write the tps to a file
tpsWriter = vtkMNITransformWriter()
tpsWriter.SetFileName(filename)
tpsWriter.SetTransform(thinPlate0)
tpsWriter.Write()
# read it back
tpsReader = vtkMNITransformReader()
if tpsReader.CanReadFile(filename) != 0:
    tpsReader.SetFileName(filename)

    thinPlate = tpsReader.GetTransform()

    # make a linear transform
    linearTransform = vtkTransform()
    linearTransform.PostMultiply()
    linearTransform.Translate(-127.5, -127.5, 0)
    linearTransform.RotateZ(30)
    linearTransform.Translate(+127.5, +127.5, 0)

    # remove the linear part of the thin plate
    tpsGeneral = vtkGeneralTransform()
    tpsGeneral.SetInput(thinPlate)
    tpsGeneral.PreMultiply()
    tpsGeneral.Concatenate(linearTransform.GetInverse().GetMatrix())

    # convert the thin plate spline into a grid
    transformToGrid = vtkTransformToGrid()
    transformToGrid.SetInput(tpsGeneral)
    transformToGrid.SetGridSpacing(16, 16, 1)
    transformToGrid.SetGridOrigin(-64.5, -64.5, 0)
    transformToGrid.SetGridExtent(0, 24, 0, 24, 0, 0)
    transformToGrid.Update()

    gridTransform = vtkGridTransform()
    gridTransform.SetDisplacementGridConnection(transformToGrid.GetOutputPort())
    gridTransform.SetInterpolationModeToCubic()

    # add back the linear part
    gridGeneral = vtkGeneralTransform()
    gridGeneral.SetInput(gridTransform)
    gridGeneral.PreMultiply()
    gridGeneral.Concatenate(linearTransform.GetMatrix())

    # invert for reslice
    gridGeneral.Inverse()
    # write to a file
    gridWriter = vtkMNITransformWriter()
    gridWriter.SetFileName('mni-grid.xfm')
    gridWriter.SetComments('TestMNITransforms output transform')
    gridWriter.SetTransform(gridGeneral)
    gridWriter.Write()
