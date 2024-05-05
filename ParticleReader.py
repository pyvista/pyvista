#!/usr/bin/env python
from vtkmodules.vtkIOGeometry import vtkParticleReader
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper

import pyvista as pv
from pyvista import examples

reader = vtkParticleReader()
filename = examples.download_particles(load=False)
reader.SetFileName(filename)
reader.SetDataByteOrderToBigEndian()
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.SetScalarRange(4, 9)
mapper.SetPiece(1)
mapper.SetNumberOfPieces(2)

pl = pv.Plotter()
actor = pv.Actor(mapper)
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(10.0)
_ = pl.add_actor(actor)
pl.background_color = "black"
pl.show()
