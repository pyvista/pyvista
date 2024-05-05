#!/usr/bin/env python
from vtkmodules.vtkIOGeometry import vtkParticleReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

from pyvista import examples

# Create the RenderWindow, Renderer and both Actors
#
ren1 = vtkRenderer()
renWin = vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
reader = vtkParticleReader()
filename = examples.download_particles(load=False)
reader.SetFileName(filename)
reader.SetDataByteOrderToBigEndian()
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.SetScalarRange(4, 9)
mapper.SetPiece(1)
mapper.SetNumberOfPieces(2)
actor = vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(2.5)
# Add the actors to the renderer, set the background and size
#
ren1.AddActor(actor)
ren1.SetBackground(0, 0, 0)
renWin.SetSize(200, 200)
# Get handles to some useful objects
#
iren.Initialize()
renWin.Render()
# prevent the tk window from showing up then start the event loop
# --- end of script --
