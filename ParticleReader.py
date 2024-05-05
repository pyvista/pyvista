#!/usr/bin/env python
from vtkmodules.vtkIOGeometry import vtkParticleReader
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper

from pyvista import examples

reader = vtkParticleReader()
filename = examples.download_particles(load=False)
reader.SetFileName(filename)
reader.SetDataByteOrderToBigEndian()
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.SetPiece(1)
mapper.SetNumberOfPieces(2)

mesh = examples.download_particles()
# pl = pv.Plotter()
mesh.plot()
# pl.add_points(mesh.points, point_size=10.0, render_points_as_spheres=True, clim=[4, 9])
# pl.show()
