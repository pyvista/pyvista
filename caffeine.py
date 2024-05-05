#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOChemistry import vtkPDBReader
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
import vtkmodules.vtkRenderingFreeType
from vtkmodules.vtkRenderingLOD import vtkLODActor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

VTK_DATA_ROOT = vtkGetDataRoot()

ren1 = vtkRenderer()
ren1.SetBackground(0, 0, 0)
renWin = vtkRenderWindow()
renWin.AddRenderer(ren1)
renWin.SetSize(300, 300)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
pdb0 = vtkPDBReader()
pdb0.SetFileName(VTK_DATA_ROOT + "/Data/caffeine.pdb")
pdb0.SetHBScale(1.0)
pdb0.SetBScale(1.0)
Sphere0 = vtkSphereSource()
Sphere0.SetCenter(0, 0, 0)
Sphere0.SetRadius(1)
Sphere0.SetThetaResolution(8)
Sphere0.SetStartTheta(0)
Sphere0.SetEndTheta(360)
Sphere0.SetPhiResolution(8)
Sphere0.SetStartPhi(0)
Sphere0.SetEndPhi(180)
Glyph0 = vtkGlyph3D()
Glyph0.SetInputConnection(pdb0.GetOutputPort())
Glyph0.SetOrient(1)
Glyph0.SetColorMode(1)
# Glyph0 ScalingOn
Glyph0.SetScaleMode(2)
Glyph0.SetScaleFactor(0.25)
Glyph0.SetSourceConnection(Sphere0.GetOutputPort())
Mapper5 = vtkPolyDataMapper()
Mapper5.SetInputConnection(Glyph0.GetOutputPort())
Mapper5.UseLookupTableScalarRangeOff()
Mapper5.SetScalarVisibility(1)
Mapper5.SetScalarModeToDefault()
Actor5 = vtkLODActor()
Actor5.SetMapper(Mapper5)
Actor5.GetProperty().SetRepresentationToSurface()
Actor5.GetProperty().SetInterpolationToGouraud()
Actor5.GetProperty().SetAmbient(0.15)
Actor5.GetProperty().SetDiffuse(0.85)
Actor5.GetProperty().SetSpecular(0.1)
Actor5.GetProperty().SetSpecularPower(100)
Actor5.GetProperty().SetSpecularColor(1, 1, 1)
Actor5.GetProperty().SetColor(1, 1, 1)
Actor5.SetNumberOfCloudPoints(30000)
ren1.AddActor(Actor5)
Tuber0 = vtkTubeFilter()
Tuber0.SetInputConnection(pdb0.GetOutputPort())
Tuber0.SetNumberOfSides(8)
Tuber0.SetCapping(0)
Tuber0.SetRadius(0.2)
Tuber0.SetVaryRadius(0)
Tuber0.SetRadiusFactor(10)
Mapper7 = vtkPolyDataMapper()
Mapper7.SetInputConnection(Tuber0.GetOutputPort())
Mapper7.UseLookupTableScalarRangeOff()
Mapper7.SetScalarVisibility(1)
Mapper7.SetScalarModeToDefault()
Actor7 = vtkLODActor()
Actor7.SetMapper(Mapper7)
Actor7.GetProperty().SetRepresentationToSurface()
Actor7.GetProperty().SetInterpolationToGouraud()
Actor7.GetProperty().SetAmbient(0.15)
Actor7.GetProperty().SetDiffuse(0.85)
Actor7.GetProperty().SetSpecular(0.1)
Actor7.GetProperty().SetSpecularPower(100)
Actor7.GetProperty().SetSpecularColor(1, 1, 1)
Actor7.GetProperty().SetColor(1, 1, 1)
ren1.AddActor(Actor7)
# enable user interface interactor
# iren SetUserMethod {wm deiconify .vtkInteract}
iren.Initialize()
# prevent the tk window from showing up then start the event loop
# --- end of script --
