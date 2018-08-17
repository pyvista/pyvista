"""
Provides an easy way of generating several geometric objects


CONTAINS
--------
vtkArrowSource
vtkCylinderSource


NEED TO ADD
-----------
vtkConeSource
vtkCubeSource
vtkCylinderSource
vtkDiskSource
vtkLineSource
vtkRegularPolygonSource
vtkSphereSource

"""
from vtkInterface import PolyData, CreateVectorPolyData
import vtk
import numpy as np


def Translate(surf, center, direction):
    normx = np.array(direction)/np.linalg.norm(direction)
    normz = np.cross(normx, np.random.random(3))
    normz /= np.linalg.norm(normz)
    normy = np.cross(normz, normx)

    trans = np.zeros((4, 4))
    trans[:3, 0] = normx
    trans[:3, 1] = normy
    trans[:3, 2] = normz
    trans[3, 3] = 1

    surf.ApplyTransformationInPlace(trans)
    surf.points += center


def Cylinder(center, direction, radius, height, resolution=100, cap_ends=True):
    """
    Create the surface of a cylinder.

    Parameters
    ----------
    center : list or np.ndarray
        Location of the centroid in [x, y, z]

    direction : list or np.ndarray
        Direction cylinder points to  in [x, y, z]

    radius : float
        Radius of the cylinder.

    height : float
        Height of the cylinder.

    resolution : int
        Number of points on the circular face of the cylinder.

    cap_ends : bool, optional
        Cap cylinder ends with polygons.  Default True

    Returns
    -------
    cylinder : vtkInterface.PolyData
        Cylinder surface.

    Examples
    --------
    >>> cylinder = Cylinder(1, 1, center=np.array([1, 2, 3]))
    >>> cylinder.Plot()
    
    """
    cylinderSource = vtk.vtkCylinderSource()
    cylinderSource.SetRadius(radius)
    cylinderSource.SetHeight(height)
    cylinderSource.SetCapping(cap_ends)
    cylinderSource.SetResolution(resolution)
    cylinderSource.Update()
    surf = PolyData(cylinderSource.GetOutput())
    surf.RotateZ(-90)
    Translate(surf, center, direction)
    return surf


def Arrow(start, direction, tip_length=0.25, tip_radius=0.1, shaft_radius=0.05,
          shaft_resolution=20):

    # Create arrow object
    arrow = vtk.vtkArrowSource()
    arrow.SetTipLength(tip_length)
    arrow.SetTipRadius(tip_radius)
    arrow.SetShaftRadius(shaft_radius)
    arrow.SetShaftResolution(shaft_resolution)
    arrow.Update()
    surf = PolyData(arrow.GetOutput())
    Translate(surf, start, direction)
    return surf

