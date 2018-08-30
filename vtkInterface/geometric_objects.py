"""
Provides an easy way of generating several geometric objects

CONTAINS
--------
vtkArrowSource
vtkCylinderSource
vtkSphereSource
vtkPlaneSource

NEED TO ADD
-----------
vtkConeSource
vtkCubeSource
vtkCylinderSource
vtkDiskSource
vtkLineSource
vtkRegularPolygonSource

"""
from vtkInterface import PolyData
import vtk
import numpy as np


def Translate(surf, center, direction):
    """
    Translates and orientates a mesh centered at the origin and
    facing in the x direction to a new center and direction
    """
    normx = np.array(direction)/np.linalg.norm(direction)
    # normz = np.cross(normx, np.random.random(3))
    normz = np.cross(normx, [0, 1.0, 0.0000001])
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
    """
    Create a vtk Arrow

    Parameters
    ----------
    start : np.ndarray
        Start location in [x, y, z]

    direction : list or np.ndarray
        Direction the arrow points to in [x, y, z]

    tip_length : float, optional
        Length of the tip.

    tip_radius : float, optional
        Radius of the tip.

    shaft_radius : float, optional
        Radius of the shaft.

    shaft_resolution : int, optional
        Number of faces around the shaft

    Returns
    -------
    arrow : vtkInterface.PolyData
        Arrow surface.
    """
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


def Sphere(radius=0.5, center=[0, 0, 0], direction=[0, 0, 1], theta_resolution=30,
           phi_resolution=30, start_theta=0, end_theta=360, start_phi=0, end_phi=180):
    """
    Create a vtk Sphere

    Parameters
    ----------
    radius : float, optional
        Sphere radius
    
    center : np.ndarray or list, optional
        Center in [x, y, z]

    direction : list or np.ndarray
        Direction the top of the sphere points to in [x, y, z]

    theta_resolution: int , optional
        Set the number of points in the longitude direction (ranging from 
        start_theta to end theta).

    phi_resolution : int, optional
        Set the number of points in the latitude direction (ranging from
        start_phi to end_phi).

    start_theta : float, optional
        Starting longitude angle.

    end_theta : float, optional
        Ending longitude angle.

    start_phi : float, optional
        Starting latitude angle.

    end_phi : float, optional
        Ending latitude angle.

    Returns
    -------
    sphere : vtkInterface.PolyData
        Sphere mesh.
    """
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(theta_resolution)
    sphere.SetPhiResolution(phi_resolution)
    sphere.SetStartTheta(start_theta)
    sphere.SetEndTheta(end_theta)
    sphere.SetStartPhi(start_phi)
    sphere.SetEndPhi(end_phi)
    sphere.Update()
    surf = PolyData(sphere.GetOutput())
    surf.RotateY(-90)
    Translate(surf, center, direction)
    return surf


def Plane(center=[0, 0, 0], direction=[0, 0, 1], i_size=1, j_size=1, i_resolution=10,
          j_resolution=10):
    """
    Create a plane

    Parameters
    ----------
    center : list or np.ndarray
        Location of the centroid in [x, y, z]

    direction : list or np.ndarray
        Direction cylinder points to  in [x, y, z]

    i_size : float
        Size of the plane in the i direction.

    j_size : float
        Size of the plane in the i direction.

    i_resolution : int
        Number of points on the plane in the i direction.

    j_resolution : int
        Number of points on the plane in the j direction.

    Returns
    -------
    plane : vtkInterface.PolyData
        Plane mesh

    """
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetXResolution(i_resolution)
    planeSource.SetYResolution(j_resolution)
    planeSource.Update()

    surf = PolyData(planeSource.GetOutput())

    surf.points[:, 0] *= i_size
    surf.points[:, 1] *= j_size
    surf.RotateY(-90)
    Translate(surf, center, direction)
    return surf
