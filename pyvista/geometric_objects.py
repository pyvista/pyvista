"""
Provides an easy way of generating several geometric objects

CONTAINS
--------
vtkArrowSource
vtkCylinderSource
vtkSphereSource
vtkPlaneSource
vtkLineSource
vtkCubeSource
vtkConeSource
vtkDiskSource
vtkRegularPolygonSource

"""
import numpy as np
import vtk

import pyvista
from pyvista import PolyData


def translate(surf, center, direction):
    """
    Translates and orientates a mesh centered at the origin and
    facing in the x direction to a new center and direction
    """
    normx = np.array(direction)/np.linalg.norm(direction)
    normz = np.cross(normx, [0, 1.0, 0.0000001])
    normz /= np.linalg.norm(normz)
    normy = np.cross(normz, normx)

    trans = np.zeros((4, 4))
    trans[:3, 0] = normx
    trans[:3, 1] = normy
    trans[:3, 2] = normz
    trans[3, 3] = 1

    surf.transform(trans)
    surf.points += np.array(center)


def Cylinder(center=(0.,0.,0.), direction=(1.,0.,0.), radius=0.5, height=1.0,
             resolution=100, **kwargs):

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

    capping : bool, optional
        Cap cylinder ends with polygons.  Default True

    Returns
    -------
    cylinder : pyvista.PolyData
        Cylinder surface.

    Examples
    --------
    >>> import pyvista
    >>> import numpy as np
    >>> cylinder = pyvista.Cylinder(np.array([1, 2, 3]), np.array([1, 1, 1]), 1, 1)
    >>> cylinder.plot() # doctest:+SKIP

    """
    capping = kwargs.get('capping', kwargs.get('cap_ends', True))
    cylinderSource = vtk.vtkCylinderSource()
    cylinderSource.SetRadius(radius)
    cylinderSource.SetHeight(height)
    cylinderSource.SetCapping(capping)
    cylinderSource.SetResolution(resolution)
    cylinderSource.Update()
    surf = PolyData(cylinderSource.GetOutput())
    surf.rotate_z(-90)
    translate(surf, center, direction)
    return surf


def Arrow(start=(0.,0.,0.), direction=(1.,0.,0.), tip_length=0.25,
          tip_radius=0.1, shaft_radius=0.05, shaft_resolution=20):
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
    arrow : pyvista.PolyData
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
    translate(surf, start, direction)
    return surf


def Sphere(radius=0.5, center=(0, 0, 0), direction=(0, 0, 1), theta_resolution=30,
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
    sphere : pyvista.PolyData
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
    surf.rotate_y(-90)
    translate(surf, center, direction)
    return surf


def Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1, j_size=1,
          i_resolution=10, j_resolution=10):
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
    plane : pyvista.PolyData
        Plane mesh

    """
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetXResolution(i_resolution)
    planeSource.SetYResolution(j_resolution)
    planeSource.Update()

    surf = PolyData(planeSource.GetOutput())

    surf.points[:, 0] *= i_size
    surf.points[:, 1] *= j_size
    surf.rotate_y(-90)
    translate(surf, center, direction)
    return surf


def Line(pointa=(-0.5, 0., 0.), pointb=(0.5, 0., 0.), resolution=1):
    """Create a line

    Parameters
    ----------
    pointa : np.ndarray or list
        Location in [x, y, z].

    pointb : np.ndarray or list
        Location in [x, y, z].

    resolution : int
        number of pieces to divide line into
    """
    if np.array(pointa).size != 3:
        raise TypeError('Point A must be a length three tuple of floats.')
    if np.array(pointb).size != 3:
        raise TypeError('Point B must be a length three tuple of floats.')
    src = vtk.vtkLineSource()
    src.SetPoint1(*pointa)
    src.SetPoint2(*pointb)
    src.SetResolution(resolution)
    src.Update()
    return pyvista.wrap(src.GetOutput())


def Cube(center=(0., 0., 0.), x_length=1.0, y_length=1.0, z_length=1.0, bounds=None):
    """Create a cube by either specifying the center and side lengths or just
    the bounds of the cube. If ``bounds`` are given, all other arguments are
    ignored.

    Parameters
    ----------
    center : np.ndarray or list
        Center in [x, y, z].

    x_length : float
        length of the cube in the x-direction.

    y_length : float
        length of the cube in the y-direction.

    z_length : float
        length of the cube in the z-direction.

    bounds : np.ndarray or list
        Specify the bounding box of the cube. If given, all other arguments are
        ignored. ``(xMin,xMax, yMin,yMax, zMin,zMax)``
    """
    src = vtk.vtkCubeSource()
    if bounds is not None:
        if np.array(bounds).size != 6:
            raise TypeError('Bounds must be given as length 6 tuple: (xMin,xMax, yMin,yMax, zMin,zMax)')
        src.SetBounds(bounds)
    else:
        src.SetCenter(center)
        src.SetXLength(x_length)
        src.SetYLength(y_length)
        src.SetZLength(z_length)
    src.Update()
    return pyvista.wrap(src.GetOutput())


def Box(bounds=(-1.,1.,-1.,1.,-1.,1.)):
    """Creates a box with solid faces for the given bounds.

    Parameters
    ----------
    bounds : np.ndarray or list
        Specify the bounding box of the cube. If given, all other arguments are
        ignored. ``(xMin,xMax, yMin,yMax, zMin,zMax)``
    """
    return Cube(bounds=bounds)


def Cone(center=(0., 0., 0.), direction=(1., 0., 0.), height=1.0, radius=0.5,
         capping=True, angle=26.6, resolution=6):
    """Create a cone

    Parameters
    ----------
    center : np.ndarray or list
        Center in [x, y, z]. middle of the axis of the cone.

    direction : np.ndarray or list
        direction vector in [x, y, z]. orientation vector of the cone.

    height : float
        height along the cone in its specified direction.

    radius : float
        base radius of the cone

    capping : bool
        Turn on/off whether to cap the base of the cone with a polygon.

    angle : float
        The angle degrees between the axis of the cone and a generatrix.

    resolution : int
        number of facets used to represent the cone
    """
    src = vtk.vtkConeSource()
    src.SetAngle(angle)
    src.SetCapping(capping)
    src.SetCenter(center)
    src.SetHeight(height)
    src.SetRadius(radius)
    src.SetResolution(resolution)
    src.Update()
    return pyvista.wrap(src.GetOutput())


def Polygon(center=(0.,0.,0.), radius=1, normal=(0,0,1), n_sides=6):
    """
    Createa a polygonal disk with a hole in the center. The disk has zero
    height. The user can specify the inner and outer radius of the disk, and
    the radial and circumferential resolution of the polygonal representation.

    Parameters
    ----------
    center : np.ndarray or list
        Center in [x, y, z]. middle of the axis of the polygon.

    radius : float
        The radius of the polygon

    normal : np.ndarray or list
        direction vector in [x, y, z]. orientation vector of the cone.

    n_sides : int
        Number of sides of the polygon
    """
    src = vtk.vtkRegularPolygonSource()
    src.SetCenter(center)
    src.SetNumberOfSides(n_sides)
    src.SetRadius(radius)
    src.SetNormal(normal)
    src.Update()
    return pyvista.wrap(src.GetOutput())


def Disc(center=(0.,0.,0.), inner=0.25, outer=0.5, normal=(0,0,1), r_res=1,
         c_res=6):
    """
    Createa a polygonal disk with a hole in the center. The disk has zero
    height. The user can specify the inner and outer radius of the disk, and
    the radial and circumferential resolution of the polygonal representation.

    Parameters
    ----------
    center : np.ndarray or list
        Center in [x, y, z]. middle of the axis of the disc.

    inner : flaot
        The inner radius

    outer : float
        The outer radius

    normal : np.ndarray or list
        direction vector in [x, y, z]. orientation vector of the cone.

    r_res: int
        number of points in radius direction.

    r_res: int
        number of points in circumferential direction.
    """
    src = vtk.vtkDiskSource()
    src.SetInnerRadius(inner)
    src.SetOuterRadius(outer)
    src.SetRadialResolution(r_res)
    src.SetCircumferentialResolution(c_res)
    src.Update()
    return pyvista.wrap(src.GetOutput())


def Text3D(string, depth=0.5):
    """ Create 3D text from a string"""
    vec_text = vtk.vtkVectorText()
    vec_text.SetText(string)

    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(vec_text.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion();
    extrude.SetVector(0, 0, 1 )
    extrude.SetScaleFactor(depth)

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(extrude.GetOutputPort())
    tri_filter.Update()
    return pyvista.wrap(tri_filter.GetOutput())
