"""Provides an easy way of generating several geometric objects.

**CONTAINS**
vtkArrowSource
vtkCylinderSource
vtkSphereSource
vtkPlaneSource
vtkLineSource
vtkCubeSource
vtkConeSource
vtkDiskSource
vtkRegularPolygonSource
vtkPyramid
vtkPlatonicSolidSource
vtkSuperquadricSource

as well as some pure-python helpers.

"""
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning

from .arrays import _coerce_pointslike_arg
from .helpers import wrap
from .misc import check_valid_vector

NORMALS = {
    'x': [1, 0, 0],
    'y': [0, 1, 0],
    'z': [0, 0, 1],
    '-x': [-1, 0, 0],
    '-y': [0, -1, 0],
    '-z': [0, 0, -1],
}


def translate(surf, center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)):
    """Translate and orient a mesh to a new center and direction.

    By default, the input mesh is considered centered at the origin
    and facing in the x direction.

    Parameters
    ----------
    surf : pyvista.core.pointset.PolyData
        Mesh to be translated and oriented.
    center : tuple, optional, default: (0.0, 0.0, 0.0)
        Center point to which the mesh should be translated.
    direction : tuple, optional, default: (1.0, 0.0, 0.0)
        Direction vector along which the mesh should be oriented.

    """
    normx = np.array(direction) / np.linalg.norm(direction)
    normy_temp = [0.0, 1.0, 0.0]

    # Adjust normy if collinear with normx since cross-product will
    # be zero otherwise
    if np.allclose(normx, [0, 1, 0]):
        normy_temp = [-1.0, 0.0, 0.0]
    elif np.allclose(normx, [0, -1, 0]):
        normy_temp = [1.0, 0.0, 0.0]

    normz = np.cross(normx, normy_temp)
    normz /= np.linalg.norm(normz)
    normy = np.cross(normz, normx)

    trans = np.zeros((4, 4))
    trans[:3, 0] = normx
    trans[:3, 1] = normy
    trans[:3, 2] = normz
    trans[3, 3] = 1

    surf.transform(trans)
    if not np.allclose(center, [0.0, 0.0, 0.0]):
        surf.points += np.array(center)


def Cylinder(
    center=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    radius=0.5,
    height=1.0,
    resolution=100,
    capping=True,
):
    """Create the surface of a cylinder.

    See also :func:`pyvista.CylinderStructured`.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Location of the centroid in ``[x, y, z]``.

    direction : sequence[float], default: (1.0, 0.0, 0.0)
        Direction cylinder points to  in ``[x, y, z]``.

    radius : float, default: 0.5
        Radius of the cylinder.

    height : float, default: 1.0
        Height of the cylinder.

    resolution : int, default: 100
        Number of points on the circular face of the cylinder.

    capping : bool, default: True
        Cap cylinder ends with polygons.

    Returns
    -------
    pyvista.PolyData
        Cylinder surface.

    Examples
    --------
    >>> import pyvista
    >>> import numpy as np
    >>> cylinder = pyvista.Cylinder(
    ...     center=[1, 2, 3], direction=[1, 1, 1], radius=1, height=2
    ... )
    >>> cylinder.plot(show_edges=True, line_width=5, cpos='xy')
    """
    cylinderSource = _vtk.vtkCylinderSource()
    cylinderSource.SetRadius(radius)
    cylinderSource.SetHeight(height)
    cylinderSource.SetCapping(capping)
    cylinderSource.SetResolution(resolution)
    cylinderSource.Update()
    surf = wrap(cylinderSource.GetOutput())
    surf.rotate_z(-90, inplace=True)
    translate(surf, center, direction)
    return surf


def CylinderStructured(
    radius=0.5,
    height=1.0,
    center=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    theta_resolution=32,
    z_resolution=10,
):
    """Create a cylinder mesh as a :class:`pyvista.StructuredGrid`.

    The end caps are left open. This can create a surface mesh if a single
    value for the ``radius`` is given or a 3D mesh if multiple radii are given
    as a list/array in the ``radius`` argument.

    Parameters
    ----------
    radius : float | sequence[float], default: 0.5
        Radius of the cylinder. If a sequence, then describes the
        radial coordinates of the cells as a range of values as
        specified by the ``radius``.

    height : float, default: 1.0
        Height of the cylinder along its Z-axis.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Location of the centroid in ``[x, y, z]``.

    direction : sequence[float], default: (1.0, 0.0, 0.0)
        Direction cylinder Z-axis in ``[x, y, z]``.

    theta_resolution : int, default: 32
        Number of points on the circular face of the cylinder.
        Ignored if ``radius`` is an iterable.

    z_resolution : int, default: 10
        Number of points along the height (Z-axis) of the cylinder.

    Returns
    -------
    pyvista.StructuredGrid
        Structured cylinder.

    Notes
    -----
    .. versionchanged:: 0.38.0
       Prior to version 0.38, this method had incorrect results, producing
       inconsistent number of points on the circular face of the cylinder.

    Examples
    --------
    Default structured cylinder

    >>> import pyvista
    >>> mesh = pyvista.CylinderStructured()
    >>> mesh.plot(show_edges=True)

    Structured cylinder with an inner radius of 1, outer of 2, with 5
    segments.

    >>> import numpy as np
    >>> mesh = pyvista.CylinderStructured(radius=np.linspace(1, 2, 5))
    >>> mesh.plot(show_edges=True)

    """
    # Define grid in polar coordinates
    r = np.array([radius]).ravel()
    nr = len(r)
    theta = np.linspace(0, 2 * np.pi, num=theta_resolution + 1)
    radius_matrix, theta_matrix = np.meshgrid(r, theta)

    # Transform to cartesian space
    X = radius_matrix * np.cos(theta_matrix)
    Y = radius_matrix * np.sin(theta_matrix)

    # Make all the nodes in the grid
    xx = np.array([X] * z_resolution).ravel()
    yy = np.array([Y] * z_resolution).ravel()
    dz = height / (z_resolution - 1)
    zz = np.empty(yy.size)
    zz = np.full((X.size, z_resolution), dz)
    zz *= np.arange(z_resolution)
    zz = zz.ravel(order='f')

    # Create the grid
    grid = pyvista.StructuredGrid()
    grid.points = np.c_[xx, yy, zz]
    grid.dimensions = [nr, theta_resolution + 1, z_resolution]

    # Orient properly in user direction
    vx = np.array([0.0, 0.0, 1.0])
    if not np.allclose(vx, direction):
        direction /= np.linalg.norm(direction)
        vx -= vx.dot(direction) * direction
        vx /= np.linalg.norm(vx)
        vy = np.cross(direction, vx)
        rmtx = np.array([vx, vy, direction])
        grid.points = grid.points.dot(rmtx)

    # Translate to given center
    grid.points -= np.array(grid.center)
    grid.points += np.array(center)
    return grid


def Arrow(
    start=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    tip_length=0.25,
    tip_radius=0.1,
    tip_resolution=20,
    shaft_radius=0.05,
    shaft_resolution=20,
    scale=None,
):
    """Create an arrow.

    Parameters
    ----------
    start : sequence[float], default: (0.0, 0.0, 0.0)
        Start location in ``[x, y, z]``.

    direction : sequence[float], default: (1.0, 0.0, 0.0)
        Direction the arrow points to in ``[x, y, z]``.

    tip_length : float, default: 0.25
        Length of the tip.

    tip_radius : float, default: 0.1
        Radius of the tip.

    tip_resolution : int, default: 20
        Number of faces around the tip.

    shaft_radius : float, default: 0.05
        Radius of the shaft.

    shaft_resolution : int, default: 20
        Number of faces around the shaft.

    scale : float | str, optional
        Scale factor of the entire object, defaults to a scale of 1.
        ``'auto'`` scales to length of direction array.

    Returns
    -------
    pyvista.PolyData
        Arrow mesh.

    Examples
    --------
    Plot a default arrow.

    >>> import pyvista
    >>> mesh = pyvista.Arrow()
    >>> mesh.plot(show_edges=True)

    """
    # Create arrow object
    arrow = _vtk.vtkArrowSource()
    arrow.SetTipLength(tip_length)
    arrow.SetTipRadius(tip_radius)
    arrow.SetTipResolution(tip_resolution)
    arrow.SetShaftRadius(shaft_radius)
    arrow.SetShaftResolution(shaft_resolution)
    arrow.Update()
    surf = wrap(arrow.GetOutput())

    if scale == 'auto':
        scale = float(np.linalg.norm(direction))
    if isinstance(scale, (float, int)):
        surf.points *= scale
    elif scale is not None:
        raise TypeError("Scale must be either float, int or 'auto'.")

    translate(surf, start, direction)
    return surf


def Sphere(
    radius=0.5,
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    theta_resolution=30,
    phi_resolution=30,
    start_theta=0.0,
    end_theta=360.0,
    start_phi=0.0,
    end_phi=180.0,
):
    """Create a sphere.

    Parameters
    ----------
    radius : float, default: 0.5
        Sphere radius.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``.

    direction : sequence[float], default: (0.0, 0.0, 1.0)
        Direction vector in ``[x, y, z]`` pointing from ``center`` to
        the sphere's north pole at zero degrees latitude.

    theta_resolution : int, default: 30
        Set the number of points in the longitude direction (ranging
        from ``start_theta`` to ``end_theta``).

    phi_resolution : int, default: 30
        Set the number of points in the latitude direction (ranging from
        ``start_phi`` to ``end_phi``).

    start_theta : float, default: 0.0
        Starting longitude angle in degrees ``[0, 360]``.

    end_theta : float, default: 360.0
        Ending longitude angle in degrees ``[0, 360]``.

    start_phi : float, default: 0.0
        Starting latitude angle in degrees ``[0, 180]``.

    end_phi : float, default: 180.0
        Ending latitude angle in degrees ``[0, 180]``.

    Returns
    -------
    pyvista.PolyData
        Sphere mesh.

    See Also
    --------
    pyvista.Icosphere

    Examples
    --------
    Create a sphere using default parameters.

    >>> import pyvista
    >>> sphere = pyvista.Sphere()
    >>> sphere.plot(show_edges=True)

    Create a quarter sphere by setting ``end_theta``.

    >>> sphere = pyvista.Sphere(end_theta=90)
    >>> out = sphere.plot(show_edges=True)

    Create a hemisphere by setting ``end_phi``.

    >>> sphere = pyvista.Sphere(end_phi=90)
    >>> out = sphere.plot(show_edges=True)

    """
    sphere = _vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(theta_resolution)
    sphere.SetPhiResolution(phi_resolution)
    sphere.SetStartTheta(start_theta)
    sphere.SetEndTheta(end_theta)
    sphere.SetStartPhi(start_phi)
    sphere.SetEndPhi(end_phi)
    sphere.Update()
    surf = wrap(sphere.GetOutput())
    surf.rotate_y(90, inplace=True)
    translate(surf, center, direction)
    return surf


def Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    i_size=1,
    j_size=1,
    i_resolution=10,
    j_resolution=10,
):
    """Create a plane.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Location of the centroid in ``[x, y, z]``.

    direction : sequence[float], default: (0.0, 0.0, 1.0)
        Direction of the plane's normal in ``[x, y, z]``.

    i_size : float, default: 1.0
        Size of the plane in the i direction.

    j_size : float, default: 1.0
        Size of the plane in the j direction.

    i_resolution : int, default: 10
        Number of points on the plane in the i direction.

    j_resolution : int, default: 10
        Number of points on the plane in the j direction.

    Returns
    -------
    pyvista.PolyData
        Plane mesh.

    Examples
    --------
    Create a default plane.

    >>> import pyvista
    >>> mesh = pyvista.Plane()
    >>> mesh.point_data.clear()
    >>> mesh.plot(show_edges=True)
    """
    planeSource = _vtk.vtkPlaneSource()
    planeSource.SetXResolution(i_resolution)
    planeSource.SetYResolution(j_resolution)
    planeSource.Update()

    surf = wrap(planeSource.GetOutput())

    surf.points[:, 0] *= i_size
    surf.points[:, 1] *= j_size
    surf.rotate_y(90, inplace=True)
    translate(surf, center, direction)
    return surf


def Line(pointa=(-0.5, 0.0, 0.0), pointb=(0.5, 0.0, 0.0), resolution=1):
    """Create a line.

    Parameters
    ----------
    pointa : sequence[float], default: (-0.5, 0.0, 0.0)
        Location in ``[x, y, z]``.

    pointb : sequence[float], default: (0.5, 0.0, 0.0)
        Location in ``[x, y, z]``.

    resolution : int, default: 1
        Number of pieces to divide line into.

    Returns
    -------
    pyvista.PolyData
        Line mesh.

    Examples
    --------
    Create a line between ``(0, 0, 0)`` and ``(0, 0, 1)``.

    >>> import pyvista
    >>> mesh = pyvista.Line((0, 0, 0), (0, 0, 1))
    >>> mesh.plot(color='k', line_width=10)

    """
    if resolution <= 0:
        raise ValueError('Resolution must be positive')
    if np.array(pointa).size != 3:
        raise TypeError('Point A must be a length three tuple of floats.')
    if np.array(pointb).size != 3:
        raise TypeError('Point B must be a length three tuple of floats.')
    src = _vtk.vtkLineSource()
    src.SetPoint1(*pointa)
    src.SetPoint2(*pointb)
    src.SetResolution(resolution)
    src.Update()
    line = wrap(src.GetOutput())
    # Compute distance of every point along line
    compute = lambda p0, p1: np.sqrt(np.sum((p1 - p0) ** 2, axis=1))
    distance = compute(np.array(pointa), line.points)
    line['Distance'] = distance
    return line


def MultipleLines(points=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]):
    """Create multiple lines.

    Parameters
    ----------
    points : array_like[float], default: [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        List of points defining a broken line.

    Returns
    -------
    pyvista.PolyData
        Line mesh.

    Examples
    --------
    Create a multiple lines between ``(0, 0, 0)``, ``(1, 1, 1)`` and ``(0, 0, 1)``.

    >>> import pyvista
    >>> mesh = pyvista.MultipleLines(
    ...     points=[[0, 0, 0], [1, 1, 1], [0, 0, 1]]
    ... )
    >>> plotter = pyvista.Plotter()
    >>> actor = plotter.add_mesh(mesh, color='k', line_width=10)
    >>> plotter.camera.azimuth = 45
    >>> plotter.camera.zoom(0.8)
    >>> plotter.show()
    """
    points, _ = _coerce_pointslike_arg(points)
    src = _vtk.vtkLineSource()
    if not (len(points) >= 2):
        raise ValueError('>=2 points need to define multiple lines.')
    src.SetPoints(pyvista.vtk_points(points))
    src.Update()
    multiple_lines = wrap(src.GetOutput())
    return multiple_lines


def Tube(pointa=(-0.5, 0.0, 0.0), pointb=(0.5, 0.0, 0.0), resolution=1, radius=1.0, n_sides=15):
    """Create a tube.

    Parameters
    ----------
    pointa : sequence[float], default: (-0.5, 0.0, 0.0)
        Location in ``[x, y, z]``.

    pointb : sequence[float], default: (0.5, 0.0, 0.0)
        Location in ``[x, y, z]``.

    resolution : int, default: 1
        Number of pieces to divide tube into.

    radius : float, default: 1.0
        Minimum tube radius (minimum because the tube radius may vary).

    n_sides : int, default: 15
        Number of sides for the tube.

    Returns
    -------
    pyvista.PolyData
        Tube mesh.

    Examples
    --------
    Create a tube between ``(0, 0, 0)`` and ``(0, 0, 1)``.

    >>> import pyvista
    >>> mesh = pyvista.Tube((0, 0, 0), (0, 0, 1))
    >>> mesh.plot()

    """
    if resolution <= 0:
        raise ValueError('Resolution must be positive.')
    if np.array(pointa).size != 3:
        raise TypeError('Point A must be a length three tuple of floats.')
    if np.array(pointb).size != 3:
        raise TypeError('Point B must be a length three tuple of floats.')
    line_src = _vtk.vtkLineSource()
    line_src.SetPoint1(*pointa)
    line_src.SetPoint2(*pointb)
    line_src.SetResolution(resolution)
    line_src.Update()

    if n_sides < 3:
        raise ValueError('Number of sides `n_sides` must be >= 3')
    tube_filter = _vtk.vtkTubeFilter()
    tube_filter.SetInputConnection(line_src.GetOutputPort())
    tube_filter.SetRadius(radius)
    tube_filter.SetNumberOfSides(n_sides)
    tube_filter.Update()

    return wrap(tube_filter.GetOutput())


def Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0, bounds=None, clean=True):
    """Create a cube.

    It's possible to specify either the center and side lengths or
    just the bounds of the cube. If ``bounds`` are given, all other
    arguments are ignored.

    .. versionchanged:: 0.33.0
        The cube is created using ``vtk.vtkCubeSource``. For
        compatibility with :func:`pyvista.PlatonicSolid`, face indices
        are also added as cell data. For full compatibility with
        :func:`PlatonicSolid() <pyvista.PlatonicSolid>`, one has to
        use ``x_length = y_length = z_length = 2 * radius / 3**0.5``.
        The cube points are also cleaned by default now, leaving only
        the 8 corners and a watertight (manifold) mesh.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``.

    x_length : float, default: 1.0
        Length of the cube in the x-direction.

    y_length : float, default: 1.0
        Length of the cube in the y-direction.

    z_length : float, default: 1.0
        Length of the cube in the z-direction.

    bounds : sequence[float], optional
        Specify the bounding box of the cube. If given, all other size
        arguments are ignored. ``(xMin, xMax, yMin, yMax, zMin, zMax)``.

    clean : bool, default: True
        Whether to clean the raw points of the mesh, making the cube
        manifold. Note that this will degrade the texture coordinates
        that come with the mesh, so if you plan to map a texture on
        the cube, consider setting this to ``False``.

        .. versionadded:: 0.33.0

    Returns
    -------
    pyvista.PolyData
        Mesh of the cube.

    Examples
    --------
    Create a default cube.

    >>> import pyvista
    >>> mesh = pyvista.Cube()
    >>> mesh.plot(show_edges=True, line_width=5)

    """
    src = _vtk.vtkCubeSource()
    if bounds is not None:
        if np.array(bounds).size != 6:
            raise TypeError(
                'Bounds must be given as length 6 tuple: (xMin, xMax, yMin, yMax, zMin, zMax)'
            )
        src.SetBounds(bounds)
    else:
        src.SetCenter(center)
        src.SetXLength(x_length)
        src.SetYLength(y_length)
        src.SetZLength(z_length)
    src.Update()
    cube = wrap(src.GetOutput())

    # add face index data for compatibility with PlatonicSolid
    # but make it inactive for backwards compatibility
    cube.cell_data.set_array([1, 4, 0, 3, 5, 2], 'FaceIndex')

    # clean duplicate points
    if clean:
        cube.clean(inplace=True)

    return cube


def Box(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), level=0, quads=True):
    """Create a box with solid faces for the given bounds.

    Parameters
    ----------
    bounds : sequence[float], default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        Specify the bounding box of the cube.
        ``(xMin, xMax, yMin, yMax, zMin, zMax)``.

    level : int, default: 0
        Level of subdivision of the faces.

    quads : bool, default: True
        Flag to tell the source to generate either a quad or two
        triangle for a set of four points.

    Returns
    -------
    pyvista.PolyData
        Mesh of the box.

    Examples
    --------
    Create a box with subdivision ``level=2``.

    >>> import pyvista
    >>> mesh = pyvista.Box(level=2)
    >>> mesh.plot(show_edges=True)

    """
    if np.array(bounds).size != 6:
        raise TypeError(
            'Bounds must be given as length 6 tuple: (xMin, xMax, yMin, yMax, zMin, zMax)'
        )
    src = _vtk.vtkTessellatedBoxSource()
    src.SetLevel(level)
    src.SetQuads(quads)
    src.SetBounds(bounds)
    src.Update()
    return wrap(src.GetOutput())


def Cone(
    center=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    height=1.0,
    radius=None,
    capping=True,
    angle=None,
    resolution=6,
):
    """Create a cone.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``. Axis of the cone passes through this
        point.

    direction : sequence[float], default: (1.0, 0.0, 0.0)
        Direction vector in ``[x, y, z]``. Orientation vector of the
        cone.

    height : float, default: 1.0
        Height along the cone in its specified direction.

    radius : float, optional
        Base radius of the cone.

    capping : bool, optional
        Enable or disable the capping the base of the cone with a
        polygon.

    angle : float, optional
        The angle in degrees between the axis of the cone and a
        generatrix.

    resolution : int, default: 6
        Number of facets used to represent the cone.

    Returns
    -------
    pyvista.PolyData
        Cone mesh.

    Examples
    --------
    Create a default Cone.

    >>> import pyvista
    >>> mesh = pyvista.Cone()
    >>> mesh.plot(show_edges=True, line_width=5)
    """
    src = _vtk.vtkConeSource()
    src.SetCapping(capping)
    src.SetDirection(direction)
    src.SetCenter(center)
    src.SetHeight(height)
    if angle and radius:
        raise ValueError("Both radius and angle specified. They are mutually exclusive.")
    elif angle and not radius:
        src.SetAngle(angle)
    elif not angle and radius:
        src.SetRadius(radius)
    elif not angle and not radius:
        src.SetRadius(0.5)
    src.SetResolution(resolution)
    src.Update()
    return wrap(src.GetOutput())


def Polygon(center=(0.0, 0.0, 0.0), radius=1.0, normal=(0.0, 0.0, 1.0), n_sides=6, fill=True):
    """Create a polygon.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``. Central axis of the polygon passes
        through this point.

    radius : float, default: 1.0
        The radius of the polygon.

    normal : sequence[float], default: (0.0, 0.0, 1.0)
        Direction vector in ``[x, y, z]``. Orientation vector of the polygon.

    n_sides : int, default: 6
        Number of sides of the polygon.

    fill : bool, default: True
        Enable or disable producing filled polygons.

    Returns
    -------
    pyvista.PolyData
        Mesh of the polygon.

    Examples
    --------
    Create an 8 sided polygon.

    >>> import pyvista
    >>> mesh = pyvista.Polygon(n_sides=8)
    >>> mesh.plot(show_edges=True, line_width=5)

    """
    src = _vtk.vtkRegularPolygonSource()
    src.SetGeneratePolygon(fill)
    src.SetCenter(center)
    src.SetNumberOfSides(n_sides)
    src.SetRadius(radius)
    src.SetNormal(normal)
    src.Update()
    return wrap(src.GetOutput())


def Disc(center=(0.0, 0.0, 0.0), inner=0.25, outer=0.5, normal=(0.0, 0.0, 1.0), r_res=1, c_res=6):
    """Create a polygonal disk with a hole in the center.

    The disk has zero height. The user can specify the inner and outer
    radius of the disk, and the radial and circumferential resolution
    of the polygonal representation.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``. Middle of the axis of the disc.

    inner : float, default: 0.25
        The inner radius.

    outer : float, default: 0.5
        The outer radius.

    normal : sequence[float], default: (0.0, 0.0, 1.0)
        Direction vector in ``[x, y, z]``. Orientation vector of the disc.

    r_res : int, default: 1
        Number of points in radial direction.

    c_res : int, default: 6
        Number of points in circumferential direction.

    Returns
    -------
    pyvista.PolyData
        Disk mesh.

    Examples
    --------
    Create a disc with 50 points in the circumferential direction.

    >>> import pyvista
    >>> mesh = pyvista.Disc(c_res=50)
    >>> mesh.plot(show_edges=True, line_width=5)

    """
    src = _vtk.vtkDiskSource()
    src.SetInnerRadius(inner)
    src.SetOuterRadius(outer)
    src.SetRadialResolution(r_res)
    src.SetCircumferentialResolution(c_res)
    src.Update()
    normal = np.array(normal)
    center = np.array(center)
    surf = wrap(src.GetOutput())
    surf.rotate_y(90, inplace=True)
    translate(surf, center, normal)
    return surf


def Text3D(string, depth=0.5):
    """Create 3D text from a string.

    Parameters
    ----------
    string : str
        String to generate 3D text from.

    depth : float, default: 0.5
        Depth of the text.

    Returns
    -------
    pyvista.PolyData
        3D text mesh.

    Examples
    --------
    >>> import pyvista
    >>> text_mesh = pyvista.Text3D('PyVista')
    >>> text_mesh.plot(cpos='xy')
    """
    from vtkmodules.vtkRenderingFreeType import vtkVectorText

    vec_text = vtkVectorText()
    vec_text.SetText(string)

    extrude = _vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(vec_text.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0, 0, 1)
    extrude.SetScaleFactor(depth)

    tri_filter = _vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(extrude.GetOutputPort())
    tri_filter.Update()
    return wrap(tri_filter.GetOutput())


def Wavelet(
    extent=(-10, 10, -10, 10, -10, 10),
    center=(0.0, 0.0, 0.0),
    maximum=255.0,
    x_freq=60.0,
    y_freq=30.0,
    z_freq=40.0,
    x_mag=10.0,
    y_mag=18.0,
    z_mag=5.0,
    std=0.5,
    subsample_rate=1,
):
    """Create a wavelet.

    Produces images with pixel values determined by
    ``Maximum*Gaussian*x_mag*sin(x_freq*x)*sin(y_freq*y)*cos(z_freq*z)``

    Values are float scalars on point data with name ``"RTData"``.

    Parameters
    ----------
    extent : sequence[int], default: (-10, 10, -10, 10, -10, 10)
        Set/Get the extent of the whole output image.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center of the wavelet.

    maximum : float, default: 255.0
        Maximum of the wavelet function.

    x_freq : float, default: 60.0
        Natural frequency in the x direction.

    y_freq : float, default: 30.0
        Natural frequency in the y direction.

    z_freq : float, default: 40.0
        Natural frequency in the z direction.

    x_mag : float, default: 10.0
        Magnitude in the x direction.

    y_mag : float, default: 18.0
        Magnitude in the y direction.

    z_mag : float, default: 5.0
        Magnitude in the z direction.

    std : float, default: 0.5
        Standard deviation.

    subsample_rate : int, default: 1
        The sub-sample rate.

    Returns
    -------
    pyvista.PolyData
        Wavelet mesh.

    Examples
    --------
    >>> import pyvista
    >>> wavelet = pyvista.Wavelet(
    ...     extent=(0, 50, 0, 50, 0, 10),
    ...     x_freq=20,
    ...     y_freq=10,
    ...     z_freq=1,
    ...     x_mag=100,
    ...     y_mag=100,
    ...     z_mag=1000,
    ... )
    >>> wavelet.plot(show_scalar_bar=False)

    Extract lower valued cells of the wavelet and create a surface from it.

    >>> thresh = wavelet.threshold(800).extract_surface()
    >>> thresh.plot(show_scalar_bar=False)

    Smooth it to create "waves"

    >>> waves = thresh.smooth(n_iter=100, relaxation_factor=0.1)
    >>> waves.plot(color='white', smooth_shading=True, show_edges=True)

    """
    wavelet_source = _vtk.vtkRTAnalyticSource()
    wavelet_source.SetWholeExtent(*extent)
    wavelet_source.SetCenter(center)
    wavelet_source.SetMaximum(maximum)
    wavelet_source.SetXFreq(x_freq)
    wavelet_source.SetYFreq(y_freq)
    wavelet_source.SetZFreq(z_freq)
    wavelet_source.SetXMag(x_mag)
    wavelet_source.SetYMag(y_mag)
    wavelet_source.SetZMag(z_mag)
    wavelet_source.SetStandardDeviation(std)
    wavelet_source.SetSubsampleRate(subsample_rate)
    wavelet_source.Update()
    return wrap(wavelet_source.GetOutput())


def CircularArc(pointa, pointb, center, resolution=100, negative=False):
    """Create a circular arc defined by two endpoints and a center.

    The number of segments composing the polyline is controlled by
    setting the object resolution.

    Parameters
    ----------
    pointa : sequence[float]
        Position of the first end point.

    pointb : sequence[float]
        Position of the other end point.

    center : sequence[float]
        Center of the circle that defines the arc.

    resolution : int, default: 100
        The number of segments of the polyline that draws the arc.
        Resolution of 1 will just create a line.

    negative : bool, default: False
        By default the arc spans the shortest angular sector between
        ``pointa`` and ``pointb``.

        By setting this to ``True``, the longest angular sector is
        used instead (i.e. the negative coterminal angle to the
        shortest one).

    Returns
    -------
    pyvista.PolyData
        Circular arc mesh.

    Examples
    --------
    Create a quarter arc centered at the origin in the xy plane.

    >>> import pyvista
    >>> arc = pyvista.CircularArc([-1, 0, 0], [0, 1, 0], [0, 0, 0])
    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(arc, color='k', line_width=10)
    >>> _ = pl.show_bounds(location='all', font_size=30, use_2d=True)
    >>> _ = pl.view_xy()
    >>> pl.show()
    """
    check_valid_vector(pointa, 'pointa')
    check_valid_vector(pointb, 'pointb')
    check_valid_vector(center, 'center')
    if not np.isclose(
        np.linalg.norm(np.array(pointa) - np.array(center)),
        np.linalg.norm(np.array(pointb) - np.array(center)),
    ):
        raise ValueError("pointa and pointb are not equidistant from center")

    # fix half-arc bug: if a half arc travels directly through the
    # center point, it becomes a line
    pointb = list(pointb)
    pointb[0] -= 1e-10
    pointb[1] -= 1e-10

    arc = _vtk.vtkArcSource()
    arc.SetPoint1(*pointa)
    arc.SetPoint2(*pointb)
    arc.SetCenter(*center)
    arc.SetResolution(resolution)
    arc.SetNegative(negative)

    arc.Update()
    angle = np.deg2rad(arc.GetAngle())
    arc = wrap(arc.GetOutput())
    # Compute distance of every point along circular arc
    center = np.array(center).ravel()
    radius = np.sqrt(np.sum((arc.points[0] - center) ** 2, axis=0))
    angles = np.arange(0.0, 1.0 + 1.0 / resolution, 1.0 / resolution) * angle
    arc['Distance'] = radius * angles
    return arc


def CircularArcFromNormal(center, resolution=100, normal=None, polar=None, angle=None):
    """Create a circular arc defined by normal to the plane of the arc, and an angle.

    The number of segments composing the polyline is controlled by
    setting the object resolution.

    Parameters
    ----------
    center : sequence[float]
        Center of the circle that defines the arc.

    resolution : int, default: 100
        The number of segments of the polyline that draws the arc.
        Resolution of 1 will just create a line.

    normal : sequence[float], optional
        The normal vector to the plane of the arc.  By default it
        points in the positive Z direction.

    polar : sequence[float], optional
        Starting point of the arc in polar coordinates.  By default it
        is the unit vector in the positive x direction.

    angle : float, optional
        Arc length (in degrees) beginning at the polar vector.  The
        direction is counterclockwise.  By default it is 90.

    Returns
    -------
    pyvista.PolyData
        Circular arc mesh.

    Examples
    --------
    Quarter arc centered at the origin in the xy plane.

    >>> import pyvista
    >>> normal = [0, 0, 1]
    >>> polar = [-1, 0, 0]
    >>> arc = pyvista.CircularArcFromNormal(
    ...     [0, 0, 0], normal=normal, polar=polar
    ... )
    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(arc, color='k', line_width=10)
    >>> _ = pl.show_bounds(location='all', font_size=30, use_2d=True)
    >>> _ = pl.view_xy()
    >>> pl.show()
    """
    check_valid_vector(center, 'center')
    if normal is None:
        normal = [0, 0, 1]
    if polar is None:
        polar = [1, 0, 0]
    if angle is None:
        angle = 90.0

    arc = _vtk.vtkArcSource()
    arc.SetCenter(*center)
    arc.SetResolution(resolution)
    arc.UseNormalAndAngleOn()
    check_valid_vector(normal, 'normal')
    arc.SetNormal(*normal)
    check_valid_vector(polar, 'polar')
    arc.SetPolarVector(*polar)
    arc.SetAngle(angle)
    arc.Update()
    angle = np.deg2rad(arc.GetAngle())
    arc = wrap(arc.GetOutput())
    # Compute distance of every point along circular arc
    center = np.array(center)
    radius = np.sqrt(np.sum((arc.points[0] - center) ** 2, axis=0))
    angles = np.linspace(0.0, angle, resolution + 1)
    arc['Distance'] = radius * angles
    return arc


def Pyramid(points=None):
    """Create a pyramid defined by 5 points.

    Parameters
    ----------
    points : array_like[float], optional
        Points of the pyramid.  Points are ordered such that the first
        four points are the four counterclockwise points on the
        quadrilateral face, and the last point is the apex.

        Defaults to pyramid in example.

    Returns
    -------
    pyvista.UnstructuredGrid
        Unstructured grid containing a single pyramid cell.

    Examples
    --------
    >>> import pyvista
    >>> pointa = [1.0, 1.0, 0.0]
    >>> pointb = [-1.0, 1.0, 0.0]
    >>> pointc = [-1.0, -1.0, 0.0]
    >>> pointd = [1.0, -1.0, 0.0]
    >>> pointe = [0.0, 0.0, 1.608]
    >>> pyramid = pyvista.Pyramid([pointa, pointb, pointc, pointd, pointe])
    >>> pyramid.plot(show_edges=True, line_width=5)
    """
    if points is None:
        points = [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, (4 - 2**0.5) ** 0.5],
        ]

    if len(points) != 5:
        raise TypeError('Points must be given as length 5 np.ndarray or list.')

    check_valid_vector(points[0], 'points[0]')
    check_valid_vector(points[1], 'points[1]')
    check_valid_vector(points[2], 'points[2]')
    check_valid_vector(points[3], 'points[3]')
    check_valid_vector(points[4], 'points[4]')

    pyramid = _vtk.vtkPyramid()
    pyramid.GetPointIds().SetId(0, 0)
    pyramid.GetPointIds().SetId(1, 1)
    pyramid.GetPointIds().SetId(2, 2)
    pyramid.GetPointIds().SetId(3, 3)
    pyramid.GetPointIds().SetId(4, 4)

    ug = _vtk.vtkUnstructuredGrid()
    ug.SetPoints(pyvista.vtk_points(np.array(points), False))
    ug.InsertNextCell(pyramid.GetCellType(), pyramid.GetPointIds())

    return wrap(ug)


def Triangle(points=None):
    """Create a triangle defined by 3 points.

    Parameters
    ----------
    points : array_like[float], optional
        Points of the triangle.  Defaults to a right isosceles
        triangle (see example).

    Returns
    -------
    pyvista.PolyData
        Triangle mesh.

    Examples
    --------
    >>> import pyvista
    >>> pointa = [0, 0, 0]
    >>> pointb = [1, 0, 0]
    >>> pointc = [0.5, 0.707, 0]
    >>> triangle = pyvista.Triangle([pointa, pointb, pointc])
    >>> triangle.plot(show_edges=True, line_width=5)
    """
    if points is None:
        points = [[0, 0, 0], [1, 0, 0], [0.5, 0.5**0.5, 0]]

    if len(points) != 3:
        raise TypeError('Points must be given as length 3 np.ndarray or list')

    check_valid_vector(points[0], 'points[0]')
    check_valid_vector(points[1], 'points[1]')
    check_valid_vector(points[2], 'points[2]')

    cells = np.array([[3, 0, 1, 2]])
    return wrap(pyvista.PolyData(points, cells))


def Rectangle(points=None):
    """Create a rectangle defined by 3 points.

    .. deprecated:: 0.39.0
       To deal with more than 3 points use :func:`pyvista.Quadrilateral()
       <pyvista.examples.cells.Quadrilateral>` instead.

    The 3 points must define an orthogonal set of vectors.

    Parameters
    ----------
    points : array_like[float], optional
        Points of the rectangle. Defaults to a unit square in xy-plane.

    Returns
    -------
    pyvista.PolyData
        Rectangle mesh.

    Examples
    --------
    >>> import pyvista
    >>> pointa = [1.0, 0.0, 0.0]
    >>> pointb = [1.0, 1.0, 0.0]
    >>> pointc = [0.0, 1.0, 0.0]
    >>> rectangle = pyvista.Rectangle([pointa, pointb, pointc])
    >>> rectangle.plot(show_edges=True, line_width=5)
    """
    if points is None:
        points = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    if len(points) == 4:
        warnings.warn(
            'Rectangle defined by 4 points is deprecated. Please use ``pyvista.Quadrilateral``.',
            PyVistaDeprecationWarning,
        )
        return Quadrilateral(points)
    if len(points) != 3:
        raise TypeError('Points must be given as length 3 np.ndarray or list')

    points, _ = _coerce_pointslike_arg(points)

    point_0 = points[0]
    point_1 = points[1]
    point_2 = points[2]

    vec_01 = point_1 - point_0
    vec_02 = point_2 - point_0
    vec_12 = point_2 - point_1

    scalar_pdct_01_02 = np.dot(vec_01, vec_02)
    scalar_pdct_01_12 = np.dot(vec_01, vec_12)
    scalar_pdct_02_12 = np.dot(vec_02, vec_12)

    null_scalar_products = [
        val
        for val in [scalar_pdct_01_02, scalar_pdct_01_12, scalar_pdct_02_12]
        if np.isclose(val, 0)
    ]
    if len(null_scalar_products) == 0:
        raise ValueError("The three points should defined orthogonal vectors")
    if len(null_scalar_products) > 1:
        raise ValueError("Unable to build a rectangle with less than three different points")

    points = np.array([point_0, point_1, point_2, point_0])
    if np.isclose(scalar_pdct_01_02, 0):
        points[3] = point_0 + vec_01 + vec_02
        cells = np.array([[4, 0, 1, 3, 2]])
    elif np.isclose(scalar_pdct_01_12, 0):
        points[3] = point_1 + vec_12 - vec_01
        cells = np.array([[4, 0, 1, 2, 3]])
    else:
        points[3] = point_2 - vec_02 - vec_12
        cells = np.array([[4, 0, 2, 1, 3]])

    return pyvista.PolyData(points, cells)


def Quadrilateral(points=None):
    """Create a quadrilateral defined by 4 points.

    Parameters
    ----------
    points : array_like[float], optional
        Points of the quadrilateral.  Defaults to a unit square in xy-plane.

    Returns
    -------
    pyvista.PolyData
        Quadrilateral mesh.

    Examples
    --------
    >>> import pyvista as pv
    >>> pointa = [1.0, 0.0, 0.0]
    >>> pointb = [1.0, 1.0, 0.0]
    >>> pointc = [0.0, 1.0, 0.0]
    >>> pointd = [0.0, 0.0, 0.0]
    >>> quadrilateral = pv.Quadrilateral([pointa, pointb, pointc, pointd])
    >>> quadrilateral.plot(show_edges=True, line_width=5)

    """
    if points is None:
        points = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    if len(points) != 4:
        raise TypeError('Points must be given as length 4 np.ndarray or list')

    points, _ = _coerce_pointslike_arg(points)

    cells = np.array([[4, 0, 1, 2, 3]])
    return wrap(pyvista.PolyData(points, cells))


def Circle(radius=0.5, resolution=100):
    """Create a single PolyData circle defined by radius in the XY plane.

    Parameters
    ----------
    radius : float, default: 0.5
        Radius of circle.

    resolution : int, default: 100
        Number of points on the circle.

    Returns
    -------
    pyvista.PolyData
        Circle mesh.

    Notes
    -----
    .. versionchanged:: 0.38.0
       Prior to version 0.38, this method had incorrect results, producing
       inconsistent edge lengths and a duplicated point which is now fixed.

    Examples
    --------
    >>> import pyvista
    >>> radius = 0.5
    >>> circle = pyvista.Circle(radius)
    >>> circle.plot(show_edges=True, line_width=5)

    """
    points = np.zeros((resolution, 3))
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    cells = np.array([np.append(np.array([resolution]), np.arange(resolution))])
    return wrap(pyvista.PolyData(points, cells))


def Ellipse(semi_major_axis=0.5, semi_minor_axis=0.2, resolution=100):
    """Create a single PolyData ellipse defined by the Semi-major and Semi-minor axes in the XY plane.

    Parameters
    ----------
    semi_major_axis : float, default: 0.5
        Semi-major axis of ellipse.

    semi_minor_axis : float, default: 0.2
        Semi-minor axis of ellipse.

    resolution : int, default: 100
        Number of points on the ellipse.

    Returns
    -------
    pyvista.PolyData
        Ellipse mesh.

    Notes
    -----
    .. versionchanged:: 0.38.0
       Prior to version 0.38, this method had incorrect results, producing
       inconsistent edge lengths and a duplicated point which is now fixed.

    Examples
    --------
    >>> import pyvista
    >>> ellipse = pyvista.Ellipse(semi_major_axis=8, semi_minor_axis=4)
    >>> ellipse.plot(show_edges=True, line_width=5)
    """
    points = np.zeros((resolution, 3))
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    points[:, 0] = semi_major_axis * np.cos(theta)
    points[:, 1] = semi_minor_axis * np.sin(theta)
    cells = np.array([np.append(np.array([resolution]), np.arange(resolution))])
    return wrap(pyvista.PolyData(points, cells))


def Superquadric(
    center=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    size=0.5,
    theta_roundness=1.0,
    phi_roundness=1.0,
    theta_resolution=16,
    phi_resolution=16,
    toroidal=False,
    thickness=1 / 3,
):
    """Create a superquadric.

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center of the superquadric in ``[x, y, z]``.

    scale : sequence[float], default: (1.0, 1.0, 1.0)
        Scale factors of the superquadric in ``[x, y, z]``.

    size : float, default: 0.5
        Superquadric isotropic size.

    theta_roundness : float, default: 1.0
        Superquadric east/west roundness.
        Values range from 0 (rectangular) to 1 (circular) to higher orders.

    phi_roundness : float, default: 1.0
        Superquadric north/south roundness.
        Values range from 0 (rectangular) to 1 (circular) to higher orders.

    theta_resolution : int, default: 16
        Number of points in the longitude direction.
        Values are rounded to nearest multiple of 4.

    phi_resolution : int, default: 16
        Number of points in the latitude direction.
        Values are rounded to nearest multiple of 8.

    toroidal : bool, default: False
        Whether or not the superquadric is toroidal (``True``)
        or ellipsoidal (``False``).

    thickness : float, default: 0.3333333333
        Superquadric ring thickness.
        Only applies if toroidal is set to ``True``.

    Returns
    -------
    pyvista.PolyData
        Superquadric mesh.

    See Also
    --------
    pyvista.ParametricSuperEllipsoid :
        Parametric superquadric if toroidal is ``False``.
    pyvista.ParametricSuperToroid :
        Parametric superquadric if toroidal is ``True``.

    Examples
    --------
    >>> import pyvista
    >>> superquadric = pyvista.Superquadric(
    ...     scale=(3.0, 1.0, 0.5),
    ...     phi_roundness=0.1,
    ...     theta_roundness=0.5,
    ... )
    >>> superquadric.plot(show_edges=True)

    """
    superquadricSource = _vtk.vtkSuperquadricSource()
    superquadricSource.SetCenter(center)
    superquadricSource.SetScale(scale)
    superquadricSource.SetSize(size)
    superquadricSource.SetThetaRoundness(theta_roundness)
    superquadricSource.SetPhiRoundness(phi_roundness)
    superquadricSource.SetThetaResolution(round(theta_resolution / 4) * 4)
    superquadricSource.SetPhiResolution(round(phi_resolution / 8) * 8)
    superquadricSource.SetToroidal(toroidal)
    superquadricSource.SetThickness(thickness)
    superquadricSource.Update()
    return wrap(superquadricSource.GetOutput())


def PlatonicSolid(kind='tetrahedron', radius=1.0, center=(0.0, 0.0, 0.0)):
    """Create a Platonic solid of a given size.

    Parameters
    ----------
    kind : str | int, default: 'tetrahedron'
        The kind of Platonic solid to create. Either the name of the
        polyhedron or an integer index:

            * ``'tetrahedron'`` or ``0``
            * ``'cube'`` or ``1``
            * ``'octahedron'`` or ``2``
            * ``'icosahedron'`` or ``3``
            * ``'dodecahedron'`` or ``4``

    radius : float, default: 1.0
        The radius of the circumscribed sphere for the solid to create.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Sequence defining the center of the solid to create.

    Returns
    -------
    pyvista.PolyData
        One of the five Platonic solids. Cell scalars are defined that
        assign integer labels to each face (with array name
        ``"FaceIndex"``).

    Examples
    --------
    Create and plot a dodecahedron.

    >>> import pyvista
    >>> dodeca = pyvista.PlatonicSolid('dodecahedron')
    >>> dodeca.plot(categories=True)

    See :ref:`platonic_example` for more examples using this filter.

    """
    kinds = {
        'tetrahedron': 0,
        'cube': 1,
        'octahedron': 2,
        'icosahedron': 3,
        'dodecahedron': 4,
    }
    if isinstance(kind, str):
        if kind not in kinds:
            raise ValueError(f'Invalid Platonic solid kind "{kind}".')
        kind = kinds[kind]
    elif isinstance(kind, int) and kind not in range(5):
        raise ValueError(f'Invalid Platonic solid index "{kind}".')
    elif not isinstance(kind, int):
        raise ValueError(f'Invalid Platonic solid index type "{type(kind).__name__}".')
    check_valid_vector(center, 'center')

    solid = _vtk.vtkPlatonicSolidSource()
    solid.SetSolidType(kind)
    solid.Update()
    solid = wrap(solid.GetOutput())
    # rename and activate cell scalars
    cell_data = solid.cell_data.get_array(0)
    solid.clear_data()
    solid.cell_data['FaceIndex'] = cell_data
    # scale and translate
    solid.scale(radius, inplace=True)
    solid.points += np.asanyarray(center) - solid.center
    return solid


def Tetrahedron(radius=1.0, center=(0.0, 0.0, 0.0)):
    """Create a tetrahedron of a given size.

    A tetrahedron is composed of four congruent equilateral triangles.

    Parameters
    ----------
    radius : float, default: 1.0
        The radius of the circumscribed sphere for the tetrahedron.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Three-length sequence defining the center of the tetrahedron.

    Returns
    -------
    pyvista.PolyData
        Mesh for the tetrahedron. Cell scalars are defined that assign
        integer labels to each face (with array name ``"FaceIndex"``).

    Examples
    --------
    Create and plot a tetrahedron.

    >>> import pyvista
    >>> tetra = pyvista.Tetrahedron()
    >>> tetra.plot(categories=True)

    See :ref:`platonic_example` for more examples using this filter.

    """
    return PlatonicSolid(kind='tetrahedron', radius=radius, center=center)


def Octahedron(radius=1.0, center=(0.0, 0.0, 0.0)):
    """Create an octahedron of a given size.

    An octahedron is composed of eight congruent equilateral
    triangles.

    Parameters
    ----------
    radius : float, default: 1.0
        The radius of the circumscribed sphere for the octahedron.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Three-length sequence defining the center of the octahedron.

    Returns
    -------
    pyvista.PolyData
        Mesh for the octahedron. Cell scalars are defined that assign
        integer labels to each face (with array name ``"FaceIndex"``).

    Examples
    --------
    Create and plot an octahedron.

    >>> import pyvista
    >>> tetra = pyvista.Octahedron()
    >>> tetra.plot(categories=True)

    See :ref:`platonic_example` for more examples using this filter.

    """
    return PlatonicSolid(kind='octahedron', radius=radius, center=center)


def Dodecahedron(radius=1.0, center=(0.0, 0.0, 0.0)):
    """Create a dodecahedron of a given size.

    A dodecahedron is composed of twelve congruent regular pentagons.

    Parameters
    ----------
    radius : float, default: 1.0
        The radius of the circumscribed sphere for the dodecahedron.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Three-length sequence defining the center of the dodecahedron.

    Returns
    -------
    pyvista.PolyData
        Mesh for the dodecahedron. Cell scalars are defined that assign
        integer labels to each face (with array name ``"FaceIndex"``).

    Examples
    --------
    Create and plot a dodecahedron.

    >>> import pyvista
    >>> tetra = pyvista.Dodecahedron()
    >>> tetra.plot(categories=True)

    See :ref:`platonic_example` for more examples using this filter.

    """
    return PlatonicSolid(kind='dodecahedron', radius=radius, center=center)


def Icosahedron(radius=1.0, center=(0.0, 0.0, 0.0)):
    """Create an icosahedron of a given size.

    An icosahedron is composed of twenty congruent equilateral
    triangles.

    Parameters
    ----------
    radius : float, default: 1.0
        The radius of the circumscribed sphere for the icosahedron.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Three-length sequence defining the center of the icosahedron.

    Returns
    -------
    pyvista.PolyData
        Mesh for the icosahedron. Cell scalars are defined that assign
        integer labels to each face (with array name ``"FaceIndex"``).

    Examples
    --------
    Create and plot an icosahedron.

    >>> import pyvista
    >>> tetra = pyvista.Icosahedron()
    >>> tetra.plot(categories=True)

    See :ref:`platonic_example` for more examples using this filter.

    """
    return PlatonicSolid(kind='icosahedron', radius=radius, center=center)


def Icosphere(radius=1.0, center=(0.0, 0.0, 0.0), nsub=3):
    """Create an icosphere.

    An icosphere is a `geodesic polyhedron
    <https://en.wikipedia.org/wiki/Geodesic_polyhedron>`_, which is a
    convex polyhedron made from triangles.

    Geodesic polyhedra are constructed by subdividing faces of simpler
    polyhedra, and then projecting the new vertices onto the surface of
    a sphere. A geodesic polyhedron has straight edges and flat faces
    that approximate a sphere,

    Parameters
    ----------
    radius : float, default: 1.0
        Radius of the icosphere.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center of the icosphere.

    nsub : int, default: 3
        This is the number of times each triangle of the original
        :func:`pyvista.Icosahedron` is subdivided.

    Returns
    -------
    pyvista.PolyData
        Mesh of the icosphere.

    See Also
    --------
    pyvista.Sphere

    Examples
    --------
    Create the icosphere and plot it with edges.

    >>> import pyvista as pv
    >>> icosphere = pv.Icosphere()
    >>> icosphere.plot(show_edges=True)

    Show how this icosphere was created.

    >>> import numpy as np
    >>> icosahedron = pv.Icosahedron()
    >>> icosahedron.clear_data()  # remove extra scalars
    >>> icosahedron_sub = icosahedron.subdivide(nsub=3)
    >>> pl = pv.Plotter(shape=(1, 3))
    >>> _ = pl.add_mesh(icosahedron, show_edges=True)
    >>> pl.subplot(0, 1)
    >>> _ = pl.add_mesh(icosahedron_sub, show_edges=True)
    >>> pl.subplot(0, 2)
    >>> _ = pl.add_mesh(icosphere, show_edges=True)
    >>> pl.show()

    Show how the triangles are not uniform in area. This is because the
    ones farther from the edges from the original triangles have farther
    to travel to the sphere.

    >>> icosphere = pv.Icosphere(nsub=4)
    >>> icosphere.compute_cell_sizes().plot(scalars='Area')

    """
    mesh = Icosahedron()
    mesh.clear_data()
    mesh = mesh.subdivide(nsub=nsub)

    # scale to desired radius and translate origin
    dist = np.linalg.norm(mesh.points, axis=1, keepdims=True)  # distance from origin
    mesh.points = mesh.points * (radius / dist) + center
    return mesh
