"""Provides an easy way of generating several geometric sources.

Also includes some pure-python helpers.

"""
from typing import Sequence

import numpy as np

from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.misc import no_new_attr

from .helpers import wrap


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


@no_new_attr
class ConeSource(_vtk.vtkConeSource):
    """Cone source algorithm class.

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

    capping : bool, default: True
        Enable or disable the capping the base of the cone with a
        polygon.

    angle : float, optional
        The angle in degrees between the axis of the cone and a
        generatrix.

    resolution : int, default: 6
        Number of facets used to represent the cone.

    Examples
    --------
    Create a default ConeSource.

    >>> import pyvista
    >>> source = pyvista.ConeSource()
    >>> source.output.plot(show_edges=True, line_width=5)
    """

    def __init__(
        self,
        center=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        height=1.0,
        radius=None,
        capping=True,
        angle=None,
        resolution=6,
    ):
        """Initialize the cone source class."""
        super().__init__()
        self.center = center
        self.direction = direction
        self.height = height
        self.capping = capping
        if angle is not None and radius is not None:
            raise ValueError(
                "Both radius and angle cannot be specified. They are mutually exclusive."
            )
        elif angle is not None and radius is None:
            self.angle = angle
        elif angle is None and radius is not None:
            self.radius = radius
        elif angle is None and radius is None:
            self.radius = 0.5
        self.resolution = resolution

    @property
    def center(self) -> Sequence[float]:
        """Get the center in ``[x, y, z]``. Axis of the cone passes through this point.

        Returns
        -------
        sequence[float]
            Center in ``[x, y, z]``. Axis of the cone passes through this
            point.
        """
        return self.GetCenter()

    @center.setter
    def center(self, center: Sequence[float]):
        """Set the center in ``[x, y, z]``. Axis of the cone passes through this point.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``. Axis of the cone passes through this
            point.
        """
        self.SetCenter(center)

    @property
    def direction(self) -> Sequence[float]:
        """Get the direction vector in ``[x, y, z]``. Orientation vector of the cone.

        Returns
        -------
        sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cone.
        """
        return self.GetDirection()

    @direction.setter
    def direction(self, direction: Sequence[float]):
        """Set the direction in ``[x, y, z]``. Axis of the cone passes through this point.

        Parameters
        ----------
        direction : sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cone.
        """
        self.SetDirection(direction)

    @property
    def height(self) -> float:
        """Get the height along the cone in its specified direction.

        Returns
        -------
        float
            Height along the cone in its specified direction.
        """
        return self.GetHeight()

    @height.setter
    def height(self, height: float):
        """Set the height of the cone.

        Parameters
        ----------
        height : float
            Height of the cone.
        """
        self.SetHeight(height)

    @property
    def radius(self) -> bool:
        """Get base radius of the cone.

        Returns
        -------
        float
            Base radius of the cone.
        """
        return self.GetRadius()

    @radius.setter
    def radius(self, radius: float):
        """Set base radius of the cone.

        Parameters
        ----------
        radius : float
            Base radius of the cone.
        """
        self.SetRadius(radius)

    @property
    def capping(self) -> bool:
        """Enable or disable the capping the base of the cone with a polygon.

        Returns
        -------
        bool
            Enable or disable the capping the base of the cone with a
            polygon.
        """
        return self.GetCapping()

    @capping.setter
    def capping(self, capping: bool):
        """Set base capping of the cone.

        Parameters
        ----------
        capping : bool, optional
            Enable or disable the capping the base of the cone with a
            polygon.
        """
        self.SetCapping(capping)

    @property
    def angle(self) -> float:
        """Get the angle in degrees between the axis of the cone and a generatrix.

        Returns
        -------
        float
            The angle in degrees between the axis of the cone and a
            generatrix.
        """
        return self.GetAngle()

    @angle.setter
    def angle(self, angle: float):
        """Set the angle in degrees between the axis of the cone and a generatrix.

        Parameters
        ----------
        angle : float, optional
            The angle in degrees between the axis of the cone and a
            generatrix.
        """
        self.SetAngle(angle)

    @property
    def resolution(self) -> int:
        """Get number of points on the circular face of the cone.

        Returns
        -------
        int
            Number of points on the circular face of the cone.
        """
        return self.GetResolution()

    @resolution.setter
    def resolution(self, resolution: int):
        """Set number of points on the circular face of the cone.

        Parameters
        ----------
        resolution : int
            Number of points on the circular face of the cone.
        """
        self.SetResolution(resolution)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Cone surface.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class CylinderSource(_vtk.vtkCylinderSource):
    """Cylinder source algorithm class.

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

    capping : bool, default: True
        Cap cylinder ends with polygons.

    resolution : int, default: 100
        Number of points on the circular face of the cylinder.

    Examples
    --------
    Create a default CylinderSource.

    >>> import pyvista
    >>> source = pyvista.CylinderSource()
    >>> source.output.plot(show_edges=True, line_width=5)
    """

    _new_attr_exceptions = ['_center', '_direction']

    def __init__(
        self,
        center=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=0.5,
        height=1.0,
        capping=True,
        resolution=100,
    ):
        """Initialize the cylinder source class."""
        super().__init__()
        self._center = center
        self._direction = direction
        self.radius = radius
        self.height = height
        self.resolution = resolution
        self.capping = capping

    @property
    def center(self) -> Sequence[float]:
        """Get location of the centroid in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Center in ``[x, y, z]``. Axis of the cylinder passes through this
            point.
        """
        return self._center

    @center.setter
    def center(self, center: Sequence[float]):
        """Set location of the centroid in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``. Axis of the cylinder passes through this
            point.
        """
        self._center = center

    @property
    def direction(self) -> Sequence[float]:
        """Get the direction vector in ``[x, y, z]``. Orientation vector of the cylinder.

        Returns
        -------
        sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cylinder.
        """
        return self._direction

    @direction.setter
    def direction(self, direction: Sequence[float]):
        """Set the direction in ``[x, y, z]``. Axis of the cylinder passes through this point.

        Parameters
        ----------
        direction : sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cylinder.
        """
        self._direction = direction

    @property
    def radius(self) -> bool:
        """Get radius of the cylinder.

        Returns
        -------
        float
            Radius of the cylinder.
        """
        return self.GetRadius()

    @radius.setter
    def radius(self, radius: float):
        """Set radius of the cylinder.

        Parameters
        ----------
        radius : float
            Radius of the cylinder.
        """
        self.SetRadius(radius)

    @property
    def height(self) -> float:
        """Get the height of the cylinder.

        Returns
        -------
        float
            Height of the cylinder.
        """
        return self.GetHeight()

    @height.setter
    def height(self, height: float):
        """Set the height of the cylinder.

        Parameters
        ----------
        height : float
            Height of the cylinder.
        """
        self.SetHeight(height)

    @property
    def resolution(self) -> int:
        """Get number of points on the circular face of the cylinder.

        Returns
        -------
        int
            Number of points on the circular face of the cone.
        """
        return self.GetResolution()

    @resolution.setter
    def resolution(self, resolution: int):
        """Set number of points on the circular face of the cone.

        Parameters
        ----------
        resolution : int
            Number of points on the circular face of the cone.
        """
        self.SetResolution(resolution)

    @property
    def capping(self) -> bool:
        """Get cap cylinder ends with polygons.

        Returns
        -------
        bool
            Cap cylinder ends with polygons.
        """
        return self.GetCapping()

    @capping.setter
    def capping(self, capping: bool):
        """Set cap cylinder ends with polygons.

        Parameters
        ----------
        capping : bool, optional
            Cap cylinder ends with polygons.
        """
        self.SetCapping(capping)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Cylinder surface.
        """
        self.Update()
        output = wrap(self.GetOutput())
        output.rotate_z(-90, inplace=True)
        translate(output, self.center, self.direction)
        return output
