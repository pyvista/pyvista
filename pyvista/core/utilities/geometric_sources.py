"""Provides an easy way of generating several geometric sources.

Also includes some pure-python helpers.

"""
from typing import Sequence, Union
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import Matrix, Vector
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import no_new_attr

from .arrays import _coerce_pointslike_arg
from .helpers import wrap


def translate(
    surf, center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), normx=(1.0, 0.0, 0.0), normy=(0.0, 1.0, 0.0)
):
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
    normx : tuple, optional, default: (1.0, 0.0, 0.0)
        Norm x vector along which the mesh should be oriented.
    normy : tuple, optional, default: (0.0, 1.0, 0.0)
        Norm y vector along which the mesh should be oriented.

    """
    if direction is not None:
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

        # Deprecated on v0.43.0, estimated removal on v0.46.0
        warnings.warn(
            '`direction` argument is deprecated. Please use `normx` and `normy`.',
            PyVistaDeprecationWarning,
        )
    else:
        normx = np.array(normx) / np.linalg.norm(normx)
        normy = np.array(normy) / np.linalg.norm(normy)
        normz = np.cross(normx, normy)

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

    >>> import pyvista as pv
    >>> source = pv.ConeSource()
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

    .. warning::
       :func:`pyvista.Cylinder` function rotates the :class:`pyvista.CylinderSource` 's
       :class:`pyvista.PolyData` in its own way.
       It rotates the :attr:`pyvista.CylinderSource.output` 90 degrees in z-axis, translates and
       orients the mesh to a new ``center`` and ``direction`` (or ``normx`` and ``normy``).

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

    normx : sequence[float], default: (1.0, 0.0, 0.0)
        Norm x cylinder points to  in ``[x, y, z]``.

        .. versionchanged:: 0.43.0
            The ``normx`` parameter has been added.

    normy : sequence[float], default: (0.0, 1.0, 0.0)
        Norm y cylinder points to  in ``[x, y, z]``.

        .. versionchanged:: 0.43.0
            The ``normy`` parameter has been added.

    Examples
    --------
    Create a default CylinderSource.

    >>> import pyvista as pv
    >>> source = pv.CylinderSource()
    >>> source.output.plot(show_edges=True, line_width=5)

    Display a 3D plot of a default :class:`CylinderSource`.

    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(pv.CylinderSource(), show_edges=True, line_width=5)
    >>> pl.show()

    Visualize the output of :class:`CylinderSource` in a 3D plot.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(
    ...     pv.CylinderSource().output, show_edges=True, line_width=5
    ... )
    >>> pl.show()

    The above examples are similar in terms of their behavior.
    """

    _new_attr_exceptions = ['_center', '_direction', '_normx', '_normy']

    def __init__(
        self,
        center=(0.0, 0.0, 0.0),
        direction=None,
        radius=0.5,
        height=1.0,
        capping=True,
        resolution=100,
        normx=(1.0, 0.0, 0.0),
        normy=(0.0, 1.0, 0.0),
    ):
        """Initialize the cylinder source class."""
        super().__init__()
        self._center = center
        if direction is not None:
            self._direction = direction
        self.radius = radius
        self.height = height
        self.resolution = resolution
        self.capping = capping
        self._normx = normx
        self._normy = normy

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
        return wrap(self.GetOutput())

    @property
    def normx(self) -> Sequence[float]:
        """Get the normx vector in ``[x, y, z]``. Orientation vector of the cylinder.

        Returns
        -------
        sequence[float]
            Norm x vector in ``[x, y, z]``. Orientation vector of the
            cylinder.
        """
        return self._normx

    @normx.setter
    def normx(self, normx: Sequence[float]):
        """Set the normx in ``[x, y, z]``. Axis of the cylinder passes through this point.

        Parameters
        ----------
        normx : sequence[float]
            Norm x vector in ``[x, y, z]``. Orientation vector of the
            cylinder.
        """
        self._normx = normx

    @property
    def normy(self) -> Sequence[float]:
        """Get the normy vector in ``[x, y, z]``. Orientation vector of the cylinder.

        Returns
        -------
        sequence[float]
            Norm y vector in ``[x, y, z]``. Orientation vector of the
            cylinder.
        """
        return self._normy

    @normy.setter
    def normy(self, normy: Sequence[float]):
        """Set the normy in ``[x, y, z]``. Axis of the cylinder passes through this point.

        Parameters
        ----------
        normy : sequence[float]
            Norm y vector in ``[x, y, z]``. Orientation vector of the
            cylinder.
        """
        self._normy = normy


@no_new_attr
class MultipleLinesSource(_vtk.vtkLineSource):
    """Multiple lines source algorithm class.

    Parameters
    ----------
    points : array_like[float], default: [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        List of points defining a broken line.
    """

    _new_attr_exceptions = ['points']

    def __init__(self, points=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]):
        """Initialize the multiple lines source class."""
        super().__init__()
        self.points = points

    @property
    def points(self) -> np.ndarray:
        """Return the points defining a broken line.

        Returns
        -------
        np.ndarray
            Points defining a broken line.
        """
        return _vtk.vtk_to_numpy(self.GetPoints().GetData())

    @points.setter
    def points(self, points: Union[Matrix, Vector]):
        """Set the list of points defining a broken line.

        Parameters
        ----------
        points : array_like[float]
            List of points defining a broken line.
        """
        points, _ = _coerce_pointslike_arg(points)
        if not (len(points) >= 2):
            raise ValueError('>=2 points need to define multiple lines.')
        self.SetPoints(pyvista.vtk_points(points))

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Line mesh.
        """
        self.Update()
        return wrap(self.GetOutput())
