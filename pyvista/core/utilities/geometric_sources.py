"""Provides an easy way of generating several geometric sources.

Also includes some pure-python helpers.

"""

from __future__ import annotations

from enum import IntEnum
import itertools
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal
from typing import get_args

import numpy as np
from vtkmodules.vtkRenderingFreeType import vtkVectorText

import pyvista
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _check_range
from pyvista.core.utilities.misc import _reciprocal
from pyvista.core.utilities.misc import no_new_attr

if TYPE_CHECKING:  # pragma: no cover
    from typing import Sequence

    from pyvista.core._typing_core import BoundsLike
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike


SINGLE_PRECISION = _vtk.vtkAlgorithm.SINGLE_PRECISION
DOUBLE_PRECISION = _vtk.vtkAlgorithm.DOUBLE_PRECISION


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
        surf.points += np.array(center, dtype=surf.points.dtype)


if _vtk.vtk_version_info < (9, 3):

    @no_new_attr
    class CapsuleSource(_vtk.vtkCapsuleSource):
        """Capsule source algorithm class.

        .. versionadded:: 0.44.0

        Parameters
        ----------
        center : sequence[float], default: (0.0, 0.0, 0.0)
            Center in ``[x, y, z]``.

        direction : sequence[float], default: (1.0, 0.0, 0.0)
            Direction of the capsule in ``[x, y, z]``.

        radius : float, default: 0.5
            Radius of the capsule.

        cylinder_length : float, default: 1.0
            Cylinder length of the capsule.

        theta_resolution : int, default: 30
            Set the number of points in the azimuthal direction (ranging
            from ``start_theta`` to ``end_theta``).

        phi_resolution : int, default: 30
            Set the number of points in the polar direction (ranging from
            ``start_phi`` to ``end_phi``).

        Examples
        --------
        Create a default CapsuleSource.

        >>> import pyvista as pv
        >>> source = pv.CapsuleSource()
        >>> source.output.plot(show_edges=True, line_width=5)
        """

        _new_attr_exceptions: ClassVar[list[str]] = ['_direction']

        def __init__(
            self,
            center=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            radius=0.5,
            cylinder_length=1.0,
            theta_resolution=30,
            phi_resolution=30,
        ):
            """Initialize the capsule source class."""
            super().__init__()
            self.center = center
            self._direction = direction
            self.radius = radius
            self.cylinder_length = cylinder_length
            self.theta_resolution = theta_resolution
            self.phi_resolution = phi_resolution

        @property
        def center(self) -> Sequence[float]:
            """Get the center in ``[x, y, z]``. Axis of the capsule passes through this point.

            Returns
            -------
            sequence[float]
                Center in ``[x, y, z]``. Axis of the capsule passes through this
                point.
            """
            return self.GetCenter()

        @center.setter
        def center(self, center: Sequence[float]):
            """Set the center in ``[x, y, z]``. Axis of the capsule passes through this point.

            Parameters
            ----------
            center : sequence[float]
                Center in ``[x, y, z]``. Axis of the capsule passes through this
                point.
            """
            self.SetCenter(center)

        @property
        def direction(self) -> Sequence[float]:
            """Get the direction vector in ``[x, y, z]``. Orientation vector of the capsule.

            Returns
            -------
            sequence[float]
                Direction vector in ``[x, y, z]``. Orientation vector of the
                capsule.
            """
            return self._direction

        @direction.setter
        def direction(self, direction: Sequence[float]):
            """Set the direction in ``[x, y, z]``. Axis of the capsule passes through this point.

            Parameters
            ----------
            direction : sequence[float]
                Direction vector in ``[x, y, z]``. Orientation vector of the
                capsule.
            """
            self._direction = direction

        @property
        def cylinder_length(self) -> float:
            """Get the cylinder length along the capsule in its specified direction.

            Returns
            -------
            float
                Cylinder length along the capsule in its specified direction.
            """
            return self.GetCylinderLength()

        @cylinder_length.setter
        def cylinder_length(self, length: float):
            """Set the cylinder length of the capsule.

            Parameters
            ----------
            length : float
                Cylinder length of the capsule.
            """
            self.SetCylinderLength(length)

        @property
        def radius(self) -> float:
            """Get base radius of the capsule.

            Returns
            -------
            float
                Base radius of the capsule.
            """
            return self.GetRadius()

        @radius.setter
        def radius(self, radius: float):
            """Set base radius of the capsule.

            Parameters
            ----------
            radius : float
                Base radius of the capsule.
            """
            self.SetRadius(radius)

        @property
        def theta_resolution(self) -> int:
            """Get the number of points in the azimuthal direction.

            Returns
            -------
            int
                The number of points in the azimuthal direction.
            """
            return self.GetThetaResolution()

        @theta_resolution.setter
        def theta_resolution(self, theta_resolution: int):
            """Set the number of points in the azimuthal direction.

            Parameters
            ----------
            theta_resolution : int
                The number of points in the azimuthal direction.
            """
            self.SetThetaResolution(theta_resolution)

        @property
        def phi_resolution(self) -> int:
            """Get the number of points in the polar direction.

            Returns
            -------
            int
                The number of points in the polar direction.
            """
            return self.GetPhiResolution()

        @phi_resolution.setter
        def phi_resolution(self, phi_resolution: int):
            """Set the number of points in the polar direction.

            Parameters
            ----------
            phi_resolution : int
                The number of points in the polar direction.
            """
            self.SetPhiResolution(phi_resolution)

        @property
        def output(self):
            """Get the output data object for a port on this algorithm.

            Returns
            -------
            pyvista.PolyData
                Capsule surface.
            """
            self.Update()
            return wrap(self.GetOutput())


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
                "Both radius and angle cannot be specified. They are mutually exclusive.",
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
    def radius(self) -> float:
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
        return bool(self.GetCapping())

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
       orients the mesh to a new ``center`` and ``direction``.

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

    _new_attr_exceptions: ClassVar[list[str]] = ['_center', '_direction']

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
    def radius(self) -> float:
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
        return bool(self.GetCapping())

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
    def capsule_cap(self) -> bool:
        """Get whether the capping should make the cylinder a capsule.

        .. versionadded:: 0.44.0

        Returns
        -------
        bool
            Capsule cap.
        """
        return bool(self.GetCapsuleCap())

    @capsule_cap.setter
    def capsule_cap(self, capsule_cap: bool):
        """Set whether the capping should make the cylinder a capsule.

        Parameters
        ----------
        capsule_cap : bool
            Capsule cap.
        """
        self.SetCapsuleCap(capsule_cap)

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


@no_new_attr
class MultipleLinesSource(_vtk.vtkLineSource):
    """Multiple lines source algorithm class.

    Parameters
    ----------
    points : array_like[float], default: [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        List of points defining a broken line.
    """

    _new_attr_exceptions: ClassVar[list[str]] = ['points']

    def __init__(self, points=None):
        """Initialize the multiple lines source class."""
        if points is None:
            points = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        super().__init__()
        self.points = points

    @property
    def points(self) -> NumpyArray[float]:
        """Return the points defining a broken line.

        Returns
        -------
        np.ndarray
            Points defining a broken line.
        """
        return _vtk.vtk_to_numpy(self.GetPoints().GetData())

    @points.setter
    def points(self, points: MatrixLike[float] | VectorLike[float]):
        """Set the list of points defining a broken line.

        Parameters
        ----------
        points : VectorLike[float] | MatrixLike[float]
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


class Text3DSource(vtkVectorText):
    """3D text from a string.

    Generate 3D text from a string with a specified width, height or depth.

    .. versionadded:: 0.43

    Parameters
    ----------
    string : str, default: ""
        Text string of the source.

    depth : float, optional
        Depth of the text. If ``None``, the depth is set to half
        the :attr:`height` by default. Set to ``0.0`` for planar
        text.

    width : float, optional
        Width of the text. If ``None``, the width is scaled
        proportional to :attr:`height`.

    height : float, optional
        Height of the text. If ``None``, the height is scaled
        proportional to :attr:`width`.

    center : Sequence[float], default: (0.0, 0.0, 0.0)
        Center of the text, defined as the middle of the axis-aligned
        bounding box of the text.

    normal : Sequence[float], default: (0.0, 0.0, 1.0)
        Normal direction of the text. The direction is parallel to the
        :attr:`depth` of the text and points away from the front surface
        of the text.

    process_empty_string : bool, default: True
        If ``True``, when :attr:`string` is empty the :attr:`output` is a
        single point located at :attr:`center` instead of an empty mesh.
        See :attr:`process_empty_string` for details.

    """

    _new_attr_exceptions: ClassVar[list[str]] = [
        '_center',
        '_height',
        '_width',
        '_depth',
        '_normal',
        '_process_empty_string',
        '_output',
        '_modified',
    ]

    def __init__(
        self,
        string=None,
        depth=None,
        width=None,
        height=None,
        center=(0, 0, 0),
        normal=(0, 0, 1),
        process_empty_string=True,
    ):
        """Initialize source."""
        super().__init__()

        self._output = pyvista.PolyData()

        # Set params
        self.string = "" if string is None else string
        self._process_empty_string = process_empty_string
        self._center = center
        self._normal = normal
        self._height = height
        self._width = width
        self._depth = depth
        self._modified = True

    def __setattr__(self, name, value):  # numpydoc ignore=GL08
        """Override to set modified flag and disable setting new attributes."""
        if hasattr(self, name) and name != '_modified':
            # Set modified flag
            old_value = getattr(self, name)
            if not np.array_equal(old_value, value):
                object.__setattr__(self, name, value)
                object.__setattr__(self, '_modified', True)
        else:
            # Do not allow setting attributes.
            # This is similar to using @no_new_attr decorator but without
            # the __setattr__ override since this class defines its own override
            # for setting the modified flag
            if name in Text3DSource._new_attr_exceptions:
                object.__setattr__(self, name, value)
            else:
                raise AttributeError(
                    f'Attribute "{name}" does not exist and cannot be added to type '
                    f'{self.__class__.__name__}',
                )

    @property
    def string(self) -> str:  # numpydoc ignore=RT01
        """Return or set the text string."""
        return self.GetText()

    @string.setter
    def string(self, string: str):  # numpydoc ignore=GL08
        self.SetText("" if string is None else string)

    @property
    def process_empty_string(self) -> bool:  # numpydoc ignore=RT01
        """Return or set flag to control behavior when empty strings are set.

        When :attr:`string` is empty or only contains whitespace, the :attr:`output`
        mesh will be empty. This can cause the bounds of the output to be undefined.

        If ``True``, the output is modified to instead have a single point located
        at :attr:`center`.

        """
        return self._process_empty_string

    @process_empty_string.setter
    def process_empty_string(self, value: bool):  # numpydoc ignore=GL08
        self._process_empty_string = value

    @property
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the center of the text.

        The center is defined as the middle of the axis-aligned bounding box
        of the text.
        """
        return self._center

    @center.setter
    def center(self, center: Sequence[float]):  # numpydoc ignore=GL08
        self._center = float(center[0]), float(center[1]), float(center[2])

    @property
    def normal(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the normal direction of the text.

        The normal direction is parallel to the :attr:`depth` of the text, and
        points away from the front surface of the text.
        """
        return self._normal

    @normal.setter
    def normal(self, normal: Sequence[float]):  # numpydoc ignore=GL08
        self._normal = float(normal[0]), float(normal[1]), float(normal[2])

    @property
    def width(self) -> float:  # numpydoc ignore=RT01
        """Return or set the width of the text."""
        return self._width

    @width.setter
    def width(self, width: float):  # numpydoc ignore=GL08
        _check_range(width, rng=(0, float('inf')), parm_name='width') if width is not None else None
        self._width = width

    @property
    def height(self) -> float:  # numpydoc ignore=RT01
        """Return or set the height of the text."""
        return self._height

    @height.setter
    def height(self, height: float):  # numpydoc ignore=GL08
        (
            _check_range(height, rng=(0, float('inf')), parm_name='height')
            if height is not None
            else None
        )
        self._height = height

    @property
    def depth(self) -> float:  # numpydoc ignore=RT01
        """Return or set the depth of the text."""
        return self._depth

    @depth.setter
    def depth(self, depth: float):  # numpydoc ignore=GL08
        _check_range(depth, rng=(0, float('inf')), parm_name='depth') if depth is not None else None
        self._depth = depth

    def update(self):
        """Update the output of the source."""
        if self._modified:
            is_empty_string = self.string == "" or self.string.isspace()
            is_2d = self.depth == 0 or (self.depth is None and self.height == 0)
            if is_empty_string or is_2d:
                # Do not apply filters
                self.Update()
                out = self.GetOutput()
            else:
                # 3D case, apply filters
                # Create output filters to make text 3D
                extrude = _vtk.vtkLinearExtrusionFilter()
                extrude.SetInputConnection(self.GetOutputPort())
                extrude.SetExtrusionTypeToNormalExtrusion()
                extrude.SetVector(0, 0, 1)

                tri_filter = _vtk.vtkTriangleFilter()
                tri_filter.SetInputConnection(extrude.GetOutputPort())
                tri_filter.Update()
                out = tri_filter.GetOutput()

            # Modify output object
            self._output.copy_from(out)

            # For empty strings, the bounds are either default values (+/- 1) initially or
            # become uninitialized (+/- VTK_DOUBLE_MAX) if set to empty a second time
            if is_empty_string and self.process_empty_string:
                # Add a single point to 'fix' the bounds
                self._output.points = (0.0, 0.0, 0.0)

            self._transform_output()
            self._modified = False

    @property
    def output(self) -> _vtk.vtkPolyData:  # numpydoc ignore=RT01
        """Get the output of the source.

        The source is automatically updated by :meth:`update` prior
        to returning the output.
        """
        self.update()
        return self._output

    def _transform_output(self):
        """Scale, rotate, and translate the output mesh."""
        # Create aliases
        out, width, height, depth = self._output, self.width, self.height, self.depth
        width_set, height_set, depth_set = width is not None, height is not None, depth is not None

        # Scale mesh
        bnds = out.bounds
        size_w, size_h, size_d = bnds[1] - bnds[0], bnds[3] - bnds[2], bnds[5] - bnds[4]
        scale_w, scale_h, scale_d = _reciprocal((size_w, size_h, size_d))

        # Scale width and height first
        if width_set and height_set:
            # Scale independently
            scale_w *= width
            scale_h *= height
        elif not width_set and height_set:
            # Scale proportional to height
            scale_h *= height
            scale_w = scale_h
        elif width_set and not height_set:
            # Scale proportional to width
            scale_w *= width
            scale_h = scale_w
        else:
            # Do not scale
            scale_w = 1
            scale_h = 1

        out.points[:, 0] *= scale_w
        out.points[:, 1] *= scale_h

        # Scale depth
        if depth_set:
            if depth == 0:
                # Do not scale since depth is already zero (no extrusion)
                scale_d = 1
            else:
                scale_d *= depth
        else:
            # Scale to half the height by default
            scale_d *= size_h * scale_h * 0.5

        out.points[:, 2] *= scale_d

        # Center points at origin
        out.points -= out.center

        # Move to final position.
        # Only rotate if non-default normal.
        if not np.array_equal(self.normal, (0, 0, 1)):
            out.rotate_x(90, inplace=True)
            out.rotate_z(90, inplace=True)
            translate(out, self.center, self.normal)
        else:
            out.points += self.center


@no_new_attr
class CubeSource(_vtk.vtkCubeSource):
    """Cube source algorithm class.

    .. versionadded:: 0.44.0

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

    point_dtype : str, default: 'float32'
        Set the desired output point types. It must be either 'float32' or 'float64'.

        .. versionadded:: 0.44.0

    Examples
    --------
    Create a default CubeSource.

    >>> import pyvista as pv
    >>> source = pv.CubeSource()
    >>> source.output.plot(show_edges=True, line_width=5)
    """

    _new_attr_exceptions: ClassVar[list[str]] = [
        "bounds",
        "_bounds",
    ]

    def __init__(
        self,
        center=(0.0, 0.0, 0.0),
        x_length=1.0,
        y_length=1.0,
        z_length=1.0,
        bounds=None,
        point_dtype='float32',
    ):
        """Initialize the cube source class."""
        super().__init__()
        if bounds is not None:
            self.bounds = bounds
        else:
            self.center = center
            self.x_length = x_length
            self.y_length = y_length
            self.z_length = z_length
        self.point_dtype = point_dtype

    @property
    def bounds(self) -> BoundsLike:  # numpydoc ignore=RT01
        """Return or set the bounding box of the cube."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: BoundsLike):  # numpydoc ignore=GL08
        if np.array(bounds).size != 6:
            raise TypeError(
                'Bounds must be given as length 6 tuple: (xMin, xMax, yMin, yMax, zMin, zMax)',
            )
        self._bounds = bounds
        self.SetBounds(bounds)

    @property
    def center(self) -> Sequence[float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Center in ``[x, y, z]``.
        """
        return self.GetCenter()

    @center.setter
    def center(self, center: Sequence[float]):
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.
        """
        self.SetCenter(center)

    @property
    def x_length(self) -> float:
        """Get the x length along the cube in its specified direction.

        Returns
        -------
        float
            XLength along the cone in its specified direction.
        """
        return self.GetXLength()

    @x_length.setter
    def x_length(self, x_length: float):
        """Set the x length of the cube.

        Parameters
        ----------
        x_length : float
            XLength of the cone.
        """
        self.SetXLength(x_length)

    @property
    def y_length(self) -> float:
        """Get the y length along the cube in its specified direction.

        Returns
        -------
        float
            YLength along the cone in its specified direction.
        """
        return self.GetYLength()

    @y_length.setter
    def y_length(self, y_length: float):
        """Set the y length of the cube.

        Parameters
        ----------
        y_length : float
            YLength of the cone.
        """
        self.SetYLength(y_length)

    @property
    def z_length(self) -> float:
        """Get the z length along the cube in its specified direction.

        Returns
        -------
        float
            ZLength along the cone in its specified direction.
        """
        return self.GetZLength()

    @z_length.setter
    def z_length(self, z_length: float):
        """Set the z length of the cube.

        Parameters
        ----------
        z_length : float
            ZLength of the cone.
        """
        self.SetZLength(z_length)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Cube surface.
        """
        self.Update()
        return wrap(self.GetOutput())

    @property
    def point_dtype(self) -> str:
        """Get the desired output point types.

        Returns
        -------
        str
            Desired output point types.
            It must be either 'float32' or 'float64'.
        """
        precision = self.GetOutputPointsPrecision()
        return {
            SINGLE_PRECISION: 'float32',
            DOUBLE_PRECISION: 'float64',
        }[precision]

    @point_dtype.setter
    def point_dtype(self, point_dtype: str):
        """Set the desired output point types.

        Parameters
        ----------
        point_dtype : str, default: 'float32'
            Set the desired output point types.
            It must be either 'float32' or 'float64'.

        Returns
        -------
        point_dtype: str
            Desired output point types.
        """
        if point_dtype not in ['float32', 'float64']:
            raise ValueError("Point dtype must be either 'float32' or 'float64'")
        precision = {
            'float32': SINGLE_PRECISION,
            'float64': DOUBLE_PRECISION,
        }[point_dtype]
        self.SetOutputPointsPrecision(precision)


@no_new_attr
class DiscSource(_vtk.vtkDiskSource):
    """Disc source algorithm class.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``. Middle of the axis of the disc.

    inner : float, default: 0.25
        The inner radius.

    outer : float, default: 0.5
        The outer radius.

    r_res : int, default: 1
        Number of points in radial direction.

    c_res : int, default: 6
        Number of points in circumferential direction.

    Examples
    --------
    Create a disc with 50 points in the circumferential direction.

    >>> import pyvista as pv
    >>> source = pv.DiscSource(c_res=50)
    >>> source.output.plot(show_edges=True, line_width=5)
    """

    _new_attr_exceptions: ClassVar[list[str]] = ["center"]

    def __init__(self, center=None, inner=0.25, outer=0.5, r_res=1, c_res=6):
        """Initialize the disc source class."""
        super().__init__()
        if center is not None:
            self.center = center
        self.inner = inner
        self.outer = outer
        self.r_res = r_res
        self.c_res = c_res

    @property
    def center(self) -> Sequence[float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Center in ``[x, y, z]``.
        """
        if pyvista.vtk_version_info >= (9, 2):  # pragma: no cover
            return self.GetCenter()
        else:  # pragma: no cover
            return (0.0, 0.0, 0.0)

    @center.setter
    def center(self, center: Sequence[float]):
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.
        """
        if pyvista.vtk_version_info >= (9, 2):  # pragma: no cover
            self.SetCenter(center)
        else:  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError(
                'To change vtkDiskSource with `center` requires VTK 9.2 or later.',
            )

    @property
    def inner(self) -> float:
        """Get the inner radius.

        Returns
        -------
        float
            The inner radius.
        """
        return self.GetInnerRadius()

    @inner.setter
    def inner(self, inner: float):
        """Set the inner radius.

        Parameters
        ----------
        inner : float
            The inner radius.
        """
        self.SetInnerRadius(inner)

    @property
    def outer(self) -> float:
        """Get the outer radius.

        Returns
        -------
        float
            The outer radius.
        """
        return self.GetOuterRadius()

    @outer.setter
    def outer(self, outer: float):
        """Set the outer radius.

        Parameters
        ----------
        outer : float
            The outer radius.
        """
        self.SetOuterRadius(outer)

    @property
    def r_res(self) -> int:
        """Get number of points in radial direction.

        Returns
        -------
        int
            Number of points in radial direction.
        """
        return self.GetRadialResolution()

    @r_res.setter
    def r_res(self, r_res: int):
        """Set number of points in radial direction.

        Parameters
        ----------
        r_res : int
            Number of points in radial direction.
        """
        self.SetRadialResolution(r_res)

    @property
    def c_res(self) -> int:
        """Get number of points in circumferential direction.

        Returns
        -------
        int
            Number of points in circumferential direction.
        """
        return self.GetCircumferentialResolution()

    @c_res.setter
    def c_res(self, c_res: int):
        """Set number of points in circumferential direction.

        Parameters
        ----------
        c_res : int
            Number of points in circumferential direction.
        """
        self.SetCircumferentialResolution(c_res)

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


@no_new_attr
class LineSource(_vtk.vtkLineSource):
    """Create a line.

    .. versionadded:: 0.44

    Parameters
    ----------
    pointa : sequence[float], default: (-0.5, 0.0, 0.0)
        Location in ``[x, y, z]``.

    pointb : sequence[float], default: (0.5, 0.0, 0.0)
        Location in ``[x, y, z]``.

    resolution : int, default: 1
        Number of pieces to divide line into.

    """

    def __init__(
        self,
        pointa=(-0.5, 0.0, 0.0),
        pointb=(0.5, 0.0, 0.0),
        resolution=1,
    ):
        """Initialize source."""
        super().__init__()
        self.pointa = pointa
        self.pointb = pointb
        self.resolution = resolution

    @property
    def pointa(self) -> Sequence[float]:
        """Location in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Location in ``[x, y, z]``.
        """
        return self.GetPoint1()

    @pointa.setter
    def pointa(self, pointa: Sequence[float]):
        """Set the Location in ``[x, y, z]``.

        Parameters
        ----------
        pointa : sequence[float]
            Location in ``[x, y, z]``.
        """
        if np.array(pointa).size != 3:
            raise TypeError('Point A must be a length three tuple of floats.')
        self.SetPoint1(*pointa)

    @property
    def pointb(self) -> Sequence[float]:
        """Location in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Location in ``[x, y, z]``.
        """
        return self.GetPoint2()

    @pointb.setter
    def pointb(self, pointb: Sequence[float]):
        """Set the Location in ``[x, y, z]``.

        Parameters
        ----------
        pointb : sequence[float]
            Location in ``[x, y, z]``.
        """
        if np.array(pointb).size != 3:
            raise TypeError('Point B must be a length three tuple of floats.')
        self.SetPoint2(*pointb)

    @property
    def resolution(self) -> int:
        """Number of pieces to divide line into.

        Returns
        -------
        int
            Number of pieces to divide line into.
        """
        return self.GetResolution()

    @resolution.setter
    def resolution(self, resolution):
        """Set number of pieces to divide line into.

        Parameters
        ----------
        resolution : int
            Number of pieces to divide line into.
        """
        if resolution <= 0:
            raise ValueError('Resolution must be positive')
        self.SetResolution(resolution)

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


@no_new_attr
class SphereSource(_vtk.vtkSphereSource):
    """Sphere source algorithm class.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    radius : float, default: 0.5
        Sphere radius.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center coordinate vector in ``[x, y, z]``.

    theta_resolution : int, default: 30
        Set the number of points in the azimuthal direction (ranging
        from ``start_theta`` to ``end_theta``).

    phi_resolution : int, default: 30
        Set the number of points in the polar direction (ranging from
        ``start_phi`` to ``end_phi``).

    start_theta : float, default: 0.0
        Starting azimuthal angle in degrees ``[0, 360]``.

    end_theta : float, default: 360.0
        Ending azimuthal angle in degrees ``[0, 360]``.

    start_phi : float, default: 0.0
        Starting polar angle in degrees ``[0, 180]``.

    end_phi : float, default: 180.0
        Ending polar angle in degrees ``[0, 180]``.

    See Also
    --------
    pyvista.Icosphere : Sphere created from projection of icosahedron.
    pyvista.SolidSphere : Sphere that fills 3D space.

    Examples
    --------
    Create a sphere using default parameters.

    >>> import pyvista as pv
    >>> sphere = pv.SphereSource()
    >>> sphere.output.plot(show_edges=True)

    Create a quarter sphere by setting ``end_theta``.

    >>> sphere = pv.SphereSource(end_theta=90)
    >>> out = sphere.output.plot(show_edges=True)

    Create a hemisphere by setting ``end_phi``.

    >>> sphere = pv.SphereSource(end_phi=90)
    >>> out = sphere.output.plot(show_edges=True)

    """

    def __init__(
        self,
        radius=0.5,
        center=None,
        theta_resolution=30,
        phi_resolution=30,
        start_theta=0.0,
        end_theta=360.0,
        start_phi=0.0,
        end_phi=180.0,
    ):
        """Initialize the sphere source class."""
        super().__init__()
        self.radius = radius
        if center is not None:  # pragma: no cover
            self.center = center
        self.theta_resolution = theta_resolution
        self.phi_resolution = phi_resolution
        self.start_theta = start_theta
        self.end_theta = end_theta
        self.start_phi = start_phi
        self.end_phi = end_phi

    @property
    def center(self) -> Sequence[float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Center in ``[x, y, z]``.
        """
        if pyvista.vtk_version_info >= (9, 2):
            return self.GetCenter()
        else:  # pragma: no cover
            return (0.0, 0.0, 0.0)

    @center.setter
    def center(self, center: Sequence[float]):
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.
        """
        if pyvista.vtk_version_info >= (9, 2):
            self.SetCenter(center)
        else:  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError(
                'To change vtkSphereSource with `center` requires VTK 9.2 or later.',
            )

    @property
    def radius(self) -> float:
        """Get sphere radius.

        Returns
        -------
        float
            Sphere radius.
        """
        return self.GetRadius()

    @radius.setter
    def radius(self, radius: float):
        """Set sphere radius.

        Parameters
        ----------
        radius : float
            Sphere radius.
        """
        self.SetRadius(radius)

    @property
    def theta_resolution(self) -> int:
        """Get the number of points in the azimuthal direction.

        Returns
        -------
        int
            The number of points in the azimuthal direction.
        """
        return self.GetThetaResolution()

    @theta_resolution.setter
    def theta_resolution(self, theta_resolution: int):
        """Set the number of points in the azimuthal direction.

        Parameters
        ----------
        theta_resolution : int
            The number of points in the azimuthal direction.
        """
        self.SetThetaResolution(theta_resolution)

    @property
    def phi_resolution(self) -> int:
        """Get the number of points in the polar direction.

        Returns
        -------
        int
            The number of points in the polar direction.
        """
        return self.GetPhiResolution()

    @phi_resolution.setter
    def phi_resolution(self, phi_resolution: int):
        """Set the number of points in the polar direction.

        Parameters
        ----------
        phi_resolution : int
            The number of points in the polar direction.
        """
        self.SetPhiResolution(phi_resolution)

    @property
    def start_theta(self) -> float:
        """Get starting azimuthal angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the azimuthal direction.
        """
        return self.GetStartTheta()

    @start_theta.setter
    def start_theta(self, start_theta: float):
        """Set starting azimuthal angle in degrees ``[0, 360]``.

        Parameters
        ----------
        start_theta : float
            The number of points in the azimuthal direction.
        """
        self.SetStartTheta(start_theta)

    @property
    def end_theta(self) -> float:
        """Get ending azimuthal angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the azimuthal direction.
        """
        return self.GetEndTheta()

    @end_theta.setter
    def end_theta(self, end_theta: float):
        """Set ending azimuthal angle in degrees ``[0, 360]``.

        Parameters
        ----------
        end_theta : float
            The number of points in the azimuthal direction.
        """
        self.SetEndTheta(end_theta)

    @property
    def start_phi(self) -> float:
        """Get starting polar angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the polar direction.
        """
        return self.GetStartPhi()

    @start_phi.setter
    def start_phi(self, start_phi: float):
        """Set starting polar angle in degrees ``[0, 360]``.

        Parameters
        ----------
        start_phi : float
            The number of points in the polar direction.
        """
        self.SetStartPhi(start_phi)

    @property
    def end_phi(self) -> float:
        """Get ending polar angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the polar direction.
        """
        return self.GetEndPhi()

    @end_phi.setter
    def end_phi(self, end_phi: float):
        """Set ending polar angle in degrees ``[0, 360]``.

        Parameters
        ----------
        end_phi : float
            The number of points in the polar direction.
        """
        self.SetEndPhi(end_phi)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Sphere surface.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class PolygonSource(_vtk.vtkRegularPolygonSource):
    """Polygon source algorithm class.

    .. versionadded:: 0.44.0

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

    Examples
    --------
    Create an 8 sided polygon.

    >>> import pyvista as pv
    >>> source = pv.PolygonSource(n_sides=8)
    >>> source.output.plot(show_edges=True, line_width=5)
    """

    def __init__(
        self,
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        normal=(0.0, 0.0, 1.0),
        n_sides=6,
        fill=True,
    ):
        """Initialize the polygon source class."""
        super().__init__()
        self.center = center
        self.radius = radius
        self.normal = normal
        self.n_sides = n_sides
        self.fill = fill

    @property
    def center(self) -> Sequence[float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Center in ``[x, y, z]``.
        """
        return self.GetCenter()

    @center.setter
    def center(self, center: Sequence[float]):
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.
        """
        self.SetCenter(center)

    @property
    def radius(self) -> float:
        """Get the radius of the polygon.

        Returns
        -------
        float
            The radius of the polygon.
        """
        return self.GetRadius()

    @radius.setter
    def radius(self, radius: float):
        """Set the radius of the polygon.

        Parameters
        ----------
        radius : float
            The radius of the polygon.
        """
        self.SetRadius(radius)

    @property
    def normal(self) -> Sequence[float]:
        """Get the normal in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Normal in ``[x, y, z]``.
        """
        return self.GetNormal()

    @normal.setter
    def normal(self, normal: Sequence[float]):
        """Set the normal in ``[x, y, z]``.

        Parameters
        ----------
        normal : sequence[float]
            Normal in ``[x, y, z]``.
        """
        self.SetNormal(normal)

    @property
    def n_sides(self) -> int:
        """Get number of sides of the polygon.

        Returns
        -------
        int
            Number of sides of the polygon.
        """
        return self.GetNumberOfSides()

    @n_sides.setter
    def n_sides(self, n_sides: int):
        """Set number of sides of the polygon.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon.
        """
        self.SetNumberOfSides(n_sides)

    @property
    def fill(self) -> bool:
        """Get enable or disable producing filled polygons.

        Returns
        -------
        bool
            Enable or disable producing filled polygons.
        """
        return bool(self.GetGeneratePolygon())

    @fill.setter
    def fill(self, fill: bool):
        """Set enable or disable producing filled polygons.

        Parameters
        ----------
        fill : bool, optional
            Enable or disable producing filled polygons.
        """
        self.SetGeneratePolygon(fill)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Polygon surface.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class PlatonicSolidSource(_vtk.vtkPlatonicSolidSource):
    """Platonic solid source algorithm class.

    .. versionadded:: 0.44.0

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

    Examples
    --------
    Create and plot a dodecahedron.

    >>> import pyvista as pv
    >>> dodeca = pv.PlatonicSolidSource('dodecahedron')
    >>> dodeca.output.plot(categories=True)

    See :ref:`platonic_example` for more examples using this filter.

    """

    _new_attr_exceptions: ClassVar[list[str]] = ['_kinds']

    def __init__(self: PlatonicSolidSource, kind='tetrahedron'):
        """Initialize the platonic solid source class."""
        super().__init__()
        self._kinds: dict[str, int] = {
            'tetrahedron': 0,
            'cube': 1,
            'octahedron': 2,
            'icosahedron': 3,
            'dodecahedron': 4,
        }
        self.kind = kind

    @property
    def kind(self) -> str:
        """Get the kind of Platonic solid to create.

        Returns
        -------
        str
            The kind of Platonic solid to create.
        """
        return list(self._kinds.keys())[self.GetSolidType()]

    @kind.setter
    def kind(self, kind: str | int):
        """Set the kind of Platonic solid to create.

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
        """
        if isinstance(kind, str):
            if kind not in self._kinds:
                raise ValueError(f'Invalid Platonic solid kind "{kind}".')
            kind = self._kinds[kind]
        elif isinstance(kind, int) and kind not in range(5):
            raise ValueError(f'Invalid Platonic solid index "{kind}".')
        elif not isinstance(kind, int):
            raise ValueError(f'Invalid Platonic solid index type "{type(kind).__name__}".')
        self.SetSolidType(kind)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            PlatonicSolid surface.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class PlaneSource(_vtk.vtkPlaneSource):
    """Create a plane source.

    .. versionadded:: 0.44

    Parameters
    ----------
    i_resolution : int, default: 10
        Number of points on the plane in the i direction.

    j_resolution : int, default: 10
        Number of points on the plane in the j direction.

    """

    def __init__(
        self,
        i_resolution=10,
        j_resolution=10,
    ):
        """Initialize source."""
        super().__init__()
        self.i_resolution = i_resolution
        self.j_resolution = j_resolution

    @property
    def i_resolution(self) -> int:
        """Number of points on the plane in the i direction.

        Returns
        -------
        int
            Number of points on the plane in the i direction.
        """
        return self.GetXResolution()

    @i_resolution.setter
    def i_resolution(self, i_resolution: int):
        """Set number of points on the plane in the i direction.

        Parameters
        ----------
        i_resolution : int
            Number of points on the plane in the i direction.
        """
        self.SetXResolution(i_resolution)

    @property
    def j_resolution(self) -> int:
        """Number of points on the plane in the j direction.

        Returns
        -------
        int
            Number of points on the plane in the j direction.
        """
        return self.GetYResolution()

    @j_resolution.setter
    def j_resolution(self, j_resolution: int):
        """Set number of points on the plane in the j direction.

        Parameters
        ----------
        j_resolution : int
            Number of points on the plane in the j direction.
        """
        self.SetYResolution(j_resolution)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class ArrowSource(_vtk.vtkArrowSource):
    """Create a arrow source.

    .. versionadded:: 0.44

    Parameters
    ----------
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
    """

    def __init__(
        self,
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
    ):
        """Initialize source."""
        self.tip_length = tip_length
        self.tip_radius = tip_radius
        self.tip_resolution = tip_resolution
        self.shaft_radius = shaft_radius
        self.shaft_resolution = shaft_resolution

    @property
    def tip_length(self) -> int:
        """Get the length of the tip.

        Returns
        -------
        int
            The length of the tip.
        """
        return self.GetTipLength()

    @tip_length.setter
    def tip_length(self, tip_length: int):
        """Set the length of the tip.

        Parameters
        ----------
        tip_length : int
            The length of the tip.
        """
        self.SetTipLength(tip_length)

    @property
    def tip_radius(self) -> int:
        """Get the radius of the tip.

        Returns
        -------
        int
            The radius of the tip.
        """
        return self.GetTipRadius()

    @tip_radius.setter
    def tip_radius(self, tip_radius: int):
        """Set the radius of the tip.

        Parameters
        ----------
        tip_radius : int
            The radius of the tip.
        """
        self.SetTipRadius(tip_radius)

    @property
    def tip_resolution(self) -> int:
        """Get the number of faces around the tip.

        Returns
        -------
        int
            The number of faces around the tip.
        """
        return self.GetTipResolution()

    @tip_resolution.setter
    def tip_resolution(self, tip_resolution: int):
        """Set the number of faces around the tip.

        Parameters
        ----------
        tip_resolution : int
            The number of faces around the tip.
        """
        self.SetTipResolution(tip_resolution)

    @property
    def shaft_resolution(self) -> int:
        """Get the number of faces around the shaft.

        Returns
        -------
        int
            The number of faces around the shaft.
        """
        return self.GetShaftResolution()

    @shaft_resolution.setter
    def shaft_resolution(self, shaft_resolution: int):
        """Set the number of faces around the shaft.

        Parameters
        ----------
        shaft_resolution : int
            The number of faces around the shaft.
        """
        self.SetShaftResolution(shaft_resolution)

    @property
    def shaft_radius(self) -> int:
        """Get the radius of the shaft.

        Returns
        -------
        int
            The radius of the shaft.
        """
        return self.GetShaftRadius()

    @shaft_radius.setter
    def shaft_radius(self, shaft_radius: int):
        """Set the radius of the shaft.

        Parameters
        ----------
        shaft_radius : int
            The radius of the shaft.
        """
        self.SetShaftRadius(shaft_radius)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class BoxSource(_vtk.vtkTessellatedBoxSource):
    """Create a box source.

    .. versionadded:: 0.44

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

    """

    _new_attr_exceptions: ClassVar[list[str]] = [
        "bounds",
        "_bounds",
    ]

    def __init__(self, bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), level=0, quads=True):
        """Initialize source."""
        super().__init__()
        self.bounds = bounds
        self.level = level
        self.quads = quads

    @property
    def bounds(self) -> BoundsLike:  # numpydoc ignore=RT01
        """Return or set the bounding box of the cube."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: BoundsLike):  # numpydoc ignore=GL08
        if np.array(bounds).size != 6:
            raise TypeError(
                'Bounds must be given as length 6 tuple: (xMin, xMax, yMin, yMax, zMin, zMax)',
            )
        self._bounds = bounds
        self.SetBounds(bounds)

    @property
    def level(self) -> int:
        """Get level of subdivision of the faces.

        Returns
        -------
        int
            Level of subdivision of the faces.
        """
        return self.GetLevel()

    @level.setter
    def level(self, level: int):
        """Set level of subdivision of the faces.

        Parameters
        ----------
        level : int
            Level of subdivision of the faces.
        """
        self.SetLevel(level)

    @property
    def quads(self) -> bool:
        """Flag to tell the source to generate either a quad or two triangle for a set of four points.

        Returns
        -------
        bool
            Flag to tell the source to generate either a quad or two
            triangle for a set of four points.
        """
        return bool(self.GetQuads())

    @quads.setter
    def quads(self, quads: bool):
        """Set flag to tell the source to generate either a quad or two triangle for a set of four points.

        Parameters
        ----------
        quads : bool, optional
            Flag to tell the source to generate either a quad or two
            triangle for a set of four points.
        """
        self.SetQuads(quads)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.
        """
        self.Update()
        return wrap(self.GetOutput())


@no_new_attr
class SuperquadricSource(_vtk.vtkSuperquadricSource):
    """Create superquadric source.

    .. versionadded:: 0.44

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
    """

    def __init__(
        self,
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
        """Initialize source."""
        super().__init__()
        self.center = center
        self.scale = scale
        self.size = size
        self.theta_roundness = theta_roundness
        self.phi_roundness = phi_roundness
        self.theta_resolution = theta_resolution
        self.phi_resolution = phi_resolution
        self.toroidal = toroidal
        self.thickness = thickness

    @property
    def center(self) -> Sequence[float]:
        """Center of the superquadric in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Center of the superquadric in ``[x, y, z]``.
        """
        return self.GetCenter()

    @center.setter
    def center(self, center: Sequence[float]):
        """Set center of the superquadric in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center of the superquadric in ``[x, y, z]``.
        """
        self.SetCenter(center)

    @property
    def scale(self) -> Sequence[float]:
        """Scale factors of the superquadric in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Scale factors of the superquadric in ``[x, y, z]``.
        """
        return self.GetScale()

    @scale.setter
    def scale(self, scale: Sequence[float]):
        """Set scale factors of the superquadric in ``[x, y, z]``.

        Parameters
        ----------
        scale : sequence[float]
           Scale factors of the superquadric in ``[x, y, z]``.
        """
        self.SetScale(scale)

    @property
    def size(self) -> float:
        """Superquadric isotropic size.

        Returns
        -------
        float
            Superquadric isotropic size.
        """
        return self.GetSize()

    @size.setter
    def size(self, size: float):
        """Set superquadric isotropic size.

        Parameters
        ----------
        size : float
            Superquadric isotropic size.
        """
        self.SetSize(size)

    @property
    def theta_roundness(self) -> float:
        """Superquadric east/west roundness.

        Returns
        -------
        float
            Superquadric east/west roundness.
        """
        return self.GetThetaRoundness()

    @theta_roundness.setter
    def theta_roundness(self, theta_roundness: float):
        """Set superquadric east/west roundness.

        Parameters
        ----------
        theta_roundness : float
            Superquadric east/west roundness.
        """
        self.SetThetaRoundness(theta_roundness)

    @property
    def phi_roundness(self) -> float:
        """Superquadric north/south roundness.

        Returns
        -------
        float
            Superquadric north/south roundness.
        """
        return self.GetPhiRoundness()

    @phi_roundness.setter
    def phi_roundness(self, phi_roundness: float):
        """Set superquadric north/south roundness.

        Parameters
        ----------
        phi_roundness : float
            Superquadric north/south roundness.
        """
        self.SetPhiRoundness(phi_roundness)

    @property
    def theta_resolution(self) -> float:
        """Number of points in the longitude direction.

        Returns
        -------
        float
            Number of points in the longitude direction.
        """
        return self.GetThetaResolution()

    @theta_resolution.setter
    def theta_resolution(self, theta_resolution: float):
        """Set number of points in the longitude direction.

        Parameters
        ----------
        theta_resolution : float
            Number of points in the longitude direction.
        """
        self.SetThetaResolution(round(theta_resolution / 4) * 4)

    @property
    def phi_resolution(self) -> float:
        """Number of points in the latitude direction.

        Returns
        -------
        float
            Number of points in the latitude direction.
        """
        return self.GetPhiResolution()

    @phi_resolution.setter
    def phi_resolution(self, phi_resolution: float):
        """Set number of points in the latitude direction.

        Parameters
        ----------
        phi_resolution : float
            Number of points in the latitude direction.
        """
        self.SetPhiResolution(round(phi_resolution / 8) * 8)

    @property
    def toroidal(self) -> bool:
        """Whether or not the superquadric is toroidal (``True``) or ellipsoidal (``False``).

        Returns
        -------
        bool
            Whether or not the superquadric is toroidal (``True``)
            or ellipsoidal (``False``).
        """
        return self.GetToroidal()

    @toroidal.setter
    def toroidal(self, toroidal: bool):
        """Set whether or not the superquadric is toroidal (``True``) or ellipsoidal (``False``).

        Parameters
        ----------
        toroidal : bool
            Whether or not the superquadric is toroidal (``True``)
            or ellipsoidal (``False``).
        """
        self.SetToroidal(toroidal)

    @property
    def thickness(self):
        """Superquadric ring thickness.

        Returns
        -------
        float
            Superquadric ring thickness.
        """
        return self.GetThickness()

    @thickness.setter
    def thickness(self, thickness: float):
        """Set superquadric ring thickness.

        Parameters
        ----------
        thickness : float
            Superquadric ring thickness.
        """
        self.SetThickness(thickness)

    @property
    def output(self):
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.
        """
        self.Update()
        return wrap(self.GetOutput())


class _AxisEnum(IntEnum):
    x = 0
    y = 1
    z = 2


class _PartEnum(IntEnum):
    shaft = 0
    tip = 1


class AxesGeometrySource:
    """Create axes geometry source.

    Source for generating fully 3-dimensional axes shaft and tip geometry.

    By default, the shafts are cylinders and the tips are cones, though other geometries
    such as spheres and cubes are also supported. The use of an arbitrary dataset
    for the shafts and/or tips is also supported.

    Unlike :class:`pyvista.AxesActor`, the output from this source is a
    :class:`pyvista.MultiBlock`, not an actor, and does not support colors or labels.
    The generated axes are "true-to-scale" by default, i.e. a shaft with a
    radius of 0.1 will truly have a radius of 0.1, and the axes may be oriented
    arbitrarily in space (this is not the case for :class:`pyvista.AxesActor`).

    Parameters
    ----------
    shaft_type : str | pyvista.DataSet, default: 'cylinder'
        Shaft type for all axes. Can be any of the following:

        - ``'cylinder'``
        - ``'sphere'``
        - ``'hemisphere'``
        - ``'cone'``
        - ``'pyramid'``
        - ``'cube'``
        - ``'octahedron'``

        Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet` may be
        specified. In this case, the dataset must be oriented such that it "points" in
        the positive z direction.

    shaft_radius : float, default: 0.025
        Radius of the axes shafts.

    shaft_length : float | VectorLike[float], default: 0.8
        Length of the shaft for each axis.

    tip_type : str | pyvista.DataSet, default: 'cone'
        Tip type for all axes. Can be any of the following:

        - ``'cylinder'``
        - ``'sphere'``
        - ``'hemisphere'``
        - ``'cone'``
        - ``'pyramid'``
        - ``'cube'``
        - ``'octahedron'``

        Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet` may be
        specified. In this case, the dataset must be oriented such that it "points" in
        the positive z direction.

    tip_radius : float, default: 0.1
        Radius of the axes tips.

    tip_length : float | VectorLike[float], default: 0.2
        Length of the tip for each axis.

    symmetric : bool, default: False
        Mirror the axes such that they extend to negative values.

    symmetric_bounds : bool, default: False
        Make the bounds of the axes symmetric. This option is similar to
        :attr:`symmetric`, except only the bounds are made to be symmetric,
        not the actual geometry. Has no effect if :attr:`symmetric` is ``True``.

    """

    GeometryTypes = Literal[
        'cylinder',
        'sphere',
        'hemisphere',
        'cone',
        'pyramid',
        'cube',
        'octahedron',
    ]
    GEOMETRY_TYPES: ClassVar[tuple[str]] = get_args(GeometryTypes)

    def __init__(
        self,
        *,
        shaft_type: GeometryTypes | pyvista.DataSet = 'cylinder',
        shaft_radius: float = 0.025,
        shaft_length: float | VectorLike[float] = 0.8,
        tip_type: GeometryTypes | pyvista.DataSet = 'cone',
        tip_radius: float = 0.1,
        tip_length: float | VectorLike[float] = 0.2,
        symmetric: bool = False,
        symmetric_bounds: bool = False,
    ):
        super().__init__()
        # Init datasets
        names = ['x_shaft', 'y_shaft', 'z_shaft', 'x_tip', 'y_tip', 'z_tip']
        polys = [pyvista.PolyData() for _ in range(len(names))]
        self._output = pyvista.MultiBlock(dict(zip(names, polys)))

        # Store shaft/tip references in separate vars for convenience
        self._shaft_datasets = (polys[0], polys[1], polys[2])
        self._tip_datasets = (polys[3], polys[4], polys[5])

        # Also store datasets for internal use
        self._shaft_datasets_normalized = [pyvista.PolyData() for _ in range(3)]
        self._tip_datasets_normalized = [pyvista.PolyData() for _ in range(3)]

        # Set geometry-dependent params
        self.shaft_type = shaft_type  # type: ignore[assignment]
        self.shaft_radius = shaft_radius
        self.shaft_length = shaft_length  # type: ignore[assignment]
        self.tip_type = tip_type  # type: ignore[assignment]
        self.tip_radius = tip_radius
        self.tip_length = tip_length  # type: ignore[assignment]

        # Set flags
        self._symmetric = symmetric
        self._symmetric_bounds = symmetric_bounds

    def __repr__(self):
        """Representation of the axes."""
        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            f"  Shaft type:                 '{self.shaft_type}'",
            f"  Shaft radius:               {self.shaft_radius}",
            f"  Shaft length:               {self.shaft_length}",
            f"  Tip type:                   '{self.tip_type}'",
            f"  Tip radius:                 {self.tip_radius}",
            f"  Tip length:                 {self.tip_length}",
            f"  Symmetric:                  {self.symmetric}",
            f"  Symmetric bounds:           {self.symmetric_bounds}",
        ]
        return '\n'.join(attr)

    @property
    def symmetric(self) -> bool:  # numpydoc ignore=RT01
        """Mirror the axes such that they extend to negative values.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource(symmetric=True)
        >>> axes_geometry_source.output.plot()
        """
        return self._symmetric

    @symmetric.setter
    def symmetric(self, val: bool):  # numpydoc ignore=GL08
        self._symmetric = val

    @property
    def symmetric_bounds(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable symmetry in the axes bounds.

        This option is similar to :attr:`symmetric`, except instead of making
        the axes parts symmetric, only the bounds of the axes are made to be
        symmetric. This is achieved by adding a single invisible cell to each tip
        dataset along each axis to simulate the symmetry. Setting this
        parameter primarily affects camera positioning and is useful if the
        axes are used as a widget, as it allows for the axes to rotate
        about its origin.

        Examples
        --------
        Get the symmetric bounds of the axes.

        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource(
        ...     symmetric_bounds=True
        ... )
        >>> axes_geometry_source.output.bounds
        (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        >>> axes_geometry_source.output.center
        array([0.0, 0.0, 0.0])

        Get the asymmetric bounds.

        >>> axes_geometry_source.symmetric_bounds = False
        >>> axes_geometry_source.output.bounds
        (-0.10000000149011612, 1.0, -0.10000000149011612, 1.0, -0.10000000149011612, 1.0)

        >>> axes_geometry_source.output.center
        array([0.45, 0.45, 0.45])

        Show the difference in camera positioning with and without
        symmetric bounds. Orientation is added for visualization.

        Create actors.
        >>> axes_sym = pv.AxesAssembly(
        ...     orientation=(90, 0, 0), symmetric_bounds=True
        ... )
        >>> axes_asym = pv.AxesAssembly(
        ...     orientation=(90, 0, 0), symmetric_bounds=False
        ... )

        Show multi-window plot.
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_text("Symmetric bounds")
        >>> _ = pl.add_actor(axes_sym)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_text("Asymmetric bounds")
        >>> _ = pl.add_actor(axes_asym)
        >>> pl.show()
        """
        return self._symmetric_bounds

    @symmetric_bounds.setter
    def symmetric_bounds(self, val: bool):  # numpydoc ignore=GL08
        self._symmetric_bounds = val

    @property
    def shaft_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Length of the shaft for each axis.

        Value must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.shaft_length
        (0.8, 0.8, 0.8)
        >>> axes_geometry_source.shaft_length = 0.7
        >>> axes_geometry_source.shaft_length
        (0.7, 0.7, 0.7)
        >>> axes_geometry_source.shaft_length = (1.0, 0.9, 0.5)
        >>> axes_geometry_source.shaft_length
        (1.0, 0.9, 0.5)
        """
        return tuple(self._shaft_length.tolist())

    @shaft_length.setter
    def shaft_length(self, length: float | VectorLike[float]):  # numpydoc ignore=GL08
        self._shaft_length: NumpyArray[float] = _validation.validate_array3(
            length,
            broadcast=True,
            must_be_in_range=[0.0, np.inf],
            name="Shaft length",
        )

    @property
    def tip_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Length of the tip for each axis.

        Value must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.tip_length
        (0.2, 0.2, 0.2)
        >>> axes_geometry_source.tip_length = 0.3
        >>> axes_geometry_source.tip_length
        (0.3, 0.3, 0.3)
        >>> axes_geometry_source.tip_length = (0.1, 0.4, 0.2)
        >>> axes_geometry_source.tip_length
        (0.1, 0.4, 0.2)
        """
        return tuple(self._tip_length.tolist())

    @tip_length.setter
    def tip_length(self, length: float | VectorLike[float]):  # numpydoc ignore=GL08
        self._tip_length: NumpyArray[float] = _validation.validate_array3(
            length,
            broadcast=True,
            must_be_in_range=[0.0, np.inf],
            name="Tip length",
        )

    @property
    def tip_radius(self) -> float:  # numpydoc ignore=RT01
        """Radius of the axes tips.

        Value must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.tip_radius
        0.1
        >>> axes_geometry_source.tip_radius = 0.2
        >>> axes_geometry_source.tip_radius
        0.2
        """
        return self._tip_radius

    @tip_radius.setter
    def tip_radius(self, radius: float):  # numpydoc ignore=GL08
        _validation.check_range(radius, (0, float('inf')), name='tip radius')
        self._tip_radius = radius

    @property
    def shaft_radius(self):  # numpydoc ignore=RT01
        """Radius of the axes shafts.

        Value must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.shaft_radius
        0.025
        >>> axes_geometry_source.shaft_radius = 0.05
        >>> axes_geometry_source.shaft_radius
        0.05
        """
        return self._shaft_radius

    @shaft_radius.setter
    def shaft_radius(self, radius):  # numpydoc ignore=GL08
        _validation.check_range(radius, (0, float('inf')), name='shaft radius')
        self._shaft_radius = radius

    @property
    def shaft_type(self) -> str:  # numpydoc ignore=RT01
        """Shaft type for all axes.

        Must be a string, e.g. ``'cylinder'`` or ``'cube'`` or any other supported
        geometry. Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet`
        may also be specified. In this case, the dataset must be oriented such that it
        "points" in the positive z direction.

        Examples
        --------
        Show a list of all shaft type options.

        >>> import pyvista as pv
        >>> pv.AxesGeometrySource.GEOMETRY_TYPES
        ('cylinder', 'sphere', 'hemisphere', 'cone', 'pyramid', 'cube', 'octahedron')

        Show the default shaft type and modify it.

        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.shaft_type
        'cylinder'
        >>> axes_geometry_source.shaft_type = 'cube'
        >>> axes_geometry_source.shaft_type
        'cube'

        Set the shaft type to any 3-dimensional dataset.

        >>> axes_geometry_source.shaft_type = pv.Superquadric()
        >>> axes_geometry_source.shaft_type
        'custom'
        """
        return self._shaft_type

    @shaft_type.setter
    def shaft_type(self, shaft_type: GeometryTypes | pyvista.DataSet):  # numpydoc ignore=GL08
        self._shaft_type = self._set_normalized_datasets(part=_PartEnum.shaft, geometry=shaft_type)

    @property
    def tip_type(self) -> str:  # numpydoc ignore=RT01
        """Tip type for all axes.

        Must be a string, e.g. ``'cone'`` or ``'sphere'`` or any other supported
        geometry. Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet`
        may also be specified. In this case, the dataset must be oriented such that it
        "points" in the positive z direction.

        Examples
        --------
        Show a list of all shaft type options.

        >>> import pyvista as pv
        >>> pv.AxesGeometrySource.GEOMETRY_TYPES
        ('cylinder', 'sphere', 'hemisphere', 'cone', 'pyramid', 'cube', 'octahedron')

        Show the default tip type and modify it.

        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.tip_type
        'cone'
        >>> axes_geometry_source.tip_type = 'sphere'
        >>> axes_geometry_source.tip_type
        'sphere'

        Set the tip type to any 3-dimensional dataset.

        >>> axes_geometry_source.tip_type = pv.Text3D('O')
        >>> axes_geometry_source.tip_type
        'custom'

        >>> axes_geometry_source.output.plot(cpos='xy')
        """
        return self._tip_type

    @tip_type.setter
    def tip_type(self, tip_type: str | pyvista.DataSet):  # numpydoc ignore=GL08
        self._tip_type = self._set_normalized_datasets(part=_PartEnum.tip, geometry=tip_type)

    def _set_normalized_datasets(self, part: _PartEnum, geometry: str | pyvista.DataSet):
        geometry_name, new_datasets = AxesGeometrySource._make_axes_parts(geometry)
        datasets = (
            self._shaft_datasets_normalized
            if part == _PartEnum.shaft
            else self._tip_datasets_normalized
        )
        datasets[_AxisEnum.x].copy_from(new_datasets[_AxisEnum.x])
        datasets[_AxisEnum.y].copy_from(new_datasets[_AxisEnum.y])
        datasets[_AxisEnum.z].copy_from(new_datasets[_AxisEnum.z])
        return geometry_name

    def _reset_shaft_and_tip_geometry(self):
        # Store local copies of properties for iterating
        shaft_radius, shaft_length = self.shaft_radius, self.shaft_length
        tip_radius, tip_length = (
            self.tip_radius,
            self.tip_length,
        )

        nested_datasets = [self._shaft_datasets, self._tip_datasets]
        nested_datasets_normalized = [
            self._shaft_datasets_normalized,
            self._tip_datasets_normalized,
        ]
        for part_type, axis in itertools.product(_PartEnum, _AxisEnum):
            # Reset part by copying from the normalized version
            part_normalized = nested_datasets_normalized[part_type][axis]
            part = nested_datasets[part_type][axis]
            part.copy_from(part_normalized)

            # Offset so axis bounds are [0, 1]
            part.points[:, axis] += 0.5

            # Scale by length along axis, scale by radius off-axis
            radius, length = (
                (shaft_radius, shaft_length)
                if part_type == _PartEnum.shaft
                else (tip_radius, tip_length)
            )
            diameter = radius * 2
            scale = [diameter] * 3
            scale[axis] = length[axis]
            part.scale(scale, inplace=True)

            if part_type == _PartEnum.tip:
                # Move tip to end of shaft
                part.points[:, axis] += shaft_length[axis]

            if self.symmetric:
                # Flip and append to part
                origin = [0, 0, 0]
                normal = [0, 0, 0]
                normal[axis] = 1
                flipped = part.flip_normal(normal=normal, point=origin)
                part.append_polydata(flipped, inplace=True)
            elif self.symmetric_bounds and part_type == _PartEnum.tip:
                # For this feature we add a single degenerate cell
                # at the tip and flip its position
                point = [0, 0, 0]
                total_length = shaft_length[axis] + tip_length[axis]
                point[axis] = total_length
                flipped_point = np.array([point]) * -1  # Flip point
                point_id = part.n_points
                new_face = [3, point_id, point_id, point_id]

                # Update mesh
                part.points = np.append(part.points, flipped_point, axis=0)
                part.faces = np.append(part.faces, new_face)

    def update(self):
        """Update the output of the source."""
        self._reset_shaft_and_tip_geometry()

    @property
    def output(self) -> pyvista.MultiBlock:
        """Get the output of the source.

        The output is a :class:`pyvista.MultiBlock` with six blocks: one for each part
        of the axes. The blocks are ordered by shafts first then tips, and in x-y-z order.
        Specifically, they are named as follows:

            (``'x_shaft'``, ``'y_shaft'``, ``'z_shaft'``, ``'x_tip'``, ``'y_tip'``, ``'z_tip'``)

        The source is automatically updated by :meth:`update` prior to returning
        the output.

        Returns
        -------
        pyvista.MultiBlock
            Composite mesh with separate shaft and tip datasets.
        """
        self.update()
        return self._output

    @staticmethod
    def _make_default_part(geometry: str) -> pyvista.PolyData:
        """Create part geometry with its length axis pointing in the +z direction."""
        resolution = 50
        if geometry == 'cylinder':
            return pyvista.Cylinder(direction=(0, 0, 1), resolution=resolution)
        elif geometry == 'sphere':
            return pyvista.Sphere(phi_resolution=resolution, theta_resolution=resolution)
        elif geometry == 'hemisphere':
            return pyvista.SolidSphere(end_phi=90).extract_geometry()
        elif geometry == 'cone':
            return pyvista.Cone(direction=(0, 0, 1), resolution=resolution)
        elif geometry == 'pyramid':
            return pyvista.Pyramid().extract_geometry()
        elif geometry == 'cube':
            return pyvista.Cube()
        elif geometry == 'octahedron':
            mesh = pyvista.Octahedron()
            mesh.cell_data.remove('FaceIndex')
            return mesh
        else:
            _validation.check_contains(
                item=geometry,
                container=AxesGeometrySource.GEOMETRY_TYPES,
                name='Geometry',
            )
            raise NotImplementedError(
                f"Geometry '{geometry}' is not implemented"
            )  # pragma: no cover

    @staticmethod
    def _make_any_part(geometry: str | pyvista.DataSet) -> tuple[str, pyvista.PolyData]:
        part: pyvista.DataSet
        part_poly: pyvista.PolyData
        if isinstance(geometry, str):
            name = geometry
            part = AxesGeometrySource._make_default_part(
                geometry,
            )
        elif isinstance(geometry, pyvista.DataSet):
            name = 'custom'
            part = geometry.copy()
        else:
            raise TypeError(
                f"Geometry must be a string or pyvista.DataSet. Got {type(geometry)}.",
            )
        part_poly = part if isinstance(part, pyvista.PolyData) else part.extract_geometry()
        part_poly = AxesGeometrySource._normalize_part(part_poly)
        return name, part_poly

    @staticmethod
    def _normalize_part(part: pyvista.PolyData) -> pyvista.PolyData:
        """Scale and translate part to have origin-centered bounding box with edge length one."""
        # Center points at origin
        # mypy ignore since pyvista_ndarray is not compatible with np.ndarray, see GH#5434
        part.points -= part.center  # type: ignore[misc]

        # Scale so bounding box edges have length one
        bnds = part.bounds
        axis_length = np.array((bnds[1] - bnds[0], bnds[3] - bnds[2], bnds[5] - bnds[4]))
        if np.any(axis_length < 1e-8):
            raise ValueError(f"Custom axes part must be 3D. Got bounds: {bnds}.")
        part.scale(np.reciprocal(axis_length), inplace=True)
        return part

    @staticmethod
    def _make_axes_parts(
        geometry: str | pyvista.DataSet,
    ) -> tuple[str, tuple[pyvista.PolyData, pyvista.PolyData, pyvista.PolyData]]:
        """Return three axis-aligned normalized parts centered at the origin."""
        name, part_z = AxesGeometrySource._make_any_part(geometry)
        part_x = part_z.copy().rotate_y(90)
        part_y = part_z.copy().rotate_x(-90)
        return name, (part_x, part_y, part_z)
