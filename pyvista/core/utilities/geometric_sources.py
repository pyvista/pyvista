"""Provides an easy way of generating several geometric sources.

Also includes some pure-python helpers.

"""

from __future__ import annotations

from enum import IntEnum
import itertools
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import cast
from typing import get_args

import numpy as np
from vtkmodules.vtkRenderingFreeType import vtkVectorText

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import BoundsTuple
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _check_range
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import _reciprocal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet
    from pyvista.core.pointset import PolyData


SINGLE_PRECISION = _vtk.vtkAlgorithm.SINGLE_PRECISION
DOUBLE_PRECISION = _vtk.vtkAlgorithm.DOUBLE_PRECISION


def translate(
    surf: DataSet,
    center: VectorLike[float] = (0.0, 0.0, 0.0),
    direction: VectorLike[float] = (1.0, 0.0, 0.0),
) -> None:
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

    surf.transform(trans, inplace=True)
    if not np.allclose(center, [0.0, 0.0, 0.0]):
        surf.points += np.array(center, dtype=surf.points.dtype)


if _vtk.vtk_version_info < (9, 3):

    class CapsuleSource(_NoNewAttrMixin, _vtk.vtkCapsuleSource):  # type: ignore[misc]
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

        @_deprecate_positional_args
        def __init__(  # noqa: PLR0917
            self: CapsuleSource,
            center: VectorLike[float] = (0.0, 0.0, 0.0),
            direction: VectorLike[float] = (1.0, 0.0, 0.0),
            radius: float = 0.5,
            cylinder_length: float = 1.0,
            theta_resolution: int = 30,
            phi_resolution: int = 30,
        ) -> None:
            """Initialize the capsule source class."""
            super().__init__()
            self.center = center
            self.direction = direction
            self.radius = radius
            self.cylinder_length = cylinder_length
            self.theta_resolution = theta_resolution
            self.phi_resolution = phi_resolution

        @property
        def center(self: CapsuleSource) -> tuple[float, float, float]:
            """Get the center in ``[x, y, z]``. Axis of the capsule passes through this point.

            Returns
            -------
            tuple[float, float, float]
                Center in ``[x, y, z]``. Axis of the capsule passes through this
                point.

            """
            return self.GetCenter()

        @center.setter
        def center(self: CapsuleSource, center: VectorLike[float]) -> None:
            """Set the center in ``[x, y, z]``. Axis of the capsule passes through this point.

            Parameters
            ----------
            center : sequence[float]
                Center in ``[x, y, z]``. Axis of the capsule passes through this
                point.

            """
            self.SetCenter(*center)

        @property
        def direction(self: CapsuleSource) -> tuple[float, float, float]:
            """Get the direction vector in ``[x, y, z]``. Orientation vector of the capsule.

            Returns
            -------
            sequence[float]
                Direction vector in ``[x, y, z]``. Orientation vector of the
                capsule.

            """
            return self._direction

        @direction.setter
        def direction(self: CapsuleSource, direction: VectorLike[float]) -> None:
            """Set the direction in ``[x, y, z]``. Axis of the capsule passes through this point.

            Parameters
            ----------
            direction : sequence[float]
                Direction vector in ``[x, y, z]``. Orientation vector of the
                capsule.

            """
            valid_direction = _validation.validate_array3(
                direction, dtype_out=float, to_tuple=True
            )
            self._direction = cast('tuple[float, float, float]', valid_direction)

        @property
        def cylinder_length(self: CapsuleSource) -> float:
            """Get the cylinder length along the capsule in its specified direction.

            Returns
            -------
            float
                Cylinder length along the capsule in its specified direction.

            """
            return self.GetCylinderLength()

        @cylinder_length.setter
        def cylinder_length(self: CapsuleSource, length: float) -> None:
            """Set the cylinder length of the capsule.

            Parameters
            ----------
            length : float
                Cylinder length of the capsule.

            """
            self.SetCylinderLength(length)

        @property
        def radius(self: CapsuleSource) -> float:
            """Get base radius of the capsule.

            Returns
            -------
            float
                Base radius of the capsule.

            """
            return self.GetRadius()

        @radius.setter
        def radius(self: CapsuleSource, radius: float) -> None:
            """Set base radius of the capsule.

            Parameters
            ----------
            radius : float
                Base radius of the capsule.

            """
            self.SetRadius(radius)

        @property
        def theta_resolution(self: CapsuleSource) -> int:
            """Get the number of points in the azimuthal direction.

            Returns
            -------
            int
                The number of points in the azimuthal direction.

            """
            return self.GetThetaResolution()

        @theta_resolution.setter
        def theta_resolution(self: CapsuleSource, theta_resolution: int) -> None:
            """Set the number of points in the azimuthal direction.

            Parameters
            ----------
            theta_resolution : int
                The number of points in the azimuthal direction.

            """
            self.SetThetaResolution(theta_resolution)

        @property
        def phi_resolution(self: CapsuleSource) -> int:
            """Get the number of points in the polar direction.

            Returns
            -------
            int
                The number of points in the polar direction.

            """
            return self.GetPhiResolution()

        @phi_resolution.setter
        def phi_resolution(self: CapsuleSource, phi_resolution: int) -> None:
            """Set the number of points in the polar direction.

            Parameters
            ----------
            phi_resolution : int
                The number of points in the polar direction.

            """
            self.SetPhiResolution(phi_resolution)

        @property
        def output(self: CapsuleSource) -> PolyData:
            """Get the output data object for a port on this algorithm.

            Returns
            -------
            pyvista.PolyData
                Capsule surface.

            """
            self.Update()
            return wrap(self.GetOutput())


class ConeSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkConeSource):
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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: ConeSource,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        direction: VectorLike[float] = (1.0, 0.0, 0.0),
        height: float = 1.0,
        radius: float | None = None,
        capping: bool = True,  # noqa: FBT001, FBT002
        angle: float | None = None,
        resolution: int = 6,
    ) -> None:
        """Initialize the cone source class."""
        super().__init__()
        self.center = center
        self.direction = direction
        self.height = height
        self.capping = capping
        if angle is not None and radius is not None:
            msg = 'Both radius and angle cannot be specified. They are mutually exclusive.'
            raise ValueError(msg)
        elif angle is not None and radius is None:
            self.angle = angle
        elif angle is None and radius is not None:
            self.radius = radius
        elif angle is None and radius is None:
            self.radius = 0.5
        self.resolution = resolution

    @property
    def center(self: ConeSource) -> tuple[float, float, float]:
        """Get the center in ``[x, y, z]``. Axis of the cone passes through this point.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``. Axis of the cone passes through this
            point.

        """
        return self.GetCenter()

    @center.setter
    def center(self: ConeSource, center: VectorLike[float]) -> None:
        """Set the center in ``[x, y, z]``. Axis of the cone passes through this point.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``. Axis of the cone passes through this
            point.

        """
        self.SetCenter(*center)

    @property
    def direction(self: ConeSource) -> tuple[float, float, float]:
        """Get the direction vector in ``[x, y, z]``. Orientation vector of the cone.

        Returns
        -------
        sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cone.

        """
        return self.GetDirection()

    @direction.setter
    def direction(self: ConeSource, direction: VectorLike[float]) -> None:
        """Set the direction in ``[x, y, z]``. Axis of the cone passes through this point.

        Parameters
        ----------
        direction : sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cone.

        """
        self.SetDirection(*direction)

    @property
    def height(self: ConeSource) -> float:
        """Get the height along the cone in its specified direction.

        Returns
        -------
        float
            Height along the cone in its specified direction.

        """
        return self.GetHeight()

    @height.setter
    def height(self: ConeSource, height: float) -> None:
        """Set the height of the cone.

        Parameters
        ----------
        height : float
            Height of the cone.

        """
        self.SetHeight(height)

    @property
    def radius(self: ConeSource) -> float:
        """Get base radius of the cone.

        Returns
        -------
        float
            Base radius of the cone.

        """
        return self.GetRadius()

    @radius.setter
    def radius(self: ConeSource, radius: float) -> None:
        """Set base radius of the cone.

        Parameters
        ----------
        radius : float
            Base radius of the cone.

        """
        self.SetRadius(radius)

    @property
    def capping(self: ConeSource) -> bool:
        """Enable or disable the capping the base of the cone with a polygon.

        Returns
        -------
        bool
            Enable or disable the capping the base of the cone with a
            polygon.

        """
        return bool(self.GetCapping())

    @capping.setter
    def capping(self: ConeSource, capping: bool) -> None:
        """Set base capping of the cone.

        Parameters
        ----------
        capping : bool, optional
            Enable or disable the capping the base of the cone with a
            polygon.

        """
        self.SetCapping(capping)

    @property
    def angle(self: ConeSource) -> float:
        """Get the angle in degrees between the axis of the cone and a generatrix.

        Returns
        -------
        float
            The angle in degrees between the axis of the cone and a
            generatrix.

        """
        return self.GetAngle()

    @angle.setter
    def angle(self: ConeSource, angle: float) -> None:
        """Set the angle in degrees between the axis of the cone and a generatrix.

        Parameters
        ----------
        angle : float, optional
            The angle in degrees between the axis of the cone and a
            generatrix.

        """
        self.SetAngle(angle)

    @property
    def resolution(self: ConeSource) -> int:
        """Get number of points on the circular face of the cone.

        Returns
        -------
        int
            Number of points on the circular face of the cone.

        """
        return self.GetResolution()

    @resolution.setter
    def resolution(self: ConeSource, resolution: int) -> None:
        """Set number of points on the circular face of the cone.

        Parameters
        ----------
        resolution : int
            Number of points on the circular face of the cone.

        """
        self.SetResolution(resolution)

    @property
    def output(self: ConeSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Cone surface.

        """
        self.Update()
        return wrap(self.GetOutput())


class CylinderSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkCylinderSource):
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
    >>> _ = pl.add_mesh(pv.CylinderSource().output, show_edges=True, line_width=5)
    >>> pl.show()

    The above examples are similar in terms of their behavior.

    """

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: CylinderSource,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        direction: VectorLike[float] = (1.0, 0.0, 0.0),
        radius: float = 0.5,
        height: float = 1.0,
        capping: bool = True,  # noqa: FBT001, FBT002
        resolution: int = 100,
    ) -> None:
        """Initialize the cylinder source class."""
        super().__init__()
        self.center = center
        self.direction = direction
        self.radius = radius
        self.height = height
        self.resolution = resolution
        self.capping = capping

    @property
    def center(self: CylinderSource) -> tuple[float, float, float]:
        """Get location of the centroid in ``[x, y, z]``.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``. Axis of the cylinder passes through this
            point.

        """
        return self._center

    @center.setter
    def center(self: CylinderSource, center: VectorLike[float]) -> None:
        """Set location of the centroid in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``. Axis of the cylinder passes through this
            point.

        """
        valid_center = _validation.validate_array3(center, dtype_out=float, to_tuple=True)
        self._center = cast('tuple[float, float, float]', valid_center)

    @property
    def direction(self: CylinderSource) -> tuple[float, float, float]:
        """Get the direction vector in ``[x, y, z]``. Orientation vector of the cylinder.

        Returns
        -------
        sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cylinder.

        """
        return self._direction

    @direction.setter
    def direction(self: CylinderSource, direction: VectorLike[float]) -> None:
        """Set the direction in ``[x, y, z]``. Axis of the cylinder passes through this point.

        Parameters
        ----------
        direction : sequence[float]
            Direction vector in ``[x, y, z]``. Orientation vector of the
            cylinder.

        """
        valid_direction = _validation.validate_array3(direction, dtype_out=float, to_tuple=True)
        self._direction = cast('tuple[float, float, float]', valid_direction)

    @property
    def radius(self: CylinderSource) -> float:
        """Get radius of the cylinder.

        Returns
        -------
        float
            Radius of the cylinder.

        """
        return self.GetRadius()

    @radius.setter
    def radius(self: CylinderSource, radius: float) -> None:
        """Set radius of the cylinder.

        Parameters
        ----------
        radius : float
            Radius of the cylinder.

        """
        self.SetRadius(radius)

    @property
    def height(self: CylinderSource) -> float:
        """Get the height of the cylinder.

        Returns
        -------
        float
            Height of the cylinder.

        """
        return self.GetHeight()

    @height.setter
    def height(self: CylinderSource, height: float) -> None:
        """Set the height of the cylinder.

        Parameters
        ----------
        height : float
            Height of the cylinder.

        """
        self.SetHeight(height)

    @property
    def resolution(self: CylinderSource) -> int:
        """Get number of points on the circular face of the cylinder.

        Returns
        -------
        int
            Number of points on the circular face of the cone.

        """
        return self.GetResolution()

    @resolution.setter
    def resolution(self: CylinderSource, resolution: int) -> None:
        """Set number of points on the circular face of the cone.

        Parameters
        ----------
        resolution : int
            Number of points on the circular face of the cone.

        """
        self.SetResolution(resolution)

    @property
    def capping(self: CylinderSource) -> bool:
        """Get cap cylinder ends with polygons.

        Returns
        -------
        bool
            Cap cylinder ends with polygons.

        """
        return bool(self.GetCapping())

    @capping.setter
    def capping(self: CylinderSource, capping: bool) -> None:
        """Set cap cylinder ends with polygons.

        Parameters
        ----------
        capping : bool, optional
            Cap cylinder ends with polygons.

        """
        self.SetCapping(capping)

    @property
    def capsule_cap(self: CylinderSource) -> bool:
        """Get whether the capping should make the cylinder a capsule.

        .. versionadded:: 0.44.0

        Returns
        -------
        bool
            Capsule cap.

        """
        return bool(self.GetCapsuleCap())

    @capsule_cap.setter
    def capsule_cap(self: CylinderSource, capsule_cap: bool) -> None:
        """Set whether the capping should make the cylinder a capsule.

        Parameters
        ----------
        capsule_cap : bool
            Capsule cap.

        """
        self.SetCapsuleCap(capsule_cap)

    @property
    def output(self: CylinderSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Cylinder surface.

        """
        self.Update()
        return wrap(self.GetOutput())


class MultipleLinesSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkLineSource):
    """Multiple lines source algorithm class.

    Parameters
    ----------
    points : array_like[float], default: [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        List of points defining a broken line.

    """

    def __init__(self: MultipleLinesSource, points: MatrixLike[float] | None = None) -> None:
        """Initialize the multiple lines source class."""
        if points is None:
            points = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        super().__init__()
        self.points = points

    @property
    def points(self: MultipleLinesSource) -> NumpyArray[float]:
        """Return the points defining a broken line.

        Returns
        -------
        np.ndarray
            Points defining a broken line.

        """
        return _vtk.vtk_to_numpy(self.GetPoints().GetData())

    @points.setter
    def points(self: MultipleLinesSource, points: MatrixLike[float] | VectorLike[float]) -> None:
        """Set the list of points defining a broken line.

        Parameters
        ----------
        points : VectorLike[float] | MatrixLike[float]
            List of points defining a broken line.

        """
        points, _ = _coerce_pointslike_arg(points)
        if not (len(points) >= 2):
            msg = '>=2 points need to define multiple lines.'
            raise ValueError(msg)
        self.SetPoints(pyvista.vtk_points(points))

    @property
    def output(self: MultipleLinesSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Line mesh.

        """
        self.Update()
        return wrap(self.GetOutput())


class Text3DSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, vtkVectorText):
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

    @_deprecate_positional_args(allowed=['string'])
    def __init__(  # noqa: PLR0917
        self: Text3DSource,
        string: str | None = None,
        depth: float | None = None,
        width: float | None = None,
        height: float | None = None,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        normal: VectorLike[float] = (0.0, 0.0, 1.0),
        process_empty_string: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize source."""
        super().__init__()

        self._output = pyvista.PolyData()

        # Set params
        self.string = '' if string is None else string
        self._process_empty_string = process_empty_string
        self.center = center
        self.normal = normal
        self._height = height
        self._width = width
        self._depth = depth
        self._modified = True

    def __setattr__(self: Text3DSource, name: str, value: Any) -> None:
        """Override to set modified flag and disable setting new attributes."""
        if hasattr(self, name) and name != '_modified':
            # Set modified flag
            old_value = getattr(self, name)
            if not np.array_equal(old_value, value):
                object.__setattr__(self, name, value)
                object.__setattr__(self, '_modified', True)
        else:
            super().__setattr__(name, value)

    @property
    def string(self: Text3DSource) -> str:  # numpydoc ignore=RT01
        """Return or set the text string."""
        return self.GetText()

    @string.setter
    def string(self: Text3DSource, string: str | None) -> None:
        self.SetText('' if string is None else string)

    @property
    def process_empty_string(self: Text3DSource) -> bool:  # numpydoc ignore=RT01
        """Return or set flag to control behavior when empty strings are set.

        When :attr:`string` is empty or only contains whitespace, the :attr:`output`
        mesh will be empty. This can cause the bounds of the output to be undefined.

        If ``True``, the output is modified to instead have a single point located
        at :attr:`center`.

        """
        return self._process_empty_string

    @process_empty_string.setter
    def process_empty_string(self: Text3DSource, value: bool) -> None:
        self._process_empty_string = value

    @property
    def center(
        self: Text3DSource,
    ) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the center of the text.

        The center is defined as the middle of the axis-aligned bounding box
        of the text.
        """
        return self._center

    @center.setter
    def center(self: Text3DSource, center: VectorLike[float]) -> None:
        valid_center = _validation.validate_array3(center, dtype_out=float, to_tuple=True)
        self._center = cast('tuple[float, float, float]', valid_center)

    @property
    def normal(
        self: Text3DSource,
    ) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the normal direction of the text.

        The normal direction is parallel to the :attr:`depth` of the text, and
        points away from the front surface of the text.
        """
        return self._normal

    @normal.setter
    def normal(self: Text3DSource, normal: VectorLike[float]) -> None:
        normal_ = _validation.validate_array3(normal, dtype_out=float, to_tuple=True)
        self._normal = cast('tuple[float, float, float]', normal_)

    @property
    def width(self: Text3DSource) -> float | None:  # numpydoc ignore=RT01
        """Return or set the width of the text."""
        return self._width

    @width.setter
    def width(self: Text3DSource, width: float | None) -> None:
        _check_range(
            width, rng=(0, float('inf')), parm_name='width'
        ) if width is not None else None
        self._width = width

    @property
    def height(self: Text3DSource) -> float | None:  # numpydoc ignore=RT01
        """Return or set the height of the text."""
        return self._height

    @height.setter
    def height(self: Text3DSource, height: float | None) -> None:
        (
            _check_range(height, rng=(0, float('inf')), parm_name='height')
            if height is not None
            else None
        )
        self._height = height

    @property
    def depth(self: Text3DSource) -> float | None:  # numpydoc ignore=RT01
        """Return or set the depth of the text."""
        return self._depth

    @depth.setter
    def depth(self: Text3DSource, depth: float | None) -> None:
        _check_range(
            depth, rng=(0, float('inf')), parm_name='depth'
        ) if depth is not None else None
        self._depth = depth

    def update(self: Text3DSource) -> None:
        """Update the output of the source."""
        if self._modified:
            is_empty_string = self.string == '' or self.string.isspace()
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
    def output(self: Text3DSource) -> PolyData:  # numpydoc ignore=RT01
        """Get the output of the source.

        The source is automatically updated by :meth:`update` prior
        to returning the output.
        """
        self.update()
        return self._output

    def _transform_output(self: Text3DSource) -> None:
        """Scale, rotate, and translate the output mesh."""
        # Create aliases
        out, width, height, depth = self._output, self.width, self.height, self.depth
        width_set, height_set, depth_set = (
            width is not None,
            height is not None,
            depth is not None,
        )

        # Scale mesh
        size_w, size_h, size_d = out.bounds_size
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


class CubeSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkCubeSource):
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
        arguments are ignored. ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: CubeSource,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        x_length: float = 1.0,
        y_length: float = 1.0,
        z_length: float = 1.0,
        bounds: VectorLike[float] | None = None,
        point_dtype: str = 'float32',
    ) -> None:
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
    def bounds(self: CubeSource) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return or set the bounding box of the cube."""
        bnds = [0.0] * 6
        self.GetBounds(bnds)
        return BoundsTuple(*bnds)

    @bounds.setter
    def bounds(self: CubeSource, bounds: VectorLike[float]) -> None:
        if np.array(bounds).size != 6:
            msg = (
                'Bounds must be given as length 6 tuple: '
                '(x_min, x_max, y_min, y_max, z_min, z_max)'
            )
            raise TypeError(msg)
        self.SetBounds(bounds)  # type: ignore[arg-type]

    @property
    def center(self: CubeSource) -> tuple[float, float, float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``.

        """
        return self.GetCenter()

    @center.setter
    def center(self: CubeSource, center: VectorLike[float]) -> None:
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.

        """
        self.SetCenter(*center)

    @property
    def x_length(self: CubeSource) -> float:
        """Get the x length along the cube in its specified direction.

        Returns
        -------
        float
            XLength along the cone in its specified direction.

        """
        return self.GetXLength()

    @x_length.setter
    def x_length(self: CubeSource, x_length: float) -> None:
        """Set the x length of the cube.

        Parameters
        ----------
        x_length : float
            XLength of the cone.

        """
        self.SetXLength(x_length)

    @property
    def y_length(self: CubeSource) -> float:
        """Get the y length along the cube in its specified direction.

        Returns
        -------
        float
            YLength along the cone in its specified direction.

        """
        return self.GetYLength()

    @y_length.setter
    def y_length(self: CubeSource, y_length: float) -> None:
        """Set the y length of the cube.

        Parameters
        ----------
        y_length : float
            YLength of the cone.

        """
        self.SetYLength(y_length)

    @property
    def z_length(self: CubeSource) -> float:
        """Get the z length along the cube in its specified direction.

        Returns
        -------
        float
            ZLength along the cone in its specified direction.

        """
        return self.GetZLength()

    @z_length.setter
    def z_length(self: CubeSource, z_length: float) -> None:
        """Set the z length of the cube.

        Parameters
        ----------
        z_length : float
            ZLength of the cone.

        """
        self.SetZLength(z_length)

    @property
    def output(self: CubeSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Cube surface.

        """
        self.Update()
        return wrap(self.GetOutput())

    @property
    def point_dtype(self: CubeSource) -> str:
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
        }[precision]  # type: ignore[index]

    @point_dtype.setter
    def point_dtype(self: CubeSource, point_dtype: str) -> None:
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
            msg = "Point dtype must be either 'float32' or 'float64'"
            raise ValueError(msg)
        precision = {
            'float32': SINGLE_PRECISION,
            'float64': DOUBLE_PRECISION,
        }[point_dtype]
        self.SetOutputPointsPrecision(precision)


class DiscSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkDiskSource):
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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: DiscSource,
        center: VectorLike[float] | None = None,
        inner: float = 0.25,
        outer: float = 0.5,
        r_res: int = 1,
        c_res: int = 6,
    ) -> None:
        """Initialize the disc source class."""
        super().__init__()
        if center is not None:
            self.center = center
        self.inner = inner
        self.outer = outer
        self.r_res = r_res
        self.c_res = c_res

    @property
    def center(self: DiscSource) -> tuple[float, float, float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``.

        """
        return self.GetCenter()

    @center.setter
    def center(self: DiscSource, center: VectorLike[float]) -> None:
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.

        """
        self.SetCenter(*center)

    @property
    def inner(self: DiscSource) -> float:
        """Get the inner radius.

        Returns
        -------
        float
            The inner radius.

        """
        return self.GetInnerRadius()

    @inner.setter
    def inner(self: DiscSource, inner: float) -> None:
        """Set the inner radius.

        Parameters
        ----------
        inner : float
            The inner radius.

        """
        self.SetInnerRadius(inner)

    @property
    def outer(self: DiscSource) -> float:
        """Get the outer radius.

        Returns
        -------
        float
            The outer radius.

        """
        return self.GetOuterRadius()

    @outer.setter
    def outer(self: DiscSource, outer: float) -> None:
        """Set the outer radius.

        Parameters
        ----------
        outer : float
            The outer radius.

        """
        self.SetOuterRadius(outer)

    @property
    def r_res(self: DiscSource) -> int:
        """Get number of points in radial direction.

        Returns
        -------
        int
            Number of points in radial direction.

        """
        return self.GetRadialResolution()

    @r_res.setter
    def r_res(self: DiscSource, r_res: int) -> None:
        """Set number of points in radial direction.

        Parameters
        ----------
        r_res : int
            Number of points in radial direction.

        """
        self.SetRadialResolution(r_res)

    @property
    def c_res(self: DiscSource) -> int:
        """Get number of points in circumferential direction.

        Returns
        -------
        int
            Number of points in circumferential direction.

        """
        return self.GetCircumferentialResolution()

    @c_res.setter
    def c_res(self: DiscSource, c_res: int) -> None:
        """Set number of points in circumferential direction.

        Parameters
        ----------
        c_res : int
            Number of points in circumferential direction.

        """
        self.SetCircumferentialResolution(c_res)

    @property
    def output(self: DiscSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Line mesh.

        """
        self.Update()
        return wrap(self.GetOutput())


class LineSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkLineSource):
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
        self: LineSource,
        pointa: VectorLike[float] = (-0.5, 0.0, 0.0),
        pointb: VectorLike[float] = (0.5, 0.0, 0.0),
        resolution: int = 1,
    ) -> None:
        """Initialize source."""
        super().__init__()
        self.pointa = pointa
        self.pointb = pointb
        self.resolution = resolution

    @property
    def pointa(self: LineSource) -> tuple[float, float, float]:
        """Location in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Location in ``[x, y, z]``.

        """
        return self.GetPoint1()

    @pointa.setter
    def pointa(self: LineSource, pointa: VectorLike[float]) -> None:
        """Set the Location in ``[x, y, z]``.

        Parameters
        ----------
        pointa : sequence[float]
            Location in ``[x, y, z]``.

        """
        self.SetPoint1(*pointa)

    @property
    def pointb(self: LineSource) -> tuple[float, float, float]:
        """Location in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Location in ``[x, y, z]``.

        """
        return self.GetPoint2()

    @pointb.setter
    def pointb(self: LineSource, pointb: VectorLike[float]) -> None:
        """Set the Location in ``[x, y, z]``.

        Parameters
        ----------
        pointb : sequence[float]
            Location in ``[x, y, z]``.

        """
        self.SetPoint2(*pointb)

    @property
    def resolution(self: LineSource) -> int:
        """Number of pieces to divide line into.

        Returns
        -------
        int
            Number of pieces to divide line into.

        """
        return self.GetResolution()

    @resolution.setter
    def resolution(self: LineSource, resolution: int) -> None:
        """Set number of pieces to divide line into.

        Parameters
        ----------
        resolution : int
            Number of pieces to divide line into.

        """
        if resolution <= 0:
            msg = 'Resolution must be positive'
            raise ValueError(msg)
        self.SetResolution(resolution)

    @property
    def output(self: LineSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Line mesh.

        """
        self.Update()
        return wrap(self.GetOutput())


class SphereSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkSphereSource):
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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: SphereSource,
        radius: float = 0.5,
        center: VectorLike[float] | None = None,
        theta_resolution: int = 30,
        phi_resolution: int = 30,
        start_theta: float = 0.0,
        end_theta: float = 360.0,
        start_phi: float = 0.0,
        end_phi: float = 180.0,
    ) -> None:
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
    def center(self: SphereSource) -> tuple[float, float, float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``.

        """
        return self.GetCenter()

    @center.setter
    def center(self: SphereSource, center: VectorLike[float]) -> None:
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.

        """
        self.SetCenter(*center)

    @property
    def radius(self: SphereSource) -> float:
        """Get sphere radius.

        Returns
        -------
        float
            Sphere radius.

        """
        return self.GetRadius()

    @radius.setter
    def radius(self: SphereSource, radius: float) -> None:
        """Set sphere radius.

        Parameters
        ----------
        radius : float
            Sphere radius.

        """
        self.SetRadius(radius)

    @property
    def theta_resolution(self: SphereSource) -> int:
        """Get the number of points in the azimuthal direction.

        Returns
        -------
        int
            The number of points in the azimuthal direction.

        """
        return self.GetThetaResolution()

    @theta_resolution.setter
    def theta_resolution(self: SphereSource, theta_resolution: int) -> None:
        """Set the number of points in the azimuthal direction.

        Parameters
        ----------
        theta_resolution : int
            The number of points in the azimuthal direction.

        """
        self.SetThetaResolution(theta_resolution)

    @property
    def phi_resolution(self: SphereSource) -> int:
        """Get the number of points in the polar direction.

        Returns
        -------
        int
            The number of points in the polar direction.

        """
        return self.GetPhiResolution()

    @phi_resolution.setter
    def phi_resolution(self: SphereSource, phi_resolution: int) -> None:
        """Set the number of points in the polar direction.

        Parameters
        ----------
        phi_resolution : int
            The number of points in the polar direction.

        """
        self.SetPhiResolution(phi_resolution)

    @property
    def start_theta(self: SphereSource) -> float:
        """Get starting azimuthal angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the azimuthal direction.

        """
        return self.GetStartTheta()

    @start_theta.setter
    def start_theta(self: SphereSource, start_theta: float) -> None:
        """Set starting azimuthal angle in degrees ``[0, 360]``.

        Parameters
        ----------
        start_theta : float
            The number of points in the azimuthal direction.

        """
        self.SetStartTheta(start_theta)

    @property
    def end_theta(self: SphereSource) -> float:
        """Get ending azimuthal angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the azimuthal direction.

        """
        return self.GetEndTheta()

    @end_theta.setter
    def end_theta(self: SphereSource, end_theta: float) -> None:
        """Set ending azimuthal angle in degrees ``[0, 360]``.

        Parameters
        ----------
        end_theta : float
            The number of points in the azimuthal direction.

        """
        self.SetEndTheta(end_theta)

    @property
    def start_phi(self: SphereSource) -> float:
        """Get starting polar angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the polar direction.

        """
        return self.GetStartPhi()

    @start_phi.setter
    def start_phi(self: SphereSource, start_phi: float) -> None:
        """Set starting polar angle in degrees ``[0, 360]``.

        Parameters
        ----------
        start_phi : float
            The number of points in the polar direction.

        """
        self.SetStartPhi(start_phi)

    @property
    def end_phi(self: SphereSource) -> float:
        """Get ending polar angle in degrees ``[0, 360]``.

        Returns
        -------
        float
            The number of points in the polar direction.

        """
        return self.GetEndPhi()

    @end_phi.setter
    def end_phi(self: SphereSource, end_phi: float) -> None:
        """Set ending polar angle in degrees ``[0, 360]``.

        Parameters
        ----------
        end_phi : float
            The number of points in the polar direction.

        """
        self.SetEndPhi(end_phi)

    @property
    def output(self: SphereSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Sphere surface.

        """
        self.Update()
        return wrap(self.GetOutput())


class PolygonSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkRegularPolygonSource):
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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: PolygonSource,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        normal: VectorLike[float] = (0.0, 0.0, 1.0),
        n_sides: int = 6,
        fill: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the polygon source class."""
        super().__init__()
        self.center = center
        self.radius = radius
        self.normal = normal
        self.n_sides = n_sides
        self.fill = fill

    @property
    def center(self: PolygonSource) -> tuple[float, float, float]:
        """Get the center in ``[x, y, z]``.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``.

        """
        return self.GetCenter()

    @center.setter
    def center(self: PolygonSource, center: VectorLike[float]) -> None:
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.

        """
        self.SetCenter(*center)

    @property
    def radius(self: PolygonSource) -> float:
        """Get the radius of the polygon.

        Returns
        -------
        float
            The radius of the polygon.

        """
        return self.GetRadius()

    @radius.setter
    def radius(self: PolygonSource, radius: float) -> None:
        """Set the radius of the polygon.

        Parameters
        ----------
        radius : float
            The radius of the polygon.

        """
        self.SetRadius(radius)

    @property
    def normal(self: PolygonSource) -> tuple[float, float, float]:
        """Get the normal in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Normal in ``[x, y, z]``.

        """
        return self.GetNormal()

    @normal.setter
    def normal(self: PolygonSource, normal: VectorLike[float]) -> None:
        """Set the normal in ``[x, y, z]``.

        Parameters
        ----------
        normal : sequence[float]
            Normal in ``[x, y, z]``.

        """
        self.SetNormal(*normal)

    @property
    def n_sides(self: PolygonSource) -> int:
        """Get number of sides of the polygon.

        Returns
        -------
        int
            Number of sides of the polygon.

        """
        return self.GetNumberOfSides()

    @n_sides.setter
    def n_sides(self: PolygonSource, n_sides: int) -> None:
        """Set number of sides of the polygon.

        Parameters
        ----------
        n_sides : int
            Number of sides of the polygon.

        """
        self.SetNumberOfSides(n_sides)

    @property
    def fill(self: PolygonSource) -> bool:
        """Get enable or disable producing filled polygons.

        Returns
        -------
        bool
            Enable or disable producing filled polygons.

        """
        return bool(self.GetGeneratePolygon())

    @fill.setter
    def fill(self: PolygonSource, fill: bool) -> None:
        """Set enable or disable producing filled polygons.

        Parameters
        ----------
        fill : bool, optional
            Enable or disable producing filled polygons.

        """
        self.SetGeneratePolygon(fill)

    @property
    def output(self: PolygonSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Polygon surface.

        """
        self.Update()
        return wrap(self.GetOutput())


class PlatonicSolidSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkPlatonicSolidSource):
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

    See :ref:`create_platonic_solids_example` for more examples using this filter.

    """

    def __init__(self: PlatonicSolidSource, kind: str = 'tetrahedron') -> None:
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
    def kind(self: PlatonicSolidSource) -> str:
        """Get the kind of Platonic solid to create.

        Returns
        -------
        str
            The kind of Platonic solid to create.

        """
        return list(self._kinds.keys())[self.GetSolidType()]

    @kind.setter
    def kind(self: PlatonicSolidSource, kind: str | int) -> None:
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
        if isinstance(kind, int):
            if kind not in range(5):
                msg = f'Invalid Platonic solid index "{kind}".'
                raise ValueError(msg)
            self.SetSolidType(kind)
            return

        if isinstance(kind, str):
            if kind not in self._kinds:
                msg = f'Invalid Platonic solid kind "{kind}".'
                raise ValueError(msg)
            self.SetSolidType(self._kinds[kind])
            return

        msg = f'Invalid Platonic solid index type "{type(kind).__name__}".'  # type: ignore[unreachable]
        raise ValueError(msg)

    @property
    def output(self: PlatonicSolidSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            PlatonicSolid surface.

        """
        self.Update()
        return wrap(self.GetOutput())


class PlaneSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkPlaneSource):
    """Create a plane source.

    The plane is defined by specifying an origin point, and then
    two other points that, together with the origin, define two
    axes for the plane (magnitude and direction). These axes do
    not have to be orthogonal - so you can create a parallelogram.
    The axes must not be parallel.

    .. versionadded:: 0.44

    Parameters
    ----------
    i_resolution : int, default: 10
        Number of points on the plane in the i direction.

    j_resolution : int, default: 10
        Number of points on the plane in the j direction.

    center : sequence[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``.

    origin : sequence[float], default: (-0.5, -0.5, 0.0)
        Origin in ``[x, y, z]``.

    point_a : sequence[float], default: (0.5, -0.5, 0.0)
        Location in ``[x, y, z]``.

    point_b : sequence[float], default: (-0.5, 0.5, 0.0)
        Location in ``[x, y, z]``.

    """

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: PlaneSource,
        i_resolution: int = 10,
        j_resolution: int = 10,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        origin: VectorLike[float] = (-0.5, -0.5, 0.0),
        point_a: VectorLike[float] = (0.5, -0.5, 0.0),
        point_b: VectorLike[float] = (-0.5, 0.5, 0.0),
    ) -> None:
        """Initialize source."""
        super().__init__()
        self.i_resolution = i_resolution
        self.j_resolution = j_resolution
        self.center = center
        self.origin = origin
        self.point_a = point_a
        self.point_b = point_b

    @property
    def i_resolution(self: PlaneSource) -> int:
        """Number of points on the plane in the i direction.

        Returns
        -------
        int
            Number of points on the plane in the i direction.

        """
        return self.GetXResolution()

    @i_resolution.setter
    def i_resolution(self: PlaneSource, i_resolution: int) -> None:
        """Set number of points on the plane in the i direction.

        Parameters
        ----------
        i_resolution : int
            Number of points on the plane in the i direction.

        """
        self.SetXResolution(i_resolution)

    @property
    def j_resolution(self: PlaneSource) -> int:
        """Number of points on the plane in the j direction.

        Returns
        -------
        int
            Number of points on the plane in the j direction.

        """
        return self.GetYResolution()

    @j_resolution.setter
    def j_resolution(self: PlaneSource, j_resolution: int) -> None:
        """Set number of points on the plane in the j direction.

        Parameters
        ----------
        j_resolution : int
            Number of points on the plane in the j direction.

        """
        self.SetYResolution(j_resolution)

    @property
    def center(self: PlaneSource) -> tuple[float, float, float]:
        """Get the center in ``[x, y, z]``.

        The center of the plane is translated to the specified point.

        Returns
        -------
        tuple[float, float, float]
            Center in ``[x, y, z]``.

        """
        return self.GetCenter()

    @center.setter
    def center(self: PlaneSource, center: VectorLike[float]) -> None:
        """Set the center in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center in ``[x, y, z]``.

        """
        self.SetCenter(*center)

    @property
    def origin(self: PlaneSource) -> tuple[float, float, float]:
        """Get the origin in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Origin in ``[x, y, z]``.

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self: PlaneSource, origin: VectorLike[float]) -> None:
        """Set the origin in ``[x, y, z]``.

        Parameters
        ----------
        origin : sequence[float]
            Origin in ``[x, y, z]``.

        """
        self.SetOrigin(*origin)

    @property
    def point_a(self: PlaneSource) -> tuple[float, float, float]:
        """Get the Location in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Location in ``[x, y, z]``.

        """
        return self.GetPoint1()

    @point_a.setter
    def point_a(self: PlaneSource, point_a: VectorLike[float]) -> None:
        """Set the Location in ``[x, y, z]``.

        Parameters
        ----------
        point_a : sequence[float]
            Location in ``[x, y, z]``.

        """
        self.SetPoint1(*point_a)

    @property
    def point_b(self: PlaneSource) -> tuple[float, float, float]:
        """Get the Location in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Location in ``[x, y, z]``.

        """
        return self.GetPoint2()

    @point_b.setter
    def point_b(self: PlaneSource, point_b: VectorLike[float]) -> None:
        """Set the Location in ``[x, y, z]``.

        Parameters
        ----------
        point_b : sequence[float]
            Location in ``[x, y, z]``.

        """
        self.SetPoint2(point_b)  # type: ignore[call-overload]

    @property
    def output(self: PlaneSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.

        """
        self.Update()
        return wrap(self.GetOutput())

    @property
    def normal(
        self: PlaneSource,
    ) -> tuple[float, float, float]:  # numpydoc ignore: RT01
        """Get the plane's normal vector."""
        origin = np.array(self.origin)
        v1 = self.point_a - origin
        v2 = self.point_b - origin
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        # Avoid div by zero and return +z normal by default
        return tuple((normal / norm).tolist()) if norm else (0.0, 0.0, 1.0)

    def flip_normal(self: PlaneSource) -> None:
        """Flip the plane's normal.

        This method modifies the plane's :attr:`point_a` and :attr:`point_b` by
        swapping them.
        """
        point_a = self.point_a
        self.point_a = self.point_b
        self.point_b = point_a

    def push(self: PlaneSource, distance: float) -> None:  # numpydoc ignore: PR01
        """Translate the plane in the direction of the normal by the distance specified."""
        _validation.validate_number(distance, dtype_out=float)
        self.center = (self.center + np.array(self.normal) * distance).tolist()


class ArrowSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkArrowSource):
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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: ArrowSource,
        tip_length: float = 0.25,
        tip_radius: float = 0.1,
        tip_resolution: int = 20,
        shaft_radius: float = 0.05,
        shaft_resolution: int = 20,
    ) -> None:
        """Initialize source."""
        self.tip_length = tip_length
        self.tip_radius = tip_radius
        self.tip_resolution = tip_resolution
        self.shaft_radius = shaft_radius
        self.shaft_resolution = shaft_resolution

    @property
    def tip_length(self: ArrowSource) -> float:
        """Get the length of the tip.

        Returns
        -------
        int
            The length of the tip.

        """
        return self.GetTipLength()

    @tip_length.setter
    def tip_length(self: ArrowSource, tip_length: float) -> None:
        """Set the length of the tip.

        Parameters
        ----------
        tip_length : int
            The length of the tip.

        """
        self.SetTipLength(tip_length)

    @property
    def tip_radius(self: ArrowSource) -> float:
        """Get the radius of the tip.

        Returns
        -------
        int
            The radius of the tip.

        """
        return self.GetTipRadius()

    @tip_radius.setter
    def tip_radius(self: ArrowSource, tip_radius: float) -> None:
        """Set the radius of the tip.

        Parameters
        ----------
        tip_radius : int
            The radius of the tip.

        """
        self.SetTipRadius(tip_radius)

    @property
    def tip_resolution(self: ArrowSource) -> int:
        """Get the number of faces around the tip.

        Returns
        -------
        int
            The number of faces around the tip.

        """
        return self.GetTipResolution()

    @tip_resolution.setter
    def tip_resolution(self: ArrowSource, tip_resolution: int) -> None:
        """Set the number of faces around the tip.

        Parameters
        ----------
        tip_resolution : int
            The number of faces around the tip.

        """
        self.SetTipResolution(tip_resolution)

    @property
    def shaft_resolution(self: ArrowSource) -> int:
        """Get the number of faces around the shaft.

        Returns
        -------
        int
            The number of faces around the shaft.

        """
        return self.GetShaftResolution()

    @shaft_resolution.setter
    def shaft_resolution(self: ArrowSource, shaft_resolution: int) -> None:
        """Set the number of faces around the shaft.

        Parameters
        ----------
        shaft_resolution : int
            The number of faces around the shaft.

        """
        self.SetShaftResolution(shaft_resolution)

    @property
    def shaft_radius(self: ArrowSource) -> float:
        """Get the radius of the shaft.

        Returns
        -------
        int
            The radius of the shaft.

        """
        return self.GetShaftRadius()

    @shaft_radius.setter
    def shaft_radius(self: ArrowSource, shaft_radius: float) -> None:
        """Set the radius of the shaft.

        Parameters
        ----------
        shaft_radius : int
            The radius of the shaft.

        """
        self.SetShaftRadius(shaft_radius)

    @property
    def output(self: ArrowSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.

        """
        self.Update()
        return wrap(self.GetOutput())


class BoxSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkTessellatedBoxSource):
    """Create a box source.

    .. versionadded:: 0.44

    Parameters
    ----------
    bounds : sequence[float], default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        Specify the bounds of the box.
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

    level : int, default: 0
        Level of subdivision of the faces.

    quads : bool, default: True
        Flag to tell the source to generate either a quad or two
        triangle for a set of four points.

    """

    @_deprecate_positional_args(allowed=['bounds'])
    def __init__(
        self: BoxSource,
        bounds: VectorLike[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        level: int = 0,
        quads: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize source."""
        super().__init__()
        self.bounds = bounds
        self.level = level
        self.quads = quads

    @property
    def bounds(self: BoxSource) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return or set the bounds of the box."""
        return BoundsTuple(*self.GetBounds())

    @bounds.setter
    def bounds(self: BoxSource, bounds: VectorLike[float]) -> None:
        if np.array(bounds).size != 6:
            msg = (
                'Bounds must be given as length 6 tuple: '
                '(x_min, x_max, y_min, y_max, z_min, z_max)'
            )
            raise TypeError(msg)
        self.SetBounds(bounds)  # type: ignore[arg-type]

    @property
    def level(self: BoxSource) -> int:
        """Get level of subdivision of the faces.

        Returns
        -------
        int
            Level of subdivision of the faces.

        """
        return self.GetLevel()

    @level.setter
    def level(self: BoxSource, level: int) -> None:
        """Set level of subdivision of the faces.

        Parameters
        ----------
        level : int
            Level of subdivision of the faces.

        """
        self.SetLevel(level)

    @property
    def quads(self: BoxSource) -> bool:
        """Flag to tell the source to generate a quad or two triangle for a set of four points.

        Returns
        -------
        bool
            Flag to tell the source to generate either a quad or two
            triangle for a set of four points.

        """
        return bool(self.GetQuads())

    @quads.setter
    def quads(self: BoxSource, quads: bool) -> None:
        """Set flag to tell the source to generate a quad or two triangle for a set of four points.

        Parameters
        ----------
        quads : bool, optional
            Flag to tell the source to generate either a quad or two
            triangle for a set of four points.

        """
        self.SetQuads(quads)

    @property
    def output(self: BoxSource) -> PolyData:
        """Get the output data object for a port on this algorithm.

        Returns
        -------
        pyvista.PolyData
            Plane mesh.

        """
        self.Update()
        return wrap(self.GetOutput())


class SuperquadricSource(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkSuperquadricSource):
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

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self: SuperquadricSource,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        scale: VectorLike[float] = (1.0, 1.0, 1.0),
        size: float = 0.5,
        theta_roundness: float = 1.0,
        phi_roundness: float = 1.0,
        theta_resolution: int = 16,
        phi_resolution: int = 16,
        toroidal: bool = False,  # noqa: FBT001, FBT002
        thickness: float = 1 / 3,
    ) -> None:
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
    def center(self: SuperquadricSource) -> tuple[float, float, float]:
        """Center of the superquadric in ``[x, y, z]``.

        Returns
        -------
        tuple[float, float, float]
            Center of the superquadric in ``[x, y, z]``.

        """
        return self.GetCenter()

    @center.setter
    def center(self: SuperquadricSource, center: VectorLike[float]) -> None:
        """Set center of the superquadric in ``[x, y, z]``.

        Parameters
        ----------
        center : sequence[float]
            Center of the superquadric in ``[x, y, z]``.

        """
        self.SetCenter(*center)

    @property
    def scale(self: SuperquadricSource) -> tuple[float, float, float]:
        """Scale factors of the superquadric in ``[x, y, z]``.

        Returns
        -------
        sequence[float]
            Scale factors of the superquadric in ``[x, y, z]``.

        """
        return self.GetScale()

    @scale.setter
    def scale(self: SuperquadricSource, scale: VectorLike[float]) -> None:
        """Set scale factors of the superquadric in ``[x, y, z]``.

        Parameters
        ----------
        scale : sequence[float]
           Scale factors of the superquadric in ``[x, y, z]``.

        """
        self.SetScale(*scale)

    @property
    def size(self: SuperquadricSource) -> float:
        """Superquadric isotropic size.

        Returns
        -------
        float
            Superquadric isotropic size.

        """
        return self.GetSize()

    @size.setter
    def size(self: SuperquadricSource, size: float) -> None:
        """Set superquadric isotropic size.

        Parameters
        ----------
        size : float
            Superquadric isotropic size.

        """
        self.SetSize(size)

    @property
    def theta_roundness(self: SuperquadricSource) -> float:
        """Superquadric east/west roundness.

        Returns
        -------
        float
            Superquadric east/west roundness.

        """
        return self.GetThetaRoundness()

    @theta_roundness.setter
    def theta_roundness(self: SuperquadricSource, theta_roundness: float) -> None:
        """Set superquadric east/west roundness.

        Parameters
        ----------
        theta_roundness : float
            Superquadric east/west roundness.

        """
        self.SetThetaRoundness(theta_roundness)

    @property
    def phi_roundness(self: SuperquadricSource) -> float:
        """Superquadric north/south roundness.

        Returns
        -------
        float
            Superquadric north/south roundness.

        """
        return self.GetPhiRoundness()

    @phi_roundness.setter
    def phi_roundness(self: SuperquadricSource, phi_roundness: float) -> None:
        """Set superquadric north/south roundness.

        Parameters
        ----------
        phi_roundness : float
            Superquadric north/south roundness.

        """
        self.SetPhiRoundness(phi_roundness)

    @property
    def theta_resolution(self: SuperquadricSource) -> float:
        """Number of points in the longitude direction.

        Returns
        -------
        float
            Number of points in the longitude direction.

        """
        return self.GetThetaResolution()

    @theta_resolution.setter
    def theta_resolution(self: SuperquadricSource, theta_resolution: float) -> None:
        """Set number of points in the longitude direction.

        Parameters
        ----------
        theta_resolution : float
            Number of points in the longitude direction.

        """
        self.SetThetaResolution(round(theta_resolution / 4) * 4)

    @property
    def phi_resolution(self: SuperquadricSource) -> float:
        """Number of points in the latitude direction.

        Returns
        -------
        float
            Number of points in the latitude direction.

        """
        return self.GetPhiResolution()

    @phi_resolution.setter
    def phi_resolution(self: SuperquadricSource, phi_resolution: float) -> None:
        """Set number of points in the latitude direction.

        Parameters
        ----------
        phi_resolution : float
            Number of points in the latitude direction.

        """
        self.SetPhiResolution(round(phi_resolution / 8) * 8)

    @property
    def toroidal(self: SuperquadricSource) -> bool:
        """Whether or not the superquadric is toroidal (``True``) or ellipsoidal (``False``).

        Returns
        -------
        bool
            Whether or not the superquadric is toroidal (``True``)
            or ellipsoidal (``False``).

        """
        return self.GetToroidal()  # type: ignore[return-value]

    @toroidal.setter
    def toroidal(self: SuperquadricSource, toroidal: bool) -> None:
        """Set whether or not the superquadric is toroidal (``True``) or ellipsoidal (``False``).

        Parameters
        ----------
        toroidal : bool
            Whether or not the superquadric is toroidal (``True``)
            or ellipsoidal (``False``).

        """
        self.SetToroidal(toroidal)

    @property
    def thickness(self: SuperquadricSource) -> float:
        """Superquadric ring thickness.

        Returns
        -------
        float
            Superquadric ring thickness.

        """
        return self.GetThickness()

    @thickness.setter
    def thickness(self: SuperquadricSource, thickness: float) -> None:
        """Set superquadric ring thickness.

        Parameters
        ----------
        thickness : float
            Superquadric ring thickness.

        """
        self.SetThickness(thickness)

    @property
    def output(self: SuperquadricSource) -> PolyData:
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


class AxesGeometrySource(_NoNewAttrMixin):
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
        self: AxesGeometrySource,
        *,
        shaft_type: GeometryTypes | DataSet = 'cylinder',
        shaft_radius: float = 0.025,
        shaft_length: float | VectorLike[float] = 0.8,
        tip_type: GeometryTypes | DataSet = 'cone',
        tip_radius: float = 0.1,
        tip_length: float | VectorLike[float] = 0.2,
        symmetric: bool = False,
        symmetric_bounds: bool = False,
    ) -> None:
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
        self.shaft_type = shaft_type
        self.shaft_radius = shaft_radius
        self.shaft_length = shaft_length
        self.tip_type = tip_type
        self.tip_radius = tip_radius
        self.tip_length = tip_length

        # Set flags
        self._symmetric = symmetric
        self._symmetric_bounds = symmetric_bounds

    def __repr__(self: AxesGeometrySource) -> str:
        """Representation of the axes."""
        attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f"  Shaft type:                 '{self.shaft_type}'",
            f'  Shaft radius:               {self.shaft_radius}',
            f'  Shaft length:               {self.shaft_length}',
            f"  Tip type:                   '{self.tip_type}'",
            f'  Tip radius:                 {self.tip_radius}',
            f'  Tip length:                 {self.tip_length}',
            f'  Symmetric:                  {self.symmetric}',
            f'  Symmetric bounds:           {self.symmetric_bounds}',
        ]
        return '\n'.join(attr)

    @property
    def symmetric(self: AxesGeometrySource) -> bool:  # numpydoc ignore=RT01
        """Mirror the axes such that they extend to negative values.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource(symmetric=True)
        >>> axes_geometry_source.output.plot()

        """
        return self._symmetric

    @symmetric.setter
    def symmetric(self: AxesGeometrySource, val: bool) -> None:
        self._symmetric = val

    @property
    def symmetric_bounds(self: AxesGeometrySource) -> bool:  # numpydoc ignore=RT01
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
        >>> axes_geometry_source = pv.AxesGeometrySource(symmetric_bounds=True)
        >>> axes_geometry_source.output.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  1.0,
                    y_min = -1.0,
                    y_max =  1.0,
                    z_min = -1.0,
                    z_max =  1.0)

        >>> axes_geometry_source.output.center
        (0.0, 0.0, 0.0)

        Get the asymmetric bounds.

        >>> axes_geometry_source.symmetric_bounds = False
        >>> axes_geometry_source.output.bounds
        BoundsTuple(x_min = -0.1,
                    x_max =  1.0,
                    y_min = -0.1,
                    y_max =  1.0,
                    z_min = -0.1,
                    z_max =  1.0)

        >>> axes_geometry_source.output.center
        (0.45, 0.45, 0.45)

        Show the difference in camera positioning with and without
        symmetric bounds. Orientation is added for visualization.

        Create actors.

        >>> axes_sym = pv.AxesAssembly(orientation=(90, 0, 0), symmetric_bounds=True)
        >>> axes_asym = pv.AxesAssembly(orientation=(90, 0, 0), symmetric_bounds=False)

        Show multi-window plot.

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_text('Symmetric bounds')
        >>> _ = pl.add_actor(axes_sym)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_text('Asymmetric bounds')
        >>> _ = pl.add_actor(axes_asym)
        >>> pl.show()

        """
        return self._symmetric_bounds

    @symmetric_bounds.setter
    def symmetric_bounds(self: AxesGeometrySource, val: bool) -> None:
        self._symmetric_bounds = val

    @property
    def shaft_length(
        self: AxesGeometrySource,
    ) -> tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def shaft_length(self: AxesGeometrySource, length: float | VectorLike[float]) -> None:
        self._shaft_length: NumpyArray[float] = _validation.validate_array3(
            length,
            broadcast=True,
            must_be_in_range=[0.0, np.inf],
            name='Shaft length',
        )

    @property
    def tip_length(
        self: AxesGeometrySource,
    ) -> tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def tip_length(self: AxesGeometrySource, length: float | VectorLike[float]) -> None:
        self._tip_length: NumpyArray[float] = _validation.validate_array3(
            length,
            broadcast=True,
            must_be_in_range=[0.0, np.inf],
            name='Tip length',
        )

    @property
    def tip_radius(self: AxesGeometrySource) -> float:  # numpydoc ignore=RT01
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
    def tip_radius(self: AxesGeometrySource, radius: float) -> None:
        _validation.check_range(radius, (0, float('inf')), name='tip radius')
        self._tip_radius = radius

    @property
    def shaft_radius(self: AxesGeometrySource) -> float:  # numpydoc ignore=RT01
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
    def shaft_radius(self: AxesGeometrySource, radius: float) -> None:
        _validation.check_range(radius, (0, float('inf')), name='shaft radius')
        self._shaft_radius = radius

    @property
    def shaft_type(self: AxesGeometrySource) -> str:  # numpydoc ignore=RT01
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
    def shaft_type(self: AxesGeometrySource, shaft_type: GeometryTypes | DataSet) -> None:
        self._shaft_type = self._set_normalized_datasets(part=_PartEnum.shaft, geometry=shaft_type)

    @property
    def tip_type(self: AxesGeometrySource) -> str:  # numpydoc ignore=RT01
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
    def tip_type(self: AxesGeometrySource, tip_type: str | DataSet) -> None:
        self._tip_type = self._set_normalized_datasets(part=_PartEnum.tip, geometry=tip_type)

    def _set_normalized_datasets(
        self: AxesGeometrySource, part: _PartEnum, geometry: str | DataSet
    ) -> str:
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

    def _reset_shaft_and_tip_geometry(self: AxesGeometrySource) -> None:
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
                point[axis] = total_length  # type: ignore[call-overload]
                flipped_point = np.array([point]) * -1  # Flip point
                point_id = part.n_points
                new_face = [3, point_id, point_id, point_id]

                # Update mesh
                part.points = np.append(part.points, flipped_point, axis=0)
                part.faces = np.append(part.faces, new_face)

    def update(self: AxesGeometrySource) -> None:
        """Update the output of the source."""
        self._reset_shaft_and_tip_geometry()

    @property
    def output(self: AxesGeometrySource) -> pyvista.MultiBlock:
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
    def _make_default_part(geometry: str) -> PolyData:
        """Create part geometry with its length axis pointing in the +z direction."""
        resolution = 50
        if geometry == 'cylinder':
            out = pyvista.Cylinder(direction=(0, 0, 1), resolution=resolution)
        elif geometry == 'sphere':
            out = pyvista.Sphere(phi_resolution=resolution, theta_resolution=resolution)
        elif geometry == 'hemisphere':
            out = pyvista.SolidSphere(end_phi=90).extract_geometry()
        elif geometry == 'cone':
            out = pyvista.Cone(direction=(0, 0, 1), resolution=resolution)
        elif geometry == 'pyramid':
            out = pyvista.Pyramid().extract_geometry()
        elif geometry == 'cube':
            out = pyvista.Cube()
        elif geometry == 'octahedron':
            mesh = pyvista.Octahedron()
            mesh.cell_data.remove('FaceIndex')
            out = mesh
        else:
            _validation.check_contains(
                AxesGeometrySource.GEOMETRY_TYPES,
                must_contain=geometry,
                name='Geometry',
            )
            msg = f"Geometry '{geometry}' is not implemented"  # pragma: no cover
            raise NotImplementedError(msg)  # pragma: no cover
        return out

    @staticmethod
    def _make_any_part(geometry: str | DataSet) -> tuple[str, PolyData]:
        part: DataSet
        part_poly: PolyData
        if isinstance(geometry, str):
            name = geometry
            part = AxesGeometrySource._make_default_part(
                geometry,
            )
        elif isinstance(geometry, pyvista.DataSet):
            name = 'custom'
            part = geometry.copy()
        else:
            msg = f'Geometry must be a string or pyvista.DataSet. Got {type(geometry)}.'  # type: ignore[unreachable]
            raise TypeError(msg)
        part_poly = part if isinstance(part, pyvista.PolyData) else part.extract_geometry()
        part_poly = AxesGeometrySource._normalize_part(part_poly)
        return name, part_poly

    @staticmethod
    def _normalize_part(part: PolyData) -> PolyData:
        """Scale and translate part to have origin-centered bounding box with edge length one."""
        # Center points at origin
        # mypy ignore since pyvista_ndarray is not compatible with np.ndarray, see GH#5434
        part.points -= part.center

        # Scale so bounding box edges have length one
        size = np.array(part.bounds_size)
        if np.any(size < 1e-8):
            msg = f'Custom axes part must be 3D. Got bounds:\n{part.bounds}.'
            raise ValueError(msg)
        part.scale(np.reciprocal(size), inplace=True)
        return part

    @staticmethod
    def _make_axes_parts(
        geometry: str | DataSet,
    ) -> tuple[str, tuple[PolyData, PolyData, PolyData]]:
        """Return three axis-aligned normalized parts centered at the origin."""
        name, part_z = AxesGeometrySource._make_any_part(geometry)
        part_x = part_z.copy().rotate_y(90)
        part_y = part_z.copy().rotate_x(-90)
        return name, (part_x, part_y, part_z)


class OrthogonalPlanesSource(_NoNewAttrMixin):
    """Orthogonal planes source.

    This source generates three orthogonal planes. The :attr:`output` is a
    :class:`~pyvista.MultiBlock` with named plane meshes ``'yz'``, ``'zx'``, ``'xy'``.
    The meshes are ordered such that the first, second, and third plane is perpendicular
    to the x, y, and z-axis, respectively.

    .. versionadded:: 0.45

    Parameters
    ----------
    bounds : VectorLike[float], default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        Specify the bounds of the planes in the form:
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
        The generated planes are centered in these bounds.

    resolution : int | VectorLike[int], default: 2
        Number of points on the planes in the x-y-z directions. Use a single number
        for a uniform resolution, or three values to set independent resolutions.

    normal_sign : '+' | '-' | sequence['+' | '-'], default: '+'
        Sign of the plane's normal vectors. Use a single value to set all normals to
        the same sign, or three values to set them independently.

    names : sequence[str], default: ('xy','yz','zx')
        Name of each plane in the generated :class:`~pyvista.MultiBlock`.

    Examples
    --------
    Generate default orthogonal planes.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> planes_source = pv.OrthogonalPlanesSource()
    >>> output = planes_source.output
    >>> output.plot()

    Modify the planes to fit a mesh's bounds.

    >>> human = examples.download_human()
    >>> planes_source.bounds = human.bounds
    >>> planes_source.update()

    Plot the mesh and the planes.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(human, scalars='Color', rgb=True)
    >>> _ = pl.add_mesh(output, opacity=0.3, show_edges=True)
    >>> pl.show()

    The planes are centered geometrically, but the frontal plane is positioned a bit
    too far forward. Use :meth:`push` to move the frontal plane.

    >>> planes_source.push(0.0, -10.0, 0)
    >>> planes_source.update()

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(human, scalars='Color', rgb=True)
    >>> _ = pl.add_mesh(output, opacity=0.3, show_edges=True, line_width=10)
    >>> pl.view_yz()
    >>> pl.show()

    """

    def __init__(
        self: OrthogonalPlanesSource,
        bounds: VectorLike[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        *,
        resolution: int | VectorLike[int] = 2,
        normal_sign: Literal['+', '-'] | Sequence[str] = '+',
        names: Sequence[str] = ('yz', 'zx', 'xy'),
    ) -> None:
        # Init sources and the output dataset
        self._output = pyvista.MultiBlock([pyvista.PolyData() for _ in range(3)])
        self.sources = tuple(pyvista.PlaneSource() for _ in range(3))

        # Init properties
        self.bounds = bounds
        self.resolution = resolution
        self.normal_sign = normal_sign
        self.names = names

    @property
    def normal_sign(
        self: OrthogonalPlanesSource,
    ) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Return or set the sign of the plane's normal vectors."""
        return cast('tuple[str, str, str]', self._normal_sign)

    @normal_sign.setter
    def normal_sign(
        self: OrthogonalPlanesSource, sign: Literal['+', '-'] | Sequence[str] = '+'
    ) -> None:
        def _check_sign(sign_: str) -> None:
            allowed = ['+', '-']
            _validation.check_contains(allowed, must_contain=sign_, name='normal sign')

        valid_sign: Sequence[str]
        _validation.check_instance(sign, (tuple, list, str), name='normal sign')
        if isinstance(sign, str):
            _check_sign(sign)
            valid_sign = [sign] * 3
        else:
            _validation.check_length(sign, exact_length=3)
            [_check_sign(s) for s in sign]  # type: ignore[func-returns-value]
            valid_sign = sign
        self._normal_sign = tuple(valid_sign)

        # Modify sources
        for source, axis_vector, sign_ in zip(self.sources, np.eye(3), valid_sign):
            has_positive_normal = np.dot(source.normal, axis_vector) > 0
            if has_positive_normal and sign_ == '-':
                source.flip_normal()

    @property
    def resolution(
        self: OrthogonalPlanesSource,
    ) -> tuple[int, int, int]:  # numpydoc ignore=RT01
        """Return or set the resolution of the planes."""
        return cast('tuple[int, int, int]', tuple(self._resolution))

    @resolution.setter
    def resolution(self: OrthogonalPlanesSource, resolution: int | VectorLike[int]) -> None:
        valid_resolution = _validation.validate_array3(
            resolution, broadcast=True, to_tuple=True, name='resolution'
        )
        self._resolution = valid_resolution

        # Modify sources
        x_res, y_res, z_res = valid_resolution
        yz_source, zx_source, xy_source = self.sources

        yz_source.i_resolution = y_res
        yz_source.j_resolution = z_res
        zx_source.i_resolution = z_res
        zx_source.j_resolution = x_res
        xy_source.i_resolution = x_res
        xy_source.j_resolution = y_res

    @property
    def bounds(self: OrthogonalPlanesSource) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return or set the bounds of the planes."""
        return self._bounds

    @bounds.setter
    def bounds(self: OrthogonalPlanesSource, bounds: VectorLike[float]) -> None:
        bounds_tuple = _validation.validate_array(
            bounds, dtype_out=float, must_have_length=6, to_tuple=True, name='bounds'
        )
        self._bounds = BoundsTuple(*bounds_tuple)

        # Modify sources
        x_min, x_max, y_min, y_max, z_min, z_max = bounds_tuple
        x_size, y_size, z_size = x_max - x_min, y_max - y_min, z_max - z_min
        center = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
        ORIGIN = (0.0, 0.0, 0.0)
        yz_source, zx_source, xy_source = self.sources

        xy_source.point_a = x_size, 0.0, 0.0
        xy_source.point_b = 0.0, y_size, 0.0
        xy_source.origin = ORIGIN
        xy_source.center = center

        yz_source.point_a = 0.0, y_size, 0.0
        yz_source.point_b = 0.0, 0.0, z_size
        yz_source.origin = ORIGIN
        yz_source.center = center

        zx_source.point_a = 0.0, 0.0, z_size
        zx_source.point_b = x_size, 0.0, 0.0
        zx_source.origin = ORIGIN
        zx_source.center = center

    @property
    def names(
        self: OrthogonalPlanesSource,
    ) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Return or set the names of the planes."""
        return self._names

    @names.setter
    def names(self: OrthogonalPlanesSource, names: Sequence[str]) -> None:
        _validation.check_instance(names, (tuple, list), name='names')
        _validation.check_iterable_items(names, str, name='names')
        _validation.check_length(names, exact_length=3, name='names')
        valid_names = cast('tuple[str, str, str]', tuple(names))
        self._names = valid_names

        output = self._output
        for i, name in enumerate(valid_names):
            output.set_block_name(i, name)

    def push(
        self: OrthogonalPlanesSource, *distance: float | VectorLike[float]
    ) -> None:  # numpydoc ignore=RT01
        """Translate each plane by the specified distance along its normal.

        Internally, this method calls :meth:`pyvista.PlaneSource.push` on each
        plane source.

        Parameters
        ----------
        *distance : float | VectorLike[float], default: (0.0, 0.0, 0.0)
            Distance to move each plane.

        """
        valid_distance = _validation.validate_array3(
            distance,  # type: ignore[arg-type]
            broadcast=True,
            dtype_out=float,
            to_tuple=True,
        )
        for source, dist in zip(self.sources, valid_distance):
            source.push(dist)

    def update(self: OrthogonalPlanesSource) -> None:
        """Update the output of the source."""
        for source, plane in zip(self.sources, self._output):
            plane.copy_from(source.output)

    @property
    def output(self: OrthogonalPlanesSource) -> pyvista.MultiBlock:
        """Get the output of the source.

        The output is a :class:`pyvista.MultiBlock` with three blocks: one for each
        plane. The blocks are named ``'yz'``, ``'zx'``, ``'xy'``, and are ordered such
        that the first, second, and third plane is perpendicular to the x, y, and
        z-axis, respectively.

        The source is automatically updated by :meth:`update` prior to returning
        the output.

        Returns
        -------
        pyvista.MultiBlock
            Composite mesh with three planes.

        """
        self.update()
        return self._output


class CubeFacesSource(CubeSource):
    """Generate the faces of a cube.

    This source generates a :class:`~pyvista.MultiBlock` with the six :class:`PolyData`
    comprising the faces of a cube.

    The faces may be shrunk or exploded to create equal-sized gaps or intersections
    between the faces. Additionally, the faces may be converted to frames with a
    constant-width border.

    .. versionadded:: 0.45.0

    Parameters
    ----------
    center : VectorLike[float], default: (0.0, 0.0, 0.0)
        Center in ``[x, y, z]``.

    x_length : float, default: 1.0
        Length of the cube in the x-direction.

    y_length : float, default: 1.0
        Length of the cube in the y-direction.

    z_length : float, default: 1.0
        Length of the cube in the z-direction.

    bounds : sequence[float], optional
        Specify the bounding box of the cube. If given, all other size
        arguments are ignored. ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

    frame_width : float, optional
        Convert the faces into frames with the specified width. If set, the center
        portion of each face is removed and the output faces will each have four quad
        cells (one for each side of the frame) instead of a single quad cell. Values
        must be between ``0.0`` (minimal frame) and ``1.0`` (large frame). The frame is
        scaled to ensure it has a constant width.

    shrink_factor : float, optional
        Shrink or grow the cube's faces. If set, this is the factor by which to shrink
        or grow each face. The amount of shrinking or growth is relative to the smallest
        edge length of the cube, and all sides of the faces are shrunk by the same
        (constant) value.

        .. note::
            - A value of ``1.0`` has no effect.
            - Values between ``0.0`` and ``1.0`` will shrink the faces.
            - Values greater than ``1.0`` will grow the faces.

        This has a similar effect to using :meth:`~pyvista.DataSetFilters.shrink`.

    explode_factor : float, optional
        Push the faces away from (or pull them toward) the cube's center. If set, this
        is the factor by which to move each face. The magnitude of the move is relative
        to the smallest edge length of the cube, and all faces are moved by the same
        (constant) amount.

        .. note::
            - A value of ``0.0`` has no effect.
            - Increasing positive values will push the faces farther away (explode).
            - Decreasing negative values will pull the faces closer together (implode).

        This has a similar effect to using :meth:`~pyvista.DataSetFilters.explode`.

    names : sequence[str], default: ('+X','-X','+Y','-Y','+Z','-Z')
        Name of each face in the generated :class:`~pyvista.MultiBlock`.

    point_dtype : str, default: 'float32'
        Set the desired output point types. It must be either 'float32' or 'float64'.

    Examples
    --------
    Generate the default faces of a cube.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> cube_faces_source = pv.CubeFacesSource()
    >>> output = cube_faces_source.output
    >>> output.plot(show_edges=True, line_width=10)

    The output is similar to that of :class:`CubeSource` except it's a
    :class:`~pyvista.MultiBlock`.

    >>> output
    MultiBlock (...)
      N Blocks:   6
      X Bounds:   -5.000e-01, 5.000e-01
      Y Bounds:   -5.000e-01, 5.000e-01
      Z Bounds:   -5.000e-01, 5.000e-01

    >>> cube_source = pv.CubeSource()
    >>> cube_source.output
    PolyData (...)
      N Cells:    6
      N Points:   24
      N Strips:   0
      X Bounds:   -5.000e-01, 5.000e-01
      Y Bounds:   -5.000e-01, 5.000e-01
      Z Bounds:   -5.000e-01, 5.000e-01
      N Arrays:   2

    Use :attr:`explode_factor` to explode the faces.

    >>> cube_faces_source.explode_factor = 0.5
    >>> cube_faces_source.update()
    >>> output.plot(show_edges=True, line_width=10)

    Use :attr:`shrink_factor` to also shrink the faces.

    >>> cube_faces_source.shrink_factor = 0.5
    >>> cube_faces_source.update()
    >>> output.plot(show_edges=True, line_width=10)

    Fit cube faces to a dataset and only plot four of them.

    >>> mesh = examples.load_airplane()
    >>> cube_faces_source = pv.CubeFacesSource(bounds=mesh.bounds)
    >>> output = cube_faces_source.output

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, color='tomato')
    >>> _ = pl.add_mesh(output['+X'], opacity=0.5)
    >>> _ = pl.add_mesh(output['-X'], opacity=0.5)
    >>> _ = pl.add_mesh(output['+Y'], opacity=0.5)
    >>> _ = pl.add_mesh(output['-Y'], opacity=0.5)
    >>> pl.show()

    Generate a frame instead of full faces.

    >>> mesh = pv.ParametricEllipsoid(5, 4, 3)
    >>> cube_faces_source = pv.CubeFacesSource(bounds=mesh.bounds, frame_width=0.1)
    >>> output = cube_faces_source.output

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, color='tomato')
    >>> _ = pl.add_mesh(output, show_edges=True, line_width=10)
    >>> pl.show()

    """

    class _FaceIndex(IntEnum):
        X_NEG = 0
        X_POS = 1
        Y_NEG = 2
        Y_POS = 3
        Z_NEG = 4
        Z_POS = 5

    def __init__(
        self: CubeFacesSource,
        *,
        center: VectorLike[float] = (0.0, 0.0, 0.0),
        x_length: float = 1.0,
        y_length: float = 1.0,
        z_length: float = 1.0,
        bounds: VectorLike[float] | None = None,
        frame_width: float | None = None,
        shrink_factor: float | None = None,
        explode_factor: float | None = None,
        names: Sequence[str] = ('+X', '-X', '+Y', '-Y', '+Z', '-Z'),
        point_dtype: str = 'float32',
    ) -> None:
        # Init CubeSource
        super().__init__(
            center=center,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            bounds=bounds,
            point_dtype=point_dtype,
        )
        # Init output
        self._output = pyvista.MultiBlock([pyvista.PolyData() for _ in range(6)])

        # Set properties
        self.frame_width = frame_width
        self.shrink_factor = shrink_factor
        self.explode_factor = explode_factor
        self.names = names  # type: ignore[assignment]

    @property
    def frame_width(self: CubeFacesSource) -> float | None:  # numpydoc ignore=RT01
        """Convert the faces into frames with the specified border width.

        If set, the center portion of each face is removed and the :attr:`output`
        :class:`pyvista.PolyData` will each have four quad cells (one for each
        side of the frame) instead of a single quad cell. Values must be between ``0.0``
        (minimal frame) and ``1.0`` (large frame). The frame is scaled to ensure it has
        a constant width.

        Examples
        --------
        >>> import pyvista as pv
        >>> cube_faces_source = pv.CubeFacesSource(
        ...     x_length=3, y_length=2, z_length=1, frame_width=0.2
        ... )
        >>> cube_faces_source.output.plot(show_edges=True, line_width=10)

        >>> cube_faces_source.frame_width = 0.8
        >>> cube_faces_source.output.plot(show_edges=True, line_width=10)

        """
        return self._frame_width

    @frame_width.setter
    def frame_width(self: CubeFacesSource, width: float | None) -> None:
        self._frame_width = (
            width
            if width is None
            else _validation.validate_number(
                width, must_be_in_range=[0.0, 1.0], name='frame width'
            )
        )

    @property
    def shrink_factor(self: CubeFacesSource) -> float | None:  # numpydoc ignore=RT01
        """Shrink or grow the cube's faces.

        If set, this is the factor by which to shrink or grow each face. The amount of
        shrinking or growth is relative to the smallest edge length of the cube, and
        all sides of the faces are shrunk by the same (constant) value.

        .. note::
            - A value of ``1.0`` has no effect.
            - Values between ``0.0`` and ``1.0`` will shrink the faces.
            - Values greater than ``1.0`` will grow the faces.

        This has a similar effect to using :meth:`~pyvista.DataSetFilters.shrink`.

        Examples
        --------
        >>> import pyvista as pv
        >>> cube_faces_source = pv.CubeFacesSource(
        ...     x_length=3, y_length=2, z_length=1, shrink_factor=0.8
        ... )
        >>> output = cube_faces_source.output
        >>> output.plot(show_edges=True, line_width=10)

        Note how all edges are shrunk by the same (constant) amount in terms of absolute
        distance. Compare this to :meth:`~pyvista.DataSetFilters.shrink` where the
        amount of shrinkage is relative to the size of the faces.

        >>> exploded = pv.merge(output).shrink(0.8)
        >>> exploded.plot(show_edges=True, line_width=10)

        """
        return self._shrink_factor

    @shrink_factor.setter
    def shrink_factor(self: CubeFacesSource, factor: float | None) -> None:
        self._shrink_factor = (
            factor
            if factor is None
            else _validation.validate_number(
                factor, must_be_in_range=[0.0, np.inf], name='shrink factor'
            )
        )

    @property
    def explode_factor(self: CubeFacesSource) -> float | None:  # numpydoc ignore=RT01
        """Push the faces away from (or pull them toward) the cube's center.

        If set, this is the factor by which to move each face. The magnitude of the
        move is relative to the smallest edge length of the cube, and all faces are
        moved by the same (constant) amount.

        .. note::
            - A value of ``0.0`` has no effect.
            - Increasing positive values will push the faces farther away (explode).
            - Decreasing negative values will pull the faces closer together (implode).

        This has a similar effect to using :meth:`~pyvista.DataSetFilters.explode`.

        Examples
        --------
        >>> import pyvista as pv
        >>> cube_faces_source = pv.CubeFacesSource(
        ...     x_length=3, y_length=2, z_length=1, explode_factor=0.2
        ... )
        >>> output = cube_faces_source.output
        >>> output.plot(show_edges=True, line_width=10)

        Note how all faces are moved by the same amount. Compare this to using
        :meth:`~pyvista.DataSetFilters.explode` where the distance each face moves
        is relative to distance of each face to the center of the cube.

        >>> exploded = pv.merge(output).explode(0.2)
        >>> exploded.plot(show_edges=True, line_width=10)

        """
        return self._explode_factor

    @explode_factor.setter
    def explode_factor(self: CubeFacesSource, factor: float | None) -> None:
        self._explode_factor = (
            factor
            if factor is None
            else _validation.validate_number(factor, name='explode factor')
        )

    @property
    def names(
        self: CubeFacesSource,
    ) -> tuple[str, str, str, str, str, str]:  # numpydoc ignore=RT01
        """Return or set the names of the faces.

        Specify three strings, one for each '+/-' face pair, or six strings, one for
        each individual face.

        If three strings, plus ``'+'`` and minus ``'-'`` characters are added to
        the names.

        Examples
        --------
        >>> import pyvista as pv
        >>> cube_faces = pv.CubeFacesSource()

        Use three strings to set the names. Plus ``'+'`` and minus ``'-'``
        characters are added automatically.

        >>> cube_faces.names = ['U', 'V', 'W']
        >>> cube_faces.names
        ('+U', '-U', '+V', '-V', '+W', '-W')

        Alternatively, use six strings to set the names explicitly.

        >>> cube_faces.names = [
        ...     'right',
        ...     'left',
        ...     'anterior',
        ...     'posterior',
        ...     'superior',
        ...     'inferior',
        ... ]
        >>> cube_faces.names
        ('right', 'left', 'anterior', 'posterior', 'superior', 'inferior')

        """
        return self._names

    @names.setter
    def names(
        self: CubeFacesSource,
        names: list[str] | tuple[str, str, str] | tuple[str, str, str, str, str, str],
    ) -> None:
        name = 'face names'
        _validation.check_instance(names, (list, tuple), name=name)
        _validation.check_iterable_items(names, str, name=name)
        _validation.check_length(names, exact_length=[3, 6], name=name)
        valid_names = (
            tuple(names)
            if len(names) == 6
            else (
                '+' + names[0],
                '-' + names[0],
                '+' + names[1],
                '-' + names[1],
                '+' + names[2],
                '-' + names[2],
            )
        )
        self._names = cast('tuple[str, str, str, str, str, str]', valid_names)

    def update(self: CubeFacesSource) -> None:
        """Update the output of the source."""

        def _scale_points(
            points_: NumpyArray[float],
            origin_: NumpyArray[float],
            scale_: NumpyArray[float],
        ) -> NumpyArray[float]:
            points_ -= origin_
            points_ *= scale_
            points_ += origin_
            return points_

        def _create_frame_from_quad_points(
            quad_points: NumpyArray[float],
            center: NumpyArray[float],
            scale: NumpyArray[float],
        ) -> tuple[NumpyArray[float], NumpyArray[float]]:
            """Create a picture-frame from 4 points defining a rectangle.

            The inner points of the frame are generated by scaling the quad_points by
            the length-3 scaling factor.
            """
            inner_points = _scale_points(quad_points.copy(), center, scale)

            # Define frame quads from outer and inner points
            quad1_points = np.vstack((quad_points[[0, 1]], inner_points[[1, 0]]))
            quad2_points = np.vstack((quad_points[[1, 2]], inner_points[[2, 1]]))
            quad3_points = np.vstack((quad_points[[2, 3]], inner_points[[3, 2]]))
            quad4_points = np.vstack((quad_points[[3, 0]], inner_points[[0, 3]]))
            frame_points = np.vstack((quad1_points, quad2_points, quad3_points, quad4_points))
            frame_faces = np.array(
                [
                    [4, 0, 1, 2, 3],
                    [4, 4, 5, 6, 7],
                    [4, 8, 9, 10, 11],
                    [4, 12, 13, 14, 15],
                ]
            ).ravel()
            return frame_points, frame_faces

        # Get initial cube output
        cube = CubeSource.output.fget(self)  # type: ignore[attr-defined]

        # Extract list of points for each face in the desired order
        cube_points, cube_faces = cube.points, cube.regular_faces
        Index = CubeFacesSource._FaceIndex
        face_points = [
            cube_points[cube_faces[Index.X_POS]],
            cube_points[cube_faces[Index.X_NEG]],
            cube_points[cube_faces[Index.Y_POS]],
            cube_points[cube_faces[Index.Y_NEG]],
            cube_points[cube_faces[Index.Z_POS]],
            cube_points[cube_faces[Index.Z_NEG]],
        ]

        # Calc lengths/properties of cube
        cube_center = np.array(cube.center)
        bnds = cube.bounds
        x_len = np.linalg.norm(bnds[1] - bnds[0])
        y_len = np.linalg.norm(bnds[3] - bnds[2])
        z_len = np.linalg.norm(bnds[5] - bnds[4])
        lengths = np.array((x_len, y_len, z_len))
        min_length = np.min(lengths)

        # Store vars for updating the output
        shrink_factor = self.shrink_factor
        explode_factor = self.explode_factor
        frame_width = self.frame_width
        output = self._output

        # Modify each face mesh of the output
        for index, (name, points) in enumerate(zip(self.names, face_points)):
            output.set_block_name(index, name)
            face_poly = output[index]
            face_center = np.mean(points, axis=0)

            if shrink_factor is not None:
                # Shrink proportional to the smallest face
                shrink_scale = shrink_factor + (1 - shrink_factor) * (1 - min_length / lengths)
                _scale_points(points, face_center, shrink_scale)

            if explode_factor is not None:
                # Move away from center by some distance proportional to the smallest face
                explode_scale = min_length * explode_factor
                direction = face_center - cube_center
                direction /= np.linalg.norm(direction)
                vector = direction * explode_scale
                points += vector  # noqa: PLW2901

            # Set poly as a single quad cell
            face_poly.points = points  # type: ignore[union-attr]
            face_poly.faces = [4, 0, 1, 2, 3]  # type: ignore[union-attr]

            if frame_width is not None:
                # Create frame proportional to the smallest face
                frame_scale = 1 - (frame_width * min_length / lengths)
                frame_points, frame_faces = _create_frame_from_quad_points(
                    points, face_center, frame_scale
                )
                # Set poly as four quad cells of the frame
                face_poly.points = frame_points  # type: ignore[union-attr]
                face_poly.faces = frame_faces  # type: ignore[union-attr]

    @property
    def output(self: CubeFacesSource) -> pyvista.MultiBlock:  # type: ignore[override]
        """Get the output of the source.

        The output is a :class:`pyvista.MultiBlock` with six blocks: one for each
        face. The blocks are named and ordered as ``('+X','-X','+Y','-Y','+Z','-Z')``.

        The source is automatically updated by :meth:`update` prior to returning
        the output.

        Returns
        -------
        pyvista.MultiBlock
            Composite mesh with six cube faces.

        """
        self.update()
        return self._output
