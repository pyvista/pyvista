"""Provides an easy way of generating several geometric sources.

Also includes some pure-python helpers.

"""
from typing import Sequence, Tuple, Union

import numpy as np
from vtkmodules.vtkRenderingFreeType import vtkVectorText

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import Matrix, Vector
from pyvista.core.utilities.misc import _check_range, _reciprocal, no_new_attr

from .arrays import _coerce_pointslike_arg
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
        surf.points += np.array(center, dtype=surf.points.dtype)


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
        return wrap(self.GetOutput())


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

    _new_attr_exceptions = [
        '_center',
        '_height',
        '_width',
        '_depth',
        '_normal',
        '_process_empty_string',
        '_output',
        '_extrude_filter',
        '_tri_filter',
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

        # Create output filters to make text 3D
        extrude = _vtk.vtkLinearExtrusionFilter()
        extrude.SetInputConnection(self.GetOutputPort())
        extrude.SetExtrusionTypeToNormalExtrusion()
        extrude.SetVector(0, 0, 1)
        self._extrude_filter = extrude

        tri_filter = _vtk.vtkTriangleFilter()
        tri_filter.SetInputConnection(extrude.GetOutputPort())
        self._tri_filter = tri_filter

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
                    f'{self.__class__.__name__}'
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
    def center(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the center of the text.

        The center is defined as the middle of the axis-aligned bounding box
        of the text.
        """
        return self._center

    @center.setter
    def center(self, center: Sequence[float]):  # numpydoc ignore=GL08
        self._center = float(center[0]), float(center[1]), float(center[2])

    @property
    def normal(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
        _check_range(
            height, rng=(0, float('inf')), parm_name='height'
        ) if height is not None else None
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
                self._tri_filter.Update()
                out = self._tri_filter.GetOutput()

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
