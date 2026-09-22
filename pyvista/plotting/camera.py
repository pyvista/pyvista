"""Module containing pyvista implementation of :vtk:`vtkCamera`."""

from __future__ import annotations

from typing import TYPE_CHECKING
import weakref
from xml.etree import ElementTree as ET

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista import _vtk
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .helpers import view_vectors

if TYPE_CHECKING:
    from pathlib import Path

    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike

    from .helpers import _ViewOptions
    from .renderer import Renderer

# OpenCV cameras look along +z with +y down; VTK looks along -z with +y up.
_OPENCV_FROM_VTK = np.diag([1.0, -1.0, -1.0, 1.0])


class Camera(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkCamera):
    """PyVista wrapper for the VTK Camera class.

    Parameters
    ----------
    renderer : pyvista.Renderer, optional
        Renderer to attach the camera to.

    Examples
    --------
    Create a camera at the pyvista module level.

    >>> import pyvista as pv
    >>> camera = pv.Camera()

    Access the active camera of a plotter and get the position of the
    camera.

    >>> pl = pv.Plotter()
    >>> pl.camera.position
    (1.0, 1.0, 1.0)

    """

    def __init__(self, renderer: Renderer | None = None) -> None:
        """Initialize a new camera descriptor."""
        self._parallel_projection = False
        self._elevation = 0.0
        self._azimuth = 0.0
        self._is_set = False
        self._focus: NumpyArray[float] | None = None  # Used by BackgroundRenderer

        if renderer:
            if not isinstance(renderer, pv.Renderer):
                msg = 'Camera only accepts a pyvista.Renderer or None as the ``renderer`` argument'  # type: ignore[unreachable]
                raise TypeError(msg)
            self._renderer = weakref.proxy(renderer)
        else:
            self._renderer = None

    def __eq__(self, other: object) -> bool:
        """Compare whether the relevant attributes of two cameras are equal."""
        if not isinstance(other, Camera):
            return NotImplemented

        # attributes which are native python types and thus implement __eq__
        native_attrs = [
            'position',
            'focal_point',
            'parallel_projection',
            'distance',
            'thickness',
            'parallel_scale',
            'clipping_range',
            'window_center',
            'explicit_aspect_ratio',
            'view_angle',
            'roll',
        ]
        for attr in native_attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return bool(np.array_equal(self.model_transform_matrix, other.model_transform_matrix))

    __hash__ = None  # type: ignore[assignment]  # https://github.com/pyvista/pyvista/pull/7671

    def __repr__(self) -> str:
        """Print a ``repr`` specifying the id of the camera and its camera type."""
        repr_str = f'{self.__class__.__name__} ({hex(id(self))})'
        repr_str += f'\n  Position:            {self.position}'
        repr_str += f'\n  Focal Point:         {self.focal_point}'
        repr_str += f'\n  Parallel Projection: {self.parallel_projection}'
        repr_str += f'\n  Distance:            {self.distance}'
        repr_str += f'\n  Thickness:           {self.thickness}'
        repr_str += f'\n  Parallel Scale:      {self.parallel_scale}'
        repr_str += f'\n  Clipping Range:      {self.clipping_range}'
        repr_str += f'\n  View Angle:          {self.view_angle}'
        repr_str += f'\n  Roll:                {self.roll}'
        return repr_str

    def __str__(self) -> str:
        """Return the object string representation."""
        return self.__repr__()

    def __del__(self) -> None:
        """Delete the camera."""
        self.RemoveAllObservers()

    @property
    def is_set(self) -> bool:  # numpydoc ignore=RT01
        """Get or set whether this camera has been configured."""
        return self._is_set

    @is_set.setter
    def is_set(self, value: bool) -> None:
        self._is_set = bool(value)

    @classmethod
    def from_paraview_pvcc(cls, filename: str | Path) -> Camera:
        """Load a ParaView camera file (.pvcc extension).

        Returns a pyvista.Camera object for which attributes has been read
        from the filename argument.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to ParaView camera file (.pvcc).

        Returns
        -------
        pyvista.Camera
            Camera from the camera file.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera = pv.Camera.from_paraview_pvcc('camera.pvcc')  # doctest:+SKIP
        >>> pl.camera.position
        (1.0, 1.0, 1.0)

        """
        to_find = {
            'CameraPosition': ('position', float),
            'CameraFocalPoint': ('focal_point', float),
            'CameraViewAngle': ('view_angle', float),
            'CameraViewUp': ('up', float),
            'CameraParallelProjection': ('parallel_projection', int),
            'CameraParallelScale': ('parallel_scale', float),
        }
        camera = cls()

        tree = ET.parse(filename)
        root = tree.getroot()[0]
        for element in root:
            attrib = element.attrib
            attrib_name = attrib['name']

            if attrib_name in to_find:
                name, typ = to_find[attrib_name]
                nelems = int(attrib['number_of_elements'])

                # Set the camera attributes
                if nelems == 3:
                    values = [typ(e.attrib['value']) for e in element]
                    setattr(camera, name, values)
                elif nelems == 1:
                    # Special case for bool since bool("0") returns True.
                    # So first convert to int from `to_find` and then apply bool
                    if 'name' in element[-1].attrib and element[-1].attrib['name'] == 'bool':
                        val = bool(typ(element[0].attrib['value']))
                    else:
                        val = typ(element[0].attrib['value'])
                    setattr(camera, name, val)

        camera.is_set = True
        return camera

    def to_paraview_pvcc(self, filename: str | Path) -> None:
        """Write the camera parameters to a ParaView camera file (.pvcc extension).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to ParaView camera file (.pvcc).

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.to_paraview_pvcc('camera.pvcc')  # doctest:+SKIP

        """
        root = ET.Element('PVCameraConfiguration')
        root.attrib['description'] = 'ParaView camera configuration'
        root.attrib['version'] = '1.0'

        dico = dict(group='views', type='RenderView', id='0', servers='21')
        proxy = ET.SubElement(root, 'Proxy', dico)

        # Add tuples
        to_find = {
            'CameraPosition': 'position',
            'CameraFocalPoint': 'focal_point',
            'CameraViewUp': 'up',
        }
        for name, attr in to_find.items():
            e = ET.SubElement(
                proxy,
                'Property',
                dict(name=name, id=f'0.{name}', number_of_elements='3'),
            )

            for i in range(3):
                tmp = ET.Element('Element')
                tmp.attrib['index'] = str(i)
                tmp.attrib['value'] = str(getattr(self, attr)[i])
                e.append(tmp)

        # Add single values
        to_find = {
            'CameraViewAngle': 'view_angle',
            'CameraParallelScale': 'parallel_scale',
            'CameraParallelProjection': 'parallel_projection',
        }

        for name, attr in to_find.items():
            e = ET.SubElement(
                proxy,
                'Property',
                dict(name=name, id=f'0.{name}', number_of_elements='1'),
            )
            tmp = ET.Element('Element')
            tmp.attrib['index'] = '0'

            val = getattr(self, attr)
            if not isinstance(val, bool):
                tmp.attrib['value'] = str(val)
                e.append(tmp)
            else:
                tmp.attrib['value'] = '1' if val else '0'
                e.append(tmp)
                e.append(ET.Element('Domain', dict(name='bool', id=f'0.{name}.bool')))

        ET.indent(root, space='\t')
        ET.ElementTree(root).write(filename, encoding='utf-8', xml_declaration=True)

    @property
    def position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the position of the camera in world coordinates.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.position
        (1.0, 1.0, 1.0)
        >>> pl.camera.position = (2.0, 1.0, 1.0)
        >>> pl.camera.position
        (2.0, 1.0, 1.0)

        """
        return self.GetPosition()

    @position.setter
    def position(self, value: VectorLike[float]) -> None:
        self.SetPosition(_validation.validate_array3(value, dtype_out=float, to_tuple=True))
        self._elevation = 0.0
        self._azimuth = 0.0
        if self._renderer:
            self.reset_clipping_range()
        self.is_set = True

    def reset_clipping_range(self) -> None:
        """Reset the camera clipping range based on the bounds of the visible actors.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.clipping_range = (1, 2)
        >>> pl.camera.reset_clipping_range()  # doctest:+SKIP
        (0.0039213485598532955, 3.9213485598532953)

        """
        if self._renderer is None:
            msg = 'Camera is must be associated with a renderer to reset its clipping range.'
            raise AttributeError(msg)
        self._renderer.reset_camera_clipping_range()

    @property
    def focal_point(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Location of the camera's focus in world coordinates.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.focal_point
        (0.0, 0.0, 0.0)
        >>> pl.camera.focal_point = (2.0, 0.0, 0.0)
        >>> pl.camera.focal_point
        (2.0, 0.0, 0.0)

        """
        return self.GetFocalPoint()

    @focal_point.setter
    def focal_point(self, point: VectorLike[float]) -> None:
        self.SetFocalPoint(_validation.validate_array3(point, dtype_out=float, to_tuple=True))
        self.is_set = True

    @property
    def model_transform_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return or set the camera's model transformation matrix.

        Examples
        --------
        >>> import pyvista as pv
        >>> import numpy as np
        >>> pl = pv.Plotter()
        >>> pl.camera.model_transform_matrix
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        >>> pl.camera.model_transform_matrix = np.array(
        ...     [
        ...         [1.0, 0.0, 0.0, 0.0],
        ...         [0.0, 1.0, 0.0, 0.0],
        ...         [0.0, 0.0, 1.0, 0.0],
        ...         [0.0, 0.0, 0.0, 0.5],
        ...     ]
        ... )
        >>> pl.camera.model_transform_matrix
        array([[1. , 0. , 0. , 0. ],
               [0. , 1. , 0. , 0. ],
               [0. , 0. , 1. , 0. ],
               [0. , 0. , 0. , 0.5]])

        """
        vtk_matrix = self.GetModelTransformMatrix()
        matrix = np.empty((4, 4))
        vtk_matrix.DeepCopy(matrix.ravel(), vtk_matrix)
        return matrix

    @model_transform_matrix.setter
    def model_transform_matrix(self, matrix: NumpyArray[float]) -> None:
        vtk_matrix = _vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(matrix.ravel().tolist())
        self.SetModelTransformMatrix(vtk_matrix)

    @property
    def distance(self) -> float:  # numpydoc ignore=RT01
        """Return or set the distance of the focal point from the camera.

        Notes
        -----
        Setting the distance keeps the camera fixed and moves the focal point.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.distance
        1.73205
        >>> pl.camera.distance = 2.0
        >>> pl.camera.distance
        2.0

        """
        return self.GetDistance()

    @distance.setter
    def distance(self, distance: float) -> None:
        self.SetDistance(distance)
        self.is_set = True

    @property
    def thickness(self) -> float:  # numpydoc ignore=RT01
        """Return or set the distance between clipping planes.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.thickness
        1000.0
        >>> pl.camera.thickness = 100
        >>> pl.camera.thickness
        100.0

        """
        return self.GetThickness()

    @thickness.setter
    def thickness(self, length: float) -> None:
        self.SetThickness(length)

    @property
    def parallel_scale(self) -> float:  # numpydoc ignore=RT01
        """Return or set the scaling used for a parallel projection.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.parallel_scale
        1.0
        >>> pl.camera.parallel_scale = 2.0
        >>> pl.camera.parallel_scale
        2.0

        """
        return self.GetParallelScale()

    @parallel_scale.setter
    def parallel_scale(self, scale: float) -> None:
        self.SetParallelScale(scale)

    def zoom(self, value: float | str) -> None:
        """Set the zoom of the camera.

        In perspective mode, decrease the view angle by the specified
        factor.

        In parallel mode, decrease the parallel scale by the specified
        factor. A value greater than 1 is a zoom-in, a value less than
        1 is a zoom-out.

        Parameters
        ----------
        value : float or str
            Zoom of the camera. If a float, must be greater than 0. Otherwise,
            if a string, must be ``"tight"``. If tight, the plot will be zoomed
            such that the actors fill the entire viewport.

        Examples
        --------
        Show the Default zoom.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.zoom(1.0)
        >>> pl.show()

        Show 2x zoom.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.zoom(2.0)
        >>> pl.show()

        Zoom so the actor fills the entire render window.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        """
        if isinstance(value, str):
            if value != 'tight':
                msg = 'If a string, ``zoom`` can only be "tight"'
                raise ValueError(msg)
            self.tight()
            return

        self.Zoom(value)
        self.is_set = True

    @property
    def up(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the "up" of the camera.

        The vector is normalized, so it must have a non-zero magnitude.

        .. versionchanged:: 0.49

            Setting a zero-length vector now raises a ``ValueError``. Previously
            it was silently replaced with ``(0, 1, 0)`` by VTK.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.up
        (0.0, 0.0, 1.0)
        >>> pl.camera.up = (0.410018, 0.217989, 0.885644)
        >>> pl.camera.up
        (0.410018, 0.217989, 0.885644)

        """
        return self.GetViewUp()

    @up.setter
    def up(self, vector: VectorLike[float]) -> None:
        # VTK normalizes the view up vector and silently substitutes (0, 1, 0) when it
        # has no magnitude, so a zero vector must be rejected before SetViewUp.
        if np.allclose(vector, 0.0):
            msg = 'Camera up vector cannot be zero.'
            raise ValueError(msg)
        self.SetViewUp(_validation.validate_array3(vector, dtype_out=float, to_tuple=True))
        self.is_set = True

    def enable_parallel_projection(self) -> None:
        """Enable parallel projection.

        The camera will have a parallel projection. Parallel
        projection is often useful when viewing images or 2D datasets,
        but will look odd when viewing 3D datasets.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import demos
        >>> pl = pv.demos.orientation_plotter()
        >>> pl.enable_parallel_projection()
        >>> pl.show()

        """
        self._parallel_projection = True
        self.SetParallelProjection(True)

    def disable_parallel_projection(self) -> None:
        """Disable the use of parallel projection.

        This is default behavior.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import demos
        >>> pl = pv.demos.orientation_plotter()
        >>> pl.disable_parallel_projection()
        >>> pl.show()

        """
        self._parallel_projection = False
        self.SetParallelProjection(False)

    @property
    def parallel_projection(self) -> bool:  # numpydoc ignore=RT01
        """Return the state of the parallel projection.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import demos
        >>> pl = pv.Plotter()
        >>> pl.disable_parallel_projection()
        >>> pl.parallel_projection
        False

        """
        return self._parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state: bool) -> None:
        if state:
            self.enable_parallel_projection()
        else:
            self.disable_parallel_projection()

    @property
    def clipping_range(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the location of the clipping planes.

        Clipping planes are the near and far clipping planes along
        the direction of projection.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.clipping_range
        (0.01, 1000.01)
        >>> pl.camera.clipping_range = (1, 10)
        >>> pl.camera.clipping_range
        (1.0, 10.0)

        """
        return self.GetClippingRange()

    @clipping_range.setter
    def clipping_range(self, points: VectorLike[float]) -> None:
        near, far = float(points[0]), float(points[1])
        if near > far:
            msg = 'Near point must be lower than the far point.'
            raise ValueError(msg)
        self.SetClippingRange(near, far)

    @property
    def view_angle(self) -> float:  # numpydoc ignore=RT01
        """Return or set the camera view angle.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.view_angle
        30.0
        >>> pl.camera.view_angle = 60.0
        >>> pl.camera.view_angle
        60.0

        """
        return self.GetViewAngle()

    @view_angle.setter
    def view_angle(self, value: float) -> None:
        self.SetViewAngle(value)

    @property
    def window_center(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the horizontal and vertical shift of the projection center.

        The two values move the optical axis away from the center of the
        viewport, as fractions of its half-width and half-height. A calibrated
        principal point ``(cx, cy)`` of an image ``width`` by ``height`` pixels
        corresponds to a window center of
        ``(-2 * (cx - width / 2) / width, 2 * (cy - height / 2) / height)``.

        .. versionadded:: 0.50

        See Also
        --------
        intrinsic_matrix

        Examples
        --------
        >>> import pyvista as pv
        >>> camera = pv.Camera()
        >>> camera.window_center
        (0.0, 0.0)
        >>> camera.window_center = (0.25, -0.1)
        >>> camera.window_center
        (0.25, -0.1)

        """
        return self.GetWindowCenter()

    @window_center.setter
    def window_center(self, value: VectorLike[float]) -> None:
        center = _validation.validate_array(
            value, must_have_shape=(2,), dtype_out=float, name='window center'
        )
        self.SetWindowCenter(*center)

    @property
    def explicit_aspect_ratio(self) -> float | None:  # numpydoc ignore=RT01
        """Return or set an aspect ratio to use in place of the viewport's own.

        The ratio is the width of the view frustum divided by its height. It is
        ``None`` when the camera takes the aspect ratio from the viewport it
        renders into, which assumes square pixels.

        .. versionadded:: 0.50

        See Also
        --------
        intrinsic_matrix

        Examples
        --------
        >>> import pyvista as pv
        >>> camera = pv.Camera()
        >>> camera.explicit_aspect_ratio is None
        True
        >>> camera.explicit_aspect_ratio = 1.25
        >>> camera.explicit_aspect_ratio
        1.25
        >>> camera.explicit_aspect_ratio = None
        >>> camera.explicit_aspect_ratio is None
        True

        """
        return self.GetExplicitAspectRatio() if self.GetUseExplicitAspectRatio() else None

    @explicit_aspect_ratio.setter
    def explicit_aspect_ratio(self, value: float | None) -> None:
        if value is None:
            self.SetUseExplicitAspectRatio(False)
            return
        ratio = _validation.validate_number(
            value,
            must_be_in_range=[0.0, np.inf],
            strict_lower_bound=True,
            name='explicit aspect ratio',
        )
        self.SetExplicitAspectRatio(ratio)
        self.SetUseExplicitAspectRatio(True)

    def _viewport_size(self) -> tuple[int, int]:
        """Return the pixel width and height of the viewport the camera renders into."""
        if self._renderer is None:
            msg = 'An intrinsic matrix requires a plotter to derive the image size from.'
            raise RuntimeError(msg)
        width, height = self._renderer.GetSize()
        if not width or not height:
            msg = (
                'An intrinsic matrix requires a plotter with a non-empty viewport, got '
                f'{width}x{height}. A closed plotter has none.'
            )
            raise RuntimeError(msg)
        return width, height

    @property
    def intrinsic_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return or set the pinhole intrinsic matrix of the camera.

        The matrix is ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` in pixels, as
        reported by a camera calibration such as ``cv2.calibrateCamera``, with
        ``cy`` measured from the top of the image. It describes the image the
        camera renders, so it is expressed in the pixel size of the viewport
        and changes with it. Axis skew cannot be represented and must be
        zero.

        Setting the matrix gives the camera a perspective projection.

        The camera has to belong to a plotter, which is what gives it an image
        to be calibrated for. Resetting the camera, as
        :meth:`~pyvista.Plotter.reset_camera` and the view directions do,
        restores its default field of view and discards ``fx`` and ``fy``. The
        principal point is kept.

        .. versionadded:: 0.50

        See Also
        --------
        extrinsic_matrix
        window_center
        explicit_aspect_ratio

        Examples
        --------
        A camera renders a square-pixel image centered on the optical axis
        until it is given a calibration.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> pl = pv.Plotter(window_size=(640, 480))
        >>> pl.camera.intrinsic_matrix.round(3)
        array([[895.692,   0.   , 320.   ],
               [  0.   , 895.692, 240.   ],
               [  0.   ,   0.   ,   1.   ]])

        >>> pl.camera.intrinsic_matrix = np.array(
        ...     [[800.0, 0.0, 310.0], [0.0, 760.0, 250.0], [0.0, 0.0, 1.0]]
        ... )
        >>> pl.camera.intrinsic_matrix
        array([[800.,   0., 310.],
               [  0., 760., 250.],
               [  0.,   0.,   1.]])

        """
        if self.parallel_projection:
            msg = 'An intrinsic matrix is only defined for a perspective projection.'
            raise ValueError(msg)
        width, height = self._viewport_size()
        projection = array_from_vtkmatrix(
            self.GetProjectionTransformMatrix(self._renderer.GetTiledAspectRatio(), -1.0, 1.0)
        )
        return np.array(
            [
                [projection[0, 0] * width / 2, 0.0, (1.0 - projection[0, 2]) * width / 2],
                [0.0, projection[1, 1] * height / 2, (1.0 + projection[1, 2]) * height / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    @intrinsic_matrix.setter
    def intrinsic_matrix(self, matrix: MatrixLike[float]) -> None:
        valid = _validation.validate_array(
            matrix, must_have_shape=(3, 3), dtype_out=float, name='intrinsic matrix'
        )
        width, height = self._viewport_size()
        if valid[0, 1] != 0.0:
            msg = 'Intrinsic matrices with axis skew are not supported.'
            raise ValueError(msg)
        if valid[1, 0] != 0.0 or not np.array_equal(valid[2], [0.0, 0.0, 1.0]):
            msg = (
                'An intrinsic matrix must be upper triangular with a last row of '
                f'(0, 0, 1), got {valid.tolist()}.'
            )
            raise ValueError(msg)
        focal_x, focal_y = valid[0, 0], valid[1, 1]
        if focal_x <= 0.0 or focal_y <= 0.0:
            msg = f'Intrinsic matrix focal lengths must be positive, got ({focal_x}, {focal_y}).'
            raise ValueError(msg)
        center_x, center_y = valid[0, 2], valid[1, 2]
        self.parallel_projection = False
        self.view_angle = np.degrees(2 * np.arctan(height / (2 * focal_y)))
        self.window_center = (
            -2 * (center_x - width / 2) / width,
            2 * (center_y - height / 2) / height,
        )
        self.explicit_aspect_ratio = (width * focal_y) / (height * focal_x)
        self.is_set = True

    @property
    def extrinsic_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return or set the pose of the camera as a 4x4 extrinsic matrix.

        The matrix maps world coordinates to camera coordinates in the OpenCV
        convention, with ``x`` to the right, ``y`` down and ``z`` along the
        viewing direction. Invert it for the camera-to-world pose. It describes
        the camera alone and does not include :attr:`model_transform_matrix`.

        Setting the matrix keeps the camera's :attr:`distance` to its focal
        point.

        .. versionadded:: 0.50

        See Also
        --------
        intrinsic_matrix

        Examples
        --------
        >>> import pyvista as pv
        >>> camera = pv.Camera()
        >>> camera.position = (0.0, 0.0, 4.0)
        >>> camera.focal_point = (0.0, 0.0, 0.0)
        >>> camera.up = (0.0, 1.0, 0.0)
        >>> camera.extrinsic_matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0., -1.,  0.,  0.],
               [ 0.,  0., -1.,  4.],
               [ 0.,  0.,  0.,  1.]])

        """
        view = array_from_vtkmatrix(self.GetViewTransformMatrix())
        return _OPENCV_FROM_VTK @ view

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, matrix: MatrixLike[float]) -> None:
        valid = _validation.validate_transform4x4(matrix, name='extrinsic matrix')
        rotation = _validation.validate_rotation(
            valid[:3, :3], must_have_handedness='right', name='extrinsic matrix rotation'
        )
        center = -rotation.T @ valid[:3, 3]
        distance = self.distance
        self.position = center
        self.focal_point = center + distance * rotation[2]
        self.up = -rotation[1]

    @property
    def direction(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Vector from the camera position to the focal point.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.direction  # doctest:+SKIP
        (-0.5773502691896257, -0.5773502691896257, -0.5773502691896257)

        """
        return self.GetDirectionOfProjection()

    def view_frustum(self, aspect: float = 1.0) -> pv.PolyData:
        """Get the view frustum.

        Parameters
        ----------
        aspect : float, default: 1.0
            The aspect of the viewport to compute the planes.

        Returns
        -------
        pyvista.PolyData
            View frustum.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> frustum = pl.camera.view_frustum(1.0)
        >>> frustum.n_points
        8
        >>> frustum.n_cells
        6

        """
        frustum_planes = [0] * 24
        self.GetFrustumPlanes(aspect, frustum_planes)  # type: ignore[arg-type]
        planes = _vtk.vtkPlanes()
        planes.SetFrustumPlanes(frustum_planes)  # type: ignore[arg-type]

        frustum_source = _vtk.vtkFrustumSource()
        frustum_source.ShowLinesOff()
        frustum_source.SetPlanes(planes)
        frustum_source.Update()

        return pv.wrap(frustum_source.GetOutput())

    @property
    def roll(self) -> float:  # numpydoc ignore=RT01
        """Return or set the roll of the camera about the direction of projection.

        This will spin the camera about its axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.roll
        -120.00000000000001
        >>> pl.camera.roll = 45.0
        >>> pl.camera.roll
        45.0

        """
        return self.GetRoll()

    @roll.setter
    def roll(self, angle: float) -> None:
        self.SetRoll(angle)
        self.is_set = True

    @property
    def elevation(self) -> float:  # numpydoc ignore=RT01
        """Return or set the vertical rotation of the scene.

        Rotate the camera about the cross product of the negative of
        the direction of projection and the view up vector, using the
        focal point as the center of rotation.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.elevation
        0.0
        >>> pl.camera.elevation = 45.0
        >>> pl.camera.elevation
        45.0

        """
        return self._elevation

    @elevation.setter
    def elevation(self, angle: float) -> None:
        if self._elevation:
            self.Elevation(-self._elevation)
        self._elevation = angle
        self.Elevation(angle)
        self.is_set = True

    @property
    def azimuth(self) -> float:  # numpydoc ignore=RT01
        """Return or set the azimuth of the camera.

        Rotate the camera about the view up vector centered at the
        focal point. Note that the view up vector is whatever was set
        via SetViewUp, and is not necessarily perpendicular to the
        direction of projection.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.azimuth
        0.0
        >>> pl.camera.azimuth = 45.0
        >>> pl.camera.azimuth
        45.0

        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, angle: float) -> None:
        if self._azimuth:
            self.Azimuth(-self._azimuth)
        self._azimuth = angle
        self.Azimuth(angle)
        self.is_set = True

    def copy(self) -> Camera:
        """Return a deep copy of the camera.

        Returns
        -------
        pyvista.Camera
            Deep copy of the camera.

        Examples
        --------
        Create a camera and check that its copy holds the same
        transformation matrix until the original is changed.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> camera = pv.Camera()
        >>> camera.model_transform_matrix = np.array(
        ...     [
        ...         [1.0, 0.0, 0.0, 0.0],
        ...         [0.0, 1.0, 0.0, 0.0],
        ...         [0.0, 0.0, 1.0, 0.0],
        ...         [0.0, 0.0, 0.0, 1.0],
        ...     ]
        ... )
        >>> copied_camera = camera.copy()
        >>> copied_camera == camera
        True
        >>> camera.model_transform_matrix = np.array(
        ...     [
        ...         [1.0, 0.0, 0.0, 0.0],
        ...         [0.0, 1.0, 0.0, 0.0],
        ...         [0.0, 0.0, 1.0, 0.0],
        ...         [0.0, 0.0, 0.0, 0.5],
        ...     ]
        ... )
        >>> copied_camera == camera
        False

        """
        immutable_attrs = [
            'position',
            'focal_point',
            'model_transform_matrix',
            'distance',
            'thickness',
            'parallel_scale',
            'up',
            'clipping_range',
            'window_center',
            'explicit_aspect_ratio',
            'view_angle',
            'roll',
            'parallel_projection',
            'is_set',
        ]
        new_camera = Camera()

        for attr in immutable_attrs:
            value = getattr(self, attr)
            setattr(new_camera, attr, value)

        return new_camera

    def tight(
        self,
        *,
        padding: float = 0.0,
        adjust_render_window: bool = True,
        view: _ViewOptions = 'xy',
        negative: bool = False,
    ) -> None:
        """Adjust the camera position so that the actors fill the entire renderer.

        The camera view direction is reoriented to be normal to the ``view``
        plane. When ``negative=False``, The first letter of ``view`` refers
        to the axis that points to the right. The second letter of ``view``
        refers to axis that points up.  When ``negative=True``, the first
        letter refers to the axis that points left.  The up direction is
        unchanged.

        Parallel projection is enabled when using this function.

        Parameters
        ----------
        padding : float, default: 0.0
            Additional padding around the actors. This is effectively a zoom,
            where a value of 0.01 results in a zoom out of 1%.

        adjust_render_window : bool, default: True
            Adjust the size of the render window as to match the dimensions of
            the visible actors.

        view : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}, default: 'xy'
            Plane to which the view is oriented.

        negative : bool, default: False
            Whether to view in opposite direction.

        Notes
        -----
        This resets the view direction to look at a plane with parallel projection.

        Examples
        --------
        .. pyvista-plot::
            :force_static:

            Display the bird image with a tight view.

            >>> import pyvista as pv
            >>> from pyvista import examples
            >>> bird = examples.download_bird()
            >>> pl = pv.Plotter(border=True, border_width=5)
            >>> _ = pl.add_mesh(bird, rgb=True)
            >>> pl.camera.tight()
            >>> pl.show()

            Set the background to blue use a 5% padding around the image.

            >>> pl = pv.Plotter()
            >>> _ = pl.add_mesh(bird, rgb=True)
            >>> pl.background_color = 'b'
            >>> pl.camera.tight(padding=0.05)
            >>> pl.show()

        """
        if self._renderer is None:
            msg = 'Camera must be associated with a renderer to fit it to the actors.'
            raise AttributeError(msg)

        # Inspired by vedo resetCamera. Thanks @marcomusy.
        x0, x1, y0, y1, z0, z1 = self._renderer.bounds

        self.enable_parallel_projection()

        self._renderer.ComputeAspect()
        aspect = self._renderer.GetAspect()

        position0 = np.array([x0, y0, z0])
        position1 = np.array([x1, y1, z1])
        objects_size = position1 - position0
        position = position0 + objects_size / 2

        direction, viewup = view_vectors(view, negative=negative)
        horizontal = np.cross(direction, viewup)

        vert_dist = abs(objects_size @ viewup)
        horiz_dist = abs(objects_size @ horizontal)

        # set focal point to objects' center
        # offset camera position from objects center by dist in opposite of viewing direction
        # (actual distance doesn't matter due to parallel projection)
        dist = 1
        camera_position = position + dist * direction

        self.SetViewUp(*viewup)
        self.SetPosition(*camera_position)
        self.SetFocalPoint(*position)

        ps = max(horiz_dist / aspect[0], vert_dist) / 2
        self.parallel_scale = ps * (1 + padding)
        self._renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)

        if adjust_render_window:
            ren_win = self._renderer.GetRenderWindow()
            size = list(ren_win.GetSize())
            size_ratio = size[0] / size[1]
            tight_ratio = horiz_dist / vert_dist
            resize_ratio = tight_ratio / size_ratio
            if resize_ratio < 1:
                size[0] = round(size[0] * resize_ratio)
            else:
                size[1] = round(size[1] / resize_ratio)

            ren_win.SetSize(size)

            # simply call tight again to reset the parallel scale due to the
            # resized window
            self.tight(padding=padding, adjust_render_window=False, view=view, negative=negative)

        self.is_set = True
