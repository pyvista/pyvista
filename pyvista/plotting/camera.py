"""Module containing pyvista implementation of :vtk:`vtkCamera`."""

from __future__ import annotations

from pathlib import Path
from weakref import proxy
import xml.dom.minidom as md
from xml.etree import ElementTree as ET

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.misc import _NoNewAttrMixin

from . import _vtk
from .helpers import view_vectors


class Camera(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkCamera):
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

    def __init__(self, renderer=None):
        """Initialize a new camera descriptor."""
        self._parallel_projection = False
        self._elevation = 0.0
        self._azimuth = 0.0
        self._is_set = False
        self._focus = None  # Used by BackgroundRenderer

        if renderer:
            if not isinstance(renderer, pyvista.Renderer):
                msg = 'Camera only accepts a pyvista.Renderer or None as the ``renderer`` argument'
                raise TypeError(msg)
            self._renderer = proxy(renderer)
        else:
            self._renderer = None  # type: ignore[assignment]

    def __eq__(self, other) -> bool:
        """Compare whether the relevant attributes of two cameras are equal."""
        # attributes which are native python types and thus implement __eq__

        native_attrs = [
            'position',
            'focal_point',
            'parallel_projection',
            'distance',
            'thickness',
            'parallel_scale',
            'clipping_range',
            'view_angle',
            'roll',
        ]
        for attr in native_attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        this_trans = self.model_transform_matrix
        that_trans = other.model_transform_matrix
        trans_count = sum(1 for trans in [this_trans, that_trans] if trans is not None)
        if trans_count == 1:
            # either but not both are None
            return False
        return not (trans_count == 2 and not np.array_equal(this_trans, that_trans))

    __hash__ = None  # type: ignore[assignment]  # https://github.com/pyvista/pyvista/pull/7671

    def __repr__(self):
        """Print a repr specifying the id of the camera and its camera type."""
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

    def __str__(self):
        """Return the object string representation."""
        return self.__repr__()

    def __del__(self):
        """Delete the camera."""
        self.RemoveAllObservers()

    @property
    def is_set(self) -> bool:  # numpydoc ignore=RT01
        """Get or set whether this camera has been configured."""
        return self._is_set

    @is_set.setter
    def is_set(self, value: bool):
        self._is_set = bool(value)

    @classmethod
    def from_paraview_pvcc(cls, filename: str | Path) -> Camera:
        """Load a Paraview camera file (.pvcc extension).

        Returns a pyvista.Camera object for which attributes has been read
        from the filename argument.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to Paraview camera file (.pvcc).

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

    def to_paraview_pvcc(self, filename: str | Path):
        """Write the camera parameters to a Paraview camera file (.pvcc extension).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to Paraview camera file (.pvcc).

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

        xmlstr = ET.tostring(root).decode()
        newxml = md.parseString(xmlstr)
        with Path(filename).open('w') as outfile:
            outfile.write(newxml.toprettyxml(indent='\t', newl='\n'))

    @property
    def position(self):  # numpydoc ignore=RT01
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
    def position(self, value):
        self.SetPosition(value)
        self._elevation = 0.0
        self._azimuth = 0.0
        if self._renderer:  # type: ignore[truthy-bool]
            self.reset_clipping_range()
        self.is_set = True

    def reset_clipping_range(self):
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
            msg = 'Camera is must be associated with a renderer to reset its clipping range.'  # type: ignore[unreachable]
            raise AttributeError(msg)
        self._renderer.reset_camera_clipping_range()

    @property
    def focal_point(self):  # numpydoc ignore=RT01
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
    def focal_point(self, point):
        self.SetFocalPoint(point)
        self.is_set = True

    @property
    def model_transform_matrix(self):  # numpydoc ignore=RT01
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
        >>>
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 0.5]])

        """
        vtk_matrix = self.GetModelTransformMatrix()
        matrix = np.empty((4, 4))
        vtk_matrix.DeepCopy(matrix.ravel(), vtk_matrix)
        return matrix

    @model_transform_matrix.setter
    def model_transform_matrix(self, matrix):
        vtk_matrix = _vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(matrix.ravel())
        self.SetModelTransformMatrix(vtk_matrix)

    @property
    def distance(self):  # numpydoc ignore=RT01
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
    def distance(self, distance):
        self.SetDistance(distance)
        self.is_set = True

    @property
    def thickness(self):  # numpydoc ignore=RT01
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
    def thickness(self, length):
        self.SetThickness(length)

    @property
    def parallel_scale(self):  # numpydoc ignore=RT01
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
    def parallel_scale(self, scale):
        self.SetParallelScale(scale)

    def zoom(self, value):
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
    def up(self):  # numpydoc ignore=RT01
        """Return or set the "up" of the camera.

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
    def up(self, vector):
        self.SetViewUp(vector)
        self.is_set = True

    def enable_parallel_projection(self):
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

    def disable_parallel_projection(self):
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
    def parallel_projection(self):  # numpydoc ignore=RT01
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
    def parallel_projection(self, state):
        if state:
            self.enable_parallel_projection()
        else:
            self.disable_parallel_projection()

    @property
    def clipping_range(self):  # numpydoc ignore=RT01
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
    def clipping_range(self, points):
        if points[0] > points[1]:
            msg = 'Near point must be lower than the far point.'
            raise ValueError(msg)
        self.SetClippingRange(points[0], points[1])

    @property
    def view_angle(self):  # numpydoc ignore=RT01
        """Return or set the camera view angle.

        Examples
        --------
        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> plotter.camera.view_angle
        30.0
        >>> plotter.camera.view_angle = 60.0
        >>> plotter.camera.view_angle
        60.0

        """
        return self.GetViewAngle()

    @view_angle.setter
    def view_angle(self, value):
        self.SetViewAngle(value)

    @property
    def direction(self):  # numpydoc ignore=RT01
        """Vector from the camera position to the focal point.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera.direction  # doctest:+SKIP
        (-0.5773502691896257, -0.5773502691896257, -0.5773502691896257)

        """
        return self.GetDirectionOfProjection()

    def view_frustum(self, aspect=1.0):
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
        >>> plotter = pv.Plotter()
        >>> frustum = plotter.camera.view_frustum(1.0)
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

        return pyvista.wrap(frustum_source.GetOutput())

    @property
    def roll(self):  # numpydoc ignore=RT01
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
    def roll(self, angle):
        self.SetRoll(angle)
        self.is_set = True

    @property
    def elevation(self):  # numpydoc ignore=RT01
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
    def elevation(self, angle):
        if self._elevation:
            self.Elevation(-self._elevation)
        self._elevation = angle
        self.Elevation(angle)
        self.is_set = True

    @property
    def azimuth(self):  # numpydoc ignore=RT01
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
    def azimuth(self, angle):
        if self._azimuth:
            self.Azimuth(-self._azimuth)
        self._azimuth = angle
        self.Azimuth(angle)
        self.is_set = True

    def copy(self):
        """Return a deep copy of the camera.

        Returns
        -------
        pyvista.Camera
            Deep copy of the camera.

        Examples
        --------
        Create a camera and check that it shares a transformation
        matrix with its shallow copy.

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

    @_deprecate_positional_args
    def tight(  # noqa: PLR0917
        self,
        padding=0.0,
        adjust_render_window: bool = True,  # noqa: FBT001, FBT002
        view='xy',
        negative: bool = False,  # noqa: FBT001, FBT002
    ):
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
            Additional padding around the actor(s). This is effectively a zoom,
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
        Display the puppy image with a tight view.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> puppy = examples.download_puppy()
        >>> pl = pv.Plotter(border=True, border_width=5)
        >>> _ = pl.add_mesh(puppy, rgb=True)
        >>> pl.camera.tight()
        >>> pl.show()

        Set the background to blue use a 5% padding around the image.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(puppy, rgb=True)
        >>> pl.background_color = 'b'
        >>> pl.camera.tight(padding=0.05)
        >>> pl.show()

        """
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
