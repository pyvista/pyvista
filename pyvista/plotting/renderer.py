"""Module containing pyvista implementation of vtkRenderer."""

import collections.abc
from weakref import proxy

import numpy as np
import vtk
from vtk import vtkRenderer

import pyvista
from pyvista.utilities import wrap, check_depth_peeling
from .theme import parse_color, parse_font_family, rcParams, MAX_N_COLOR_BARS
from .tools import create_axes_orientation_box, create_axes_marker
from .camera import Camera


def scale_point(camera, point, invert=False):
    """Scale a point using the camera's transform matrix.

    Parameters
    ----------
    camera : Camera
        The camera who's matrix to use.

    point : tuple(float)
        Length 3 tuple of the point coordinates.

    invert : bool
        If True, invert the matrix to transform the point out of the
        camera's transformed space. Default is False to transform a
        point from world coordinates to the camera's transformed space.

    """
    if invert:
        mtx = vtk.vtkMatrix4x4()
        mtx.DeepCopy(camera.GetModelTransformMatrix())
        mtx.Invert()
    else:
        mtx = camera.GetModelTransformMatrix()
    scaled = mtx.MultiplyDoublePoint((point[0], point[1], point[2], 0.0))
    return (scaled[0], scaled[1], scaled[2])


class CameraPosition:
    """Container to hold camera location attributes."""

    def __init__(self, position, focal_point, viewup):
        """Initialize a new camera position descriptor."""
        self._position = position
        self._focal_point = focal_point
        self._viewup = viewup

    def to_list(self):
        """Convert to a list of the position, focal point, and viewup."""
        return [self._position, self._focal_point, self._viewup]

    def __repr__(self):
        """List representation method."""
        return "[{},\n {},\n {}]".format(*self.to_list())

    def __getitem__(self, index):
        """Fetch a component by index location like a list."""
        return self.to_list()[index]

    def __eq__(self, other):
        """Comparison operator to act on list version of CameraPosition object."""
        if isinstance(other, CameraPosition):
            return self.to_list() == other.to_list()
        return self.to_list() == other

    @property
    def position(self):
        """Location of the camera in world coordinates."""
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def focal_point(self):
        """Location of the camera's focus in world coordinates."""
        return self._focal_point

    @focal_point.setter
    def focal_point(self, value):
        self._focal_point = value

    @property
    def viewup(self):
        """Viewup vector of the camera."""
        return self._viewup

    @viewup.setter
    def viewup(self, value):
        self._viewup = value


class Renderer(vtkRenderer):
    """Renderer class."""

    # map camera_position string to an attribute
    CAMERA_STR_ATTR_MAP = {'xy': 'view_xy', 'xz': 'view_xz',
                           'yz': 'view_yz', 'yx': 'view_yx',
                           'zx': 'view_zx', 'zy': 'view_zy',
                           'iso': 'view_isometric'}

    def __init__(self, parent, border=True, border_color=(1, 1, 1),
                 border_width=2.0):
        """Initialize the renderer."""
        super().__init__()
        self._actors = {}
        self.parent = parent
        self.camera_set = False
        self.bounding_box_actor = None
        self.scale = [1.0, 1.0, 1.0]
        self.AutomaticLightCreationOff()
        self._floors = []
        self._floor_kwargs = []
        # this keeps track of lights added manually to prevent garbage collection
        self._lights = []
        self._camera = Camera()
        self.SetActiveCamera(self._camera)

        # This is a private variable to keep track of how many colorbars exist
        # This allows us to keep adding colorbars without overlapping
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}

        if border:
            self.add_border(border_color, border_width)

    #### Properties ####

    @property
    def camera_position(self):
        """Return camera position of active render window."""
        return CameraPosition(
            scale_point(self.camera, self.camera.position, invert=True),
            scale_point(self.camera, self.camera.focal_point, invert=True),
            self.camera.up)

    @camera_position.setter
    def camera_position(self, camera_location):
        """Set camera position of all active render windows."""
        if camera_location is None:
            return
        elif isinstance(camera_location, str):
            camera_location = camera_location.lower()
            if camera_location not in self.CAMERA_STR_ATTR_MAP:
                err = pyvista.core.errors.InvalidCameraError
                raise err('Invalid view direction.  '
                          'Use one of the following:\n   '
                          f'{", ".join(self.CAMERA_STR_ATTR_MAP)}')

            getattr(self, self.CAMERA_STR_ATTR_MAP[camera_location])()

        elif isinstance(camera_location[0], (int, float)):
            if len(camera_location) != 3:
                raise pyvista.core.errors.InvalidCameraError
            self.view_vector(camera_location)
        else:
            # check if a valid camera position
            if not isinstance(camera_location, CameraPosition):
                if not len(camera_location) == 3:
                    raise pyvista.core.errors.InvalidCameraError
                elif any([len(item) != 3 for item in camera_location]):
                    raise pyvista.core.errors.InvalidCameraError

            # everything is set explicitly
            self.camera.position = scale_point(self.camera, camera_location[0], invert=False)
            self.camera.focal_point = scale_point(self.camera, camera_location[1], invert=False)
            self.camera.up = camera_location[2]

        # reset clipping range
        self.ResetCameraClippingRange()
        self.camera_set = True
        self.Modified()

    @property
    def camera(self):
        """Return the active camera for the rendering scene."""
        return self._camera

    @camera.setter
    def camera(self, source):
        """Set the active camera for the rendering scene."""
        self._camera = source
        self.SetActiveCamera(self._camera)
        self.camera_position = CameraPosition(
            scale_point(source, source.position, invert=True),
            scale_point(source, source.focal_point, invert=True),
            source.up
        )
        self.Modified()
        self.camera_set = True

    @property
    def bounds(self):
        """Return the bounds of all actors present in the rendering window."""
        the_bounds = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])

        def _update_bounds(bounds):
            def update_axis(ax):
                if bounds[ax*2] < the_bounds[ax*2]:
                    the_bounds[ax*2] = bounds[ax*2]
                if bounds[ax*2+1] > the_bounds[ax*2+1]:
                    the_bounds[ax*2+1] = bounds[ax*2+1]
            for ax in range(3):
                update_axis(ax)
            return

        for actor in self._actors.values():
            if isinstance(actor, vtk.vtkCubeAxesActor):
                continue
            if (hasattr(actor, 'GetBounds') and actor.GetBounds() is not None
                 and id(actor) != id(self.bounding_box_actor)):
                _update_bounds(actor.GetBounds())

        if np.any(np.abs(the_bounds)):
            the_bounds[the_bounds == np.inf] = -1.0
            the_bounds[the_bounds == -np.inf] = 1.0

        return the_bounds.tolist()

    @property
    def length(self):
        """Return the length of the diagonal of the bounding box of the scene."""
        return pyvista.Box(self.bounds).length

    @property
    def center(self):
        """Return the center of the bounding box around all data present in the scene."""
        bounds = self.bounds
        x = (bounds[1] + bounds[0])/2
        y = (bounds[3] + bounds[2])/2
        z = (bounds[5] + bounds[4])/2
        return [x, y, z]

    @property
    def background_color(self):
        """Return the background color of this renderer."""
        return self.GetBackground()

    @background_color.setter
    def background_color(self, color):
        """Set the background color of this renderer."""
        self.set_background(color)
        self.Modified()

    #### Everything else ####

    def enable_depth_peeling(self, number_of_peels=None, occlusion_ratio=None):
        """Enable depth peeling to improve rendering of translucent geometry.

        Parameters
        ----------
        number_of_peels : int
            The maximum number of peeling layers. Initial value is 4 and is set
            in the ``rcParams``. A special value of 0 means no maximum limit.
            It has to be a positive value.

        occlusion_ratio : float
            The threshold under which the deepth peeling algorithm stops to
            iterate over peel layers. This is the ratio of the number of pixels
            that have been touched by the last layer over the total number of
            pixels of the viewport area. Initial value is 0.0, meaning
            rendering have to be exact. Greater values may speed-up the
            rendering with small impact on the quality.

        """
        if number_of_peels is None:
            number_of_peels = rcParams["depth_peeling"]["number_of_peels"]
        if occlusion_ratio is None:
            occlusion_ratio = rcParams["depth_peeling"]["occlusion_ratio"]
        depth_peeling_supported = check_depth_peeling(number_of_peels,
                                                      occlusion_ratio)
        if depth_peeling_supported:
            self.SetUseDepthPeeling(True)
            self.SetMaximumNumberOfPeels(number_of_peels)
            self.SetOcclusionRatio(occlusion_ratio)
        self.Modified()
        return depth_peeling_supported

    def disable_depth_peeling(self):
        """Disable depth peeling."""
        self.SetUseDepthPeeling(False)
        self.Modified()

    def enable_anti_aliasing(self):
        """Enable anti-aliasing FXAA."""
        self.SetUseFXAA(True)
        self.Modified()

    def disable_anti_aliasing(self):
        """Disable anti-aliasing FXAA."""
        self.SetUseFXAA(False)
        self.Modified()

    def add_border(self, color=[1, 1, 1], width=2.0):
        """Add borders around the frame."""
        points = np.array([[1., 1., 0.],
                           [0., 1., 0.],
                           [0., 0., 0.],
                           [1., 0., 0.]])

        lines = np.array([[2, 0, 1],
                          [2, 1, 2],
                          [2, 2, 3],
                          [2, 3, 0]]).ravel()

        poly = pyvista.PolyData()
        poly.points = points
        poly.lines = lines

        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToNormalizedViewport()

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        mapper.SetTransformCoordinate(coordinate)

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(parse_color(color))
        actor.GetProperty().SetLineWidth(width)

        self.AddViewProp(actor)
        self.Modified()
        return actor

    def add_actor(self, uinput, reset_camera=False, name=None, culling=False,
                  pickable=True, render=True):
        """Add an actor to render window.

        Creates an actor if input is a mapper.

        Parameters
        ----------
        uinput : vtk.vtkMapper or vtk.vtkActor
            vtk mapper or vtk actor to be added.

        reset_camera : bool, optional
            Resets the camera when true.

        culling : str, optional
            Does not render faces that are culled. Options are ``'front'`` or
            ``'back'``. This can be helpful for dense surface meshes,
            especially when edges are visible, but can cause flat
            meshes to be partially displayed.  Default False.

        Returns
        -------
        actor : vtk.vtkActor
            The actor.

        actor_properties : vtk.Properties
            Actor properties.

        """
        # Remove actor by that name if present
        rv = self.remove_actor(name, reset_camera=False, render=render)

        if isinstance(uinput, vtk.vtkMapper):
            actor = vtk.vtkActor()
            actor.SetMapper(uinput)
        else:
            actor = uinput

        self.AddActor(actor)
        actor.renderer = proxy(self)

        if name is None:
            name = actor.GetAddressAsString("")

        self._actors[name] = actor

        if reset_camera:
            self.reset_camera(render)
        elif not self.camera_set and reset_camera is None and not rv:
            self.reset_camera(render)
        elif render:
            self.parent.render()

        self.update_bounds_axes()

        if isinstance(culling, str):
            culling = culling.lower()

        if culling:
            if culling in [True, 'back', 'backface', 'b']:
                try:
                    actor.GetProperty().BackfaceCullingOn()
                except AttributeError:  # pragma: no cover
                    pass
            elif culling in ['front', 'frontface', 'f']:
                try:
                    actor.GetProperty().FrontfaceCullingOn()
                except AttributeError:  # pragma: no cover
                    pass
            else:
                raise ValueError(f'Culling option ({culling}) not understood.')

        actor.SetPickable(pickable)

        self.ResetCameraClippingRange()

        self.Modified()

        return actor, actor.GetProperty()

    def add_axes_at_origin(self, x_color=None, y_color=None, z_color=None,
                           xlabel='X', ylabel='Y', zlabel='Z', line_width=2,
                           labels_off=False):
        """Add axes actor at origin.

        Returns
        -------
        marker_actor : vtk.vtkAxesActor
            vtkAxesActor actor

        """
        self.marker_actor = create_axes_marker(line_width=line_width,
            x_color=x_color, y_color=y_color, z_color=z_color,
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, labels_off=labels_off)
        self.AddActor(self.marker_actor)
        memory_address = self.marker_actor.GetAddressAsString("")
        self._actors[memory_address] = self.marker_actor
        self.Modified()
        return self.marker_actor

    def add_orientation_widget(self, actor, interactive=None, color=None,
                               opacity=1.0):
        """Use the given actor in an orientation marker widget.

        Color and opacity are only valid arguments if a mesh is passed.

        Parameters
        ----------
        actor : vtk.vtkActor or pyvista.Common
            The mesh or actor to use as the marker.

        color : string, optional
            The color of the actor.

        opacity : int or float, optional
            Opacity of the marker.

        """
        if isinstance(actor, pyvista.Common):
            mapper = vtk.vtkDataSetMapper()
            mesh = actor.copy()
            mesh.clear_arrays()
            mapper.SetInputData(mesh)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            if color is not None:
                prop.SetColor(parse_color(color))
            prop.SetOpacity(opacity)
        if hasattr(self, 'axes_widget'):
            # Delete the old one
            self.axes_widget.EnabledOff()
            self.Modified()
            del self.axes_widget
        if interactive is None:
            interactive = rcParams['interactive']
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(actor)
        if hasattr(self.parent, 'iren'):
            self.axes_widget.SetInteractor(self.parent.iren)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.SetInteractive(interactive)
        self.axes_widget.SetCurrentRenderer(self)
        self.Modified()
        return self.axes_widget

    def add_axes(self, interactive=None, line_width=2,
                 color=None, x_color=None, y_color=None, z_color=None,
                 xlabel='X', ylabel='Y', zlabel='Z', labels_off=False,
                 box=None, box_args=None):
        """Add an interactive axes widget in the bottom left corner.

        Parameters
        ----------
        interacitve : bool
            Enable this orientation widget to be moved by the user.

        line_width : int
            The width of the marker lines

        box : bool
            Show a box orientation marker. Use ``box_args`` to adjust.
            See :any:`pyvista.create_axes_orientation_box` for details.

        opacity : int or float, optional
            The opacity of the marker.
        """
        if interactive is None:
            interactive = rcParams['interactive']
        if hasattr(self, 'axes_widget'):
            self.axes_widget.EnabledOff()
            self.Modified()
            del self.axes_widget
        if box is None:
            box = rcParams['axes']['box']
        if box:
            if box_args is None:
                box_args = {}
            self.axes_actor = create_axes_orientation_box(
                label_color=color, line_width=line_width,
                x_color=x_color, y_color=y_color, z_color=z_color,
                xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                labels_off=labels_off, **box_args)
        else:
            self.axes_actor = create_axes_marker(
                label_color=color, line_width=line_width,
                x_color=x_color, y_color=y_color, z_color=z_color,
                xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, labels_off=labels_off)
        self.add_orientation_widget(self.axes_actor, interactive=interactive,
                                    color=None)
        return self.axes_actor

    def hide_axes(self):
        """Hide the axes orientation widget."""
        if hasattr(self, 'axes_widget') and self.axes_widget.GetEnabled():
            self.axes_widget.EnabledOff()
            self.Modified()

    def show_axes(self):
        """Show the axes orientation widget."""
        if hasattr(self, 'axes_widget'):
            self.axes_widget.EnabledOn()
            self.axes_widget.SetCurrentRenderer(self)
        else:
            self.add_axes()
        self.Modified()

    def show_bounds(self, mesh=None, bounds=None, show_xaxis=True,
                    show_yaxis=True, show_zaxis=True, show_xlabels=True,
                    show_ylabels=True, show_zlabels=True,
                    bold=True, font_size=None,
                    font_family=None, color=None,
                    xlabel='X Axis', ylabel='Y Axis', zlabel='Z Axis',
                    use_2d=False, grid=None, location='closest', ticks=None,
                    all_edges=False, corner_factor=0.5, fmt=None,
                    minor_ticks=False, padding=0.0, render=None):
        """Add bounds axes.

        Shows the bounds of the most recent input mesh unless mesh is specified.

        Parameters
        ----------
        mesh : vtkPolydata or unstructured grid, optional
            Input mesh to draw bounds axes around

        bounds : list or tuple, optional
            Bounds to override mesh bounds.
            [xmin, xmax, ymin, ymax, zmin, zmax]

        show_xaxis : bool, optional
            Makes x axis visible.  Default True.

        show_yaxis : bool, optional
            Makes y axis visible.  Default True.

        show_zaxis : bool, optional
            Makes z axis visible.  Default True.

        show_xlabels : bool, optional
            Shows x labels.  Default True.

        show_ylabels : bool, optional
            Shows y labels.  Default True.

        show_zlabels : bool, optional
            Shows z labels.  Default True.

        italic : bool, optional
            Italicises axis labels and numbers.  Default False.

        bold : bool, optional
            Bolds axis labels and numbers.  Default True.

        shadow : bool, optional
            Adds a black shadow to the text.  Default False.

        font_size : float, optional
            Sets the size of the label font.  Defaults to 16.

        font_family : string, optional
            Font family.  Must be either courier, times, or arial.

        color : string or 3 item list, optional
            Color of all labels and axis titles.  Default white.
            Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        xlabel : string, optional
            Title of the x axis.  Default "X Axis"

        ylabel : string, optional
            Title of the y axis.  Default "Y Axis"

        zlabel : string, optional
            Title of the z axis.  Default "Z Axis"

        use_2d : bool, optional
            A bug with vtk 6.3 in Windows seems to cause this function
            to crash this can be enabled for smoother plotting for
            other environments.

        grid : bool or str, optional
            Add grid lines to the backface (``True``, ``'back'``, or
            ``'backface'``) or to the frontface (``'front'``,
            ``'frontface'``) of the axes actor.

        location : str, optional
            Set how the axes are drawn: either static (``'all'``),
            closest triad (``front``), furthest triad (``'back'``),
            static closest to the origin (``'origin'``), or outer
            edges (``'outer'``) in relation to the camera
            position. Options include: ``'all', 'front', 'back',
            'origin', 'outer'``

        ticks : str, optional
            Set how the ticks are drawn on the axes grid. Options include:
            ``'inside', 'outside', 'both'``

        all_edges : bool, optional
            Adds an unlabeled and unticked box at the boundaries of
            plot. Useful for when wanting to plot outer grids while
            still retaining all edges of the boundary.

        corner_factor : float, optional
            If ``all_edges````, this is the factor along each axis to
            draw the default box. Dafuault is 0.5 to show the full box.

        padding : float, optional
            An optional percent padding along each axial direction to cushion
            the datasets in the scene from the axes annotations. Defaults to
            have no padding

        Returns
        -------
        cube_axes_actor : vtk.vtkCubeAxesActor
            Bounds actor

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = pyvista.Sphere()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(mesh)
        >>> _ = plotter.show_bounds(grid='front', location='outer', all_edges=True)
        >>> plotter.show() # doctest:+SKIP

        """
        self.remove_bounds_axes()

        if font_family is None:
            font_family = rcParams['font']['family']
        if font_size is None:
            font_size = rcParams['font']['size']
        if color is None:
            color = rcParams['font']['color']
        if fmt is None:
            fmt = rcParams['font']['fmt']

        color = parse_color(color)

        # Use the bounds of all data in the rendering window
        if mesh is None and bounds is None:
            bounds = self.bounds

        # create actor
        cube_axes_actor = vtk.vtkCubeAxesActor()
        if use_2d or not np.allclose(self.scale, [1.0, 1.0, 1.0]):
            cube_axes_actor.SetUse2DMode(True)
        else:
            cube_axes_actor.SetUse2DMode(False)

        if grid:
            if isinstance(grid, str) and grid.lower() in ('front', 'frontface'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_CLOSEST)
            if isinstance(grid, str) and grid.lower() in ('both', 'all'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_ALL)
            else:
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
            # Only show user desired grid lines
            cube_axes_actor.SetDrawXGridlines(show_xaxis)
            cube_axes_actor.SetDrawYGridlines(show_yaxis)
            cube_axes_actor.SetDrawZGridlines(show_zaxis)
            # Set the colors
            cube_axes_actor.GetXAxesGridlinesProperty().SetColor(color)
            cube_axes_actor.GetYAxesGridlinesProperty().SetColor(color)
            cube_axes_actor.GetZAxesGridlinesProperty().SetColor(color)

        if isinstance(ticks, str):
            ticks = ticks.lower()
            if ticks in ('inside'):
                cube_axes_actor.SetTickLocationToInside()
            elif ticks in ('outside'):
                cube_axes_actor.SetTickLocationToOutside()
            elif ticks in ('both'):
                cube_axes_actor.SetTickLocationToBoth()
            else:
                raise ValueError(f'Value of ticks ({ticks}) not understood.')

        if isinstance(location, str):
            location = location.lower()
            if location in ('all'):
                cube_axes_actor.SetFlyModeToStaticEdges()
            elif location in ('origin'):
                cube_axes_actor.SetFlyModeToStaticTriad()
            elif location in ('outer'):
                cube_axes_actor.SetFlyModeToOuterEdges()
            elif location in ('default', 'closest', 'front'):
                cube_axes_actor.SetFlyModeToClosestTriad()
            elif location in ('furthest', 'back'):
                cube_axes_actor.SetFlyModeToFurthestTriad()
            else:
                raise ValueError(f'Value of location ({location}) not understood.')

        # set bounds
        if bounds is None:
            bounds = np.array(mesh.GetBounds())
        if isinstance(padding, (int, float)) and 0.0 <= padding < 1.0:
            if not np.any(np.abs(bounds) == np.inf):
                cushion = np.array([np.abs(bounds[1] - bounds[0]),
                                    np.abs(bounds[3] - bounds[2]),
                                    np.abs(bounds[5] - bounds[4])]) * padding
                bounds[::2] -= cushion
                bounds[1::2] += cushion
        else:
            raise ValueError(f'padding ({padding}) not understood. Must be float between 0 and 1')
        cube_axes_actor.SetBounds(bounds)

        # show or hide axes
        cube_axes_actor.SetXAxisVisibility(show_xaxis)
        cube_axes_actor.SetYAxisVisibility(show_yaxis)
        cube_axes_actor.SetZAxisVisibility(show_zaxis)

        # disable minor ticks
        if not minor_ticks:
            cube_axes_actor.XAxisMinorTickVisibilityOff()
            cube_axes_actor.YAxisMinorTickVisibilityOff()
            cube_axes_actor.ZAxisMinorTickVisibilityOff()

        cube_axes_actor.SetCamera(self.camera)

        # set color
        cube_axes_actor.GetXAxesLinesProperty().SetColor(color)
        cube_axes_actor.GetYAxesLinesProperty().SetColor(color)
        cube_axes_actor.GetZAxesLinesProperty().SetColor(color)

        # empty arr
        empty_str = vtk.vtkStringArray()
        empty_str.InsertNextValue('')

        # show lines
        if show_xaxis:
            cube_axes_actor.SetXTitle(xlabel)
        else:
            cube_axes_actor.SetXTitle('')
            cube_axes_actor.SetAxisLabels(0, empty_str)

        if show_yaxis:
            cube_axes_actor.SetYTitle(ylabel)
        else:
            cube_axes_actor.SetYTitle('')
            cube_axes_actor.SetAxisLabels(1, empty_str)

        if show_zaxis:
            cube_axes_actor.SetZTitle(zlabel)
        else:
            cube_axes_actor.SetZTitle('')
            cube_axes_actor.SetAxisLabels(2, empty_str)

        # show labels
        if not show_xlabels:
            cube_axes_actor.SetAxisLabels(0, empty_str)

        if not show_ylabels:
            cube_axes_actor.SetAxisLabels(1, empty_str)

        if not show_zlabels:
            cube_axes_actor.SetAxisLabels(2, empty_str)

        # set font
        font_family = parse_font_family(font_family)
        for i in range(3):
            cube_axes_actor.GetTitleTextProperty(i).SetFontSize(font_size)
            cube_axes_actor.GetTitleTextProperty(i).SetColor(color)
            cube_axes_actor.GetTitleTextProperty(i).SetFontFamily(font_family)
            cube_axes_actor.GetTitleTextProperty(i).SetBold(bold)

            cube_axes_actor.GetLabelTextProperty(i).SetFontSize(font_size)
            cube_axes_actor.GetLabelTextProperty(i).SetColor(color)
            cube_axes_actor.GetLabelTextProperty(i).SetFontFamily(font_family)
            cube_axes_actor.GetLabelTextProperty(i).SetBold(bold)

        self.add_actor(cube_axes_actor, reset_camera=False, pickable=False)
        self.cube_axes_actor = cube_axes_actor

        if all_edges:
            self.add_bounding_box(color=color, corner_factor=corner_factor)

        if fmt is not None:
            cube_axes_actor.SetXLabelFormat(fmt)
            cube_axes_actor.SetYLabelFormat(fmt)
            cube_axes_actor.SetZLabelFormat(fmt)

        self.Modified()
        return cube_axes_actor

    def show_grid(self, **kwargs):
        """Show gridlines and axes labels.

        A wrapped implementation of ``show_bounds`` to change default
        behaviour to use gridlines and showing the axes labels on the outer
        edges. This is intended to be silimar to ``matplotlib``'s ``grid``
        function.

        """
        kwargs.setdefault('grid', 'back')
        kwargs.setdefault('location', 'outer')
        kwargs.setdefault('ticks', 'both')
        return self.show_bounds(**kwargs)

    def remove_bounding_box(self, render=True):
        """Remove bounding box."""
        if hasattr(self, '_box_object'):
            actor = self.bounding_box_actor
            self.bounding_box_actor = None
            del self._box_object
            self.remove_actor(actor, reset_camera=False, render=render)
            self.Modified()

    def add_bounding_box(self, color="grey", corner_factor=0.5, line_width=None,
                         opacity=1.0, render_lines_as_tubes=False,
                         lighting=None, reset_camera=None, outline=True,
                         culling='front'):
        """Add an unlabeled and unticked box at the boundaries of plot.

        Useful for when wanting to plot outer grids while still retaining all
        edges of the boundary.

        Parameters
        ----------
        corner_factor : float, optional
            If ``all_edges``, this is the factor along each axis to
            draw the default box. Dafuault is 0.5 to show the full
            box.

        corner_factor : float, optional
            This is the factor along each axis to draw the default
            box. Dafuault is 0.5 to show the full box.

        line_width : float, optional
            Thickness of lines.

        opacity : float, optional
            Opacity of mesh.  Should be between 0 and 1.  Default 1.0

        outline : bool
            Default is ``True``. when False, a box with faces is shown with
            the specified culling

        culling : str, optional
            Does not render faces that are culled. Options are ``'front'`` or
            ``'back'``. Default is ``'front'`` for bounding box.

        """
        if lighting is None:
            lighting = rcParams['lighting']

        self.remove_bounding_box()
        if color is None:
            color = rcParams['outline_color']
        rgb_color = parse_color(color)
        if outline:
            self._bounding_box = vtk.vtkOutlineCornerSource()
            self._bounding_box.SetCornerFactor(corner_factor)
        else:
            self._bounding_box = vtk.vtkCubeSource()
        self._bounding_box.SetBounds(self.bounds)
        self._bounding_box.Update()
        self._box_object = wrap(self._bounding_box.GetOutput())
        name = f'BoundingBox({hex(id(self._box_object))})'

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self._box_object)
        self.bounding_box_actor, prop = self.add_actor(mapper,
                                                       reset_camera=reset_camera,
                                                       name=name, culling=culling,
                                                       pickable=False)

        prop.SetColor(rgb_color)
        prop.SetOpacity(opacity)
        if render_lines_as_tubes:
            prop.SetRenderLinesAsTubes(render_lines_as_tubes)

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        prop.SetRepresentationToSurface()
        self.Modified()
        return self.bounding_box_actor

    def add_floor(self, face='-z', i_resolution=10, j_resolution=10,
                  color=None, line_width=None, opacity=1.0, show_edges=False,
                  lighting=False, edge_color=None, reset_camera=None, pad=0.0,
                  offset=0.0, pickable=False, store_floor_kwargs=True):
        """Show a floor mesh.

        This generates planes at the boundaries of the scene to behave like
        floors or walls.

        Parameters
        ----------
        face : str
            The face at which to place the plane. Options are (-z, -y,
            -x, +z, +y, and +z). Where the -/+ sign indicates on which
            side of the axis the plane will lie.  For example,
            ``'-z'`` would generate a floor on the XY-plane and the
            bottom of the scene (minimum z).

        i_resolution : int
            Number of points on the plane in the i direction.

        j_resolution : int
            Number of points on the plane in the j direction.

        color : string or 3 item list, optional
            Color of all labels and axis titles.  Default gray.
            Either a string, rgb list, or hex color string.

        line_width : int
            Thickness of the edges. Only if ``show_edges`` is ``True``

        opacity : float
            The opacity of the generated surface

        show_edges : bool
            Flag on whether to show the mesh edges for tiling.

        ine_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default None.

        lighting : bool, optional
            Enable or disable view direction lighting.  Default False.

        edge_color : string or 3 item list, optional
            Color of of the edges of the mesh.

        pad : float
            Percentage padding between 0 and 1

        offset : float
            Percentage offset along plane normal
        """
        if store_floor_kwargs:
            kwargs = locals()
            kwargs.pop('self')
            self._floor_kwargs.append(kwargs)
        ranges = np.array(self.bounds).reshape(-1, 2).ptp(axis=1)
        ranges += (ranges * pad)
        center = np.array(self.center)
        if face.lower() in '-z':
            center[2] = self.bounds[4] - (ranges[2] * offset)
            normal = (0,0,1)
            i_size = ranges[0]
            j_size = ranges[1]
        elif face.lower() in '-y':
            center[1] = self.bounds[2] - (ranges[1] * offset)
            normal = (0,1,0)
            i_size = ranges[0]
            j_size = ranges[2]
        elif face.lower() in '-x':
            center[0] = self.bounds[0] - (ranges[0] * offset)
            normal = (1,0,0)
            i_size = ranges[2]
            j_size = ranges[1]
        elif face.lower() in '+z':
            center[2] = self.bounds[5] + (ranges[2] * offset)
            normal = (0,0,-1)
            i_size = ranges[0]
            j_size = ranges[1]
        elif face.lower() in '+y':
            center[1] = self.bounds[3] + (ranges[1] * offset)
            normal = (0,-1,0)
            i_size = ranges[0]
            j_size = ranges[2]
        elif face.lower() in '+x':
            center[0] = self.bounds[1] + (ranges[0] * offset)
            normal = (-1,0,0)
            i_size = ranges[2]
            j_size = ranges[1]
        else:
            raise NotImplementedError(f'Face ({face}) not implementd')
        self._floor = pyvista.Plane(center=center, direction=normal,
                                    i_size=i_size, j_size=j_size,
                                    i_resolution=i_resolution,
                                    j_resolution=j_resolution)
        name = f'Floor({face})'
        # use floor
        if lighting is None:
            lighting = rcParams['lighting']

        if edge_color is None:
            edge_color = rcParams['edge_color']

        self.remove_bounding_box()
        if color is None:
            color = rcParams['floor_color']
        rgb_color = parse_color(color)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self._floor)
        actor, prop = self.add_actor(mapper,
                                     reset_camera=reset_camera,
                                     name=name, pickable=pickable)

        prop.SetColor(rgb_color)
        prop.SetOpacity(opacity)

        # edge display style
        if show_edges:
            prop.EdgeVisibilityOn()
        prop.SetEdgeColor(parse_color(edge_color))

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        prop.SetRepresentationToSurface()
        self._floors.append(actor)
        return actor

    def remove_floors(self, clear_kwargs=True, render=True):
        """Remove all floor actors."""
        if getattr(self, '_floor', None) is not None:
            self._floor.ReleaseData()
            self._floor = None
        for actor in self._floors:
            self.remove_actor(actor, reset_camera=False, render=render)
        self._floors.clear()
        if clear_kwargs:
            self._floor_kwargs.clear()

    def remove_bounds_axes(self):
        """Remove bounds axes."""
        if hasattr(self, 'cube_axes_actor'):
            self.remove_actor(self.cube_axes_actor)
            self.Modified()

    def add_light(self, light):
        """Add a Light to the renderer."""
        if not isinstance(light, pyvista.Light):
            raise TypeError('Expected Light instance, got {type(light).__name__} instead.')
        self._lights.append(light)
        self.AddLight(light)
        self.AddActor(light._actor)
        self.Modified()

    @property
    def lights(self):
        """Return a list of all lights in the renderer."""
        return list(self.GetLights())

    def remove_all_lights(self):
        """Remove all lights from the renderer."""
        self.RemoveAllLights()
        self._lights.clear()

    def clear(self):
        """Remove all actors and properties."""
        if self._actors:
            for actor in list(self._actors):
                try:
                    self.remove_actor(actor, reset_camera=False, render=False)
                except KeyError:
                    pass

        self.remove_all_lights()
        self.RemoveAllViewProps()
        self.Modified()

    def set_focus(self, point):
        """Set focus to a point."""
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.focus = scale_point(self.camera, point, invert=False)
        self.Modified()

    def set_position(self, point, reset=False):
        """Set camera position to a point."""
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.position = scale_point(self.camera, point, invert=False)
        if reset:
            self.reset_camera()
        self.camera_set = True
        self.Modified()

    def set_viewup(self, vector):
        """Set camera viewup vector."""
        if isinstance(vector, np.ndarray):
            if vector.ndim != 1:
                vector = vector.ravel()
        self.camera.up = vector
        self.Modified()

    def enable_parallel_projection(self):
        """Enable parallel projection.

        The camera will have a parallel projection. Parallel projection is
        often useful when viewing images or 2D datasets.

        """
        self.camera.enable_parallel_projection()
        self.Modified()

    def disable_parallel_projection(self):
        """Reset the camera to use perspective projection."""
        self.camera.disable_parallel_projection()
        self.Modified()

    @property
    def parallel_projection(self):
        """Return parallel projection state of active render window."""
        return self.camera.is_parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state):
        """Set parallel projection state of all active render windows."""
        self.camera.enable_parallel_projection(state)
        self.Modified()

    @property
    def parallel_scale(self):
        """Return parallel scale of active render window."""
        return self.camera.parallel_scale

    @parallel_scale.setter
    def parallel_scale(self, value):
        """Set parallel scale of all active render windows."""
        self.camera.parallel_scale = value
        self.Modified()

    def remove_actor(self, actor, reset_camera=False, render=True):
        """Remove an actor from the Renderer.

        Parameters
        ----------
        actor : str, vtk.vtkActor, list or tuple
            If the type is ``str``, removes the previously added actor with
            the given name. If the type is ``vtk.vtkActor``, removes the actor
            if it's previously added to the Renderer. If ``list`` or ``tuple``,
            removes iteratively each actor.

        reset_camera : bool, optional
            Resets camera so all actors can be seen.

        render : bool, optional
            Render upon actor removal.  Set this to ``False`` to stop
            the render window from rendering when an actor is removed.

        Returns
        -------
        success : bool
            True when actor removed.  False when actor has not been
            removed.

        """
        name = None
        if isinstance(actor, str):
            name = actor
            keys = list(self._actors.keys())
            names = []
            for k in keys:
                if k.startswith(f'{name}-'):
                    names.append(k)
            if len(names) > 0:
                self.remove_actor(names, reset_camera=reset_camera)
            try:
                actor = self._actors[name]
            except KeyError:
                # If actor of that name is not present then return success
                return False
        if isinstance(actor, collections.abc.Iterable):
            success = False
            for a in actor:
                rv = self.remove_actor(a, reset_camera=reset_camera)
                if rv or success:
                    success = True
            return success
        if actor is None:
            return False

        # First remove this actor's mapper from _scalar_bar_mappers
        _remove_mapper_from_plotter(self.parent, actor, False, render=render)
        self.RemoveActor(actor)

        if name is None:
            for k, v in self._actors.items():
                if v == actor:
                    name = k
        self._actors.pop(name, None)
        self.update_bounds_axes()
        if reset_camera:
            self.reset_camera()
        elif not self.camera_set and reset_camera is None:
            self.reset_camera()
        elif render:
            self.parent.render()

        self.Modified()
        return True

    def set_scale(self, xscale=None, yscale=None, zscale=None, reset_camera=True):
        """Scale all the datasets in the scene.

        Scaling in performed independently on the X, Y and Z axis.
        A scale of zero is illegal and will be replaced with one.

        """
        if xscale is None:
            xscale = self.scale[0]
        if yscale is None:
            yscale = self.scale[1]
        if zscale is None:
            zscale = self.scale[2]
        self.scale = [xscale, yscale, zscale]

        # Update the camera's coordinate system
        transform = vtk.vtkTransform()
        transform.Scale(xscale, yscale, zscale)
        self.camera.SetModelTransformMatrix(transform.GetMatrix())
        self.parent.render()
        if reset_camera:
            self.update_bounds_axes()
            self.reset_camera()
        self.Modified()

    def get_default_cam_pos(self, negative=False):
        """Return the default focal points and viewup.

        Uses ResetCamera to make a useful view.

        """
        focal_pt = self.center
        if any(np.isnan(focal_pt)):
            focal_pt = (0.0, 0.0, 0.0)
        position = np.array(rcParams['camera']['position']).astype(float)
        if negative:
            position *= -1
        position = position / np.array(self.scale).astype(float)
        cpos = [position + np.array(focal_pt),
                focal_pt, rcParams['camera']['viewup']]
        return cpos

    def update_bounds_axes(self):
        """Update the bounds axes of the render window."""
        if (hasattr(self, '_box_object') and self._box_object is not None
                and self.bounding_box_actor is not None):
            if not np.allclose(self._box_object.bounds, self.bounds):
                color = self.bounding_box_actor.GetProperty().GetColor()
                self.remove_bounding_box()
                self.add_bounding_box(color=color)
                self.remove_floors(clear_kwargs=False)
                for floor_kwargs in self._floor_kwargs:
                    floor_kwargs['store_floor_kwargs'] = False
                    self.add_floor(**floor_kwargs)
        if hasattr(self, 'cube_axes_actor'):
            self.cube_axes_actor.SetBounds(self.bounds)
            if not np.allclose(self.scale, [1.0, 1.0, 1.0]):
                self.cube_axes_actor.SetUse2DMode(True)
            else:
                self.cube_axes_actor.SetUse2DMode(False)
            self.Modified()

    def reset_camera(self, render=True):
        """Reset the camera of the active render window.

        The camera slides along the vector defined from camera
        position to focal point until all of the actors can be seen.

        Parameters
        ----------
        render : bool
            Trigger a render after resetting the camera.

        """
        self.ResetCamera()
        if render:
            self.parent.render()
        self.Modified()

    def isometric_view(self):
        """Reset the camera to a default isometric view.

        DEPRECATED: Please use ``view_isometric``.

        """
        return self.view_isometric()

    def view_isometric(self, negative=False):
        """Reset the camera to a default isometric view.

        The view will show all the actors in the scene.

        """
        self.camera_position = CameraPosition(*self.get_default_cam_pos(negative=negative))
        self.camera_set = False
        return self.reset_camera()

    def view_vector(self, vector, viewup=None):
        """Point the camera in the direction of the given vector."""
        focal_pt = self.center
        if viewup is None:
            viewup = rcParams['camera']['viewup']
        cpos = CameraPosition(vector + np.array(focal_pt),
                focal_pt, viewup)
        self.camera_position = cpos
        return self.reset_camera()

    def view_xy(self, negative=False):
        """View the XY plane."""
        vec = np.array([0,0,1])
        viewup = np.array([0,1,0])
        if negative:
            vec *= -1
        return self.view_vector(vec, viewup)

    def view_yx(self, negative=False):
        """View the YX plane."""
        vec = np.array([0,0,-1])
        viewup = np.array([1,0,0])
        if negative:
            vec *= -1
        return self.view_vector(vec, viewup)

    def view_xz(self, negative=False):
        """View the XZ plane."""
        vec = np.array([0,-1,0])
        viewup = np.array([0,0,1])
        if negative:
            vec *= -1
        return self.view_vector(vec, viewup)

    def view_zx(self, negative=False):
        """View the ZX plane."""
        vec = np.array([0,1,0])
        viewup = np.array([1,0,0])
        if negative:
            vec *= -1
        return self.view_vector(vec, viewup)

    def view_yz(self, negative=False):
        """View the YZ plane."""
        vec = np.array([1,0,0])
        viewup = np.array([0,0,1])
        if negative:
            vec *= -1
        return self.view_vector(vec, viewup)

    def view_zy(self, negative=False):
        """View the ZY plane."""
        vec = np.array([-1,0,0])
        viewup = np.array([0,1,0])
        if negative:
            vec *= -1
        return self.view_vector(vec, viewup)

    def disable(self):
        """Disable this renderer's camera from being interactive."""
        return self.SetInteractive(0)

    def enable(self):
        """Enable this renderer's camera to be interactive."""
        return self.SetInteractive(1)

    def enable_eye_dome_lighting(self):
        """Enable eye dome lighting (EDL)."""
        if hasattr(self, 'edl_pass'):
            return self
        # create the basic VTK render steps
        basic_passes = vtk.vtkRenderStepsPass()
        # blur the resulting image
        # The blur delegates rendering the unblured image to the basic_passes
        self.edl_pass = vtk.vtkEDLShading()
        self.edl_pass.SetDelegatePass(basic_passes)

        # tell the renderer to use our render pass pipeline
        self.glrenderer = vtk.vtkOpenGLRenderer.SafeDownCast(self)
        self.glrenderer.SetPass(self.edl_pass)
        self.Modified()
        return self.glrenderer

    def disable_eye_dome_lighting(self):
        """Disable eye dome lighting (EDL)."""
        if not hasattr(self, 'edl_pass'):
            return
        self.SetPass(None)
        self.edl_pass.ReleaseGraphicsResources(self.parent.ren_win)
        del self.edl_pass
        self.Modified()
        return

    def get_pick_position(self):
        """Get the pick position/area as x0, y0, x1, y1."""
        x0 = int(self.GetPickX1())
        x1 = int(self.GetPickX2())
        y0 = int(self.GetPickY1())
        y1 = int(self.GetPickY2())
        return x0, y0, x1, y1

    def set_background(self, color, top=None):
        """Set the background color.

        Parameters
        ----------
        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        top : string or 3 item list, optional, defaults to None
            If given, this will enable a gradient background where the
            ``color`` argument is at the bottom and the color given in ``top``
            will be the color at the top of the renderer.

        """
        if color is None:
            color = rcParams['background']

        use_gradient = False
        if top is not None:
            use_gradient = True

        self.SetBackground(parse_color(color))
        if use_gradient:
            self.GradientBackgroundOn()
            self.SetBackground2(parse_color(top))
        else:
            self.GradientBackgroundOff()
        self.Modified()
        return

    def close(self):
        """Close out widgets and sensitive elements."""
        self.RemoveAllObservers()
        if hasattr(self, 'axes_widget'):
            self.hide_axes()  # Necessary to avoid segfault
            self.axes_actor = None
            del self.axes_widget

    def deep_clean(self, render=False):
        """Clean the renderer of the memory."""
        if hasattr(self, 'cube_axes_actor'):
            del self.cube_axes_actor
        if hasattr(self, 'edl_pass'):
            del self.edl_pass
        if hasattr(self, '_box_object'):
            self.remove_bounding_box(render=render)

        self.remove_floors(render=render)
        self.RemoveAllViewProps()
        self._actors = {}
        self._camera = None
        # remove reference to parent last
        self.parent = None
        return

    def __del__(self):
        """Delete the renderer."""
        self.deep_clean()


def _remove_mapper_from_plotter(plotter, actor, reset_camera, render=False):
    """Remove this actor's mapper from the given plotter's _scalar_bar_mappers."""
    try:
        mapper = actor.GetMapper()
    except AttributeError:
        return
    for name in list(plotter._scalar_bar_mappers.keys()):
        try:
            plotter._scalar_bar_mappers[name].remove(mapper)
        except ValueError:
            pass
        if len(plotter._scalar_bar_mappers[name]) < 1:
            slot = plotter._scalar_bar_slot_lookup.pop(name, None)
            if slot is not None:
                plotter._scalar_bar_mappers.pop(name)
                plotter._scalar_bar_ranges.pop(name)
                plotter.remove_actor(plotter._scalar_bar_actors.pop(name),
                                     reset_camera=reset_camera,
                                     render=render)
                plotter._scalar_bar_slots.add(slot)
    return
