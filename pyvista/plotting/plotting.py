"""
pyvista plotting module
"""
import collections
import logging
import os
import time
from threading import Thread

import imageio
import numpy as np
import scooby
import vtk
from vtk.util import numpy_support as VN
import warnings

import pyvista
from pyvista.utilities import (convert_array, convert_string_array,
                               get_array, is_pyvista_dataset, numpy_to_texture,
                               raise_not_matching, wrap)

from .colors import get_cmap_safe, PARAVIEW_BACKGROUND
from .export_vtkjs import export_plotter_vtkjs
from .mapper import make_mapper
from .picking import PickingHelper
from .tools import update_axes_label_color, create_axes_orientation_box, create_axes_marker
from .tools import normalize, opacity_transfer_function
from .theme import rcParams, parse_color, parse_font_family
from .theme import FONT_KEYS, MAX_N_COLOR_BARS
from .widgets import WidgetHelper

try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

_ALL_PLOTTERS = {}

def close_all():
    """Close all open/active plotters and clean up memory"""
    for key, p in _ALL_PLOTTERS.items():
        p.close()
        p.deep_clean()
    _ALL_PLOTTERS.clear()
    return True


log = logging.getLogger(__name__)
log.setLevel('CRITICAL')



class BasePlotter(PickingHelper, WidgetHelper):
    """
    To be used by the Plotter and QtInteractor classes.

    Parameters
    ----------
    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one renderer.
        Can also accept a shape as string descriptor. E.g.:
            shape="3|1" means 3 plots on the left and 1 on the right,
            shape="4/2" means 4 plots on top of 2 at bottom.

    border : bool, optional
        Draw a border around each render window.  Default False.

    border_color : string or 3 item list, optional, defaults to white
        Either a string, rgb list, or hex color string.  For example:
            color='white'
            color='w'
            color=[1, 1, 1]
            color='#FFFFFF'

    border_width : float, optional
        Width of the border in pixels when enabled.

    """

    def __new__(cls, *args, **kwargs):
        if cls is BasePlotter:
            raise TypeError("pyvista.BasePlotter is an abstract class and may not be instantiated.")
        return object.__new__(cls)

    def __init__(self, shape=(1, 1), border=None, border_color='k',
                 border_width=2.0, title=None, splitting_position=None):
        """ Initialize base plotter """
        self.image_transparent_background = rcParams['transparent_background']

        if title is None:
            title = rcParams['title']
        self.title = str(title)

        # by default add border for multiple plots
        if border is None:
            if shape != (1, 1):
                border = True
            else:
                border = False

        # add render windows
        self._active_renderer_index = 0
        self.renderers = []

        if isinstance(shape, str):

            if '|' in shape:
                n = int(shape.split('|')[0])
                m = int(shape.split('|')[1])
                rangen = reversed(range(n))
                rangem = reversed(range(m))
            else:
                m = int(shape.split('/')[0])
                n = int(shape.split('/')[1])
                rangen = range(n)
                rangem = range(m)

            if splitting_position is None:
                splitting_position = rcParams['multi_rendering_splitting_position']

            if splitting_position is None:
                if n >= m:
                    xsplit = m/(n+m)
                else:
                    xsplit = 1-n/(n+m)
            else:
                xsplit = splitting_position

            for i in rangen:
                arenderer = pyvista.Renderer(self, border, border_color, border_width)
                if '|' in shape:
                    arenderer.SetViewport(0, i/n, xsplit, (i+1)/n)
                else:
                    arenderer.SetViewport(i/n, 0, (i+1)/n, xsplit)
                self.renderers.append(arenderer)
            for i in rangem:
                arenderer = pyvista.Renderer(self, border, border_color, border_width)
                if '|' in shape:
                    arenderer.SetViewport(xsplit, i/m, 1, (i+1)/m)
                else:
                    arenderer.SetViewport(i/m, xsplit, (i+1)/m, 1)
                self.renderers.append(arenderer)

                self.shape = (n+m,)

        else:

            assert_str = '"shape" should be a list, tuple or string descriptor'
            assert isinstance(shape, collections.Iterable), assert_str
            assert shape[0] > 0, '"shape" must be positive'
            assert shape[1] > 0, '"shape" must be positive'
            self.shape = shape
            for i in reversed(range(shape[0])):
                for j in range(shape[1]):
                    renderer = pyvista.Renderer(self, border, border_color, border_width)
                    x0 = i/shape[0]
                    y0 = j/shape[1]
                    x1 = (i+1)/shape[0]
                    y1 = (j+1)/shape[1]
                    renderer.SetViewport(y0, x0, y1, x1)
                    self.renderers.append(renderer)


        # This keeps track of scalar names already plotted and their ranges
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._scalar_bar_actors = {}
        self._scalar_bar_widgets = {}
        # track if the camera has been setup
        # self.camera_set = False
        self._first_time = True
        # Keep track of the scale
        self._labels = []
        # Set default style
        self._style = vtk.vtkInteractorStyleRubberBandPick()

        # Add self to open plotters
        _ALL_PLOTTERS[str(hex(id(self)))] = self

        # lighting style
        self.lighting = vtk.vtkLightKit()
        # self.lighting.SetHeadLightWarmth(1.0)
        # self.lighting.SetHeadLightWarmth(1.0)
        for renderer in self.renderers:
            self.lighting.AddLightsToRenderer(renderer)
            renderer.LightFollowCameraOn()

        # Key bindings
        self.reset_key_events()


    def add_key_event(self, key, callback):
        """Add a function to callback when the given key is pressed. These are
        non-unique - thus a key could map to many callback functions.

        The callback function must not have any arguments.

        Parameters
        ----------
        key : str
            The key to trigger the event

        callback : callable
            A callable that takes no arguments
        """
        if not hasattr(callback, '__call__'):
            raise TypeError('callback must be callable.')
        self._key_press_event_callbacks[key].append(callback)


    def clear_events_for_key(self, key):
        self._key_press_event_callbacks.pop(key)


    def reset_key_events(self):
        """Reset all of the key press events to their defaults."""
        self._key_press_event_callbacks = collections.defaultdict(list)

        def _close_callback():
            """ Make sure a screenhsot is acquired before closing"""
            self.q_pressed = True
            # Grab screenshot right before renderer closes
            self.last_image = self.screenshot(True, return_img=True)
            self.last_image_depth = self.get_image_depth()

        self.add_key_event('q', _close_callback)
        b_left_down_callback = lambda: self.iren.AddObserver('LeftButtonPressEvent', self.left_button_down)
        self.add_key_event('b', b_left_down_callback)
        self.add_key_event('v', lambda: self.isometric_view_interactive())


    def key_press_event(self, obj, event):
        """ Listens for key press event """
        key = self.iren.GetKeySym()
        log.debug('Key %s pressed' % key)
        if key in self._key_press_event_callbacks.keys():
            # Note that defaultdict's will never throw a key error
            callbacks = self._key_press_event_callbacks[key]
            for func in callbacks:
                func()


    def left_button_down(self, obj, event_type):
        """Register the event for a left button down click"""
        # Get 2D click location on window
        click_pos = self.iren.GetEventPosition()

        # Get corresponding click location in the 3D plot
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        self.pickpoint = np.asarray(picker.GetPickPosition()).reshape((-1, 3))
        if np.any(np.isnan(self.pickpoint)):
            self.pickpoint[:] = 0


    def update_style(self):
        if not hasattr(self, '_style'):
            self._style = vtk.vtkInteractorStyleTrackballCamera()
        if hasattr(self, 'iren'):
            return self.iren.SetInteractorStyle(self._style)

    def enable_trackball_style(self):
        """ sets the interactive style to trackball - the default syle """
        self._style = vtk.vtkInteractorStyleTrackballCamera()
        return self.update_style()

    def enable_image_style(self):
        """ sets the interactive style to image

        Controls:
         - Left Mouse button triggers window level events
         - CTRL Left Mouse spins the camera around its view plane normal
         - SHIFT Left Mouse pans the camera
         - CTRL SHIFT Left Mouse dollys (a positional zoom) the camera
         - Middle mouse button pans the camera
         - Right mouse button dollys the camera.
         - SHIFT Right Mouse triggers pick events
        """
        self._style = vtk.vtkInteractorStyleImage()
        return self.update_style()

    def enable_joystick_style(self):
        """ sets the interactive style to joystick

        allows the user to move (rotate, pan, etc.) the camera, the point of
        view for the scene.  The position of the mouse relative to the center of
        the scene determines the speed at which the camera moves, and the speed
        of the mouse movement determines the acceleration of the camera, so the
        camera continues to move even if the mouse if not moving.

        For a 3-button mouse, the left button is for rotation, the right button
        for zooming, the middle button for panning, and ctrl + left button for
        spinning.  (With fewer mouse buttons, ctrl + shift + left button is
        for zooming, and shift + left button is for panning.)
        """
        self._style = vtk.vtkInteractorStyleJoystickCamera()
        return self.update_style()

    def enable_zoom_style(self):
        """ sets the interactive style to rubber band zoom

        This interactor style allows the user to draw a rectangle in the render
        window using the left mouse button.  When the mouse button is released,
        the current camera zooms by an amount determined from the shorter side
        of the drawn rectangle.
        """
        self._style = vtk.vtkInteractorStyleRubberBandZoom()
        return self.update_style()

    def enable_terrain_style(self):
        """ sets the interactive style to terrain

        Used to manipulate a camera which is viewing a scene with a natural
        view up, e.g., terrain. The camera in such a scene is manipulated by
        specifying azimuth (angle around the view up vector) and elevation
        (the angle from the horizon).
        """
        self._style = vtk.vtkInteractorStyleTerrain()
        return self.update_style()

    def enable_rubber_band_style(self):
        """ sets the interactive style to rubber band picking

        This interactor style allows the user to draw a rectangle in the render
        window by hitting 'r' and then using the left mouse button.
        When the mouse button is released, the attached picker operates on the
        pixel in the center of the selection rectangle. If the picker happens to
        be a vtkAreaPicker it will operate on the entire selection rectangle.
        When the 'p' key is hit the above pick operation occurs on a 1x1
        rectangle. In other respects it behaves the same as its parent class.
        """
        self._style = vtk.vtkInteractorStyleRubberBandPick()
        return self.update_style()

    def set_focus(self, point):
        """ sets focus to a point """
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.SetFocalPoint(point)
        self._render()

    def set_position(self, point, reset=False):
        """ sets camera position to a point """
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.SetPosition(point)
        if reset:
            self.reset_camera()
        self.camera_set = True
        self._render()

    def set_viewup(self, vector):
        """ sets camera viewup vector """
        if isinstance(vector, np.ndarray):
            if vector.ndim != 1:
                vector = vector.ravel()
        self.camera.SetViewUp(vector)
        self._render()

    def _render(self):
        """ redraws render window if the render window exists """
        if hasattr(self, 'ren_win'):
            if hasattr(self, 'render_trigger'):
                self.render_trigger.emit()
            elif not self._first_time:
                self.render()

    def add_axes(self, interactive=None, line_width=2,
                 color=None, x_color=None, y_color=None, z_color=None,
                 xlabel='X', ylabel='Y', zlabel='Z', labels_off=False,
                 box=None, box_args=None):
        """ Add an interactive axes widget """
        if interactive is None:
            interactive = rcParams['interactive']
        if hasattr(self, 'axes_widget'):
            self.axes_widget.SetInteractive(interactive)
            update_axes_label_color(color)
            return
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
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        if hasattr(self, 'iren'):
            self.axes_widget.SetInteractor(self.iren)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.SetInteractive(interactive)
        return

    def hide_axes(self):
        """Hide the axes orientation widget"""
        if hasattr(self, 'axes_widget'):
            self.axes_widget.EnabledOff()

    def show_axes(self):
        """Show the axes orientation widget"""
        if hasattr(self, 'axes_widget'):
            self.axes_widget.EnabledOn()
        else:
            self.add_axes()

    def isometric_view_interactive(self):
        """ sets the current interactive render window to isometric view """
        interactor = self.iren.GetInteractorStyle()
        renderer = interactor.GetCurrentRenderer()
        renderer.view_isometric()

    def update(self, stime=1, force_redraw=True):
        """
        Update window, redraw, process messages query

        Parameters
        ----------
        stime : int, optional
            Duration of timer that interrupt vtkRenderWindowInteractor in
            milliseconds.

        force_redraw : bool, optional
            Call vtkRenderWindowInteractor.Render() immediately.
        """

        if stime <= 0:
            stime = 1

        curr_time = time.time()
        if Plotter.last_update_time > curr_time:
            Plotter.last_update_time = curr_time

        if not hasattr(self, 'iren'):
            return

        update_rate = self.iren.GetDesiredUpdateRate()
        if (curr_time - Plotter.last_update_time) > (1.0/update_rate):
            self.right_timer_id = self.iren.CreateRepeatingTimer(stime)

            self.iren.Start()
            self.iren.DestroyTimer(self.right_timer_id)

            self._render()
            Plotter.last_update_time = curr_time
        else:
            if force_redraw:
                self.iren.Render()

    def add_mesh(self, mesh, color=None, style=None, scalars=None,
                 clim=None, show_edges=None, edge_color=None,
                 point_size=5.0, line_width=None, opacity=1.0,
                 flip_scalars=False, lighting=None, n_colors=256,
                 interpolate_before_map=True, cmap=None, label=None,
                 reset_camera=None, scalar_bar_args=None, show_scalar_bar=None,
                 stitle=None, multi_colors=False, name=None, texture=None,
                 render_points_as_spheres=None, render_lines_as_tubes=False,
                 smooth_shading=False, ambient=0.0, diffuse=1.0, specular=0.0,
                 specular_power=100.0, nan_color=None, nan_opacity=1.0,
                 loc=None, culling=None, rgb=False, categories=False,
                 use_transparency=False, below_color=None, above_color=None,
                 annotations=None, pickable=True, **kwargs):
        """
        Adds any PyVista/VTK mesh or dataset that PyVista can wrap to the
        scene. This method using a mesh representation to view the surfaces
        and/or geometry of datasets. For volume rendering, see
        :func:`pyvista.BasePlotter.add_volume`.

        Parameters
        ----------
        mesh : pyvista.Common or pyvista.MultiBlock
            Any PyVista or VTK mesh is supported. Also, any dataset
            that :func:`pyvista.wrap` can handle including NumPy arrays of XYZ
            points.

        color : string or 3 item list, optional, defaults to white
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1, 1, 1]``, or
            ``color='#FFFFFF'``. Color will be overridden if scalars are
            specified.

        style : string, optional
            Visualization style of the mesh.  One of the following:
            ``style='surface'``, ``style='wireframe'``, ``style='points'``.
            Defaults to ``'surface'``. Note that ``'wireframe'`` only shows a
            wireframe of the outer geometry.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name of an
            array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            ``color`` and ``scalars`` are ``None``, then the active scalars are
            used.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        show_edges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        edge_color : string or 3 item list, optional, defaults to black
            The solid color to give the edges when ``show_edges=True``.
            Either a string, RGB list, or hex color string.

        point_size : float, optional
            Point size of any nodes in the dataset plotted. Also applicable
            when style='points'. Default ``5.0``

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default None.

        opacity : float, str, array-like
            Opacity of the mesh. If a siblge float value is given, it will be
            the global opacity of the mesh and uniformly applied everywhere -
            should be between 0 and 1. A string can also be specified to map
            the scalar range to a predefined opacity transfer function
            (options include: 'linear', 'linear_r', 'geom', 'geom_r').
            A string could also be used to map a scalar array from the mesh to
            the the opacity (must have same number of elements as the
            ``scalars`` argument). Or you can pass a custum made trasfer
            function that is an aray either ``n_colors`` in length or shorter.

        flip_scalars : bool, optional
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        lighting : bool, optional
            Enable or disable view direction lighting. Default False.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        interpolate_before_map : bool, optional
            Enabling makes for a smoother scalar display.  Default is True.
            When False, OpenGL will interpolate the mapped colors which can
            result is showing colors that are not present in the color map.

        cmap : str, optional
           Name of the Matplotlib colormap to us when mapping the ``scalars``.
           See available Matplotlib colormaps.  Only applicable for when
           displaying ``scalars``. Requires Matplotlib to be installed.
           ``colormap`` is also an accepted alias for this. If ``colorcet`` or
           ``cmocean`` are installed, their colormaps can be specified by name.

        label : str, optional
            String label to use when adding a legend to the scene with
            :func:`pyvista.BasePlotter.add_legend`

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the scalar bar
            to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If False, a scalar bar will not be added to the scene. Defaults
            to ``True``.

        stitle : string, optional
            Scalar bar title. By default the scalar bar is given a title of the
            the scalar array used to color the mesh.
            To create a bar with no title, use an empty string (i.e. '').

        multi_colors : bool, optional
            If a ``MultiBlock`` dataset is given this will color each
            block by a solid color using matplotlib's color cycler.

        name : str, optional
            The name for the added mesh/actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        texture : vtk.vtkTexture or np.ndarray or boolean, optional
            A texture to apply if the input mesh has texture
            coordinates.  This will not work with MultiBlock
            datasets. If set to ``True``, the first avaialble texture
            on the object will be used. If a string name is given, it
            will pull a texture with that name associated to the input
            mesh.

        render_points_as_spheres : bool, optional

        render_lines_as_tubes : bool, optional

        smooth_shading : bool, optional

        ambient : float, optional
            When lighting is enabled, this is the amount of light from
            0 to 1 that reaches the actor when not directed at the
            light source emitted from the viewer.  Default 0.0

        diffuse : float, optional
            The diffuse lighting coefficient. Default 1.0

        specular : float, optional
            The specular lighting coefficient. Default 0.0

        specular_power : float, optional
            The specular power. Bewteen 0.0 and 128.0

        nan_color : string or 3 item list, optional, defaults to gray
            The color to use for all ``NaN`` values in the plotted scalar
            array.

        nan_opacity : float, optional
            Opacity of ``NaN`` values.  Should be between 0 and 1.
            Default 1.0

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.

        culling : str, optional
            Does not render faces that are culled. Options are ``'front'`` or
            ``'back'``. This can be helpful for dense surface meshes,
            especially when edges are visible, but can cause flat
            meshes to be partially displayed.  Defaults ``False``.

        rgb : bool, optional
            If an 2 dimensional array is passed as the scalars, plot those
            values as RGB(A) colors! ``rgba`` is also accepted alias for this.
            Opacity (the A) is optional.

        categories : bool, optional
            If set to ``True``, then the number of unique values in the scalar
            array will be used as the ``n_colors`` argument.

        use_transparency : bool, optional
            Invert the opacity mappings and make the values correspond to
            transperency.

        below_color : string or 3 item list, optional
            Solid color for values below the scalar range (``clim``). This will
            automatically set the scalar bar ``below_label`` to ``'Below'``

        above_color : string or 3 item list, optional
            Solid color for values below the scalar range (``clim``). This will
            automatically set the scalar bar ``above_label`` to ``'Above'``

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float values in the
            scalar range to annotate on the scalar bar and the values are the
            the string annotations.

        pickable : bool
            Set whether this mesh is pickable

        Returns
        -------
        actor: vtk.vtkActor
            VTK actor of the mesh.
        """
        # Convert the VTK data object to a pyvista wrapped object if neccessary
        if not is_pyvista_dataset(mesh):
            mesh = wrap(mesh)
            if not is_pyvista_dataset(mesh):
                raise TypeError('Object type ({}) not supported for plotting in PyVista.'.format(type(mesh)))

        ##### Parse arguments to be used for all meshes #####

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if show_edges is None:
            show_edges = rcParams['show_edges']

        if edge_color is None:
            edge_color = rcParams['edge_color']

        if show_scalar_bar is None:
            show_scalar_bar = rcParams['show_scalar_bar']

        if lighting is None:
            lighting = rcParams['lighting']

        if clim is None:
            clim = kwargs.get('rng', None)

        if render_points_as_spheres is None:
            render_points_as_spheres = rcParams['render_points_as_spheres']

        if name is None:
            name = '{}({})'.format(type(mesh).__name__, str(hex(id(mesh))))

        if nan_color is None:
            nan_color = rcParams['nan_color']
        nanr, nanb, nang = parse_color(nan_color)
        nan_color = nanr, nanb, nang, nan_opacity
        if color is True:
            color = rcParams['color']

        if texture is False:
            texture = None

        if culling is None:
            culling = kwargs.get("backface_culling", False)
            if culling is True:
                culling = 'backface'

        ##### Handle composite datasets #####

        if isinstance(mesh, pyvista.MultiBlock):
            # frist check the scalars
            if clim is None and scalars is not None:
                # Get the data range across the array for all blocks
                # if scalar specified
                if isinstance(scalars, str):
                    clim = mesh.get_data_range(scalars)
                else:
                    # TODO: an array was given... how do we deal with
                    #       that? Possibly a 2D arrays or list of
                    #       arrays where first index corresponds to
                    #       the block? This could get complicated real
                    #       quick.
                    raise RuntimeError('Scalar array must be given as a string name for multiblock datasets.')

            the_arguments = locals()
            the_arguments.update(kwargs)
            the_arguments.pop('self')
            the_arguments.pop('mesh')

            if multi_colors:
                # Compute unique colors for each index of the block
                if has_matplotlib:
                    from itertools import cycle
                    cycler = matplotlib.rcParams['axes.prop_cycle']
                    colors = cycle(cycler)
                else:
                    multi_colors = False
                    logging.warning('Please install matplotlib for color cycles')

            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(mesh.GetNumberOfBlocks()):
                if mesh[idx] is None:
                    continue
                # Get a good name to use
                next_name = '{}-{}'.format(name, idx)
                # Get the data object
                if not is_pyvista_dataset(mesh[idx]):
                    data = wrap(mesh.GetBlock(idx))
                    if not is_pyvista_dataset(mesh[idx]):
                        continue # move on if we can't plot it
                else:
                    data = mesh.GetBlock(idx)
                if data is None or (not isinstance(data, pyvista.MultiBlock) and data.n_points < 1):
                    # Note that a block can exist but be None type
                    # or it could have zeros points (be empty) after filtering
                    continue
                # Now check that scalars is available for this dataset
                if isinstance(data, vtk.vtkMultiBlockDataSet) or get_array(data, scalars) is None:
                    ts = None
                else:
                    ts = scalars
                if multi_colors:
                    color = next(colors)['color']

                ## Add to the scene
                the_arguments['color'] = color
                the_arguments['scalars'] = ts
                the_arguments['name'] = next_name
                the_arguments['texture'] = None
                a = self.add_mesh(data, **the_arguments)
                actors.append(a)

                if (reset_camera is None and not self.camera_set) or reset_camera:
                    cpos = self.get_default_cam_pos()
                    self.camera_position = cpos
                    self.camera_set = False
                    self.reset_camera()
            return actors

        ##### Plot a single PyVista mesh #####

        # Compute surface normals if using smooth shading
        if smooth_shading:
            # extract surface if mesh is exterior
            if not isinstance(mesh, pyvista.PolyData):
                grid = mesh
                mesh = grid.extract_surface()
                ind = mesh.point_arrays['vtkOriginalPointIds']
                # remap scalars
                if isinstance(scalars, np.ndarray):
                    scalars = scalars[ind]

            mesh.compute_normals(cell_normals=False, inplace=True)

        if mesh.n_points < 1:
            raise RuntimeError('Empty meshes cannot be plotted. Input mesh has zero points.')

        # set main values
        self.mesh = mesh
        self.mapper = make_mapper(vtk.vtkDataSetMapper)
        self.mapper.SetInputData(self.mesh)
        if isinstance(scalars, str):
            self.mapper.SetArrayName(scalars)

        actor, prop = self.add_actor(self.mapper,
                                     reset_camera=reset_camera,
                                     name=name, loc=loc, culling=culling)

        # Try to plot something if no preference given
        if scalars is None and color is None and texture is None:
            # Prefer texture first
            if len(list(mesh.textures.keys())) > 0:
                texture = True
            # If no texture, plot any active scalar
            else:
                # Make sure scalar components are not vectors/tuples
                scalars = mesh.active_scalar_name
                # Don't allow plotting of string arrays by default
                if scalars is not None:# and np.issubdtype(mesh.active_scalar.dtype, np.number):
                    if stitle is None:
                        stitle = scalars
                else:
                    scalars = None

        # set main values
        self.mesh = mesh
        self.mapper = make_mapper(vtk.vtkDataSetMapper)
        self.mapper.SetInputData(self.mesh)
        self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
        if interpolate_before_map:
            self.mapper.InterpolateScalarsBeforeMappingOn()

        actor, prop = self.add_actor(self.mapper,
                                     reset_camera=reset_camera,
                                     name=name, loc=loc, culling=culling,
                                     pickable=pickable)

        # Make sure scalars is a numpy array after this point
        original_scalar_name = None
        if isinstance(scalars, str):
            self.mapper.SetArrayName(scalars)
            original_scalar_name = scalars
            scalars = get_array(mesh, scalars,
                                preference=kwargs.get('preference', 'cell'), err=True)
            if stitle is None:
                stitle = original_scalar_name


        if texture is True or isinstance(texture, (str, int)):
            texture = mesh._activate_texture(texture)

        if texture:
            if isinstance(texture, np.ndarray):
                texture = numpy_to_texture(texture)
            if not isinstance(texture, (vtk.vtkTexture, vtk.vtkOpenGLTexture)):
                raise TypeError('Invalid texture type ({})'.format(type(texture)))
            if mesh.GetPointData().GetTCoords() is None:
                raise AssertionError('Input mesh does not have texture coordinates to support the texture.')
            actor.SetTexture(texture)
            # Set color to white by default when using a texture
            if color is None:
                color = 'white'
            if scalars is None:
                show_scalar_bar = False
            self.mapper.SetScalarModeToUsePointFieldData()

        # Handle making opacity array =========================================

        _custom_opac = False
        if isinstance(opacity, str):
            try:
                # Get array from mesh
                opacity = get_array(mesh, opacity,
                                    preference=kwargs.get('preference', 'cell'), err=True)
                opacity = normalize(opacity)
                _custom_opac = True
            except:
                # Or get opacity trasfer function
                opacity = opacity_transfer_function(opacity, n_colors)
            else:
                if scalars.shape[0] != opacity.shape[0]:
                    raise RuntimeError('Opacity array and scalars array must have the same number of elements.')
        elif isinstance(opacity, (np.ndarray, list, tuple)):
            opacity = np.array(opacity)
            if scalars.shape[0] == opacity.shape[0]:
                # User could pass an array of opacities for every point/cell
                pass
            else:
                opacity = opacity_transfer_function(opacity, n_colors)

        if use_transparency and np.max(opacity) <= 1.0:
            opacity = 1 - opacity
        elif use_transparency and isinstance(opacity, np.ndarray):
            opacity = 255 - opacity

        # Scalar formatting ===================================================
        if cmap is None: # grab alias for cmaps: colormap
            cmap = kwargs.get('colormap', None)
        if cmap is None: # Set default map if matplotlib is avaialble
            if has_matplotlib:
                cmap = rcParams['cmap']
        # Set the array title for when it is added back to the mesh
        if _custom_opac:
            title = '__custom_rgba'
        elif stitle is None:
            title = 'Data'
        else:
            title = stitle
        if scalars is not None:
            # if scalars is a string, then get the first array found with that name
            set_active = True

            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            _using_labels = False
            if not np.issubdtype(scalars.dtype, np.number):
                # raise TypeError('Non-numeric scalars are currently not supported for plotting.')
                # TODO: If str array, digitive and annotate
                cats, scalars = np.unique(scalars.astype('|S'), return_inverse=True)
                values = np.unique(scalars)
                clim = [np.min(values) - 0.5, np.max(values) + 0.5]
                title = '{}-digitized'.format(title)
                n_colors = len(cats)
                scalar_bar_args.setdefault('n_labels', 0)
                _using_labels = True

            if rgb is False or rgb is None:
                rgb = kwargs.get('rgba', False)
            if rgb:
                if scalars.ndim != 2 or scalars.shape[1] < 3 or scalars.shape[1] > 4:
                    raise ValueError('RGB array must be n_points/n_cells by 3/4 in shape.')

            if scalars.ndim != 1:
                if rgb:
                    pass
                elif scalars.ndim == 2 and (scalars.shape[0] == mesh.n_points or scalars.shape[0] == mesh.n_cells):
                    scalars = np.linalg.norm(scalars.copy(), axis=1)
                    title = '{}-normed'.format(title)
                else:
                    scalars = scalars.ravel()

            if scalars.dtype == np.bool:
                scalars = scalars.astype(np.float)

            def prepare_mapper(scalars):
                # Scalar interpolation approach
                if scalars.shape[0] == mesh.n_points:
                    self.mesh._add_point_array(scalars, title, set_active)
                    self.mapper.SetScalarModeToUsePointData()
                elif scalars.shape[0] == mesh.n_cells:
                    self.mesh._add_cell_array(scalars, title, set_active)
                    self.mapper.SetScalarModeToUseCellData()
                else:
                    raise_not_matching(scalars, mesh)
                # Common tasks
                self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
                if interpolate_before_map:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
                if rgb or _custom_opac:
                    self.mapper.SetColorModeToDirectScalars()
                else:
                    self.mapper.SetColorModeToMapScalars()
                return


            prepare_mapper(scalars)
            table = self.mapper.GetLookupTable()

            if _using_labels:
                table.SetAnnotations(convert_array(values), convert_string_array(cats))

            if isinstance(annotations, dict):
                for val, anno in annotations.items():
                    table.SetAnnotation(float(val), str(anno))

            # Set scalar range
            if clim is None:
                clim = [np.nanmin(scalars), np.nanmax(scalars)]
            elif isinstance(clim, float) or isinstance(clim, int):
                clim = [-clim, clim]

            if np.any(clim) and not rgb:
                self.mapper.scalar_range = clim[0], clim[1]

            table.SetNanColor(nan_color)
            if above_color:
                table.SetUseAboveRangeColor(True)
                table.SetAboveRangeColor(*parse_color(above_color, opacity=1))
                scalar_bar_args.setdefault('above_label', 'Above')
            if below_color:
                table.SetUseBelowRangeColor(True)
                table.SetBelowRangeColor(*parse_color(below_color, opacity=1))
                scalar_bar_args.setdefault('below_label', 'Below')

            if cmap is not None:
                if not has_matplotlib:
                    cmap = None
                    logging.warning('Please install matplotlib for color maps.')

                cmap = get_cmap_safe(cmap)
                if categories:
                    if categories is True:
                        n_colors = len(np.unique(scalars))
                    elif isinstance(categories, int):
                        n_colors = categories
                ctable = cmap(np.linspace(0, 1, n_colors))*255
                ctable = ctable.astype(np.uint8)
                # Set opactities
                if isinstance(opacity, np.ndarray) and not _custom_opac:
                    ctable[:,-1] = opacity
                if flip_scalars:
                    ctable = np.ascontiguousarray(ctable[::-1])
                table.SetTable(VN.numpy_to_vtk(ctable))
                if _custom_opac:
                    hue = normalize(scalars, minimum=clim[0], maximum=clim[1])
                    scalars = cmap(hue)[:, :3]
                    # combine colors and alpha into a Nx4 matrix
                    scalars = np.concatenate((scalars, opacity[:, None]), axis=1)
                    scalars = (scalars * 255).astype(np.uint8)
                    prepare_mapper(scalars)

            else:  # no cmap specified
                if flip_scalars:
                    table.SetHueRange(0.0, 0.66667)
                else:
                    table.SetHueRange(0.66667, 0.0)
        else:
            self.mapper.SetScalarModeToUseFieldData()


        # Set actor properties ================================================

        # select view style
        if not style:
            style = 'surface'
        style = style.lower()
        if style == 'wireframe':
            prop.SetRepresentationToWireframe()
            if color is None:
                color = rcParams['outline_color']
        elif style == 'points':
            prop.SetRepresentationToPoints()
        elif style == 'surface':
            prop.SetRepresentationToSurface()
        else:
            raise Exception('Invalid style.  Must be one of the following:\n'
                            '\t"surface"\n'
                            '\t"wireframe"\n'
                            '\t"points"\n')

        prop.SetPointSize(point_size)
        prop.SetAmbient(ambient)
        prop.SetDiffuse(diffuse)
        prop.SetSpecular(specular)
        prop.SetSpecularPower(specular_power)

        if smooth_shading:
            prop.SetInterpolationToPhong()
        else:
            prop.SetInterpolationToFlat()
        # edge display style
        if show_edges:
            prop.EdgeVisibilityOn()

        rgb_color = parse_color(color)
        prop.SetColor(rgb_color)
        if isinstance(opacity, (float, int)):
            prop.SetOpacity(opacity)
        prop.SetEdgeColor(parse_color(edge_color))

        if render_points_as_spheres:
            prop.SetRenderPointsAsSpheres(render_points_as_spheres)
        if render_lines_as_tubes:
            prop.SetRenderLinesAsTubes(render_lines_as_tubes)

        # legend label
        if label:
            if not isinstance(label, str):
                raise AssertionError('Label must be a string')
            geom = pyvista.single_triangle()
            if scalars is not None:
                geom = pyvista.Box()
                rgb_color = parse_color('black')
            self._labels.append([geom, label, rgb_color])

        # lighting display style
        if not lighting:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        # Add scalar bar if available
        if stitle is not None and show_scalar_bar and (not rgb or _custom_opac):
            self.add_scalar_bar(stitle, **scalar_bar_args)

        return actor


    def add_volume(self, volume, scalars=None, clim=None, resolution=None,
                   opacity='linear', n_colors=256, cmap=None, flip_scalars=False,
                   reset_camera=None, name=None, ambient=0.0, categories=False,
                   loc=None, culling=False, multi_colors=False,
                   blending='composite', mapper='fixed_point',
                   stitle=None, scalar_bar_args=None, show_scalar_bar=None,
                   annotations=None, pickable=True, **kwargs):
        """
        Adds a volume, rendered using a fixed point ray cast mapper by default.

        Requires a 3D :class:`numpy.ndarray` or :class:`pyvista.UniformGrid`.

        Parameters
        ----------
        volume : 3D numpy.ndarray or pyvista.UnformGrid
            The input volume to visualize. 3D numpy arrays are accepted.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name of an
            array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            ``color`` and ``scalars`` are ``None``, then the active scalars are
            used.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        opacity : string or numpy.ndarray, optional
            Opacity mapping for the scalars array.
            A string can also be specified to map the scalar range to a
            predefined opacity transfer function (options include: 'linear',
            'linear_r', 'geom', 'geom_r'). Or you can pass a custum made
            trasfer function that is an aray either ``n_colors`` in length or
            shorter.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        flip_scalars : bool, optional
            Flip direction of cmap.

        n_colors : int, optional
            Number of colors to use when displaying scalars.  Default
            256.

        cmap : str, optional
           Name of the Matplotlib colormap to us when mapping the ``scalars``.
           See available Matplotlib colormaps.  Only applicable for when
           displaying ``scalars``. Requires Matplotlib to be installed.
           ``colormap`` is also an accepted alias for this. If ``colorcet`` or
           ``cmocean`` are installed, their colormaps can be specified by name.

        flip_scalars : bool, optional
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        ambient : float, optional
            When lighting is enabled, this is the amount of light from
            0 to 1 that reaches the actor when not directed at the
            light source emitted from the viewer.  Default 0.0.

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.

        culling : str, optional
            Does not render faces that are culled. Options are ``'front'`` or
            ``'back'``. This can be helpful for dense surface meshes,
            especially when edges are visible, but can cause flat
            meshes to be partially displayed.  Defaults ``False``.

        categories : bool, optional
            If set to ``True``, then the number of unique values in the scalar
            array will be used as the ``n_colors`` argument.

        multi_colors : bool, optional
            Whether or not to use multiple colors when plotting MultiBlock
            object. Blocks will be colored sequentially as 'Reds', 'Greens',
            'Blues', and 'Grays'.

        blending : str, optional
            Blending mode for visualisation of the input object(s). Can be
            one of 'additive', 'maximum', 'minimum', 'composite', or
            'average'. Defaults to 'additive'.

        mapper : str, optional
            Volume mapper to use given by name. Options include:
            ``'fixed_point'``, ``'gpu'``, ``'open_gl'``, and ``'smart'``.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the scalar bar
            to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If False, a scalar bar will not be added to the scene. Defaults
            to ``True``.

        stitle : string, optional
            Scalar bar title. By default the scalar bar is given a title of the
            the scalar array used to color the mesh.
            To create a bar with no title, use an empty string (i.e. '').

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float values in the
            scalar range to annotate on the scalar bar and the values are the
            the string annotations.

        Returns
        -------
        actor: vtk.vtkVolume
            VTK volume of the input data.
        """

        # Handle default arguments

        if name is None:
            name = '{}({})'.format(type(volume).__name__, str(hex(id(volume))))

        if clim is None:
            clim = kwargs.get('rng', None)

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if show_scalar_bar is None:
            show_scalar_bar = rcParams['show_scalar_bar']

        if culling is None:
            culling = kwargs.get("backface_culling", False)
            if culling is True:
                culling = 'backface'

        # Convert the VTK data object to a pyvista wrapped object if neccessary
        if not is_pyvista_dataset(volume):
            if isinstance(volume, np.ndarray):
                volume = wrap(volume)
                if resolution is None:
                    resolution = [1,1,1]
                elif len(resolution) != 3:
                    raise ValueError('Invalid resolution dimensions.')
                volume.spacing = resolution
            else:
                volume = wrap(volume)
                if not is_pyvista_dataset(volume):
                    raise TypeError('Object type ({}) not supported for plotting in PyVista.'.format(type(volume)))
        else:
            # HACK: Make a copy so the original object is not altered
            volume = volume.copy()


        if isinstance(volume, pyvista.MultiBlock):
            from itertools import cycle
            cycler = cycle(['Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples'])
            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(volume.GetNumberOfBlocks()):
                if volume[idx] is None:
                    continue
                # Get a good name to use
                next_name = '{}-{}'.format(name, idx)
                # Get the data object
                block = wrap(volume.GetBlock(idx))
                if resolution is None:
                    try:
                        block_resolution = block.GetSpacing()
                    except AttributeError:
                        block_resolution = resolution
                else:
                    block_resolution = resolution
                if multi_colors:
                    color = next(cycler)
                else:
                    color = cmap

                a = self.add_volume(block, resolution=block_resolution, opacity=opacity,
                                    n_colors=n_colors, cmap=color, flip_scalars=flip_scalars,
                                    reset_camera=reset_camera, name=next_name,
                                    ambient=ambient, categories=categories, loc=loc,
                                    culling=culling, clim=clim,
                                    mapper=mapper, pickable=pickable, **kwargs)

                actors.append(a)
            return actors

        if not isinstance(volume, pyvista.UniformGrid):
            raise TypeError('Type ({}) not supported for volume rendering at this time. Use `pyvista.UniformGrid`.')


        if scalars is None:
            # Make sure scalar components are not vectors/tuples
            scalars = volume.active_scalar
            # Don't allow plotting of string arrays by default
            if scalars is not None and np.issubdtype(scalars.dtype, np.number):
                if stitle is None:
                    stitle = volume.active_scalar_info[1]
            else:
                raise RuntimeError('No scalars to use for volume rendering.')
        elif isinstance(scalars, str):
            pass

        ##############

        title = 'Data' if stitle is None else stitle
        set_active = False
        if isinstance(scalars, str):
            title = scalars
            scalars = get_array(volume, scalars,
                                preference=kwargs.get('preference', 'point'), err=True)
            if stitle is None:
                stitle = title
        else:
            set_active = True

        if not isinstance(scalars, np.ndarray):
            scalars = np.asarray(scalars)

        if not np.issubdtype(scalars.dtype, np.number):
            raise TypeError('Non-numeric scalars are currently not supported for volume rendering.')


        if scalars.ndim != 1:
            scalars = scalars.ravel()

        if scalars.dtype == np.bool or scalars.dtype == np.uint8:
            scalars = scalars.astype(np.float)

        # Define mapper, volume, and add the correct properties
        mappers = {
            'fixed_point': vtk.vtkFixedPointVolumeRayCastMapper,
            'gpu': vtk.vtkGPUVolumeRayCastMapper,
            'open_gl': vtk.vtkOpenGLGPUVolumeRayCastMapper,
            'smart': vtk.vtkSmartVolumeMapper,
        }
        if not isinstance(mapper, str) or mapper not in mappers.keys():
            raise RuntimeError('Mapper ({}) unknown. Available volume mappers include: {}'.format(mapper, ', '.join(mappers.keys())))
        self.mapper = make_mapper(mappers[mapper])

        # Scalar interpolation approach
        if scalars.shape[0] == volume.n_points:
            volume._add_point_array(scalars, title, set_active)
            self.mapper.SetScalarModeToUsePointData()
        elif scalars.shape[0] == volume.n_cells:
            volume._add_cell_array(scalars, title, set_active)
            self.mapper.SetScalarModeToUseCellData()
        else:
            raise_not_matching(scalars, volume)

        # Set scalar range
        if clim is None:
            clim = [np.nanmin(scalars), np.nanmax(scalars)]
        elif isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]

        ###############

        scalars = scalars.astype(np.float)
        idxs0 = scalars < clim[0]
        idxs1 = scalars > clim[1]
        scalars[idxs0] = np.nan
        scalars[idxs1] = np.nan
        scalars = ((scalars - np.nanmin(scalars)) / (np.nanmax(scalars) - np.nanmin(scalars))) * 255
        # scalars = scalars.astype(np.uint8)
        volume[title] = scalars

        self.mapper.scalar_range = clim

        # Set colormap and build lookup table
        table = vtk.vtkLookupTable()
        # table.SetNanColor(nan_color) # NaN's are chopped out with current implementation
        # above/below colors not supported with volume rendering

        if isinstance(annotations, dict):
            for val, anno in annotations.items():
                table.SetAnnotation(float(val), str(anno))

        if cmap is None: # grab alias for cmaps: colormap
            cmap = kwargs.get('colormap', None)
            if cmap is None: # Set default map if matplotlib is avaialble
                if has_matplotlib:
                    cmap = rcParams['cmap']

        if cmap is not None:
            if not has_matplotlib:
                cmap = None
                raise RuntimeError('Please install matplotlib for volume rendering.')

            cmap = get_cmap_safe(cmap)
            if categories:
                if categories is True:
                    n_colors = len(np.unique(scalars))
                elif isinstance(categories, int):
                    n_colors = categories
        if flip_scalars:
            cmap = cmap.reversed()


        color_tf = vtk.vtkColorTransferFunction()
        for ii in range(n_colors):
            color_tf.AddRGBPoint(ii, *cmap(ii)[:-1])

        # Set opacities
        if isinstance(opacity, (float, int)):
            opacity_values = [opacity] * n_colors
        elif isinstance(opacity, str):
            opacity_values = pyvista.opacity_transfer_function(opacity, n_colors)
        elif isinstance(opacity, (np.ndarray, list, tuple)):
            opacity = np.array(opacity)
            opacity_values = opacity_transfer_function(opacity, n_colors)

        opacity_tf = vtk.vtkPiecewiseFunction()
        for ii in range(n_colors):
            opacity_tf.AddPoint(ii, opacity_values[ii] / n_colors)


        # Now put color tf and opacity tf into a lookup table for the scalar bar
        table.SetNumberOfTableValues(n_colors)
        lut = cmap(np.array(range(n_colors))) * 255
        lut[:,3] = opacity_values
        lut = lut.astype(np.uint8)
        table.SetTable(VN.numpy_to_vtk(lut))
        table.SetRange(*clim)
        self.mapper.lookup_table = table

        self.mapper.SetInputData(volume)

        blending = blending.lower()
        if blending in ['additive', 'add', 'sum']:
            self.mapper.SetBlendModeToAdditive()
        elif blending in ['average', 'avg', 'average_intensity']:
            self.mapper.SetBlendModeToAverageIntensity()
        elif blending in ['composite', 'comp']:
            self.mapper.SetBlendModeToComposite()
        elif blending in ['maximum', 'max', 'maximum_intensity']:
            self.mapper.SetBlendModeToMaximumIntensity()
        elif blending in ['minimum', 'min', 'minimum_intensity']:
            self.mapper.SetBlendModeToMinimumIntensity()
        else:
            raise ValueError('Blending mode \'{}\' invalid. '.format(blending) +
                             'Please choose one ' + 'of \'additive\', '
                             '\'composite\', \'minimum\' or ' + '\'maximum\'.')
        self.mapper.Update()

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        prop.SetAmbient(ambient)
        self.volume.SetProperty(prop)

        actor, prop = self.add_actor(self.volume, reset_camera=reset_camera,
                                     name=name, loc=loc, culling=culling,
                                     pickable=pickable)


        # Add scalar bar
        if stitle is not None and show_scalar_bar:
            self.add_scalar_bar(stitle, **scalar_bar_args)


        return actor


    def update_scalar_bar_range(self, clim, name=None):
        """Update the value range of the active or named scalar bar.

        Parameters
        ----------
        2 item list
            The new range of scalar bar. Example: ``[-1, 2]``.

        name : str, optional
            The title of the scalar bar to update
        """
        if isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]
        if len(clim) != 2:
            raise TypeError('clim argument must be a length 2 iterable of values: (min, max).')
        if name is None:
            if not hasattr(self, 'mapper'):
                raise RuntimeError('This plotter does not have an active mapper.')
            self.mapper.scalar_range = clim
            return

        # Use the name to find the desired actor
        def update_mapper(mapper_helper):
            mapper_helper.scalar_range = clim
            return

        try:
            for mh in self._scalar_bar_mappers[name]:
                update_mapper(mh)
        except KeyError:
            raise KeyError('Name ({}) not valid/not found in this plotter.')
        return


    @property
    def camera_set(self):
        """ Returns if the camera of the active renderer has been set """
        return self.renderer.camera_set

    def get_default_cam_pos(self, negative=False):
        """ Return the default camera position of the active renderer """
        return self.renderer.get_default_cam_pos(negative=negative)

    @camera_set.setter
    def camera_set(self, is_set):
        """ Sets if the camera has been set on the active renderer"""
        self.renderer.camera_set = is_set

    @property
    def renderer(self):
        """ simply returns the active renderer """
        return self.renderers[self._active_renderer_index]

    @property
    def bounds(self):
        """ Returns the bounds of the active renderer """
        return self.renderer.bounds

    @property
    def length(self):
        """Returns the length of the diagonal of the bounding box of the scene
        """
        return pyvista.Box(self.bounds).length

    @property
    def center(self):
        """ Returns the center of the active renderer """
        return self.renderer.center

    def update_bounds_axes(self):
        """ Update the bounds of the active renderer """
        return self.renderer.update_bounds_axes()

    @property
    def _scalar_bar_slots(self):
        return self.renderer._scalar_bar_slots

    @property
    def _scalar_bar_slot_lookup(self):
        return self.renderer._scalar_bar_slot_lookup

    @_scalar_bar_slots.setter
    def _scalar_bar_slots(self, value):
        self.renderer._scalar_bar_slots = value

    @_scalar_bar_slot_lookup.setter
    def _scalar_bar_slot_lookup(self, value):
        self.renderer._scalar_bar_slot_lookup = value

    def clear(self):
        """ Clears plot by removing all actors and properties """
        for renderer in self.renderers:
            renderer.RemoveAllViewProps()
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._scalar_bar_actors = {}
        self._scalar_bar_widgets = {}

    def remove_actor(self, actor, reset_camera=False):
        """
        Removes an actor from the Plotter.

        Parameters
        ----------
        actor : vtk.vtkActor
            Actor that has previously added to the Renderer.

        reset_camera : bool, optional
            Resets camera so all actors can be seen.

        Returns
        -------
        success : bool
            True when actor removed.  False when actor has not been
            removed.
        """
        for renderer in self.renderers:
            renderer.remove_actor(actor, reset_camera)
        return True

    def add_actor(self, uinput, reset_camera=False, name=None, loc=None,
                  culling=False, pickable=True):
        """
        Adds an actor to render window.  Creates an actor if input is
        a mapper.

        Parameters
        ----------
        uinput : vtk.vtkMapper or vtk.vtkActor
            vtk mapper or vtk actor to be added.

        reset_camera : bool, optional
            Resets the camera when true.

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.

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
        # add actor to the correct render window
        self._active_renderer_index = self.loc_to_index(loc)
        renderer = self.renderers[self._active_renderer_index]
        return renderer.add_actor(uinput=uinput, reset_camera=reset_camera,
                                  name=name, culling=culling, pickable=pickable)

    def loc_to_index(self, loc):
        """
        Return index of the render window given a location index.

        Parameters
        ----------
        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.

        Returns
        -------
        idx : int
            Index of the render window.

        """
        if loc is None:
            return self._active_renderer_index
        elif isinstance(loc, int):
            return loc
        elif isinstance(loc, collections.Iterable):
            if not len(loc) == 2:
                raise AssertionError('"loc" must contain two items')
            index_row = loc[0]
            index_column = loc[1]
            if index_row < 0 or index_row >= self.shape[0]:
                raise IndexError('Row index is out of range ({})'.format(self.shape[0]))
            if index_column < 0 or index_column >= self.shape[1]:
                raise IndexError('Column index is out of range ({})'.format(self.shape[1]))
            sz = int(self.shape[0] * self.shape[1])
            idxs = np.array([i for i in range(sz)], dtype=int).reshape(self.shape)
            return idxs[index_row, index_column]

    def index_to_loc(self, index):
        """Convert a 1D index location to the 2D location on the plotting grid
        """
        if len(self.shape) == 1:
            return index
        sz = int(self.shape[0] * self.shape[1])
        idxs = np.array([i for i in range(sz)], dtype=int).reshape(self.shape)
        args = np.argwhere(idxs == index)
        if len(args) < 1:
            raise RuntimeError('Index ({}) is out of range.')
        return args[0]


    @property
    def camera(self):
        """ The active camera of the active renderer """
        return self.renderer.camera

    @camera.setter
    def camera(self, camera):
        """Set the active camera for the rendering scene"""
        self.renderer.camera = camera


    def enable_parallel_projection(self):
        """Set use parallel projection. The camera will have a parallel
        projection. Parallel projection is often useful when viewing images or
        2D datasets.
        """
        return self.renderer.enable_parallel_projection()


    def disable_parallel_projection(self):
        """Reset the camera to use perspective projection."""
        return self.renderer.disable_parallel_projection()

    def add_axes_at_origin(self, x_color=None, y_color=None, z_color=None,
                           xlabel='X', ylabel='Y', zlabel='Z', line_width=2,
                           labels_off=False, loc=None):
        """
        Add axes actor at the origin of a render window.

        Parameters
        ----------
        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  When None, defaults to the
            active render window.

        Returns
        --------
        marker_actor : vtk.vtkAxesActor
            vtkAxesActor actor
        """
        kwargs = locals()
        _ = kwargs.pop('self')
        _ = kwargs.pop('loc')
        self._active_renderer_index = self.loc_to_index(loc)
        return self.renderers[self._active_renderer_index].add_axes_at_origin(**kwargs)

    def show_bounds(self, mesh=None, bounds=None, show_xaxis=True,
                    show_yaxis=True, show_zaxis=True, show_xlabels=True,
                    show_ylabels=True, show_zlabels=True, italic=False,
                    bold=True, shadow=False, font_size=None,
                    font_family=None, color=None,
                    xlabel='X Axis', ylabel='Y Axis', zlabel='Z Axis',
                    use_2d=False, grid=None, location='closest', ticks=None,
                    all_edges=False, corner_factor=0.5, fmt=None,
                    minor_ticks=False, loc=None, padding=0.0):
        """
        Adds bounds axes.  Shows the bounds of the most recent input
        mesh unless mesh is specified.

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

                color='white'
                color='w'
                color=[1, 1, 1]
                color='#FFFFFF'

        xlabel : string, optional
            Title of the x axis.  Default "X Axis"

        ylabel : string, optional
            Title of the y axis.  Default "Y Axis"

        zlabel : string, optional
            Title of the z axis.  Default "Z Axis"

        use_2d : bool, optional
            A bug with vtk 6.3 in Windows seems to cause this function
            to crash this can be enabled for smoother plotting for
            other enviornments.

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

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.

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
        kwargs = locals()
        _ = kwargs.pop('self')
        _ = kwargs.pop('loc')
        self._active_renderer_index = self.loc_to_index(loc)
        renderer = self.renderers[self._active_renderer_index]
        renderer.show_bounds(**kwargs)

    def add_bounds_axes(self, *args, **kwargs):
        """Deprecated"""
        logging.warning('`add_bounds_axes` is deprecated. Use `show_bounds` or `show_grid`.')
        return self.show_bounds(*args, **kwargs)

    def add_bounding_box(self, color=None, corner_factor=0.5, line_width=None,
                         opacity=1.0, render_lines_as_tubes=False,
                         lighting=None, reset_camera=None, outline=True,
                         culling='front', loc=None):
        """
        Adds an unlabeled and unticked box at the boundaries of
        plot.  Useful for when wanting to plot outer grids while
        still retaining all edges of the boundary.

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

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.

        """
        kwargs = locals()
        _ = kwargs.pop('self')
        _ = kwargs.pop('loc')
        self._active_renderer_index = self.loc_to_index(loc)
        renderer = self.renderers[self._active_renderer_index]
        return renderer.add_bounding_box(**kwargs)

    def remove_bounding_box(self, loc=None):
        """
        Removes bounding box from the active renderer.

        Parameters
        ----------
        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.
        """
        self._active_renderer_index = self.loc_to_index(loc)
        renderer = self.renderers[self._active_renderer_index]
        renderer.remove_bounding_box()

    def remove_bounds_axes(self, loc=None):
        """
        Removes bounds axes from the active renderer.

        Parameters
        ----------
        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If None, selects the last
            active Renderer.
        """
        self._active_renderer_index = self.loc_to_index(loc)
        renderer = self.renderers[self._active_renderer_index]
        renderer.remove_bounds_axes()

    def subplot(self, index_row, index_column=None):
        """
        Sets the active subplot.

        Parameters
        ----------
        index_row : int
            Index of the subplot to activate along the rows.

        index_column : int
            Index of the subplot to activate along the columns.

        """
        if len(self.shape) == 1:
            self._active_renderer_index = index_row
            return

        if index_row < 0 or index_row >= self.shape[0]:
            raise IndexError('Row index is out of range ({})'.format(self.shape[0]))
        if index_column < 0 or index_column >= self.shape[1]:
            raise IndexError('Column index is out of range ({})'.format(self.shape[1]))
        self._active_renderer_index = self.loc_to_index((index_row, index_column))

    def link_views(self, views=0):
        """
        Links the views' cameras.

        Parameters
        ----------
        views : int | tuple or list
            If ``views`` is int, link the views to the given view
            index or if ``views`` is a tuple or a list, link the given
            views cameras.

        """
        if isinstance(views, int):
            for renderer in self.renderers:
                renderer.camera = self.renderers[views].camera
        elif isinstance(views, collections.Iterable):
            for view_index in views:
                self.renderers[view_index].camera = \
                    self.renderers[views[0]].camera
        else:
            raise TypeError('Expected type is int, list or tuple:'
                            '{} is given'.format(type(views)))

    def unlink_views(self, views=None):
        """
        Unlinks the views' cameras.

        Parameters
        ----------
        views : None | int | tuple or list
            If ``views`` is None unlink all the views, if ``views``
            is int unlink the selected view's camera or if ``views``
            is a tuple or a list, unlink the given views cameras.

        """
        if views is None:
            for renderer in self.renderers:
                renderer.camera = vtk.vtkCamera()
                renderer.reset_camera()
        elif isinstance(views, int):
            self.renderers[views].camera = vtk.vtkCamera()
            self.renderers[views].reset_camera()
        elif isinstance(views, collections.Iterable):
            for view_index in views:
                self.renderers[view_index].camera = vtk.vtkCamera()
                self.renderers[view_index].reset_camera()
        else:
            raise TypeError('Expected type is None, int, list or tuple:'
                            '{} is given'.format(type(views)))

    def show_grid(self, **kwargs):
        """
        A wrapped implementation of ``show_bounds`` to change default
        behaviour to use gridlines and showing the axes labels on the outer
        edges. This is intended to be silimar to ``matplotlib``'s ``grid``
        function.
        """
        kwargs.setdefault('grid', 'back')
        kwargs.setdefault('location', 'outer')
        kwargs.setdefault('ticks', 'both')
        return self.show_bounds(**kwargs)

    def set_scale(self, xscale=None, yscale=None, zscale=None, reset_camera=True):
        """
        Scale all the datasets in the scene of the active renderer.

        Scaling in performed independently on the X, Y and Z axis.
        A scale of zero is illegal and will be replaced with one.

        Parameters
        ----------
        xscale : float, optional
            Scaling of the x axis.  Must be greater than zero.

        yscale : float, optional
            Scaling of the y axis.  Must be greater than zero.

        zscale : float, optional
            Scaling of the z axis.  Must be greater than zero.

        reset_camera : bool, optional
            Resets camera so all actors can be seen.

        """
        self.renderer.set_scale(xscale, yscale, zscale, reset_camera)

    @property
    def scale(self):
        """ The scaling of the active renderer. """
        return self.renderer.scale


    def add_scalar_bar(self, title=None, n_labels=5, italic=False,
                       bold=True, title_font_size=None,
                       label_font_size=None, color=None,
                       font_family=None, shadow=False, mapper=None,
                       width=None, height=None, position_x=None,
                       position_y=None, vertical=None,
                       interactive=False, fmt=None, use_opacity=True,
                       outline=False, nan_annotation=False,
                       below_label=None, above_label=None,
                       background_color=None, n_colors=None):
        """
        Creates scalar bar using the ranges as set by the last input
        mesh.

        Parameters
        ----------
        title : string, optional
            Title of the scalar bar.  Default None

        n_labels : int, optional
            Number of labels to use for the scalar bar.

        italic : bool, optional
            Italicises title and bar labels.  Default False.

        bold  : bool, optional
            Bolds title and bar labels.  Default True

        title_font_size : float, optional
            Sets the size of the title font.  Defaults to None and is sized
            automatically.

        label_font_size : float, optional
            Sets the size of the title font.  Defaults to None and is sized
            automatically.

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:
                color='white'
                color='w'
                color=[1, 1, 1]
                color='#FFFFFF'

        font_family : string, optional
            Font family.  Must be either courier, times, or arial.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        width : float, optional
            The percentage (0 to 1) width of the window for the colorbar

        height : float, optional
            The percentage (0 to 1) height of the window for the colorbar

        position_x : float, optional
            The percentage (0 to 1) along the windows's horizontal
            direction to place the bottom left corner of the colorbar

        position_y : float, optional
            The percentage (0 to 1) along the windows's vertical
            direction to place the bottom left corner of the colorbar

        interactive : bool, optional
            Use a widget to control the size and location of the scalar bar.

        use_opacity : bool, optional
            Optionally disply the opacity mapping on the scalar bar

        outline : bool, optional
            Optionally outline the scalar bar to make opacity mappings more
            obvious.

        nan_annotation : bool, optional
            Annotate the NaN color

        below_label : str, optional
            String annotation for values below the scalar range

        above_label : str, optional
            String annotation for values above the scalar range

        background_color: array, optional
            The color used for the background in RGB format.

        n_colors: int, optional
            The maximum number of color displayed in the scalar bar.

        Notes
        -----
        Setting title_font_size, or label_font_size disables automatic font
        sizing for both the title and label.


        """
        if font_family is None:
            font_family = rcParams['font']['family']
        if label_font_size is None:
            label_font_size = rcParams['font']['label_size']
        if title_font_size is None:
            title_font_size = rcParams['font']['title_size']
        if color is None:
            color = rcParams['font']['color']
        if fmt is None:
            fmt = rcParams['font']['fmt']
        if vertical is None:
            if rcParams['colorbar_orientation'].lower() == 'vertical':
                vertical = True
        # Automatically choose size if not specified
        if width is None:
            if vertical:
                width = rcParams['colorbar_vertical']['width']
            else:
                width = rcParams['colorbar_horizontal']['width']
        if height is None:
            if vertical:
                height = rcParams['colorbar_vertical']['height']
            else:
                height = rcParams['colorbar_horizontal']['height']

        # check if maper exists
        if mapper is None:
            if not hasattr(self, 'mapper') or self.mapper is None:
                raise Exception('Mapper does not exist.  '
                                'Add a mesh with scalars first.')
            mapper = self.mapper

        if title:
            # Check that this data hasn't already been plotted
            if title in list(self._scalar_bar_ranges.keys()):
                clim = list(self._scalar_bar_ranges[title])
                newrng = mapper.scalar_range
                oldmappers = self._scalar_bar_mappers[title]
                # get max for range and reset everything
                if newrng[0] < clim[0]:
                    clim[0] = newrng[0]
                if newrng[1] > clim[1]:
                    clim[1] = newrng[1]
                for mh in oldmappers:
                    mh.scalar_range = clim[0], clim[1]
                mapper.scalar_range = clim[0], clim[1]
                self._scalar_bar_mappers[title].append(mapper)
                self._scalar_bar_ranges[title] = clim
                # Color bar already present and ready to be used so returning
                return

        # Automatically choose location if not specified
        if position_x is None or position_y is None:
            try:
                slot = min(self._scalar_bar_slots)
                self._scalar_bar_slots.remove(slot)
                self._scalar_bar_slot_lookup[title] = slot
            except:
                raise RuntimeError('Maximum number of color bars reached.')
            if position_x is None:
                if vertical:
                    position_x = rcParams['colorbar_vertical']['position_x']
                    position_x -= slot * (width + 0.2 * width)
                else:
                    position_x = rcParams['colorbar_horizontal']['position_x']

            if position_y is None:
                if vertical:
                    position_y = rcParams['colorbar_vertical']['position_y']
                else:
                    position_y = rcParams['colorbar_horizontal']['position_y']
                    position_y += slot * height
        # Adjust to make sure on the screen
        if position_x + width > 1:
            position_x -= width
        if position_y + height > 1:
            position_y -= height

        # parse color
        color = parse_color(color)

        # Create scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        if background_color is not None:
            from ..core.common import vtk_to_numpy, numpy_to_vtk
            if not isinstance(background_color, collections.Iterable):
                raise TypeError('Expected type for `background_color`'
                                'is list, tuple or np.ndarray: '
                                '{} is given'.format(type(background_color)))
            if len(background_color) != 3:
                raise ValueError('Expected length for `background_color` is 3: '
                                 '{} is given'.format(len(background_color)))
            background_color = np.asarray(background_color)
            background_color = np.append(background_color, 1.0) * 255.

            lut = vtk.vtkLookupTable()
            lut.DeepCopy(mapper.lookup_table)
            ctable = vtk_to_numpy(lut.GetTable())
            alphas = ctable[:, -1][:, np.newaxis] / 255.
            use_table = ctable.copy()
            use_table[:, -1] = 255.
            ctable = (use_table * alphas) + background_color * (1 - alphas)
            lut.SetTable(numpy_to_vtk(ctable, array_type=vtk.VTK_UNSIGNED_CHAR))
        else:
            lut = mapper.lookup_table
        self.scalar_bar.SetLookupTable(lut)
        if n_colors is not None:
            self.scalar_bar.SetMaximumNumberOfColors(n_colors)

        if n_labels < 1:
            self.scalar_bar.DrawTickLabelsOff()
        else:
            self.scalar_bar.SetNumberOfLabels(n_labels)

        if nan_annotation:
            self.scalar_bar.DrawNanAnnotationOn()

        if above_label:
            self.scalar_bar.DrawAboveRangeSwatchOn()
            self.scalar_bar.SetAboveRangeAnnotation(above_label)
        if below_label:
            self.scalar_bar.DrawBelowRangeSwatchOn()
            self.scalar_bar.SetBelowRangeAnnotation(below_label)

        # edit the size of the colorbar
        self.scalar_bar.SetHeight(height)
        self.scalar_bar.SetWidth(width)
        self.scalar_bar.SetPosition(position_x, position_y)

        if fmt is not None:
            self.scalar_bar.SetLabelFormat(fmt)

        if vertical:
            self.scalar_bar.SetOrientationToVertical()
        else:
            self.scalar_bar.SetOrientationToHorizontal()

        if label_font_size is None or title_font_size is None:
            self.scalar_bar.UnconstrainedFontSizeOn()
            self.scalar_bar.AnnotationTextScalingOn()

        label_text = self.scalar_bar.GetLabelTextProperty()
        anno_text = self.scalar_bar.GetAnnotationTextProperty()
        label_text.SetColor(color)
        anno_text.SetColor(color)
        label_text.SetShadow(shadow)
        anno_text.SetShadow(shadow)

        # Set font
        label_text.SetFontFamily(parse_font_family(font_family))
        anno_text.SetFontFamily(parse_font_family(font_family))
        label_text.SetItalic(italic)
        anno_text.SetItalic(italic)
        label_text.SetBold(bold)
        anno_text.SetBold(bold)
        if label_font_size:
            label_text.SetFontSize(label_font_size)
            anno_text.SetFontSize(label_font_size)

        # Set properties
        if title:
            clim = mapper.scalar_range
            self._scalar_bar_ranges[title] = clim
            self._scalar_bar_mappers[title] = [mapper]

            self.scalar_bar.SetTitle(title)
            title_text = self.scalar_bar.GetTitleTextProperty()

            title_text.SetJustificationToCentered()

            title_text.SetItalic(italic)
            title_text.SetBold(bold)
            title_text.SetShadow(shadow)
            if title_font_size:
                title_text.SetFontSize(title_font_size)

            # Set font
            title_text.SetFontFamily(parse_font_family(font_family))

            # set color
            title_text.SetColor(color)

            self._scalar_bar_actors[title] = self.scalar_bar

        if interactive is None:
            interactive = rcParams['interactive']
            if self.shape != (1, 1):
                interactive = False
        elif interactive and self.shape != (1, 1):
            err_str = 'Interactive scalar bars disabled for multi-renderer plots'
            raise Exception(err_str)

        if interactive and hasattr(self, 'iren'):
            self.scalar_widget = vtk.vtkScalarBarWidget()
            self.scalar_widget.SetScalarBarActor(self.scalar_bar)
            self.scalar_widget.SetInteractor(self.iren)
            self.scalar_widget.SetEnabled(1)
            rep = self.scalar_widget.GetRepresentation()
            # self.scalar_widget.On()
            if vertical is True or vertical is None:
                rep.SetOrientation(1)  # 0 = Horizontal, 1 = Vertical
            else:
                rep.SetOrientation(0)  # 0 = Horizontal, 1 = Vertical
            self._scalar_bar_widgets[title] = self.scalar_widget

        if use_opacity:
            self.scalar_bar.SetUseOpacity(True)

        if outline:
            self.scalar_bar.SetDrawFrame(True)
            frame_prop = self.scalar_bar.GetFrameProperty()
            frame_prop.SetColor(color)
        else:
            self.scalar_bar.SetDrawFrame(False)

        self.add_actor(self.scalar_bar, reset_camera=False, pickable=False)

    def update_scalars(self, scalars, mesh=None, render=True):
        """
        Updates scalars of the an object in the plotter.

        Parameters
        ----------
        scalars : np.ndarray
            Scalars to replace existing scalars.

        mesh : vtk.PolyData or vtk.UnstructuredGrid, optional
            Object that has already been added to the Plotter.  If
            None, uses last added mesh.

        render : bool, optional
            Forces an update to the render window.  Default True.

        """
        if mesh is None:
            mesh = self.mesh

        if isinstance(mesh, (collections.Iterable, pyvista.MultiBlock)):
            # Recursive if need to update scalars on many meshes
            for m in mesh:
                self.update_scalars(scalars, mesh=m, render=False)
            if render:
                self.ren_win.Render()
            return

        if isinstance(scalars, str):
            # Grab scalar array if name given
            scalars = get_array(mesh, scalars)

        if scalars is None:
            if render:
                self.ren_win.Render()
            return

        if scalars.shape[0] == mesh.GetNumberOfPoints():
            data = mesh.GetPointData()
        elif scalars.shape[0] == mesh.GetNumberOfCells():
            data = mesh.GetCellData()
        else:
            raise_not_matching(scalars, mesh)

        vtk_scalars = data.GetScalars()
        if vtk_scalars is None:
            raise Exception('No active scalars')
        s = convert_array(vtk_scalars)
        s[:] = scalars
        data.Modified()
        try:
            # Why are the points updated here? Not all datasets have points
            # and only the scalar array is modified by this function...
            mesh.GetPoints().Modified()
        except:
            pass

        if render:
            self.ren_win.Render()

    def update_coordinates(self, points, mesh=None, render=True):
        """
        Updates the points of the an object in the plotter.

        Parameters
        ----------
        points : np.ndarray
            Points to replace existing points.

        mesh : vtk.PolyData or vtk.UnstructuredGrid, optional
            Object that has already been added to the Plotter.  If
            None, uses last added mesh.

        render : bool, optional
            Forces an update to the render window.  Default True.

        """
        if mesh is None:
            mesh = self.mesh

        mesh.points = points

        if render:
            self._render()

    def close(self):
        """ closes render window """
        # must close out widgets first
        super(BasePlotter, self).close()

        # Grab screenshots of last render
        self.last_image = self.screenshot(None, return_img=True)
        self.last_image_depth = self.get_image_depth()

        if hasattr(self, 'axes_widget'):
            del self.axes_widget

        if hasattr(self, 'scalar_widget'):
            del self.scalar_widget

        # reset scalar bar stuff
        self.clear()

        if hasattr(self, 'ren_win'):
            self.ren_win.Finalize()
            del self.ren_win

        if hasattr(self, '_style'):
            del self._style

        if hasattr(self, 'iren'):
            self.iren.RemoveAllObservers()
            self.iren.TerminateApp()
            del self.iren

        if hasattr(self, 'textActor'):
            del self.textActor

        # end movie
        if hasattr(self, 'mwriter'):
            try:
                self.mwriter.close()
            except BaseException:
                pass

    def deep_clean(self):
        for renderer in self.renderers:
            renderer.deep_clean()
        # Do not remove the renderers on the clean
        self.mesh = None
        self.mapper = None

    def add_text(self, text, position='upper_left', font_size=18, color=None,
                 font=None, shadow=False, name=None, loc=None, viewport=False):
        """
        Adds text to plot object in the top left corner by default

        Parameters
        ----------
        text : str
            The text to add the the rendering

        position : str, tuple(float)
            Position to place the bottom left corner of the text box.
            If tuple is used, the position of the text uses the pixel
            coordinate system (default). In this case,
            it returns a more general `vtkOpenGLTextActor`.
            If string name is used, it returns a `vtkCornerAnnotation`
            object normally used for fixed labels (like title or xlabel).
            Default is to find the top left corner of the renderering window
            and place text box up there. Available position: ``'lower_left'``,
            ``'lower_right'``, ``'upper_left'``, ``'upper_right'``,
            ``'lower_edge'``, ``'upper_edge'``, ``'right_edge'``, and
            ``'left_edge'``

        font : string, optional
            Font name may be courier, times, or arial

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.

        viewport: bool
            If True and position is a tuple of float, uses
            the normalized viewport coordinate system (values between 0.0
            and 1.0 and support for HiDPI).

        Returns
        -------
        textActor : vtk.vtkTextActor
            Text actor added to plot

        """
        if font is None:
            font = rcParams['font']['family']
        if font_size is None:
            font_size = rcParams['font']['size']
        if color is None:
            color = rcParams['font']['color']
        if position is None:
            # Set the position of the text to the top left corner
            window_size = self.window_size
            x = (window_size[0] * 0.02) / self.shape[0]
            y = (window_size[1] * 0.85) / self.shape[0]
            position = [x, y]

        corner_mappings = {
            'lower_left': vtk.vtkCornerAnnotation.LowerLeft,
            'lower_right': vtk.vtkCornerAnnotation.LowerRight,
            'upper_left': vtk.vtkCornerAnnotation.UpperLeft,
            'upper_right': vtk.vtkCornerAnnotation.UpperRight,
            'lower_edge': vtk.vtkCornerAnnotation.LowerEdge,
            'upper_edge': vtk.vtkCornerAnnotation.UpperEdge,
            'left_edge': vtk.vtkCornerAnnotation.LeftEdge,
            'right_edge': vtk.vtkCornerAnnotation.RightEdge,

        }
        corner_mappings['ll'] = corner_mappings['lower_left']
        corner_mappings['lr'] = corner_mappings['lower_right']
        corner_mappings['ul'] = corner_mappings['upper_left']
        corner_mappings['ur'] = corner_mappings['upper_right']
        corner_mappings['top'] = corner_mappings['upper_edge']
        corner_mappings['bottom'] = corner_mappings['lower_edge']
        corner_mappings['right'] = corner_mappings['right_edge']
        corner_mappings['r'] = corner_mappings['right_edge']
        corner_mappings['left'] = corner_mappings['left_edge']
        corner_mappings['l'] = corner_mappings['left_edge']

        if isinstance(position, (int, str, bool)):
            if isinstance(position, str):
                position = corner_mappings[position]
            elif position is True:
                position = corner_mappings['upper_left']
            self.textActor = vtk.vtkCornerAnnotation()
            # This is how you set the font size with this actor
            self.textActor.SetLinearFontScaleFactor(font_size // 2)
            self.textActor.SetText(position, text)
        else:
            self.textActor = vtk.vtkTextActor()
            self.textActor.SetInput(text)
            self.textActor.SetPosition(position)
            if viewport:
                self.textActor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                self.textActor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            self.textActor.GetTextProperty().SetFontSize(int(font_size * 2))

        self.textActor.GetTextProperty().SetColor(parse_color(color))
        self.textActor.GetTextProperty().SetFontFamily(FONT_KEYS[font])
        self.textActor.GetTextProperty().SetShadow(shadow)

        self.add_actor(self.textActor, reset_camera=False, name=name, loc=loc, pickable=False)
        return self.textActor

    def open_movie(self, filename, framerate=24):
        """
        Establishes a connection to the ffmpeg writer

        Parameters
        ----------
        filename : str
            Filename of the movie to open.  Filename should end in mp4,
            but other filetypes may be supported.  See "imagio.get_writer"

        framerate : int, optional
            Frames per second.

        """
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        self.mwriter = imageio.get_writer(filename, fps=framerate)

    def open_gif(self, filename):
        """
        Open a gif file.

        Parameters
        ----------
        filename : str
            Filename of the gif to open.  Filename must end in gif.

        """
        if filename[-3:] != 'gif':
            raise Exception('Unsupported filetype.  Must end in .gif')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        self._gif_filename = os.path.abspath(filename)
        self.mwriter = imageio.get_writer(filename, mode='I')

    def write_frame(self):
        """ Writes a single frame to the movie file """
        if not hasattr(self, 'mwriter'):
            raise AssertionError('This plotter has not opened a movie or GIF file.')
        self.mwriter.append_data(self.image)

    @property
    def window_size(self):
        """ returns render window size """
        return list(self.ren_win.GetSize())


    @window_size.setter
    def window_size(self, window_size):
        """ set the render window size """
        self.ren_win.SetSize(window_size[0], window_size[1])

    def _run_image_filter(self, ifilter):
        # Update filter and grab pixels
        ifilter.Modified()
        ifilter.Update()
        image = pyvista.wrap(ifilter.GetOutput())
        img_size = image.dimensions
        img_array = pyvista.utilities.point_scalar(image, 'ImageScalars')

        # Reshape and write
        tgt_size = (img_size[1], img_size[0], -1)
        return img_array.reshape(tgt_size)[::-1]


    def get_image_depth(self,
                        fill_value=np.nan,
                        reset_camera_clipping_range=True):
        """ Returns a depth image representing current render window

        Parameters
        ----------
        fill_value : float
            Fill value for points in image that don't include objects in scene.
            To not use a fill value, pass ``None``.

        reset_camera_clipping_range : bool
            Reset the camera clipping range to include data in view?

        Returns
        -------
        image_depth : numpy.ndarray
            Image of depth values from camera orthogonal to image plane

        Notes
        -----
        Values in image_depth are negative to adhere to a
        right-handed coordinate system.

        """
        if not hasattr(self, 'ren_win') and hasattr(self, 'last_image_depth'):
            zval = self.last_image_depth.copy()
            if fill_value is not None:
                zval[self._image_depth_null] = fill_value
            return zval

        # Ensure points in view are within clipping range of renderer?
        if reset_camera_clipping_range:
            self.renderer.ResetCameraClippingRange()

        # Get the z-buffer image
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.ren_win)
        ifilter.ReadFrontBufferOff()
        ifilter.SetInputBufferTypeToZBuffer()
        zbuff = self._run_image_filter(ifilter)[:, :, 0]

        # Convert z-buffer values to depth from camera
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            near, far = self.camera.GetClippingRange()
            if self.camera.GetParallelProjection():
                zval = (zbuff - near) / (far - near)
            else:
                zval = 2 * near * far / ((zbuff - 0.5) * 2 * (far - near) - near - far)

            # Consider image values outside clipping range as nans
            args = np.logical_or(zval < -far, np.isclose(zval, -far))
        self._image_depth_null = args
        if fill_value is not None:
            zval[args] = fill_value

        return zval


    @property
    def image_depth(self):
        """Helper attribute for ``get_image_depth``"""
        return self.get_image_depth()


    @property
    def image(self):
        """ Returns an image array of current render window """
        if not hasattr(self, 'ren_win') and hasattr(self, 'last_image'):
            return self.last_image
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.ren_win)
        ifilter.ReadFrontBufferOff()
        if self.image_transparent_background:
            ifilter.SetInputBufferTypeToRGBA()
        else:
            ifilter.SetInputBufferTypeToRGB()
        return self._run_image_filter(ifilter)

    def enable_eye_dome_lighting(self):
        """Enable eye dome lighting (EDL) for active renderer"""
        return self.renderer.enable_eye_dome_lighting()

    def disable_eye_dome_lighting(self):
        """Disable eye dome lighting (EDL) for active renderer"""
        return self.renderer.disable_eye_dome_lighting()

    def add_lines(self, lines, color=(1, 1, 1), width=5, label=None, name=None):
        """
        Adds lines to the plotting object.

        Parameters
        ----------
        lines : np.ndarray or pyvista.PolyData
            Points representing line segments.  For example, two line segments
            would be represented as:

            np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:
                color='white'
                color='w'
                color=[1, 1, 1]
                color='#FFFFFF'

        width : float, optional
            Thickness of lines

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        Returns
        -------
        actor : vtk.vtkActor
            Lines actor.

        """
        if not isinstance(lines, np.ndarray):
            raise Exception('Input should be an array of point segments')

        lines = pyvista.lines_from_points(lines)

        # Create mapper and add lines
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(lines)

        rgb_color = parse_color(color)

        # legend label
        if label:
            if not isinstance(label, str):
                raise AssertionError('Label must be a string')
            self._labels.append([lines, label, rgb_color])

        # Create actor
        self.scalar_bar = vtk.vtkActor()
        self.scalar_bar.SetMapper(mapper)
        self.scalar_bar.GetProperty().SetLineWidth(width)
        self.scalar_bar.GetProperty().EdgeVisibilityOn()
        self.scalar_bar.GetProperty().SetEdgeColor(rgb_color)
        self.scalar_bar.GetProperty().SetColor(rgb_color)
        self.scalar_bar.GetProperty().LightingOff()

        # Add to renderer
        self.add_actor(self.scalar_bar, reset_camera=False, name=name, pickable=False)
        return self.scalar_bar

    def remove_scalar_bar(self):
        """ Removes scalar bar """
        if hasattr(self, 'scalar_bar'):
            self.remove_actor(self.scalar_bar, reset_camera=False)


    def add_point_labels(self, points, labels, italic=False, bold=True,
                         font_size=None, text_color=None,
                         font_family=None, shadow=False,
                         show_points=True, point_color=None, point_size=5,
                         name=None, shape_color='grey', shape='rounded_rect',
                         fill_shape=True, margin=3, shape_opacity=1.0,
                         pickable=False, render_points_as_spheres=False,
                         tolerance=0.001, **kwargs):
        """
        Creates a point actor with one label from list labels assigned to
        each point.

        Parameters
        ----------
        points : np.ndarray or pyvista.Common
            n x 3 numpy array of points or pyvista dataset with points

        labels : list or str
            List of labels.  Must be the same length as points. If a string name
            is given with a pyvista.Common input for points, then these are fetched.

        italic : bool, optional
            Italicises title and bar labels.  Default False.

        bold : bool, optional
            Bolds title and bar labels.  Default True

        font_size : float, optional
            Sets the size of the title font.  Defaults to 16.

        text_color : string or 3 item list, optional
            Color of text. Either a string, rgb list, or hex color string.

                text_color='white'
                text_color='w'
                text_color=[1, 1, 1]
                text_color='#FFFFFF'

        font_family : string, optional
            Font family.  Must be either courier, times, or arial.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        show_points : bool, optional
            Controls if points are visible.  Default True

        point_color : string or 3 item list, optional. Color of points (if visible).
            Either a string, rgb list, or hex color string.  For example:

                text_color='white'
                text_color='w'
                text_color=[1, 1, 1]
                text_color='#FFFFFF'

        point_size : float, optional
            Size of points (if visible)

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.


        shape_color : string or 3 item list, optional. Color of points (if visible).
            Either a string, rgb list, or hex color string.  For example:

        shape : str, optional
            The string name of the shape to use. Options are ``'rect'`` or
            ``'rounded_rect'``. If you want no shape, pass ``None``

        fill_shape : bool, optional
            Fill the shape with the ``shape_color``. Outlines if ``False``.

        margin : int, optional
            The size of the margin on the label background shape. Default is 3.

        shape_opacity : flaot
            The opacity of the shape between zero and one.

        tolerance : float
            a tolerance to use to determine whether a point label is visible.
            A tolerance is usually required because the conversion from world
            space to display space during rendering introduces numerical
            round-off.

        Returns
        -------
        labelMapper : vtk.vtkvtkLabeledDataMapper
            VTK label mapper.  Can be used to change properties of the labels.

        """
        if font_family is None:
            font_family = rcParams['font']['family']
        if font_size is None:
            font_size = rcParams['font']['size']
        if point_color is None and text_color is None and kwargs.get('color', None) is not None:
            point_color = kwargs.get('color', None)
            text_color = kwargs.get('color', None)
        if point_color is None:
            point_color = rcParams['color']
        if text_color is None:
            text_color = rcParams['font']['color']

        if isinstance(points, (list, tuple)):
            points = np.array(points)

        if isinstance(points, np.ndarray):
            vtkpoints = pyvista.PolyData(points) # Cast to poly data
        elif is_pyvista_dataset(points):
            vtkpoints = pyvista.PolyData(points.points)
            if isinstance(labels, str):
                labels = points.point_arrays[labels].astype(str)
        else:
            raise TypeError('Points type not useable: {}'.format(type(points)))

        if len(vtkpoints.points) != len(labels):
            raise Exception('There must be one label for each point')

        if name is None:
            name = '{}({})'.format(type(vtkpoints).__name__, str(hex(id(vtkpoints))))

        vtklabels = vtk.vtkStringArray()
        vtklabels.SetName('labels')
        for item in labels:
            vtklabels.InsertNextValue(str(item))
        vtkpoints.GetPointData().AddArray(vtklabels)

        # Only show visible points
        vis_points = vtk.vtkSelectVisiblePoints()
        vis_points.SetInputData(vtkpoints)
        vis_points.SetRenderer(self.renderer)
        vis_points.SetTolerance(tolerance)

        # Create heirarchy
        hier = vtk.vtkPointSetToLabelHierarchy()
        hier.SetInputConnection(vis_points.GetOutputPort())
        hier.SetLabelArrayName('labels')

        # create label mapper
        labelMapper = vtk.vtkLabelPlacementMapper()
        labelMapper.SetInputConnection(hier.GetOutputPort())
        if not isinstance(shape, str):
            labelMapper.SetShapeToNone()
        elif shape.lower() in 'rect':
            labelMapper.SetShapeToRect()
        elif shape.lower() in 'rounded_rect':
            labelMapper.SetShapeToRoundedRect()
        else:
            raise RuntimeError('Shape ({}) not understood'.format(shape))
        if fill_shape:
            labelMapper.SetStyleToFilled()
        else:
            labelMapper.SetStyleToOutline()
        labelMapper.SetBackgroundColor(parse_color(shape_color))
        labelMapper.SetBackgroundOpacity(shape_opacity)
        labelMapper.SetMargin(margin)

        textprop = hier.GetTextProperty()
        textprop.SetItalic(italic)
        textprop.SetBold(bold)
        textprop.SetFontSize(font_size)
        textprop.SetFontFamily(parse_font_family(font_family))
        textprop.SetColor(parse_color(text_color))
        textprop.SetShadow(shadow)

        self.remove_actor('{}-points'.format(name), reset_camera=False)
        self.remove_actor('{}-labels'.format(name), reset_camera=False)

        # add points
        if show_points:
            style = 'points'
        else:
            style = 'surface'
        self.add_mesh(vtkpoints, style=style, color=point_color,
                      point_size=point_size, name='{}-points'.format(name),
                      pickable=pickable,
                      render_points_as_spheres=render_points_as_spheres)

        labelActor = vtk.vtkActor2D()
        labelActor.SetMapper(labelMapper)
        self.add_actor(labelActor, reset_camera=False,
                       name='{}-labels'.format(name), pickable=False)

        return labelMapper


    def add_point_scalar_labels(self, points, labels, fmt=None, preamble='', **kwargs):
        """Wrapper for :func:`pyvista.BasePlotter.add_point_labels` that will label
        points from a dataset with their scalar values.

        Parameters
        ----------
        points : np.ndarray or pyvista.Common
            n x 3 numpy array of points or pyvista dataset with points

        labels : str
            String name of the point data array to use.

        fmt : str
            String formatter used to format numerical data
        """
        if not is_pyvista_dataset(points):
            raise TypeError('input points must be a pyvista dataset, not: {}'.format(type(points)))
        if not isinstance(labels, str):
            raise TypeError('labels must be a string name of the scalar array to use')
        if fmt is None:
            fmt = rcParams['font']['fmt']
        if fmt is None:
            fmt = '%.6e'
        scalars = points.point_arrays[labels]
        phrase = '{} {}'.format(preamble, '%.3e')
        labels = [phrase % val for val in scalars]
        return self.add_point_labels(points, labels, **kwargs)


    def add_points(self, points, **kwargs):
        """ Add points to a mesh """
        kwargs['style'] = 'points'
        self.add_mesh(points, **kwargs)

    def add_arrows(self, cent, direction, mag=1, **kwargs):
        """ Adds arrows to plotting object """
        direction = direction.copy()
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        direction[:,0] *= mag
        direction[:,1] *= mag
        direction[:,2] *= mag

        pdata = pyvista.vector_poly_data(cent, direction)
        # Create arrow object
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = wrap(glyph3D.GetOutput())

        return self.add_mesh(arrows, **kwargs)


    @staticmethod
    def _save_image(image, filename, return_img=None):
        """Internal helper for saving a NumPy image array"""
        if not image.size:
            raise Exception('Empty image.  Have you run plot() first?')
        # write screenshot to file
        supported_formats = [".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"]
        if isinstance(filename, str):
            if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
                filename = os.path.join(pyvista.FIGURE_PATH, filename)
            if not any([filename.lower().endswith(ext) for ext in supported_formats]):
                filename += ".png"
            filename = os.path.abspath(os.path.expanduser(filename))
            w = imageio.imwrite(filename, image)
            if not return_img:
                return w
        return image


    def save_graphic(self, filename, title='PyVista Export', raster=True, painter=True):
        """Save a screenshot of the rendering window as a graphic file:
        '.svg', '.eps', '.ps', '.pdf', '.tex'
        """
        if not hasattr(self, 'ren_win'):
            raise AttributeError('This plotter is closed and unable to save a screenshot.')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        filename = os.path.abspath(os.path.expanduser(filename))
        extension = pyvista.fileio.get_ext(filename)
        valid = ['.svg', '.eps', '.ps', '.pdf', '.tex']
        if extension not in valid:
            raise RuntimeError('Extension ({}) is an invalid choice. Valid options include: {}'.format(extension, ', '.join(valid)))
        writer = vtk.vtkGL2PSExporter()
        modes = {
            '.svg': writer.SetFileFormatToSVG,
            '.eps': writer.SetFileFormatToEPS,
            '.ps': writer.SetFileFormatToPS,
            '.pdf': writer.SetFileFormatToPDF,
            '.tex': writer.SetFileFormatToTeX,
        }
        writer.CompressOff()
        writer.SetFilePrefix(filename.replace(extension, ''))
        writer.SetInput(self.ren_win)
        modes[extension]()
        writer.SetTitle(title)
        writer.SetWrite3DPropsAsRasterImage(raster)
        if painter:
            writer.UsePainterSettings()
        writer.Update()
        return


    def screenshot(self, filename=None, transparent_background=None,
                   return_img=None, window_size=None):
        """
        Takes screenshot at current camera position

        Parameters
        ----------
        filename : str, optional
            Location to write image to.  If None, no image is written.

        transparent_background : bool, optional
            Makes the background transparent.  Default False.

        return_img : bool, optional
            If a string filename is given and this is true, a NumPy array of
            the image will be returned.

        Returns
        -------
        img :  numpy.ndarray
            Array containing pixel RGB and alpha.  Sized:
            [Window height x Window width x 3] for transparent_background=False
            [Window height x Window width x 4] for transparent_background=True

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(sphere)
        >>> plotter.screenshot('screenshot.png') # doctest:+SKIP
        """
        if window_size is not None:
            self.window_size = window_size

        # configure image filter
        if transparent_background is None:
            transparent_background = rcParams['transparent_background']
        self.image_transparent_background = transparent_background

        # This if statement allows you to save screenshots of closed plotters
        # This is needed for the sphinx-gallery work
        if not hasattr(self, 'ren_win'):
            # If plotter has been closed...
            # check if last_image exists
            if hasattr(self, 'last_image'):
                # Save last image
                return self._save_image(self.last_image, filename, return_img)
            # Plotter hasn't been rendered or was improperly closed
            raise AttributeError('This plotter is closed and unable to save a screenshot.')

        if isinstance(self, Plotter):
            # TODO: we need a consistent rendering function
            self.render()
        else:
            self._render()

        # debug: this needs to be called twice for some reason,
        img = self.image
        img = self.image

        return self._save_image(img, filename, return_img)

    def add_legend(self, labels=None, bcolor=(0.5, 0.5, 0.5), border=False,
                   size=None, name=None):
        """
        Adds a legend to render window.  Entries must be a list
        containing one string and color entry for each item.

        Parameters
        ----------
        labels : list, optional
            When set to None, uses existing labels as specified by

            - add_mesh
            - add_lines
            - add_points

            List contianing one entry for each item to be added to the
            legend.  Each entry must contain two strings, [label,
            color], where label is the name of the item to add, and
            color is the color of the label to add.

        bcolor : list or string, optional
            Background color, either a three item 0 to 1 RGB color
            list, or a matplotlib color string (e.g. 'w' or 'white'
            for a white color).  If None, legend background is
            disabled.

        border : bool, optional
            Controls if there will be a border around the legend.
            Default False.

        size : list, optional
            Two float list, each float between 0 and 1.  For example
            [0.1, 0.1] would make the legend 10% the size of the
            entire figure window.

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        Returns
        -------
        legend : vtk.vtkLegendBoxActor
            Actor for the legend.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> othermesh = examples.load_uniform()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(mesh, label='My Mesh')
        >>> _ = plotter.add_mesh(othermesh, 'k', label='My Other Mesh')
        >>> _ = plotter.add_legend()
        >>> plotter.show() # doctest:+SKIP

        Alternative manual example

        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> othermesh = examples.load_uniform()
        >>> legend_entries = []
        >>> legend_entries.append(['My Mesh', 'w'])
        >>> legend_entries.append(['My Other Mesh', 'k'])
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(mesh)
        >>> _ = plotter.add_mesh(othermesh, 'k')
        >>> _ = plotter.add_legend(legend_entries)
        >>> plotter.show() # doctest:+SKIP
        """
        self.legend = vtk.vtkLegendBoxActor()

        if labels is None:
            # use existing labels
            if not self._labels:
                raise Exception('No labels input.\n\n'
                                'Add labels to individual items when adding them to'
                                'the plotting object with the "label=" parameter.  '
                                'or enter them as the "labels" parameter.')

            self.legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels):
                self.legend.SetEntry(i, vtk_object, text, parse_color(color))

        else:
            self.legend.SetNumberOfEntries(len(labels))
            legendface = pyvista.single_triangle()
            for i, (text, color) in enumerate(labels):
                self.legend.SetEntry(i, legendface, text, parse_color(color))

        if size:
            self.legend.SetPosition2(size[0], size[1])

        if bcolor is None:
            self.legend.UseBackgroundOff()
        else:
            self.legend.UseBackgroundOn()
            self.legend.SetBackgroundColor(bcolor)

        if border:
            self.legend.BorderOn()
        else:
            self.legend.BorderOff()

        # Add to renderer
        self.add_actor(self.legend, reset_camera=False, name=name, pickable=False)
        return self.legend

    @property
    def camera_position(self):
        """ Returns camera position of the active render window """
        return self.renderers[self._active_renderer_index].camera_position

    @camera_position.setter
    def camera_position(self, camera_location):
        """ Set camera position of the active render window """
        self.renderers[self._active_renderer_index].camera_position = camera_location

    def reset_camera(self):
        """
        Reset camera so it slides along the vector defined from camera
        position to focal point until all of the actors can be seen.
        """
        self.renderers[self._active_renderer_index].reset_camera()
        self._render()

    def isometric_view(self):
        """DEPRECATED: Please use ``view_isometric``"""
        return self.view_isometric()

    def view_isometric(self, negative=False):
        """
        Resets the camera to a default isometric view showing all the
        actors in the scene.
        """
        return self.renderer.view_isometric(negative=negative)

    def view_vector(self, vector, viewup=None):
        return self.renderer.view_vector(vector, viewup=viewup)

    def view_xy(self, negative=False):
        """View the XY plane"""
        return self.renderer.view_xy(negative=negative)

    def view_yx(self, negative=False):
        """View the YX plane"""
        return self.renderer.view_yx(negative=negative)

    def view_xz(self, negative=False):
        """View the XZ plane"""
        return self.renderer.view_xz(negative=negative)

    def view_zx(self, negative=False):
        """View the ZX plane"""
        return self.renderer.view_zx(negative=negative)

    def view_yz(self, negative=False):
        """View the YZ plane"""
        return self.renderer.view_yz(negative=negative)

    def view_zy(self, negative=False):
        """View the ZY plane"""
        return self.renderer.view_zy(negative=negative)

    def disable(self):
        """Disable this renderer's camera from being interactive"""
        return self.renderer.disable()

    def enable(self):
        """Enable this renderer's camera to be interactive"""
        return self.renderer.enable()

    def set_background(self, color, loc='all', top=None):
        """
        Sets background color

        Parameters
        ----------
        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:
                color='white'
                color='w'
                color=[1, 1, 1]
                color='#FFFFFF'

        loc : int, tuple, list, or str, optional
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.  If ``loc='all'`` then all
            render windows will have their background set.

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

        if loc == 'all':
            for renderer in self.renderers:
                renderer.SetBackground(parse_color(color))
                if use_gradient:
                    renderer.GradientBackgroundOn()
                    renderer.SetBackground2(parse_color(top))
                else:
                    renderer.GradientBackgroundOff()
        else:
            renderer = self.renderers[self.loc_to_index(loc)]
            renderer.SetBackground(parse_color(color))
            if use_gradient:
                renderer.GradientBackgroundOn()
                renderer.SetBackground2(parse_color(top))
            else:
                renderer.GradientBackgroundOff()

    @property
    def background_color(self):
        """ Returns background color of the first render window """
        return self.renderers[0].GetBackground()

    @background_color.setter
    def background_color(self, color):
        """ Sets the background color of all the render windows """
        self.set_background(color)

    def remove_legend(self):
        """ Removes legend actor """
        if hasattr(self, 'legend'):
            self.remove_actor(self.legend, reset_camera=False)
            self._render()


    def generate_orbital_path(self, factor=3., n_points=20, viewup=None, shift=0.0):
        """Genrates an orbital path around the data scene

        Parameters
        ----------
        factor : float
            A scaling factor when biulding the orbital extent

        n_points : int
            number of points on the orbital path

        viewup : list(float)
            the normal to the orbital plane

        shift : float, optional
            shift the plane up/down from the center of the scene by this amount
        """
        if viewup is None:
            viewup = rcParams['camera']['viewup']
        center = np.array(self.center)
        bnds = np.array(self.bounds)
        radius = (bnds[1] - bnds[0]) * factor
        y = (bnds[3] - bnds[2]) * factor
        if y > radius:
            radius = y
        center += np.array(viewup) * shift
        return pyvista.Polygon(center=center, radius=radius, normal=viewup, n_sides=n_points)


    def fly_to(self, point):
        """Given a position point, move the current camera's focal point to that
        point. The movement is animated over the number of frames specified in
        NumberOfFlyFrames. The LOD desired frame rate is used.
        """
        if not hasattr(self, 'iren'):
            raise AttributeError('This plotter does not have an interactive window')
        return self.iren.FlyTo(self.renderer, *point)


    def orbit_on_path(self, path=None, focus=None, step=0.5, viewup=None,
                      bkg=True, write_frames=False):
        """Orbit on the given path focusing on the focus point

        Parameters
        ----------
        path : pyvista.PolyData
            Path of orbital points. The order in the points is the order of
            travel

        focus : list(float) of length 3, optional
            The point ot focus the camera.

        step : float, optional
            The timestep between flying to each camera position

        viewup : list(float)
            the normal to the orbital plane

        write_frames : bool
            Assume a file is open and write a frame on each camera view during
            the orbit.
        """
        if focus is None:
            focus = self.center
        if viewup is None:
            viewup = rcParams['camera']['viewup']
        if path is None:
            path = self.generate_orbital_path(viewup=viewup)
        if not is_pyvista_dataset(path):
            path = pyvista.PolyData(path)
        points = path.points

        # Make sure the whole scene is visible
        self.camera.SetThickness(path.length)

        def orbit():
            """Internal thread for running the orbit"""
            for point in points:
                self.set_position(point)
                self.set_focus(focus)
                self.set_viewup(viewup)
                if bkg:
                    time.sleep(step)
                if write_frames:
                    self.write_frame()


        if bkg and isinstance(self, pyvista.BackgroundPlotter):
            thread = Thread(target=orbit)
            thread.start()
        else:
            bkg = False
            orbit()
        return


    def export_vtkjs(self, filename, compress_arrays=False):
        """
        Export the current rendering scene as a VTKjs scene for
        rendering in a web browser
        """
        if not hasattr(self, 'ren_win'):
            raise RuntimeError('Export must be called before showing/closing the scene.')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))
        return export_plotter_vtkjs(self, filename, compress_arrays=compress_arrays)



    def export_obj(self, filename):
        """Export scene to OBJ format"""
        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter must still have a render window open.")
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))
        exporter = vtk.vtkOBJExporter()
        exporter.SetFilePrefix(filename)
        exporter.SetRenderWindow(self.ren_win)
        return exporter.Write()


    def __del__(self):
        self.close()
        self.deep_clean()
        del self.renderers


class Plotter(BasePlotter):
    """ Plotting object to display vtk meshes or numpy arrays.

    Example
    -------
    >>> import pyvista
    >>> from pyvista import examples
    >>> mesh = examples.load_hexbeam()
    >>> another_mesh = examples.load_uniform()
    >>> plotter = pyvista.Plotter()
    >>> _ = plotter.add_mesh(mesh, color='red')
    >>> _ = plotter.add_mesh(another_mesh, color='blue')
    >>> plotter.show() # doctest:+SKIP

    Parameters
    ----------
    off_screen : bool, optional
        Renders off screen when False.  Useful for automated screenshots.

    notebook : bool, optional
        When True, the resulting plot is placed inline a jupyter notebook.
        Assumes a jupyter console is active.  Automatically enables off_screen.

    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one render window.
        Can also accept a shape as string descriptor. E.g.:
            shape="3|1" means 3 plots on the left and 1 on the right,
            shape="4/2" means 4 plots on top of 2 at bottom.

    border : bool, optional
        Draw a border around each render window.  Default False.

    border_color : string or 3 item list, optional, defaults to white
        Either a string, rgb list, or hex color string.  For example:
            color='white'
            color='w'
            color=[1, 1, 1]
            color='#FFFFFF'

    window_size : list, optional
        Window size in pixels.  Defaults to [1024, 768]

    multi_samples : int
        The number of multi-samples used to mitigate aliasing. 4 is a good
        default but 8 will have better results with a potential impact on
        perfromance.

    line_smoothing : bool
        If True, enable line smothing

    point_smoothing : bool
        If True, enable point smothing

    polygon_smoothing : bool
        If True, enable polygon smothing

    """
    last_update_time = 0.0
    q_pressed = False
    right_timer_id = -1

    def __init__(self, off_screen=None, notebook=None, shape=(1, 1),
                 border=None, border_color='k', border_width=2.0,
                 window_size=None, multi_samples=None, line_smoothing=False,
                 point_smoothing=False, polygon_smoothing=False,
                 splitting_position=None, title=None):
        """
        Initialize a vtk plotting object
        """
        super(Plotter, self).__init__(shape=shape, border=border,
                                      border_color=border_color,
                                      border_width=border_width,
                                      splitting_position=splitting_position,
                                      title=title)
        log.debug('Initializing')

        def on_timer(iren, event_id):
            """ Exit application if interactive renderer stops """
            if event_id == 'TimerEvent':
                self.iren.TerminateApp()

        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN

        if notebook is None:
            notebook = scooby.in_ipykernel()

        self.notebook = notebook
        if self.notebook:
            off_screen = True
        self.off_screen = off_screen

        if window_size is None:
            window_size = rcParams['window_size']

        if multi_samples is None:
            multi_samples = rcParams['multi_samples']

        # initialize render window
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetMultiSamples(multi_samples)
        self.ren_win.SetBorders(True)
        if line_smoothing:
            self.ren_win.LineSmoothingOn()
        if point_smoothing:
            self.ren_win.PointSmoothingOn()
        if polygon_smoothing:
            self.ren_win.PolygonSmoothingOn()

        for renderer in self.renderers:
            self.ren_win.AddRenderer(renderer)

        if self.off_screen:
            self.ren_win.SetOffScreenRendering(1)
        else:  # Allow user to interact
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.LightFollowCameraOff()
            self.iren.SetDesiredUpdateRate(30.0)
            self.iren.SetRenderWindow(self.ren_win)
            self.enable_trackball_style()
            self.iren.AddObserver("KeyPressEvent", self.key_press_event)
            self.update_style()

            # for renderer in self.renderers:
            #     self.iren.SetRenderWindow(renderer)

        # Set background
        self.set_background(rcParams['background'])

        # Set window size
        self.window_size = window_size

        # add timer event if interactive render exists
        if hasattr(self, 'iren'):
            self.iren.AddObserver(vtk.vtkCommand.TimerEvent, on_timer)

    def show(self, title=None, window_size=None, interactive=True,
             auto_close=None, interactive_update=False, full_screen=False,
             screenshot=False, return_img=False, use_panel=None, cpos=None,
             height=400):
        """
        Creates plotting window

        Parameters
        ----------
        title : string, optional
            Title of plotting window.

        window_size : list, optional
            Window size in pixels.  Defaults to [1024, 768]

        interactive : bool, optional
            Enabled by default.  Allows user to pan and move figure.

        auto_close : bool, optional
            Enabled by default.  Exits plotting session when user
            closes the window when interactive is True.

        interactive_update: bool, optional
            Disabled by default.  Allows user to non-blocking draw,
            user should call Update() in each iteration.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores
            window_size.  Default False.

        use_panel : bool, optional
            If False, the interactive rendering from panel will not be used in
            notebooks

        cpos : list(tuple(floats))
            The camera position to use

        height : int, optional
            height for panel pane. Only used with panel.

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up

        """
        if use_panel is None:
            use_panel = rcParams['use_panel']

        if auto_close is None:
            auto_close = rcParams['auto_close']

        # reset unless camera for the first render unless camera is set
        if self._first_time:  # and not self.camera_set:
            for renderer in self.renderers:
                if not renderer.camera_set and cpos is None:
                    renderer.camera_position = renderer.get_default_cam_pos()
                    renderer.ResetCamera()
                elif cpos is not None:
                    renderer.camera_position = cpos
            self._first_time = False


        # if full_screen:
        if full_screen:
            self.ren_win.SetFullScreen(True)
            self.ren_win.BordersOn()  # super buggy when disabled
        else:
            if window_size is None:
                window_size = self.window_size
            self.ren_win.SetSize(window_size[0], window_size[1])

        # Render
        log.debug('Rendering')
        self.ren_win.Render()

        # This has to be after the first render for some reason
        if title is None:
            title = self.title
        if title:
            self.ren_win.SetWindowName(title)
            self.title = title

        # Keep track of image for sphinx-gallery
        self.last_image = self.screenshot(screenshot, return_img=True)
        self.last_image_depth = self.get_image_depth()
        disp = None

        if interactive and (not self.off_screen):
            try:  # interrupts will be caught here
                log.debug('Starting iren')
                self.update_style()
                self.iren.Initialize()
                if not interactive_update:
                    self.iren.Start()
            except KeyboardInterrupt:
                log.debug('KeyboardInterrupt')
                self.close()
                raise KeyboardInterrupt
        elif self.notebook and use_panel and not hasattr(self, 'volume'):
            try:
                from panel.pane import VTK as panel_display
                disp = panel_display(self.ren_win, sizing_mode='stretch_width',
                                     height=height)
            except:
                pass
        # NOTE: after this point, nothing from the render window can be accessed
        #       as if a user presed the close button, then it destroys the
        #       the render view and a stream of errors will kill the Python
        #       kernel if code here tries to access that renderer.
        #       See issues #135 and #186 for insight before editing the
        #       remainder of this function.

        # Get camera position before closing
        cpos = self.camera_position

        # NOTE: our conversion to panel currently does not support mult-view
        #       so we should display the static screenshot in notebooks for
        #       multi-view plots until we implement this feature
        # If notebook is true and panel display failed:
        if self.notebook and (disp is None or self.shape != (1,1)):
            import PIL.Image
            # sanity check
            try:
                import IPython
            except ImportError:
                raise Exception('Install IPython to display image in a notebook')
            disp = IPython.display.display(PIL.Image.fromarray(self.last_image))

        # Cleanup
        if auto_close:
            self.close()

        # Return the notebook display: either panel object or image display
        if self.notebook:
            return disp

        # If user asked for screenshot, return as numpy array after camera
        # position
        if return_img or screenshot is True:
            return cpos, self.last_image

        # default to returning last used camera position
        return cpos

    def plot(self, *args, **kwargs):
        """ Present for backwards compatibility. Use `show()` instead """
        logging.warning("`.plot()` is deprecated. Please use `.show()` instead.")
        return self.show(*args, **kwargs)

    def render(self):
        """ renders main window """
        self.ren_win.Render()
