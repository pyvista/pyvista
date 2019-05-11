"""
pyvista plotting module
"""
import collections
import ctypes
import logging
import os
import time
from threading import Thread
from subprocess import PIPE, Popen

import imageio
import numpy as np
import vtk
from vtk.util import numpy_support as VN

import pyvista
from pyvista.export import export_plotter_vtkjs
from pyvista.utilities import (get_scalar, is_pyvista_obj, numpy_to_texture, wrap,
                            _raise_not_matching, convert_array)

_ALL_PLOTTERS = {}

def close_all():
    """Close all open/active plotters"""
    for key, p in _ALL_PLOTTERS.items():
        p.close()
    _ALL_PLOTTERS.clear()
    return True

MAX_N_COLOR_BARS = 10
PV_BACKGROUND = [82/255., 87/255., 110/255.]
FONT_KEYS = {'arial': vtk.VTK_ARIAL,
             'courier': vtk.VTK_COURIER,
             'times': vtk.VTK_TIMES}


log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


rcParams = {
    'background' : [0.3, 0.3, 0.3],
    'camera' : {
        'position' : [1, 1, 1],
        'viewup' : [0, 0, 1],
    },
    'window_size' : [1024, 768],
    'font' : {
        'family' : 'courier',
        'size' : 12,
        'title_size': None,
        'label_size' : None,
        'color' : [1, 1, 1],
        'fmt' : None,
    },
    'cmap' : 'jet',
    'color' : 'white',
    'nan_color' : 'darkgray',
    'edge_color' : 'black',
    'outline_color' : 'white',
    'colorbar_orientation' : 'horizontal',
    'colorbar_horizontal' : {
        'width' : 0.60,
        'height' : 0.08,
        'position_x' : 0.35,
        'position_y' : 0.02,
    },
    'colorbar_vertical' : {
        'width' : 0.1,
        'height' : 0.8,
        'position_x' : 0.85,
        'position_y' : 0.1,
    },
    'show_scalar_bar' : True,
    'show_edges' : False,
    'lighting' : True,
    'interactive' : False,
    'render_points_as_spheres' : False,
    'use_panel' : False, #True,
    'transparent_background' : False
}

DEFAULT_THEME = dict(rcParams)

def set_plot_theme(theme):
    """Set the plotting parameters to a predefined theme"""
    if theme.lower() in ['paraview', 'pv']:
        rcParams['background'] = PV_BACKGROUND
        rcParams['cmap'] = 'coolwarm'
        rcParams['font']['family'] = 'arial'
        rcParams['font']['label_size'] = 16
        rcParams['show_edges'] = False
    elif theme.lower() in ['document', 'doc', 'paper', 'report']:
        rcParams['background'] = 'white'
        rcParams['cmap'] = 'viridis'
        rcParams['font']['size'] = 18
        rcParams['font']['title_size'] = 18
        rcParams['font']['label_size'] = 18
        rcParams['font']['color'] = 'black'
        rcParams['show_edges'] = False
        rcParams['color'] = 'tan'
        rcParams['outline_color'] = 'black'
    elif theme.lower() in ['night', 'dark']:
        rcParams['background'] = 'black'
        rcParams['cmap'] = 'viridis'
        rcParams['font']['color'] = 'white'
        rcParams['show_edges'] = False
        rcParams['color'] = 'tan'
        rcParams['outline_color'] = 'white'
        rcParams['edge_color'] = 'white'
    elif theme.lower() in ['default']:
        for k,v in DEFAULT_THEME.items():
            rcParams[k] = v


def run_from_ipython():
    """ returns True when run from IPython """
    try:
        py = __IPYTHON__
        return True
    except NameError:
        return False



def opacity_transfer_function(key, n_colors):
    """Get the opacity transfer function results: range from 0 to 255
    """
    transfer_func = {
        'linear': np.linspace(0, 255, n_colors, dtype=np.uint8),
        'linear_r': np.linspace(0, 255, n_colors, dtype=np.uint8)[::-1],
        'geom': np.geomspace(1e-6, 255, n_colors, dtype=np.uint8),
        'geom_r': np.geomspace(255, 1e-6, n_colors, dtype=np.uint8),
    }
    try:
        return transfer_func[key]
    except KeyError:
        raise KeyError('opactiy transfer function ({}) unknown.'.format(key))


def plot(var_item, off_screen=None, full_screen=False, screenshot=None,
         interactive=True, cpos=None, window_size=None,
         show_bounds=False, show_axes=True, notebook=None, background=None,
         text='', return_img=False, eye_dome_lighting=False, use_panel=None,
         **kwargs):
    """
    Convenience plotting function for a vtk or numpy object.

    Parameters
    ----------
    item : vtk or numpy object
        VTK object or numpy array to be plotted.

    off_screen : bool
        Plots off screen when True.  Helpful for saving screenshots
        without a window popping up.

    full_screen : bool, optional
        Opens window in full screen.  When enabled, ignores window_size.
        Default False.

    screenshot : str or bool, optional
        Saves screenshot to file when enabled.  See:
        help(pyvistanterface.Plotter.screenshot).  Default disabled.

        When True, takes screenshot and returns numpy array of image.

    window_size : list, optional
        Window size in pixels.  Defaults to [1024, 768]

    show_bounds : bool, optional
        Shows mesh bounds when True.  Default False. Alias ``show_grid`` also
        accepted.

    notebook : bool, optional
        When True, the resulting plot is placed inline a jupyter notebook.
        Assumes a jupyter console is active.

    show_axes : bool, optional
        Shows a vtk axes widget.  Enabled by default.

    text : str, optional
        Adds text at the bottom of the plot.

    **kwargs : optional keyword arguments
        See help(Plotter.add_mesh) for additional options.

    Returns
    -------
    cpos : list
        List of camera position, focal point, and view up.

    img :  numpy.ndarray
        Array containing pixel RGB and alpha.  Sized:
        [Window height x Window width x 3] for transparent_background=False
        [Window height x Window width x 4] for transparent_background=True
        Returned only when screenshot enabled

    """
    if notebook is None:
        if run_from_ipython():
            try:
                notebook = type(get_ipython()).__module__.startswith('ipykernel.')
            except NameError:
                pass

    if notebook:
        off_screen = notebook
    plotter = Plotter(off_screen=off_screen, notebook=notebook)
    if show_axes:
        plotter.add_axes()

    plotter.set_background(background)

    if isinstance(var_item, list):
        if len(var_item) == 2:  # might be arrows
            isarr_0 = isinstance(var_item[0], np.ndarray)
            isarr_1 = isinstance(var_item[1], np.ndarray)
            if isarr_0 and isarr_1:
                plotter.add_arrows(var_item[0], var_item[1])
            else:
                for item in var_item:
                    plotter.add_mesh(item, **kwargs)
        else:
            for item in var_item:
                plotter.add_mesh(item, **kwargs)
    else:
        plotter.add_mesh(var_item, **kwargs)

    if text:
        plotter.add_text(text)

    if show_bounds or kwargs.get('show_grid', False):
        if kwargs.get('show_grid', False):
            plotter.show_grid()
        else:
            plotter.show_bounds()

    if cpos is None:
        cpos = plotter.get_default_cam_pos()
        plotter.camera_position = cpos
        plotter.camera_set = False
    else:
        plotter.camera_position = cpos

    if eye_dome_lighting:
        plotter.enable_eye_dome_lighting()

    result = plotter.show(window_size=window_size,
                          auto_close=False,
                          interactive=interactive,
                          full_screen=full_screen,
                          screenshot=screenshot,
                          return_img=return_img,
                          use_panel=use_panel)

    # close and return camera position and maybe image
    plotter.close()

    # Result will be handled by plotter.show(): cpos or [cpos, img]
    return result


def plot_arrows(cent, direction, **kwargs):
    """
    Plots arrows as vectors

    Parameters
    ----------
    cent : np.ndarray
        Accepts a single 3d point or array of 3d points.

    directions : np.ndarray
        Accepts a single 3d point or array of 3d vectors.
        Must contain the same number of items as cent.

    **kwargs : additional arguments, optional
        See help(pyvista.Plot)

    Returns
    -------
    Same as Plot.  See help(pyvista.Plot)

    """
    return plot([cent, direction], **kwargs)


def system_supports_plotting():
    """
    Check if x server is running

    Returns
    -------
    system_supports_plotting : bool
        True when on Linux and running an xserver.  Returns None when
        on a non-linux platform.

    """
    try:
        if os.environ['ALLOW_PLOTTING'].lower() == 'true':
            return True
    except KeyError:
        pass
    try:
        p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
        p.communicate()
        return p.returncode == 0
    except:
        return False


class BasePlotter(object):
    """
    To be used by the Plotter and QtInteractor classes.

    Parameters
    ----------
    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one renderer.

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
                 border_width=1.0):
        """ Initialize base plotter """
        self.image_transparent_background = rcParams['transparent_background']

        # by default add border for multiple plots
        if border is None:
            if shape != (1, 1):
                border = True
            else:
                border = False

        # add render windows
        self.renderers = []
        self._active_renderer_index = 0
        assert_str = '"shape" should be a list or tuple'
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
        self._actors = {}
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

    def add_axes(self, interactive=None, color=None):
        """ Add an interactive axes widget """
        if interactive is None:
            interactive = rcParams['interactive']
        if hasattr(self, 'axes_widget'):
            self.axes_widget.SetInteractive(interactive)
            self._update_axes_color(color)
            return
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        if hasattr(self, 'iren'):
            self.axes_widget.SetInteractor(self.iren)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.SetInteractive(interactive)
        # Set the color
        self._update_axes_color(color)

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

    def key_press_event(self, obj, event):
        """ Listens for key press event """
        key = self.iren.GetKeySym()
        log.debug('Key %s pressed' % key)
        if key == 'q':
            self.q_pressed = True
            # Grab screenshot right before renderer closes
            self.last_image = self.screenshot(True, return_img=True)
        elif key == 'b':
            self.observer = self.iren.AddObserver('LeftButtonPressEvent',
                                                  self.left_button_down)
        elif key == 'v':
            self.isometric_view_interactive()

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
                 rng=None, stitle=None, show_edges=None,
                 point_size=5.0, opacity=1.0, line_width=None,
                 flip_scalars=False, lighting=None, n_colors=256,
                 interpolate_before_map=False, cmap=None, label=None,
                 reset_camera=None, scalar_bar_args=None,
                 multi_colors=False, name=None, texture=None,
                 render_points_as_spheres=None,
                 render_lines_as_tubes=False, edge_color=None,
                 ambient=0.0, show_scalar_bar=None, nan_color=None,
                 nan_opacity=1.0, loc=None, backface_culling=False,
                 rgb=False, categories=False, **kwargs):
        """
        Adds a unstructured, structured, or surface mesh to the
        plotting object.

        Also accepts a 3D numpy.ndarray

        Parameters
        ----------
        mesh : vtk unstructured, structured, polymesh, or 3D numpy.ndarray
            A vtk unstructured, structured, or polymesh to plot.

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:
                color='white'
                color='w'
                color=[1, 1, 1]
                color='#FFFFFF'

            Color will be overridden when scalars are input.

        style : string, optional
            Visualization style of the vtk mesh.  One for the following:
                style='surface'
                style='wireframe'
                style='points'

            Defaults to 'surface'

        scalars : numpy array, optional
            Scalars used to "color" the mesh.  Accepts an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            color and scalars are None, then the active scalars are
            used

        rng : 2 item list, optional
            Range of mapper for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``clim``
            is also an accepted alias for this.

        stitle : string, optional
            Scalar title.  By default there is no scalar legend bar.
            Setting this creates the legend bar and adds a title to
            it.  To create a bar with no title, use an empty string
            (i.e. '').

        show_edges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        point_size : float, optional
            Point size.  Applicable when style='points'.  Default 5.0

        opacity : float, optional
            Opacity of mesh.  Should be between 0 and 1.  Default 1.0.
            A string option can also be specified to map the scalar range
            to the opacity. Options are: linear, linear_r, geom, geom_r

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default None.

        flip_scalars : bool, optional
            Flip direction of cmap.

        lighting : bool, optional
            Enable or disable view direction lighting.  Default False.

        n_colors : int, optional
            Number of colors to use when displaying scalars.  Default
            256.

        interpolate_before_map : bool, optional
            Enabling makes for a smoother scalar display.  Default
            False

        cmap : str, optional
           cmap string.  See available matplotlib cmaps.  Only
           applicable for when displaying scalars.  Defaults None
           (rainbow).  Requires matplotlib.

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

        ambient : float, optional
            When lighting is enabled, this is the amount of light from
            0 to 1 that reaches the actor when not directed at the
            light source emitted from the viewer.  Default 0.2.

        nan_color : string or 3 item list, optional, defaults to gray
            The color to use for all NaN values in the plotted scalar
            array.

        nan_opacity : float, optional
            Opacity of NaN values.  Should be between 0 and 1.
            Default 1.0

        backface_culling : bool optional
            Does not render faces that should not be visible to the
            plotter.  This can be helpful for dense surface meshes,
            especially when edges are visible, but can cause flat
            meshes to be partially displayed.  Default False.

        rgb : bool, optional
            If an 2 dimensional array is passed as the scalars, plot those
            values as RGB+A colors! ``rgba`` is also accepted alias for this.

        categories : bool, optional
            If fetching a colormap from matplotlib, this is the number of
            categories to use in that colormap. If set to ``True``, then
            the number of unique values in the scalar array will be used.

        Returns
        -------
        actor: vtk.vtkActor
            VTK actor of the mesh.
        """
        # fixes lighting issue when using precalculated normals
        if isinstance(mesh, vtk.vtkPolyData):
            if mesh.GetPointData().HasArray('Normals'):
                mesh.point_arrays['Normals'] = mesh.point_arrays.pop('Normals')

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if isinstance(mesh, np.ndarray):
            mesh = pyvista.PolyData(mesh)
            style = 'points'

        # Convert the VTK data object to a pyvista wrapped object if neccessary
        if not is_pyvista_obj(mesh):
            mesh = wrap(mesh)

        if show_edges is None:
            show_edges = rcParams['show_edges']

        if edge_color is None:
            edge_color = rcParams['edge_color']

        if show_scalar_bar is None:
            show_scalar_bar = rcParams['show_scalar_bar']

        if lighting is None:
            lighting = rcParams['lighting']

        if rng is None:
            rng = kwargs.get('clim', None)

        if render_points_as_spheres is None:
            render_points_as_spheres = rcParams['render_points_as_spheres']

        if name is None:
            name = '{}({})'.format(type(mesh).__name__, str(hex(id(mesh))))

        if isinstance(mesh, pyvista.MultiBlock):
            self.remove_actor(name, reset_camera=reset_camera)
            # frist check the scalars
            if rng is None and scalars is not None:
                # Get the data range across the array for all blocks
                # if scalar specified
                if isinstance(scalars, str):
                    rng = mesh.get_data_range(scalars)
                else:
                    # TODO: an array was given... how do we deal with
                    #       that? Possibly a 2D arrays or list of
                    #       arrays where first index corresponds to
                    #       the block? This could get complicated real
                    #       quick.
                    raise RuntimeError('Scalar array must be given as a string name for multiblock datasets.')
            if multi_colors:
                # Compute unique colors for each index of the block
                try:
                    import matplotlib as mpl
                    from itertools import cycle
                    cycler = mpl.rcParams['axes.prop_cycle']
                    colors = cycle(cycler)
                except ImportError:
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
                if not is_pyvista_obj(mesh[idx]):
                    data = wrap(mesh.GetBlock(idx))
                    if not is_pyvista_obj(mesh[idx]):
                        continue # move on if we can't plot it
                else:
                    data = mesh.GetBlock(idx)
                if data is None:
                    # Note that a block can exist but be None type
                    continue
                # Now check that scalars is available for this dataset
                if isinstance(data, vtk.vtkMultiBlockDataSet) or get_scalar(data, scalars) is None:
                    ts = None
                else:
                    ts = scalars
                if multi_colors:
                    color = next(colors)['color']
                a = self.add_mesh(data, color=color, style=style,
                                  scalars=ts, rng=rng, stitle=stitle,
                                  show_edges=show_edges,
                                  point_size=point_size, opacity=opacity,
                                  line_width=line_width,
                                  flip_scalars=flip_scalars,
                                  lighting=lighting, n_colors=n_colors,
                                  interpolate_before_map=interpolate_before_map,
                                  cmap=cmap, label=label,
                                  scalar_bar_args=scalar_bar_args,
                                  reset_camera=reset_camera, name=next_name,
                                  texture=None,
                                  render_points_as_spheres=render_points_as_spheres,
                                  render_lines_as_tubes=render_lines_as_tubes,
                                  edge_color=edge_color,
                                  show_scalar_bar=show_scalar_bar, nan_color=nan_color,
                                  nan_opacity=nan_opacity,
                                  loc=loc, rgb=rgb, **kwargs)
                actors.append(a)
                if (reset_camera is None and not self.camera_set) or reset_camera:
                    cpos = self.get_default_cam_pos()
                    self.camera_position = cpos
                    self.camera_set = False
                    self.reset_camera()
            return actors

        if nan_color is None:
            nan_color = rcParams['nan_color']
        nanr, nanb, nang = parse_color(nan_color)
        nan_color = nanr, nanb, nang, nan_opacity
        if color is True:
            color = rcParams['color']

        if mesh.n_points < 1:
            raise RuntimeError('Empty meshes cannot be plotted. Input mesh has zero points.')

        # set main values
        self.mesh = mesh
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(self.mesh)
        if isinstance(scalars, str):
            self.mapper.SetArrayName(scalars)

        actor, prop = self.add_actor(self.mapper,
                                     reset_camera=reset_camera,
                                     name=name, loc=loc, culling=backface_culling)

        # Try to plot something if no preference given
        if scalars is None and color is None and texture is None:
            # Prefer texture first
            if len(list(mesh.textures.keys())) > 0:
                texture = True
            # If no texture, plot any active scalar
            else:
                # Make sure scalar components are not vectors/tuples
                scalars = mesh.active_scalar
                # Don't allow plotting of string arrays by default
                if scalars is not None and np.issubdtype(scalars.dtype, np.number):
                    if stitle is None:
                        stitle = mesh.active_scalar_info[1]
                else:
                    scalars = None

        if texture == True or isinstance(texture, (str, int)):
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

        # Scalar formatting ===================================================
        if cmap is None: # grab alias for cmaps: colormap
            cmap = kwargs.get('colormap', None)
        if cmap is None: # Set default map if matplotlib is avaialble
            try:
                import matplotlib
                cmap = rcParams['cmap']
            except ImportError:
                pass
        title = 'Data' if stitle is None else stitle
        if scalars is not None:
            # if scalars is a string, then get the first array found with that name
            append_scalars = True
            if isinstance(scalars, str):
                title = scalars
                scalars = get_scalar(mesh, scalars,
                        preference=kwargs.get('preference', 'cell'), err=True)
                if stitle is None:
                    stitle = title
                #append_scalars = False

            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            if not np.issubdtype(scalars.dtype, np.number):
                raise TypeError('Non-numeric scalars are currently not supported for plotting.')

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

            # Scalar interpolation approach
            if scalars.shape[0] == mesh.n_points:
                self.mesh._add_point_scalar(scalars, title, append_scalars)
                self.mapper.SetScalarModeToUsePointData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
                if interpolate_before_map:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
            elif scalars.shape[0] == mesh.n_cells:
                self.mesh._add_cell_scalar(scalars, title, append_scalars)
                self.mapper.SetScalarModeToUseCellData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
                if interpolate_before_map:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
            else:
                _raise_not_matching(scalars, mesh)

            # Set scalar range
            if rng is None:
                rng = [np.nanmin(scalars), np.nanmax(scalars)]
            elif isinstance(rng, float) or isinstance(rng, int):
                rng = [-rng, rng]

            if np.any(rng) and not rgb:
                self.mapper.SetScalarRange(rng[0], rng[1])

            # Flip if requested
            table = self.mapper.GetLookupTable()
            table.SetNanColor(nan_color)
            if cmap is not None:
                try:
                    from matplotlib.cm import get_cmap
                except ImportError:
                    cmap = None
                    logging.warning('Please install matplotlib for color maps.')
            if cmap is not None:
                try:
                    from matplotlib.cm import get_cmap
                except ImportError:
                    raise Exception('cmap requires matplotlib')
                if isinstance(cmap, str):
                    if categories:
                        if categories is True:
                            categories = len(np.unique(scalars))
                        cmap = get_cmap(cmap, categories)
                    else:
                        cmap = get_cmap(cmap)
                    # ELSE: assume cmap is callable
                ctable = cmap(np.linspace(0, 1, n_colors))*255
                ctable = ctable.astype(np.uint8)
                # Set opactities
                if isinstance(opacity, str):
                    ctable[:,-1] = opacity_transfer_function(opacity, n_colors)
                if flip_scalars:
                    ctable = np.ascontiguousarray(ctable[::-1])
                table.SetTable(VN.numpy_to_vtk(ctable))

            else:  # no cmap specified
                if flip_scalars:
                    table.SetHueRange(0.0, 0.66667)
                else:
                    table.SetHueRange(0.66667, 0.0)

        else:
            self.mapper.SetScalarModeToUseFieldData()

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
            raise Exception('Invalid style.  Must be one of the following:\n' +
                            '\t"surface"\n' +
                            '\t"wireframe"\n' +
                            '\t"points"\n')

        prop.SetPointSize(point_size)
        prop.SetAmbient(ambient)
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
            geom = single_triangle()
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
        if stitle is not None and show_scalar_bar and not rgb:
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
            return self.mapper.SetScalarRange(*clim)
        # Use the name to find the desired actor
        def update_mapper(mapper):
            return mapper.SetScalarRange(*clim)
        try:
            for m in self._scalar_bar_mappers[name]:
                update_mapper(m)
        except KeyError:
            raise KeyError('Name ({}) not valid/not found in this plotter.')
        return


    @property
    def camera_set(self):
        """ Returns if the camera of the active renderer has been set """
        return self.renderer.camera_set

    def get_default_cam_pos(self):
        """ Return the default camera position of the active renderer """
        return self.renderer.get_default_cam_pos()

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
                  culling=False):
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

        culling : bool optional
            Does not render faces that should not be visible to the
            plotter.  This can be helpful for dense surface meshes,
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
        return renderer.add_actor(uinput, reset_camera, name, culling)

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
            assert len(loc) == 2, '"loc" must contain two items'
            return loc[0]*self.shape[0] + loc[1]

    def index_to_loc(self, index):
        """Convert a 1D index location to the 2D location on the plotting grid
        """
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

    def add_axes_at_origin(self, loc=None):
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
        self._active_renderer_index = self.loc_to_index(loc)
        return self.renderers[self._active_renderer_index].add_axes_at_origin()

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
                         opacity=1.0, render_lines_as_tubes=False, lighting=None,
                         reset_camera=None, loc=None):
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

    def subplot(self, index_x, index_y):
        """
        Sets the active subplot.

        Parameters
        ----------
        index_x : int
            Index of the subplot to activate in the x direction.

        index_y : int
            Index of the subplot to activate in the y direction.

        """
        self._active_renderer_index = self.loc_to_index((index_x, index_y))

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

    def _update_axes_color(self, color):
        """Internal helper to set the axes label color"""
        prop_x = self.axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty()
        prop_y = self.axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty()
        prop_z = self.axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty()
        if color is None:
            color = rcParams['font']['color']
        color = parse_color(color)
        for prop in [prop_x, prop_y, prop_z]:
            prop.SetColor(color[0], color[1], color[2])
            prop.SetShadow(False)
        return

    def add_scalar_bar(self, title=None, n_labels=5, italic=False,
                       bold=True, title_font_size=None,
                       label_font_size=None, color=None,
                       font_family=None, shadow=False, mapper=None,
                       width=None, height=None, position_x=None,
                       position_y=None, vertical=None,
                       interactive=False, fmt=None, use_opacity=True,
                       outline=False):
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
            if not hasattr(self, 'mapper'):
                raise Exception('Mapper does not exist.  ' +
                                'Add a mesh with scalars first.')
            mapper = self.mapper

        if title:
            # Check that this data hasn't already been plotted
            if title in list(self._scalar_bar_ranges.keys()):
                rng = list(self._scalar_bar_ranges[title])
                newrng = mapper.GetScalarRange()
                oldmappers = self._scalar_bar_mappers[title]
                # get max for range and reset everything
                if newrng[0] < rng[0]:
                    rng[0] = newrng[0]
                if newrng[1] > rng[1]:
                    rng[1] = newrng[1]
                for m in oldmappers:
                    m.SetScalarRange(rng[0], rng[1])
                mapper.SetScalarRange(rng[0], rng[1])
                self._scalar_bar_mappers[title].append(mapper)
                self._scalar_bar_ranges[title] = rng
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
                    position_x -= slot * width
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
        self.scalar_bar.SetLookupTable(mapper.GetLookupTable())
        self.scalar_bar.SetNumberOfLabels(n_labels)

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

        if n_labels:
            label_text = self.scalar_bar.GetLabelTextProperty()
            label_text.SetColor(color)
            label_text.SetShadow(shadow)

            # Set font
            label_text.SetFontFamily(parse_font_family(font_family))
            label_text.SetItalic(italic)
            label_text.SetBold(bold)
            if label_font_size:
                label_text.SetFontSize(label_font_size)

        # Set properties
        if title:
            rng = mapper.GetScalarRange()
            self._scalar_bar_ranges[title] = rng
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
            if shape != (1, 1):
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

        self.add_actor(self.scalar_bar, reset_camera=False)

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
            scalars = get_scalar(mesh, scalars)

        if scalars is None:
            if render:
                self.ren_win.Render()
            return

        if scalars.shape[0] == mesh.GetNumberOfPoints():
            data = mesh.GetPointData()
        elif scalars.shape[0] == mesh.GetNumberOfCells():
            data = mesh.GetCellData()
        else:
            _raise_not_matching(scalars, mesh)

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
        # must close out axes marker
        if hasattr(self, 'axes_widget'):
            del self.axes_widget

        # reset scalar bar stuff
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}

        if hasattr(self, 'ren_win'):
            self.ren_win.Finalize()
            del self.ren_win

        if hasattr(self, '_style'):
            del self._style

        if hasattr(self, 'iren'):
            self.iren.RemoveAllObservers()
            del self.iren

        if hasattr(self, 'textActor'):
            del self.textActor

        # end movie
        if hasattr(self, 'mwriter'):
            try:
                self.mwriter.close()
            except BaseException:
                pass

    def add_text(self, text, position=None, font_size=50, color=None,
                 font=None, shadow=False, name=None, loc=None):
        """
        Adds text to plot object in the top left corner by default

        Parameters
        ----------
        text : str
            The text to add the the rendering

        position : tuple(float)
            Length 2 tuple of the pixelwise position to place the bottom
            left corner of the text box. Default is to find the top left corner
            of the renderering window and place text box up there.

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

        self.textActor = vtk.vtkTextActor()
        self.textActor.SetPosition(position)
        self.textActor.GetTextProperty().SetFontSize(font_size)
        self.textActor.GetTextProperty().SetColor(parse_color(color))
        self.textActor.GetTextProperty().SetFontFamily(FONT_KEYS[font])
        self.textActor.GetTextProperty().SetShadow(shadow)
        self.textActor.SetInput(text)
        self.add_actor(self.textActor, reset_camera=False, name=name, loc=loc)
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

    @property
    def image_depth(self):
        """ Returns an image array of current render window """
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.ren_win)
        ifilter.ReadFrontBufferOff()
        ifilter.SetInputBufferTypeToZBuffer()
        return self._run_image_filter(ifilter)

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
        self.add_actor(self.scalar_bar, reset_camera=False, name=name)
        return self.scalar_bar

    def remove_scalar_bar(self):
        """ Removes scalar bar """
        if hasattr(self, 'scalar_bar'):
            self.remove_actor(self.scalar_bar, reset_camera=False)


    def add_point_labels(self, points, labels, italic=False, bold=True,
                         font_size=None, text_color=None,
                         font_family=None, shadow=False,
                         show_points=True, point_color=None, point_size=5,
                         name=None, **kwargs):
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

        if isinstance(points, np.ndarray):
            vtkpoints = pyvista.PolyData(points) # Cast to poly data
        elif is_pyvista_obj(points):
            vtkpoints = pyvista.PolyData(points.points)
            if isinstance(labels, str):
                labels = points.point_arrays[labels].astype(str)
        else:
            raise TypeError('Points type not useable: {}'.format(type(points)))

        if len(vtkpoints.points) != len(labels):
            raise Exception('There must be one label for each point')

        vtklabels = vtk.vtkStringArray()
        vtklabels.SetName('labels')
        for item in labels:
            vtklabels.InsertNextValue(str(item))
        vtkpoints.GetPointData().AddArray(vtklabels)

        # create label mapper
        labelMapper = vtk.vtkLabeledDataMapper()
        labelMapper.SetInputData(vtkpoints)
        textprop = labelMapper.GetLabelTextProperty()
        textprop.SetItalic(italic)
        textprop.SetBold(bold)
        textprop.SetFontSize(font_size)
        textprop.SetFontFamily(parse_font_family(font_family))
        textprop.SetColor(parse_color(text_color))
        textprop.SetShadow(shadow)
        labelMapper.SetLabelModeToLabelFieldData()
        labelMapper.SetFieldDataName('labels')

        labelActor = vtk.vtkActor2D()
        labelActor.SetMapper(labelMapper)

        # add points
        if show_points:
            style = 'points'
        else:
            style = 'surface'
        self.add_mesh(vtkpoints, style=style, color=point_color,
                      point_size=point_size)

        self.add_actor(labelActor, reset_camera=False, name=name)
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
        if not is_pyvista_obj(points):
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
        if isinstance(filename, str):
            if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
                filename = os.path.join(pyvista.FIGURE_PATH, filename)
            if not return_img:
                return imageio.imwrite(filename, image)
            imageio.imwrite(filename, image)

        return image

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
            raise AttributeError('This plotter is unable to save a screenshot.')

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
                raise Exception('No labels input.\n\n' +
                                'Add labels to individual items when adding them to' +
                                'the plotting object with the "label=" parameter.  ' +
                                'or enter them as the "labels" parameter.')

            self.legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels):
                self.legend.SetEntry(i, vtk_object, text, parse_color(color))

        else:
            self.legend.SetNumberOfEntries(len(labels))
            legendface = single_triangle()
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
        self.add_actor(self.legend, reset_camera=False, name=name)
        return self.legend

    @property
    def camera_position(self):
        """ Returns camera position of the active render window """
        return self.renderer.camera_position

    @camera_position.setter
    def camera_position(self, camera_location):
        """ Set camera position of the active render window """
        self.renderer.camera_position = camera_location

    def reset_camera(self):
        """
        Reset camera so it slides along the vector defined from camera
        position to focal point until all of the actors can be seen.
        """
        self.renderer.reset_camera()
        self._render()

    def isometric_view(self):
        """DEPRECATED: Please use ``view_isometric``"""
        return self.view_isometric()

    def view_isometric(self):
        """
        Resets the camera to a default isometric view showing all the
        actors in the scene.
        """
        return self.renderer.view_isometric()

    def view_vector(self, vector, viewup=None):
        return self.renderer.view_vector(vector, viewup=viewup)

    def view_xy(self, negative=False):
        """View the XY plane"""
        return self.renderer.view_xy(negative=negative)

    def view_xz(self, negative=False):
        """View the XZ plane"""
        return self.renderer.view_xz(negative=negative)

    def view_yz(self, negative=False):
        """View the YZ plane"""
        return self.renderer.view_yz(negative=negative)

    def disable(self):
        """Disable this renderer's camera from being interactive"""
        return self.renderer.disable()

    def enable(self):
        """Enable this renderer's camera to be interactive"""
        return self.renderer.enable()

    def set_background(self, color, loc='all'):
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

        """
        if color is None:
            color = rcParams['background']
        if isinstance(color, str):
            if color.lower() in 'paraview' or color.lower() in 'pv':
                # Use the default ParaView background color
                color = PV_BACKGROUND
            else:
                color = pyvista.string_to_rgb(color)

        if loc =='all':
            for renderer in self.renderers:
                renderer.SetBackground(color)
        else:
            renderer = self.renderers[self.loc_to_index(loc)]
            renderer.SetBackground(color)

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

    def enable_cell_picking(self, mesh=None, callback=None):
        """
        Enables picking of cells.  Press r to enable retangle based
        selection.  Press "r" again to turn it off.  Selection will be
        saved to self.picked_cells.

        Uses last input mesh for input

        Parameters
        ----------
        mesh : vtk.UnstructuredGrid, optional
            UnstructuredGrid grid to select cells from.  Uses last
            input grid by default.

        callback : function, optional
            When input, calls this function after a selection is made.
            The picked_cells are input as the first parameter to this function.

        """
        if mesh is None:
            if not hasattr(self, 'mesh'):
                raise Exception('Input a mesh into the Plotter class first or '
                                + 'or set it in this function')
            mesh = self.mesh

        def pick_call_back(picker, event_id):
            extract = vtk.vtkExtractGeometry()
            mesh.cell_arrays['orig_extract_id'] = np.arange(mesh.n_cells)
            extract.SetInputData(mesh)
            extract.SetImplicitFunction(picker.GetFrustum())
            extract.Update()
            self.picked_cells = pyvista.wrap(extract.GetOutput())

            if callback is not None:
                callback(self.picked_cells)

        area_picker = vtk.vtkAreaPicker()
        area_picker.AddObserver(vtk.vtkCommand.EndPickEvent, pick_call_back)

        self.enable_rubber_band_style()
        self.iren.SetPicker(area_picker)


    def generate_orbital_path(self, factor=3., n_points=20, viewup=None, z_shift=None):
        """Genrates an orbital path around the data scene

        Parameters
        ----------
        facotr : float
            A scaling factor when biulding the orbital extent

        n_points : int
            number of points on the orbital path

        viewup : list(float)
            the normal to the orbital plane

        z_shift : float, optional
            shift the plane up/down from the center of the scene by this amount
        """
        if viewup is None:
            viewup = rcParams['camera']['viewup']
        center = list(self.center)
        bnds = list(self.bounds)
        if z_shift is None:
            z_shift = (bnds[5] - bnds[4]) * factor
        center[2] = center[2] + z_shift
        radius = (bnds[1] - bnds[0]) * factor
        y = (bnds[3] - bnds[2]) * factor
        if y > radius:
            radius = y
        return pyvista.Polygon(center=center, radius=radius, normal=viewup, n_sides=n_points)


    def fly_to(point):
        """Given a position point, move the current camera's focal point to that
        point. The movement is animated over the number of frames specified in
        NumberOfFlyFrames. The LOD desired frame rate is used.
        """
        return self.iren.FlyTo(self.renderer, *point)


    def orbit_on_path(self, path=None, focus=None, step=0.5, viewup=None, bkg=True):
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
        """
        if focus is None:
            focus = self.center
        if viewup is None:
            viewup = rcParams['camera']['viewup']
        if path is None:
            path = self.generate_orbital_path(viewup=viewup)
        if not is_pyvista_obj(path):
            path = pyvista.PolyData(path)
        points = path.points

        def orbit():
            """Internal thread for running the orbit"""
            for point in points:
                self.set_position(point)
                self.set_focus(focus)
                self.set_viewup(viewup)
                time.sleep(step)


        if bkg:
            thread = Thread(target=orbit)
            thread.start()
        else:
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
        return export_plotter_vtkjs(self, filename, compress_arrays=compress_arrays)


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
        with ``shape=(2, 2)``.  By default there is only one render
        window.

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

    """
    last_update_time = 0.0
    q_pressed = False
    right_timer_id = -1

    def __init__(self, off_screen=None, notebook=None, shape=(1, 1),
                 border=None, border_color='k', border_width=1.0,
                 window_size=None):
        """
        Initialize a vtk plotting object
        """
        super(Plotter, self).__init__(shape=shape, border=border,
                                      border_color=border_color,
                                      border_width=border_width)
        log.debug('Initializing')

        def on_timer(iren, event_id):
            """ Exit application if interactive renderer stops """
            if event_id == 'TimerEvent':
                self.iren.TerminateApp()

        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN

        if notebook is None:
            if run_from_ipython():
                try:
                    notebook = type(get_ipython()).__module__.startswith('ipykernel.')
                except NameError:
                    pass

        self.notebook = notebook
        if self.notebook:
            off_screen = True
        self.off_screen = off_screen

        if window_size is None:
            window_size = pyvista.rcParams['window_size']

        # initialize render window
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetBorders(True)
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
             auto_close=True, interactive_update=False, full_screen=False,
             screenshot=False, return_img=False, use_panel=None):
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

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up

        """
        if use_panel is None:
            use_panel = rcParams['use_panel']
        # reset unless camera for the first render unless camera is set
        if self._first_time:  # and not self.camera_set:
            for renderer in self.renderers:
                if not renderer.camera_set:
                    renderer.camera_position = renderer.get_default_cam_pos()
                    renderer.ResetCamera()
            self._first_time = False

        if title:
            self.ren_win.SetWindowName(title)

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

        # Keep track of image for sphinx-gallery
        self.last_image = self.screenshot(screenshot, return_img=True)
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
        elif self.notebook and use_panel:
            try:
                from panel.pane import VTK as panel_display
                disp = panel_display(self.ren_win, sizing_mode='stretch_width',
                                     height=400)
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
        if return_img or screenshot == True:
            return cpos, self.last_image

        # default to returning last used camera position
        return cpos

    def plot(self, *args, **kwargs):
        """ Present for backwards compatibility. Use `show()` instead """
        return self.show(*args, **kwargs)

    def render(self):
        """ renders main window """
        self.ren_win.Render()



def single_triangle():
    """ A single PolyData triangle """
    points = np.zeros((3, 3))
    points[1] = [1, 0, 0]
    points[2] = [0.5, 0.707, 0]
    cells = np.array([[3, 0, 1, 2]], ctypes.c_long)
    return pyvista.PolyData(points, cells)


def parse_color(color):
    """ Parses color into a vtk friendly rgb list """
    if color is None:
        color = rcParams['color']
    if isinstance(color, str):
        return pyvista.string_to_rgb(color)
    elif len(color) == 3:
        return color
    else:
        raise Exception("""
    Invalid color input
    Must ba string, rgb list, or hex color string.  For example:
        color='white'
        color='w'
        color=[1, 1, 1]
        color='#FFFFFF'""")


def parse_font_family(font_family):
    """ checks font name """
    # check font name
    font_family = font_family.lower()
    if font_family not in ['courier', 'times', 'arial']:
        raise Exception('Font must be either "courier", "times" ' +
                        'or "arial"')

    return FONT_KEYS[font_family]


def plot_compare_four(data_a, data_b, data_c, data_d, disply_kwargs=None,
                      plotter_kwargs=None, show_kwargs=None, screenshot=None,
                      camera_position=None, outline=None, outline_color='k',
                      labels=('A', 'B', 'C', 'D')):
    """Plot a 2 by 2 comparison of data objects. Plotting parameters and camera
    positions will all be the same.
    """
    datasets = [[data_a, data_b], [data_c, data_d]]
    labels = [labels[0:2], labels[2:4]]

    if plotter_kwargs is None:
        plotter_kwargs = {}
    if disply_kwargs is None:
        disply_kwargs = {}
    if show_kwargs is None:
        show_kwargs = {}

    p = pyvista.Plotter(shape=(2,2), **plotter_kwargs)

    for i in range(2):
        for j in range(2):
            p.subplot(i, j)
            p.add_mesh(datasets[i][j], **disply_kwargs)
            p.add_text(labels[i][j])
            if is_pyvista_obj(outline):
                p.add_mesh(outline, color=outline_color)
            if camera_position is not None:
                p.camera_position = camera_position

    return p.show(screenshot=screenshot, **show_kwargs)
