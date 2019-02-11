"""
vtki plotting module
"""
import time
import logging
import ctypes
import PIL.Image
from subprocess import Popen, PIPE
import os
import colorsys
import collections

import vtk
from vtk.util import numpy_support as VN

import numpy as np
import vtki
from vtki.utilities import get_scalar, wrap, is_vtki_obj, numpy_to_texture
from vtki.export import export_plotter_vtkjs
import imageio


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
    },
    'cmap' : 'jet',
    'color' : 'white',
    'colorbar' : {
        'width' : 0.60,
        'height' : 0.08,
        'position_x' : 0.35,
        'position_y' : 0.02,
    },
    'show_edges' : False,
}

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
        rcParams['cmap'] = 'coolwarm'
        rcParams['font']['family'] = 'arial'
        rcParams['font']['label_size'] = 16
        rcParams['font']['color'] = 'black'
        rcParams['show_edges'] = False
        rcParams['color'] = 'yellow'


def run_from_ipython():
    """ returns True when run from IPython """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def _raise_not_matching(scalars, mesh):
    raise Exception('Number of scalars (%d) ' % scalars.size +
                    'must match either the number of points ' +
                    '(%d) ' % mesh.GetNumberOfPoints() +
                    'or the number of cells ' +
                    '(%d) ' % mesh.GetNumberOfCells())

def _remove_mapper_from_plotter(plotter, actor, reset_camera):
    """removes this actor's mapper from the given plotter's _scalar_bar_mappers"""
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
            plotter._scalar_bar_mappers.pop(name)
            plotter._scalar_bar_ranges.pop(name)
            plotter.remove_actor(plotter._scalar_bar_actors.pop(name), reset_camera=reset_camera)
            plotter._scalar_bar_slots.add(plotter._scalar_bar_slot_lookup.pop(name))
    return


def plot(var_item, off_screen=False, full_screen=False, screenshot=None,
         interactive=True, cpos=None, window_size=None,
         show_bounds=False, show_axes=True, notebook=None, background=None,
         text='', **kwargs):
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
        help(vtkinterface.Plotter.screenshot).  Default disabled.

        When True, takes screenshot and returns numpy array of image.

    window_size : list, optional
        Window size in pixels.  Defaults to [1024, 768]

    show_bounds : bool, optional
        Shows mesh bounds when True.  Default False.

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
            notebook = type(get_ipython()).__module__.startswith('ipykernel.')

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

    if show_bounds:
        plotter.add_bounds_axes()

    if cpos is None:
        cpos = plotter.get_default_cam_pos()
        plotter.camera_position = cpos
        plotter.camera_set = False
    else:
        plotter.camera_position = cpos

    result = plotter.show(window_size=window_size,
                        auto_close=False,
                        interactive=interactive,
                        full_screen=full_screen,
                        screenshot=screenshot)

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
        See help(vtki.Plot)

    Returns
    -------
    Same as Plot.  See help(vtki.Plot)

    """
    return plot([cent, direction], **kwargs)


def running_xserver():
    """
    Check if x server is running

    Returns
    -------
    running_xserver : bool
        True when on Linux and running an xserver.  Returns None when
        on a non-linux platform.

    """
    try:
        p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
        p.communicate()
        return p.returncode == 0
    except:
        False


class BasePlotter(object):
    """
    Base plotter class to be used by the Plotter and VtkInteractor
    classes
    """

    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        # This is a private variable to keep track of how many colorbars exist
        # This allows us to keep adding colorbars without overlapping
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}
        # This keeps track of scalar names already plotted and their ranges
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._scalar_bar_actors = {}
        self._actors = {}
        # track if the camera has been setup
        self.camera_set = False
        self.first_time = True
        # Keep track of the scale
        self.scale = [1.0, 1.0, 1.0]
        self._labels = []


    def update_bounds_axes(self):
        """Update the bounds axes of the render window """
        # Update the bounds for the axes labels if present
        if hasattr(self, 'cube_axes_actor'):
            self.cube_axes_actor.SetBounds(self.bounds)
        return

    @property
    def bounds(self):
        """ Bounds of all actors present in the rendering window """
        the_bounds = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]

        def _update_bounds(bounds):
            def update_axis(ax):
                if bounds[ax*2] < the_bounds[ax*2]:
                    the_bounds[ax*2] = bounds[ax*2]
                if bounds[ax*2+1] > the_bounds[ax*2+1]:
                    the_bounds[ax*2+1] = bounds[ax*2+1]
            for ax in range(3):
                update_axis(ax)
            return

        for name, actor in self._actors.items():
            if isinstance(actor, vtk.vtkCubeAxesActor):
                continue
            if hasattr(actor, 'GetBounds') and actor.GetBounds() is not None:
                _update_bounds(actor.GetBounds())

        return the_bounds

    @property
    def center(self):
        bounds = self.bounds
        x = (bounds[1] + bounds[0])/2
        y = (bounds[3] + bounds[2])/2
        z = (bounds[5] + bounds[4])/2
        return [x, y, z]

    def clear(self):
        """ Clears plot by removing all actors and properties """
        self.renderer.RemoveAllViewProps()
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._scalar_bar_actors = {}

    def enable_trackball_style(self):
        """ sets the interacto style to trackball """
        istyle = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(istyle)

    def set_focus(self, point):
        """ sets focus to a point """
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.SetFocalPoint(point)
        self._render()

    def _render(self):
        """ redraws render window if the render window exists """
        if hasattr(self, 'ren_win'):
            if hasattr(self, 'render_trigger'):
                self.render_trigger.emit()
            elif not self.first_time:
                self.render()

    def add_axes(self, interactive=False):
        """ Add an interactive axes widget """
        if hasattr(self, 'axes_widget'):
            raise Exception('Plotter already has an axes widget')
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        if hasattr(self, 'iren'):
            self.axes_widget.SetInteractor(self.iren)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.SetInteractive(interactive)


    def get_default_cam_pos(self):
        """
        Returns the default focal points and viewup. Uses ResetCamera to
        make a useful view.
        """
        focal_pt = self.center
        return [np.array(rcParams['camera']['position']) + np.array(focal_pt),
                focal_pt, rcParams['camera']['viewup']]

    def key_press_event(self, obj, event):
        """ Listens for key press event """
        key = self.iren.GetKeySym()
        log.debug('Key %s pressed' % key)
        if key == 'q':
            self.q_pressed = True
        elif key == 'b':
            self.observer = self.iren.AddObserver('LeftButtonPressEvent',
                                                  self.left_button_down)

    def left_button_down(self, obj, event_type):
        # Get 2D click location on window
        clickPos = self.iren.GetEventPosition()

        # Get corresponding click location in the 3D plot
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
        self.pickpoint = np.asarray(picker.GetPickPosition()).reshape((-1, 3))
        if np.any(np.isnan(self.pickpoint)):
            self.pickpoint[:] = 0

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

    def add_mesh(self, mesh, color=None, style=None,
                 scalars=None, rng=None, stitle=None, show_edges=None,
                 point_size=5.0, opacity=1, line_width=None, flip_scalars=False,
                 lighting=True, n_colors=256, interpolate_before_map=False,
                 cmap=None, label=None, reset_camera=None, scalar_bar_args=None,
                 multi_colors=False, name=None, texture=None,
                 render_points_as_spheres=False, render_lines_as_tubes=False,
                 edge_color='black', **kwargs):
        """
        Adds a unstructured, structured, or surface mesh to the plotting object.

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
            Scalars used to "color" the mesh.  Accepts an array equal to the
            number of cells or the number of points in the mesh.  Array should
            be sized as a single vector. If both color and scalars are None,
            then the active scalars are used

        rng : 2 item list, optional
            Range of mapper for scalars.  Defaults to minimum and maximum of
            scalars array.  Example: ``[-1, 2]``

        stitle : string, optional
            Scalar title.  By default there is no scalar legend bar.  Setting
            this creates the legend bar and adds a title to it.  To create a
            bar with no title, use an empty string (i.e. '').

        show_edges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        point_size : float, optional
            Point size.  Applicable when style='points'.  Default 5.0

        opacity : float, optional
            Opacity of mesh.  Should be between 0 and 1.  Default 1.0

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
            If a ``MultiBlock`` dataset is given this will color each block by
            a solid color using matplotlib's color cycler.

        name : str, optional
            The name for the added mesh/actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        texture : vtk.vtkTexture or np.ndarray or boolean, optional
            A texture to apply if the input mesh has texture coordinates.
            This will not work with MultiBlock datasets. If set to ``True``,
            the first avaialble texture on the object will be used. If a string
            name is given, it will pull a texture with that name associated to
            the input mesh.

        Returns
        -------
        actor: vtk.vtkActor
            VTK actor of the mesh.
        """
        if scalar_bar_args is None:
            scalar_bar_args = {}

        if isinstance(mesh, np.ndarray):
            mesh = vtki.PolyData(mesh)
            style = 'points'

        # Convert the VTK data object to a vtki wrapped object if neccessary
        if not is_vtki_obj(mesh):
            mesh = wrap(mesh)

        if show_edges is None:
            show_edges = rcParams['show_edges']

        if name is None:
            name = '{}({})'.format(type(mesh).__name__, str(hex(id(mesh))))


        if isinstance(mesh, vtki.MultiBlock):
            self.remove_actor(name, reset_camera=reset_camera)
            # frist check the scalars
            if rng is None and scalars is not None:
                # Get the data range across the array for all blocks if scalar specified
                if isinstance(scalars, str):
                    rng = mesh.get_data_range(scalars)
                else:
                    # TODO: an array was given... how do we deal with that? Possibly
                    #       a 2D arrays or list of arrays  where first index
                    #       corresponds to the block? This could get complicated real quick.
                    raise RuntimeError('Scalar array must be given as a string name for multiblock datasets.')
            if multi_colors:
                # Compute unique colors for each index of the block
                import matplotlib as mpl
                from itertools import cycle
                cycler = mpl.rcParams['axes.prop_cycle']
                colors = cycle(cycler)
            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(mesh.GetNumberOfBlocks()):
                if mesh[idx] is None:
                    continue
                # Get a good name to use
                nm = '{}-{}'.format(name, idx)
                # Get the data object
                if not is_vtki_obj(mesh[idx]):
                    data = wrap(mesh.GetBlock(idx))
                    if not is_vtki_obj(mesh[idx]):
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
                             scalars=ts, rng=rng, stitle=stitle, show_edges=show_edges,
                             point_size=point_size, opacity=opacity, line_width=line_width,
                             flip_scalars=flip_scalars, lighting=lighting,
                             n_colors=n_colors, interpolate_before_map=interpolate_before_map,
                             cmap=cmap, label=label,
                             scalar_bar_args=scalar_bar_args, reset_camera=reset_camera,
                             name=nm, texture=None,
                             render_points_as_spheres=render_points_as_spheres, render_lines_as_tubes=render_lines_as_tubes,
                             edge_color=edge_color, **kwargs)
                actors.append(a)
                if (reset_camera is None and not self.camera_set) or reset_camera:
                    cpos = self.get_default_cam_pos()
                    self.camera_position = cpos
                    self.camera_set = False
                    self.reset_camera()
            return actors


        # set main values
        self.mesh = mesh
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(self.mesh)
        if isinstance(scalars, str):
            self.mapper.SetArrayName(scalars)
        actor, prop = self.add_actor(self.mapper, reset_camera=reset_camera, name=name)

        if texture == True or isinstance(texture, str):
            texture = mesh._activate_texture(texture)

        if texture:
            if isinstance(texture, np.ndarray):
                texture = numpy_to_texture(texture)
            if not isinstance(texture, vtk.vtkTexture):
                raise TypeError('Invalid texture type ({})'.format(type(texture)))
            if mesh.GetPointData().GetTCoords() is None:
                raise AssertionError('Input mesh does not have texture coordinates to support the texture.')
            actor.SetTexture(texture)


        # Attempt get the active scalars if no preference given
        if scalars is None and color is None and texture is None:
            scalars = mesh.active_scalar
            # Make sure scalar components are not vectors/tuples
            if scalars is None or scalars.ndim != 1:
                scalars = None
            else:
                if stitle is None:
                    stitle = mesh.active_scalar_info[1]

        # Scalar formatting ===================================================
        if cmap is None:
            cmap = rcParams['cmap']
        title = 'Data' if stitle is None else stitle
        if scalars is not None:
            # if scalars is a string, then get the first array found with that name
            append_scalars = True
            if isinstance(scalars, str):
                title = scalars
                scalars = get_scalar(mesh, scalars)
                if stitle is None:
                    stitle = title
                #append_scalars = False

            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            if scalars.ndim != 1:
                scalars = scalars.ravel()

            if scalars.dtype == np.bool:
                scalars = scalars.astype(np.float)

            # Scalar interpolation approach
            if scalars.size == mesh.GetNumberOfPoints():
                self.mesh._add_point_scalar(scalars, title, append_scalars)
                self.mapper.SetScalarModeToUsePointData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
                if interpolate_before_map:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
            elif scalars.size == mesh.GetNumberOfCells():
                self.mesh._add_cell_scalar(scalars, title, append_scalars)
                self.mapper.SetScalarModeToUseCellData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
                if interpolate_before_map:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
            else:
                _raise_not_matching(scalars, mesh)

            # Set scalar range
            if not rng:
                rng = [np.nanmin(scalars), np.nanmax(scalars)]
            elif isinstance(rng, float) or isinstance(rng, int):
                rng = [-rng, rng]

            if np.any(rng):
                self.mapper.SetScalarRange(rng[0], rng[1])

            # Flip if requested
            table = self.mapper.GetLookupTable()
            if cmap is not None:
                try:
                    from matplotlib.cm import get_cmap
                except ImportError:
                    raise Exception('cmap requires matplotlib')
                cmap = get_cmap(cmap)
                ctable = cmap(np.linspace(0, 1, n_colors))*255
                ctable = ctable.astype(np.uint8)
                if flip_scalars:
                    ctable = np.ascontiguousarray(ctable[::-1])
                table.SetTable(VN.numpy_to_vtk(ctable))

            else:  # no cmap specified
                if flip_scalars:
                    self.mapper.GetLookupTable().SetHueRange(0.0, 0.66667)
                else:
                    self.mapper.GetLookupTable().SetHueRange(0.66667, 0.0)

        else:
            self.mapper.SetScalarModeToUseFieldData()

        # select view style
        if not style:
            style = 'surface'
        style = style.lower()
        if style == 'wireframe':
            prop.SetRepresentationToWireframe()
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

        # edge display style
        if show_edges:
            prop.EdgeVisibilityOn()

        rgb_color = parse_color(color)
        prop.SetColor(rgb_color)
        prop.SetOpacity(opacity)
        prop.SetEdgeColor(parse_color(edge_color))

        if render_points_as_spheres:
            prop.SetRenderPointsAsSpheres(render_points_as_spheres)
        if render_lines_as_tubes:
            prop.SetRenderLinesAsTubes(render_lines_as_tubes)

        # legend label
        if label:
            assert isinstance(label, str), 'Label must be a string'
            self._labels.append([single_triangle(), label, rgb_color])

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        # Add scalar bar if available
        if stitle is not None:
            self.add_scalar_bar(stitle, **scalar_bar_args)

        return actor

    def add_actor(self, uinput, reset_camera=False, name=None):
        """
        Adds an actor to render window.  Creates an actor if input is
        a mapper.

        Parameters
        ----------
        uinput : vtk.vtkMapper or vtk.vtkActor
            vtk mapper or vtk actor to be added.

        reset_camera : bool, optional
            Resets the camera when true.

        Returns
        -------
        actor : vtk.vtkActor
            The actor.

        actor_properties : vtk.Properties
            Actor properties.

        """
        # Remove actor by that name if present
        rv = self.remove_actor(name, reset_camera=False)

        if isinstance(uinput, vtk.vtkMapper):
            actor = vtk.vtkActor()
            actor.SetMapper(uinput)
        else:
            actor = uinput

        # Make sure scale is consistent with rest of scene
        if hasattr(actor, 'SetScale'):
            actor.SetScale(self.scale[0], self.scale[1], self.scale[2])

        self.renderer.AddActor(actor)
        if name is None:
            name = str(hex(id(actor)))

        self._actors[name] = actor

        if reset_camera:
            self.reset_camera()
        elif not self.camera_set and reset_camera is None and not rv:
            self.reset_camera()
        else:
            self._render()

        self.update_bounds_axes()

        return actor, actor.GetProperty()

    @property
    def camera(self):
        return self.renderer.GetActiveCamera()

    def remove_actor(self, actor, reset_camera=False):
        """
        Removes an actor from the Plotter.

        Parameters
        ----------
        actor : vtk.vtkActor
            Actor that has previously added to the Plotter.
        """
        name = None
        if isinstance(actor, str):
            name = actor
            keys = list(self._actors.keys())
            names = []
            for k in keys:
                if k.startswith('{}-'.format(name)):
                    names.append(k)
            if len(names) > 0:
                self.remove_actor(names, reset_camera=reset_camera)
            try:
                actor = self._actors[name]
            except KeyError:
                # If actor of that name is not present then return success
                return False
        if isinstance(actor, collections.Iterable):
            success = False
            for a in actor:
                rv = self.remove_actor(a, reset_camera=reset_camera)
                if rv or success:
                    success = True
            return success
        if actor is None:
            return False
        # First remove this actor's mapper from _scalar_bar_mappers
        _remove_mapper_from_plotter(self, actor, False)
        self.renderer.RemoveActor(actor)
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
        else:
            self._render()
        return True

    def add_axes_at_origin(self):
        """
        Add axes actor at origin

        Returns
        --------
        marker_actor : vtk.vtkAxesActor
            vtkAxesActor actor
        """
        self.marker_actor = vtk.vtkAxesActor()
        self.renderer.AddActor(self.marker_actor)
        self._actors[str(hex(id(self.marker_actor)))] = self.marker_actor
        return self.marker_actor

    def add_bounds_axes(self, mesh=None, bounds=None, show_xaxis=True,
                        show_yaxis=True, show_zaxis=True, show_xlabels=True,
                        show_ylabels=True, show_zlabels=True, italic=False,
                        bold=True, shadow=False, font_size=None,
                        font_family=None, color=None,
                        xlabel='X Axis', ylabel='Y Axis', zlabel='Z Axis',
                        use_2d=True, grid=None, location='closest', ticks=None):
        """
        Adds bounds axes.  Shows the bounds of the most recent input mesh
        unless mesh is specified.

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

        bold  : bool, optional
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
            A bug with vtk 6.3 in Windows seems to cause this function to crash
            this can be enabled for smoother plotting for other enviornments.

        grid : bool or str, optional
            Add grid lines to the backface (``True``, ``'back'``, or
            ``'backface'``) or to the frontface (``'front'``, ``'frontface'``)
            of the axes actor.

        location : str, optional
            Set how the axes are drawn: either static (``'all'``), closest triad
            (``front``), furthest triad (``'back'``), static closest to the
            origin (``'origin'``), or outer edges (``'outer'``) in relation to
            the camera position. Options include:
            ``'all', 'front', 'back', 'origin', 'outer'``

        ticks : str, option
            Set how the ticks are drawn on the axes grid. Options include:
            ``'inside', 'outside', 'both'``

        Returns
        -------
        cube_axes_actor : vtk.vtkCubeAxesActor
            Bounds actor

        """
        # If one is already present, just use that one
        if hasattr(self, 'cube_axes_actor'):
            self.remove_actor(self.cube_axes_actor)
            del self.cube_axes_actor

        if font_family is None:
            font_family = rcParams['font']['family']
        if font_size is None:
            font_size = rcParams['font']['size']
        if color is None:
            color = rcParams['font']['color']

        color = parse_color(color)

        # Use the bounds of all data in the rendering window
        if not mesh and not bounds:
            bounds = self.bounds

        # create actor
        cube_axes_actor = vtk.vtkCubeAxesActor()
        cube_axes_actor.SetUse2DMode(False)
        cube_axes_actor.SetScale(self.scale[0], self.scale[1], self.scale[2])

        if grid:
            if isinstance(grid, str) and grid.lower() in ('front', 'frontface'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_CLOSEST)
            if isinstance(grid, str) and grid.lower() in ('both', 'all'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_ALL)
            else:
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
            cube_axes_actor.DrawXGridlinesOn()
            cube_axes_actor.DrawYGridlinesOn()
            cube_axes_actor.DrawZGridlinesOn()
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
                raise ValueError('Value of ticks ({}) not understood.'.format(ticks))

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
                raise ValueError('Value of location ({}) not understood.'.format(location))

        # set bounds
        if not bounds:
            bounds = mesh.GetBounds()
        cube_axes_actor.SetBounds(bounds)

        # show or hide axes
        cube_axes_actor.SetXAxisVisibility(show_xaxis)
        cube_axes_actor.SetYAxisVisibility(show_yaxis)
        cube_axes_actor.SetZAxisVisibility(show_zaxis)

        # disable minor ticks
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

        self.add_actor(cube_axes_actor, reset_camera=False)
        self.cube_axes_actor = cube_axes_actor
        return cube_axes_actor

    def set_scale(self, xscale=1.0, yscale=1.0, zscale=1.0, reset_camera=True):
        """
        Scale all the datasets in the scene.
        Scaling in performed independently on the X, Y and Z axis.
        A scale of zero is illegal and will be replaced with one.
        """
        self.scale = [xscale, yscale, zscale]
        for name, actor in self._actors.items():
            if hasattr(actor, 'SetScale'):
                actor.SetScale(xscale, yscale, zscale)
        self.update_bounds_axes()
        self._render()
        if reset_camera:
            self.reset_camera()

    def add_scalar_bar(self, title=None, n_labels=5, italic=False, bold=True,
                       title_font_size=None, label_font_size=None, color=None,
                       font_family=None, shadow=False, mapper=None,
                       width=None, height=None, position_x=None, position_y=None):
        """
        Creates scalar bar using the ranges as set by the last input mesh.

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
            The percentage (0 to 1) width of the window fo the colorbar

        height : float, optional
            The percentage (0 to 1) height of the window for the colorbar

        position_x : float, optional
            The percentage (0 to 1) along the winow's horizontal direction to
            place the bottom left corner of the colorbar

        position_y : float, optional
            The percentage (0 to 1) along the winow's vertical direction to
            place the bottom left corner of the colorbar

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
        # Automatically choose size if not specified
        if width is None:
            width = rcParams['colorbar']['width']
        if height is None:
            height = rcParams['colorbar']['height']

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
            except:
                raise RuntimeError('Maximum number of color bars reached.')
            if position_x is None:
                position_x = rcParams['colorbar']['position_x']
            if position_y is None:
                position_y = rcParams['colorbar']['position_y'] + slot * height
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
            self._scalar_bar_slot_lookup[title] = slot

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

        if isinstance(mesh, (collections.Iterable, vtki.MultiBlock)):
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
        s = VN.vtk_to_numpy(vtk_scalars)
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

        if hasattr(self, 'ifilter'):
            del self.ifilter

    def add_text(self, text, position=None, font_size=50, color=None,
                font=None, shadow=False, name=None):
        """
        Adds text to plot object in the top left corner by default

        Parameters
        ----------
        text : str
            The text to add the the rendering

        position : tuple(float)
            Length 2 tuple of the pixelwise position to place the bottom
            left corner of the text box. Default is to find the top right corner
            of the renderering window and place text box up there.

        font : string, optional
            Font name may be courier, times, or arial

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        Returns
        -------
        textActor : vtk.vtkTextActor
            Text actor added to plot

        """
        if font is None:
            font = rcParams['font']['family']
        if font_size is None:
            font_size = rcParams['font']['size']
        if position is None:
            # Set the position of the text to the top left corner
            window_size = self.window_size
            x = window_size[0] * 0.02
            y = window_size[1] * 0.90
            position = [x, y]

        self.textActor = vtk.vtkTextActor()
        self.textActor.SetPosition(position)
        self.textActor.GetTextProperty().SetFontSize(font_size)
        self.textActor.GetTextProperty().SetColor(parse_color(color))
        self.textActor.GetTextProperty().SetFontFamily(FONT_KEYS[font])
        self.textActor.GetTextProperty().SetShadow(shadow)
        self.textActor.SetInput(text)
        self.add_actor(self.textActor, reset_camera=False, name=name)
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
        self.mwriter = imageio.get_writer(filename, mode='I')

    def write_frame(self):
        """ Writes a single frame to the movie file """
        self.mwriter.append_data(self.image)

    @property
    def window_size(self):
        """ returns render window size """
        return list(self.ren_win.GetSize())

    @property
    def image(self):
        """ Returns an image array of current render window """
        if not hasattr(self, 'ifilter'):
            self.start_image_filter()
        # Update filter and grab pixels
        self.ifilter.Modified()
        self.ifilter.Update()
        image = self.ifilter.GetOutput()
        img_array = vtki.utilities.point_scalar(image, 'ImageScalars')

        # Reshape and write
        tgt_size = (self.window_size[1], self.window_size[0], -1)
        return img_array.reshape(tgt_size)[::-1]

    def add_lines(self, lines, color=[1, 1, 1], width=5, label=None, name=None):
        """
        Adds lines to the plotting object.

        Parameters
        ----------
        lines : np.ndarray or vtki.PolyData
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

        lines = vtki.lines_from_points(lines)

        # Create mapper and add lines
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(lines)

        rgb_color = parse_color(color)

        # legend label
        if label:
            assert isinstance(label, str), 'Label must be a string'
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
                         font_size=None, text_color='k',
                         font_family=None, shadow=False,
                         show_points=True, point_color='k', point_size=5,
                         name=None):
        """
        Creates a point actor with one label from list labels assigned to
        each point.

        Parameters
        ----------
        points : np.ndarray
            n x 3 numpy array of points.

        labels : list
            List of labels.  Must be the same length as points.

        italic : bool, optional
            Italicises title and bar labels.  Default False.

        bold : bool, optional
            Bolds title and bar labels.  Default True

        font_size : float, optional
            Sets the size of the title font.  Defaults to 16.

        text_color : string or 3 item list, optional, defaults to black
            Color of text.
            Either a string, rgb list, or hex color string.  For example:

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

        point_color : string or 3 item list, optional, defaults to black
            Color of points (if visible).
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

        if len(points) != len(labels):
            raise Exception('There must be one label for each point')

        vtkpoints = vtki.PolyData(points)

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

    def add_points(self, points, **kwargs):
        """ Add points to a mesh """
        kwargs['style'] = 'points'
        self.add_mesh(points, **kwargs)

    def add_arrows(self, cent, direction, mag=1, reset_camera=None, name=None):
        """ Adds arrows to plotting object """

        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        pdata = vtki.vector_poly_data(cent, direction * mag)
        arrows = arrows_actor(pdata)
        self.add_actor(arrows, reset_camera=reset_camera, name=name)

        return arrows, pdata

    def screenshot(self, filename=None, transparent_background=False):
        """
        Takes screenshot at current camera position

        Parameters
        ----------
        filename : str, optional
            Location to write image to.  If None, no image is written.

        transparent_background : bool, optional
            Makes the background transparent.  Default False.

        Returns
        -------
        img :  numpy.ndarray
            Array containing pixel RGB and alpha.  Sized:
            [Window height x Window width x 3] for transparent_background=False
            [Window height x Window width x 4] for transparent_background=True

        Examples
        --------
        >>> import vtki
        >>> sphere = vtki.Sphere()
        >>> plotter = vtki.Plotter()
        >>> _ = plotter.add_mesh(sphere)
        >>> _ = plotter.screenshot('screenshot.png') # doctest:+SKIP
        """
        if not hasattr(self, 'ifilter'):
            self.start_image_filter()
        # configure image filter
        if transparent_background:
            self.ifilter.SetInputBufferTypeToRGBA()
        else:
            self.ifilter.SetInputBufferTypeToRGB()

        # this needs to be called twice for some reason,  debug later
        if isinstance(self, Plotter):
            # TODO: we need a consistent rendering function
            self.render()
        else:
            self._render()
        img = self.image
        img = self.image

        if not img.size:
            raise Exception('Empty image.  Have you run plot() first?')

        # write screenshot to file
        if filename:
            imageio.imwrite(filename, img)

        return img

    def add_legend(self, labels=None, bcolor=[0.5, 0.5, 0.5], border=False,
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
        >>> import vtki
        >>> from vtki import examples
        >>> mesh = examples.load_hexbeam()
        >>> othermesh = examples.load_uniform()
        >>> plotter = vtki.Plotter()
        >>> _ = plotter.add_mesh(mesh, label='My Mesh')
        >>> _ = plotter.add_mesh(othermesh, 'k', label='My Other Mesh')
        >>> _ = plotter.add_legend()
        >>> plotter.show() # doctest:+SKIP

        Alternative manual example

        >>> import vtki
        >>> from vtki import examples
        >>> mesh = examples.load_hexbeam()
        >>> othermesh = examples.load_uniform()
        >>> legend_entries = []
        >>> legend_entries.append(['My Mesh', 'w'])
        >>> legend_entries.append(['My Other Mesh', 'k'])
        >>> plotter = vtki.Plotter()
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
        """ Returns camera position of active render window """
        return [self.camera.GetPosition(),
                self.camera.GetFocalPoint(),
                self.camera.GetViewUp()]

    @camera_position.setter
    def camera_position(self, camera_location):
        """ Set camera position of active render window """
        if camera_location is None:
            return

        # everything is set explicitly
        self.camera.SetPosition(camera_location[0])
        self.camera.SetFocalPoint(camera_location[1])
        self.camera.SetViewUp(camera_location[2])

        # reset clipping range
        self.renderer.ResetCameraClippingRange()
        self.camera_set = True

    def reset_camera(self):
        """
        Reset camera so it slides along the vector defined from camera
        position to focal point until all of the actors can be seen.
        """
        self.renderer.ResetCamera()
        self._render()

    def isometric_view(self):
        """
        Resets the camera to a default isometric view showing all the
        actors in the scene.
        """
        self.camera_position = self.get_default_cam_pos()
        self.camera_set = False
        return self.reset_camera()

    def set_background(self, color):
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

        """
        if color is None:
            color = rcParams['background']
        elif isinstance(color, str):
            if color.lower() in 'paraview' or color.lower() in 'pv':
                # Use the default ParaView background color
                color = PV_BACKGROUND
            else:
                color = vtki.string_to_rgb(color)
        self.renderer.SetBackground(color)

    @property
    def background_color(self):
        """ Returns background color """
        return self.renderer.GetBackground()

    @background_color.setter
    def background_color(self, color):
        """ Sets background color """
        self.set_background(color)

    def start_image_filter(self):
        """ creates an image filter """
        self.ifilter = vtk.vtkWindowToImageFilter()
        self.ifilter.SetInput(self.ren_win)
        self.ifilter.SetInputBufferTypeToRGB()
        self.ifilter.ReadFrontBufferOff()

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
            self.picked_cells = vtki.wrap(extract.GetOutput())

            if callback is not None:
                callback(self.picked_cells)

        area_picker = vtk.vtkAreaPicker()
        area_picker.AddObserver(vtk.vtkCommand.EndPickEvent, pick_call_back)

        style = vtk.vtkInteractorStyleRubberBandPick()
        self.iren.SetInteractorStyle(style)
        self.iren.SetPicker(area_picker)


    def export_vtkjs(self, filename, compress_arrays=False):
        """Export the current rendering scene as a VTKjs scene for rendering
        in a web browser"""
        if not hasattr(self, 'ren_win'):
            raise RuntimeError('Export must be called before showing/closing the scene.')
        return export_plotter_vtkjs(self, filename, compress_arrays=compress_arrays)


class Plotter(BasePlotter):
    """
    Plotting object to display vtk meshes or numpy arrays.

    Example
    -------
    >>> import vtki
    >>> from vtki import examples
    >>> mesh = examples.load_hexbeam()
    >>> another_mesh = examples.load_uniform()
    >>> plotter = vtki.Plotter()
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

    """
    last_update_time = 0.0
    q_pressed = False
    right_timer_id = -1

    def __init__(self, off_screen=False, notebook=None):
        """
        Initialize a vtk plotting object
        """
        super(Plotter, self).__init__()
        log.debug('Initializing')
        def onTimer(iren, eventId):
            if 'TimerEvent' == eventId:
                self.iren.TerminateApp()

        if vtki.TESTING_OFFSCREEN:
            off_screen = True

        if notebook is None:
            if run_from_ipython():
                notebook = type(get_ipython()).__module__.startswith('ipykernel.')

        self.notebook = notebook
        if self.notebook:
            off_screen = True
        self.off_screen = off_screen

        # initialize render window
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.AddRenderer(self.renderer)

        if self.off_screen:
            self.ren_win.SetOffScreenRendering(1)
        else:  # Allow user to interact
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetDesiredUpdateRate(30.0)
            self.iren.SetRenderWindow(self.ren_win)
            self.enable_trackball_style()
            self.iren.AddObserver("KeyPressEvent", self.key_press_event)

        # Set background
        self.set_background(rcParams['background'])

        # add timer event if interactive render exists
        if hasattr(self, 'iren'):
            self.iren.AddObserver(vtk.vtkCommand.TimerEvent, onTimer)


    def show(self, title=None, window_size=None, interactive=True,
             auto_close=True, interactive_update=False, full_screen=False,
             screenshot=False):
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
            Enabled by default.  Exits plotting session when user closes the
            window when interactive is True.

        interactive_update: bool, optional
            Disabled by default.  Allows user to non-blocking draw,
            user should call Update() in each iteration.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores window_size.
            Default False.

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up

        """
        # reset unless camera for the first render unless camera is set
        if self.first_time and not self.camera_set:
            self.camera_position = self.get_default_cam_pos()
            self.renderer.ResetCamera()
            self.first_time = False

        if title:
            self.ren_win.SetWindowName(title)

        # if full_screen:
        if full_screen:
            self.ren_win.SetFullScreen(True)
            self.ren_win.BordersOn()  # super buggy when disabled
        else:
            if window_size is None:
                window_size = rcParams['window_size']
            self.ren_win.SetSize(window_size[0], window_size[1])

        # Render
        log.debug('Rendering')
        self.ren_win.Render()

        if interactive and (not self.off_screen):
            try:  # interrupts will be caught here
                log.debug('Starting iren')
                self.iren.Initialize()
                if not interactive_update:
                    self.iren.Start()
            except KeyboardInterrupt:
                log.debug('KeyboardInterrupt')
                self.close()
                raise KeyboardInterrupt

        # Get camera position before closing
        cpos = self.camera_position

        if self.notebook:
            # sanity check
            try:
                import IPython
            except ImportError:
                raise Exception('Install IPython to display image in a notebook')

            img = PIL.Image.fromarray(self.screenshot())
            disp = IPython.display.display(img)

        # take screenshot
        if screenshot:
            if screenshot == True:
                img = self.screenshot()
            else:
                img = self.screenshot(screenshot)

        if auto_close:
            self.close()

        if self.notebook:
            return disp

        if screenshot:
            return cpos, img

        return cpos

    def plot(self, *args, **kwargs):
        """ Present for backwards compatibility. Use `show()` instead """
        return self.show(*args, **kwargs)

    def render(self):
        """ renders main window """
        self.ren_win.Render()


def arrows_actor(pdata):
    """ Creates an actor composed of arrows """

    # Create arrow object
    arrow = vtk.vtkArrowSource()
    arrow.Update()
    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceData(arrow.GetOutput())
    glyph3D.SetInputData(pdata)
    glyph3D.SetVectorModeToUseVector()
    glyph3D.Update()

    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())

    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()

    return actor


def single_triangle():
    """ A single PolyData triangle """
    points = np.zeros((3, 3))
    points[1] = [1, 0, 0]
    points[2] = [0.5, 0.707, 0]
    cells = np.array([[3, 0, 1, 2]], ctypes.c_long)
    return vtki.PolyData(points, cells)


def parse_color(color):
    """ Parses color into a vtk friendly rgb list """
    if color is None:
        color = rcParams['color']
    if isinstance(color, str):
        return vtki.string_to_rgb(color)
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
