"""
vtki plotting module
"""
import time
import logging
import ctypes
import PIL.Image
from subprocess import Popen, PIPE
import os
from multiprocessing import Process
import colorsys

import vtk
from vtk.util import numpy_support as VN

import numpy as np
import vtki
from vtki.utilities import get_scalar, wrap, is_vtki_obj
from vtki.container import MultiBlock
import imageio


FONT_KEYS = {'arial': vtk.VTK_ARIAL,
             'courier': vtk.VTK_COURIER,
             'times': vtk.VTK_TIMES}


log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

DEFAULT_WINDOW_SIZE = [1024, 768]
DEFAULT_BACKGROUND = [0.3, 0.3, 0.3]
DEFAULT_POSITION = [1, 1, 1]
DEFAULT_VIEWUP = [0, 0, 1]


def run_from_ipython():
    """ returns True when run from iPython """
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


def plot(var_item, off_screen=False, full_screen=False, screenshot=None,
         interactive=True, cpos=None, window_size=DEFAULT_WINDOW_SIZE,
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
    if is_vtki_obj(var_item):
        plotter._update_bounds(var_item.GetBounds())

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
    cpos = plotter.plot(window_size=window_size,
                        autoclose=False,
                        interactive=interactive,
                        full_screen=full_screen)

    # take screenshot
    img = plotter.screenshot()
    if screenshot:
        if screenshot == True:
            pass
        else:
            img = plotter.screenshot(screenshot)

    # close and return camera position and maybe image
    plotter.close()

    if screenshot:
        return cpos, img
    else:
        return cpos


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


class Plotter(object):
    """
    Plotting object to display vtk meshes or numpy arrays.

    Example
    -------
    plotter = Plotter()
    plotter.add_mesh(mesh, color='red')
    plotter.add_mesh(another_mesh, color='blue')
    plotter.plot()

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
        log.debug('Initializing')
        def onTimer(iren, eventId):
            if 'TimerEvent' == eventId:
                # TODO: python binding didn't provide
                # third parameter, which indicate right timer id
                # timer_id = iren.GetCommand(44)
                # if timer_id != self.right_timer_id:
                #     return
                self.iren.TerminateApp()

        self._labels = []

        if notebook is None:
            if run_from_ipython():
                notebook = type(get_ipython()).__module__.startswith('ipykernel.')

        self.notebook = notebook
        if self.notebook:
            off_screen = True
        self.off_screen = off_screen

        # initialize render window
        self.renderer = vtk.vtkRenderer()
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.AddRenderer(self.renderer)

        if self.off_screen:
            self.ren_win.SetOffScreenRendering(1)
        else:  # Allow user to interact
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetDesiredUpdateRate(30.0)
            self.iren.SetRenderWindow(self.ren_win)
            istyle = vtk.vtkInteractorStyleTrackballCamera()
            self.iren.SetInteractorStyle(istyle)
            self.iren.AddObserver("KeyPressEvent", self.key_press_event)

        # Set background
        self.set_background(DEFAULT_BACKGROUND)

        # initialize image filter
        self.ifilter = vtk.vtkWindowToImageFilter()
        self.ifilter.SetInput(self.ren_win)
        self.ifilter.SetInputBufferTypeToRGB()
        self.ifilter.ReadFrontBufferOff()

        # add timer event if interactive render exists
        if hasattr(self, 'iren'):
            self.iren.AddObserver(vtk.vtkCommand.TimerEvent, onTimer)

        # track if the camera has been setup
        self.camera_set = False
        self.first_time = True
        self.bounds = [0,1, 0,1, 0,1]

    def _update_bounds(self, bounds):
        def update_axis(ax):
            if bounds[ax*2] < self.bounds[ax*2]:
                self.bounds[ax*2] = bounds[ax*2]
            if bounds[ax*2+1] > self.bounds[ax*2+1]:
                self.bounds[ax*2+1] = bounds[ax*2+1]
        for ax in range(3):
            update_axis(ax)
        return

    def get_default_cam_pos(self):
        """
        Returns the default focal points and viewup. Uses ResetCamera to
        make a useful view.
        """
        bounds = self.bounds
        x = (bounds[1] + bounds[0])/2
        y = (bounds[3] + bounds[2])/2
        z = (bounds[5] + bounds[4])/2
        focal_pt = [x, y, z]
        return [np.array(DEFAULT_POSITION) + np.array(focal_pt),
                focal_pt, DEFAULT_VIEWUP]

    def key_press_event(self, obj, event):
        """ Listens for key press event """
        key = self.iren.GetKeySym()
        log.debug('Key %s pressed' % key)
        if key == 'q':
            self.q_pressed = True
        elif key == 'b':
            self.observer = self.iren.AddObserver('LeftButtonPressEvent',
                                                  self.left_button_down)

    def left_button_down(self, obj, eventType):
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
            self.render()
            Plotter.last_update_time = curr_time
        else:
            if force_redraw:
                self.iren.Render()

    def add_mesh(self, mesh, color=None, style=None,
                 scalars=None, rng=None, stitle=None, showedges=True,
                 psize=5.0, opacity=1, linethick=None, flipscalars=False,
                 lighting=False, ncolors=256, interpolatebeforemap=False,
                 colormap=None, label=None, **kwargs):
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
            be sized as a single vector.

        rng : 2 item list, optional
            Range of mapper for scalars.  Defaults to minimum and maximum of
            scalars array.  Example: [-1, 2]

        stitle : string, optional
            Scalar title.  By default there is no scalar legend bar.  Setting
            this creates the legend bar and adds a title to it.  To create a
            bar with no title, use an empty string (i.e. '').

        showedges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        psize : float, optional
            Point size.  Applicable when style='points'.  Default 5.0

        opacity : float, optional
            Opacity of mesh.  Should be between 0 and 1.  Default 1.0

        linethick : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default None.

        flipscalars : bool, optional
            Flip direction of colormap.

        lighting : bool, optional
            Enable or disable view direction lighting.  Default False.

        ncolors : int, optional
            Number of colors to use when displaying scalars.  Default
            256.

        interpolatebeforemap : bool, optional
            Enabling makes for a smoother scalar display.  Default
            False

        colormap : str, optional
           Colormap string.  See available matplotlib colormaps.  Only
           applicable for when displaying scalars.  Defaults None
           (rainbow).  Requires matplotlib.

        Returns
        -------
        actor: vtk.vtkActor
            VTK actor of the mesh.
        """
        if isinstance(mesh, np.ndarray):
            mesh = vtki.PolyData(mesh)
            style = 'points'

        # Convert the VTK data object to a vtki wrapped object if neccessary
        if not is_vtki_obj(mesh):
            mesh = wrap(mesh)


        if isinstance(mesh, MultiBlock):
            for idx in range(mesh.GetNumberOfBlocks()):
                data = wrap(mesh.GetBlock(idx))
                self.add_mesh(data, color=color, style=style,
                             scalars=scalars, rng=rng, stitle=stitle, showedges=showedges,
                             psize=psize, opacity=opacity, linethick=linethick, flipscalars=flipscalars,
                             lighting=lighting, ncolors=ncolors, interpolatebeforemap=interpolatebeforemap,
                             colormap=colormap, label=label, **kwargs)
            return


        # set main values
        self.mesh = mesh
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(self.mesh)
        actor, prop = self.add_actor(self.mapper)
        if is_vtki_obj(mesh):
            self._update_bounds(mesh.GetBounds())

        # Scalar formatting ===================================================
        title = 'Data' if stitle is None else stitle
        if scalars is not None:
            # if scalars is a string, then get the first array found with that name
            append_scalars = True
            if isinstance(scalars, str):
                title = scalars
                scalars = get_scalar(mesh, scalars)
                if stitle is None:
                    stitle = title
                append_scalars = False

            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            if scalars.ndim != 1:
                scalars = scalars.ravel()

            if scalars.dtype == np.bool:
                scalars = scalars.astype(np.float)

            # Scalar interpolation approach
            if scalars.size == mesh.GetNumberOfPoints():
                self.mesh._add_point_scalar(scalars, title, True)
                self.mapper.SetScalarModeToUsePointData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(ncolors)
                if interpolatebeforemap:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
            elif scalars.size == mesh.GetNumberOfCells():
                self.mesh._add_cell_scalar(scalars, title, True)
                self.mapper.SetScalarModeToUseCellData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(ncolors)
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
            if colormap is not None:
                try:
                    from matplotlib.cm import get_cmap
                except ImportError:
                    raise Exception('colormap requires matplotlib')
                cmap = get_cmap(colormap)
                ctable = cmap(np.linspace(0, 1, ncolors))*255
                ctable = ctable.astype(np.uint8)
                if flipscalars:
                    ctable = np.ascontiguousarray(ctable[::-1])
                table.SetTable(VN.numpy_to_vtk(ctable))

            else:  # no colormap specified
                if flipscalars:
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

        prop.SetPointSize(psize)

        # edge display style
        if showedges:
            prop.EdgeVisibilityOn()

        rgb_color = parse_color(color)
        prop.SetColor(rgb_color)
        prop.SetOpacity(opacity)

        # legend label
        if label:
            assert isinstance(label, str), 'Label must be a string'
            self._labels.append([single_triangle(), label, rgb_color])

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if linethick:
            prop.SetLineWidth(linethick)

        # Add scalar bar if available
        if stitle is not None:
            self.add_scalar_bar(stitle)

        return actor

    def add_actor(self, uinput):
        """adds an actor to render window.  creates an actor if input is a
        mapper"""
        if isinstance(uinput, vtk.vtkMapper):
            actor = vtk.vtkActor()
            actor.SetMapper(uinput)
        else:
            actor = uinput
        self.renderer.AddActor(actor)

        return actor, actor.GetProperty()

    @property
    def camera(self):
        return self.renderer.GetActiveCamera()

    def add_bounds_axes(self, mesh=None, bounds=None, show_xaxis=True,
                        show_yaxis=True, show_zaxis=True, show_xlabels=True,
                        show_ylabels=True, show_zlabels=True, italic=False,
                        bold=True, shadow=False, fontsize=16,
                        font_family='courier', color='w',
                        xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis',
                        use_2dmode=True):
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

        fontsize : float, optional
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

        xtitle : string, optional
            Title of the x axis.  Default "X Axis"

        ytitle : string, optional
            Title of the y axis.  Default "Y Axis"

        ztitle : string, optional
            Title of the z axis.  Default "Z Axis"

        use_2dmode : bool, optional
            A bug with vtk 6.3 in Windows seems to cause this function to crash
            this can be enabled for smoother plotting for other enviornments.

        Returns
        -------
        cubeAxesActor : vtk.vtkCubeAxesActor
            Bounds actor

        """

        # Use last input mesh if availble
        if not mesh and not bounds:
            if not hasattr(self, 'mesh'):
                raise Exception('Specify bounds or first input a mesh')
            mesh = self.mesh

        # create actor
        cubeAxesActor = vtk.vtkCubeAxesActor()
        cubeAxesActor.SetUse2DMode(False)

        # set bounds
        if not bounds:
            bounds = mesh.GetBounds()
        cubeAxesActor.SetBounds(mesh.GetBounds())

        # show or hide axes
        cubeAxesActor.SetXAxisVisibility(show_xaxis)
        cubeAxesActor.SetYAxisVisibility(show_yaxis)
        cubeAxesActor.SetZAxisVisibility(show_zaxis)

        # disable minor ticks
        cubeAxesActor.XAxisMinorTickVisibilityOff()
        cubeAxesActor.YAxisMinorTickVisibilityOff()
        cubeAxesActor.ZAxisMinorTickVisibilityOff()

        cubeAxesActor.SetCamera(self.camera)

        # set color
        color = parse_color(color)
        cubeAxesActor.GetXAxesLinesProperty().SetColor(color)
        cubeAxesActor.GetYAxesLinesProperty().SetColor(color)
        cubeAxesActor.GetZAxesLinesProperty().SetColor(color)

        # empty arr
        empty_str = vtk.vtkStringArray()
        empty_str.InsertNextValue('')

        # show lines
        if show_xaxis:
            cubeAxesActor.SetXTitle(xtitle)
        else:
            cubeAxesActor.SetXTitle('')
            cubeAxesActor.SetAxisLabels(0, empty_str)

        if show_yaxis:
            cubeAxesActor.SetYTitle(ytitle)
        else:
            cubeAxesActor.SetYTitle('')
            cubeAxesActor.SetAxisLabels(1, empty_str)

        if show_zaxis:
            cubeAxesActor.SetZTitle(ztitle)
        else:
            cubeAxesActor.SetZTitle('')
            cubeAxesActor.SetAxisLabels(2, empty_str)

        # show labels
        if not show_xlabels:
            cubeAxesActor.SetAxisLabels(0, empty_str)

        if not show_ylabels:
            cubeAxesActor.SetAxisLabels(1, empty_str)

        if not show_zlabels:
            cubeAxesActor.SetAxisLabels(2, empty_str)

        # set font
        font_family = parse_font_family(font_family)
        for i in range(3):
            cubeAxesActor.GetTitleTextProperty(i).SetFontSize(fontsize)
            cubeAxesActor.GetTitleTextProperty(i).SetColor(color)
            cubeAxesActor.GetTitleTextProperty(i).SetFontFamily(font_family)
            cubeAxesActor.GetTitleTextProperty(i).SetBold(bold)

            cubeAxesActor.GetLabelTextProperty(i).SetFontSize(fontsize)
            cubeAxesActor.GetLabelTextProperty(i).SetColor(color)
            cubeAxesActor.GetLabelTextProperty(i).SetFontFamily(font_family)
            cubeAxesActor.GetLabelTextProperty(i).SetBold(bold)

        self.add_actor(cubeAxesActor)
        self.cubeAxesActor = cubeAxesActor
        return cubeAxesActor

    def add_scalar_bar(self, title=None, nlabels=5, italic=False, bold=True,
                       title_fontsize=None, label_fontsize=None, color=None,
                       font_family='courier', shadow=False):
        """
        Creates scalar bar using the ranges as set by the last input mesh.

        Parameters
        ----------
        title : string, optional
            Title of the scalar bar.  Default None

        nlabels : int, optional
            Number of labels to use for the scalar bar.

        italic : bool, optional
            Italicises title and bar labels.  Default False.

        bold  : bool, optional
            Bolds title and bar labels.  Default True

        title_fontsize : float, optional
            Sets the size of the title font.  Defaults to None and is sized
            automatically.

        label_fontsize : float, optional
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

        Notes
        -----
        Setting title_fontsize, or label_fontsize disables automatic font
        sizing for both the title and label.


        """
        # check if maper exists
        if not hasattr(self, 'mapper'):
            raise Exception('Mapper does not exist.  ' +
                            'Add a mesh with scalars first.')

        # parse color
        color = parse_color(color)

        # Create scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.mapper.GetLookupTable())
        self.scalar_bar.SetNumberOfLabels(nlabels)

        if label_fontsize or title_fontsize:
            self.scalar_bar.UnconstrainedFontSizeOn()

        if nlabels:
            label_text = self.scalar_bar.GetLabelTextProperty()
            label_text.SetColor(color)
            label_text.SetShadow(shadow)

            # Set font
            label_text.SetFontFamily(parse_font_family(font_family))
            label_text.SetItalic(italic)
            label_text.SetBold(bold)
            if label_fontsize:
                label_text.SetFontSize(label_fontsize)

        # Set properties
        if title:
            self.scalar_bar.SetTitle(title)
            title_text = self.scalar_bar.GetTitleTextProperty()

            title_text.SetItalic(italic)
            title_text.SetBold(bold)
            title_text.SetShadow(shadow)
            if title_fontsize:
                title_text.SetFontSize(title_fontsize)

            # Set font
            title_text.SetFontFamily(parse_font_family(font_family))

            # set color
            title_text.SetColor(color)

        self.add_actor(self.scalar_bar)

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
        mesh.GetPoints().Modified()

        if render:
            self.render()

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
            self.render()

    def close(self):
        """ closes render window """

        # must close out axes marker
        if hasattr(self, 'axes_widget'):
            del self.axes_widget

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

    def add_text(self, text, position=[10, 10], fontsize=50, color=None,
                font='courier', shadow=False):
        """
        Adds text to plot object

        Parameters
        ----------
        font : string, optional
            Font name may be courier, times, or arial

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        Returns
        -------
        textActor : vtk.vtkTextActor
            Text actor added to plot

        """
        self.textActor = vtk.vtkTextActor()
        self.textActor.SetPosition(position)
        self.textActor.GetTextProperty().SetFontSize(fontsize)
        self.textActor.GetTextProperty().SetColor(parse_color(color))
        self.textActor.GetTextProperty().SetFontFamily(FONT_KEYS[font])
        self.textActor.GetTextProperty().SetShadow(shadow)
        self.textActor.SetInput(text)
        self.add_actor(self.textActor)
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
    def image(self):
        """ Returns an image array of current render window """
        window_size = self.ren_win.GetSize()

        # Update filter and grab pixels
        self.ifilter.Modified()
        self.ifilter.Update()
        image = self.ifilter.GetOutput()
        img_array = vtki.utilities.point_scalar(image, 'ImageScalars')

        # Reshape and write
        return img_array.reshape((window_size[1], window_size[0], -1))[::-1]

    def add_lines(self, lines, color=[1, 1, 1], width=5, label=None):
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
            # single_line = lines.ExtractSelectionCells([0])
            self._labels.append([lines, label, rgb_color])

        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(rgb_color)
        actor.GetProperty().SetColor(rgb_color)
        actor.GetProperty().LightingOff()

        # Add to renderer
        self.renderer.AddActor(actor)

        return actor

    def add_point_labels(self, points, labels, italic=False, bold=True,
                         fontsize=16, textcolor='k',
                         font_family='courier', shadow=False,
                         showpoints=True, pointcolor='k', pointsize=5):
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

        fontsize : float, optional
            Sets the size of the title font.  Defaults to 16.

        textcolor : string or 3 item list, optional, defaults to black
            Color of text.
            Either a string, rgb list, or hex color string.  For example:
                textcolor='white'
                textcolor='w'
                textcolor=[1, 1, 1]
                textcolor='#FFFFFF'

        font_family : string, optional
            Font family.  Must be either courier, times, or arial.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        showpoints : bool, optional
            Controls if points are visible.  Default True

        pointcolor : string or 3 item list, optional, defaults to black
            Color of points (if visible).
            Either a string, rgb list, or hex color string.  For example:
                textcolor='white'
                textcolor='w'
                textcolor=[1, 1, 1]
                textcolor='#FFFFFF'

        pointsize : float, optional
            Size of points (if visible)

        Returns
        -------
        labelMapper : vtk.vtkvtkLabeledDataMapper
            VTK label mapper.  Can be used to change properties of the labels.

        """
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
        textprop.SetFontSize(fontsize)
        textprop.SetFontFamily(parse_font_family(font_family))
        textprop.SetColor(parse_color(textcolor))
        textprop.SetShadow(shadow)
        labelMapper.SetLabelModeToLabelFieldData()
        labelMapper.SetFieldDataName('labels')

        labelActor = vtk.vtkActor2D()
        labelActor.SetMapper(labelMapper)

        # add points
        if showpoints:
            style = 'points'
        else:
            style = 'surface'
        self.add_mesh(vtkpoints, style=style, color=pointcolor,
                      psize=pointsize)

        self.add_actor(labelActor)
        return labelMapper

    def add_points(self, points, **kwargs):
        """ Add points to a mesh """
        kwargs['style'] = 'points'
        self.add_mesh(points, **kwargs)

    def add_arrows(self, cent, direction, mag=1):
        """ Adds arrows to plotting object """

        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        pdata = vtki.vector_poly_data(cent, direction * mag)
        arrows = arrows_actor(pdata)
        self.add_actor(arrows)

        return arrows, pdata

    # def AddLineSegments(self, points, edges, color=None, scalars=None,
    #                     ncolors=256):
    #     """ Adds arrows to plotting object """

    #     cent = (points[edges[:, 0]] + points[edges[:, 1]]) / 2
    #     direction = points[edges[:, 1]] - points[edges[:, 0]]
    #     pdata = vtki.vector_poly_data(cent, direction)
    #     arrows, mapper = line_segments_actor(pdata)

    #     # set color
    #     if isinstance(color, str):
    #         color = vtki.string_to_rgb(color)
    #         mapper.ScalarVisibilityOff()
    #         arrows.GetProperty().SetColor(color)

    #     if scalars is not None:
    #         if scalars.size == edges.shape[0]:
    #             pdata.add_cell_scalars(scalars, '', True)
    #             mapper.SetScalarModeToUseCellData()
    #             mapper.GetLookupTable().SetNumberOfTableValues(ncolors)
    #             # if interpolatebeforemap:
    #                 # self.mapper.InterpolateScalarsBeforeMappingOn()
    #         else:
    #             raise Exception('Number of scalars must match number of edges')

    #     # add to rain class
    #     self.add_actor(arrows)
    #     return arrows

    @property
    def camera_position(self):
        """ Returns camera position of active render window """
        return [self.camera.GetPosition(),
                self.camera.GetFocalPoint(),
                self.camera.GetViewUp()]

    @camera_position.setter
    def camera_position(self, cameraloc):
        """ Set camera position of active render window """
        if cameraloc is None:
            return

        # everything is set explicitly
        self.camera.SetPosition(cameraloc[0])
        self.camera.SetFocalPoint(cameraloc[1])
        self.camera.SetViewUp(cameraloc[2])
        # Rest camera so it slides along the vector defined from camera position
        #   to focal point until all of the actors can be seen.
        self.renderer.ResetCamera()
        # reset clipping range
        self.renderer.ResetCameraClippingRange()
        self.camera_set = True

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
            color = DEFAULT_BACKGROUND
        elif isinstance(color, str):
            color = vtki.string_to_rgb(color)

        self.renderer.SetBackground(color)

    def add_legend(self, labels=None, bcolor=[0.5, 0.5, 0.5], border=False,
                   size=None):
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

        Returns
        -------
        legend : vtk.vtkLegendBoxActor
            Actor for the legend.

        Examples
        --------
        >>> import vtki
        >>> plotter = vtki.Plotter()
        >>> plotter.add_mesh(mesh, label='My Mesh')
        >>> plotter.add_mesh(othermesh, 'k', label='My Other Mesh')
        >>> plotter.add_legend()
        >>> plotter.plot()

        Alternative manual example

        >>> import vtki
        >>> legend_entries = []
        >>> legend_entries.append(['My Mesh', 'w'])
        >>> legend_entries.append(['My Other Mesh', 'k'])
        >>> plotter = vtki.Plotter()
        >>> plotter.add_mesh(mesh)
        >>> plotter.add_mesh(othermesh, 'k')
        >>> plotter.add_legend(legend_entries)
        >>> plotter.plot()
        """
        legend = vtk.vtkLegendBoxActor()

        if labels is None:
            # use existing labels
            if not self._labels:
                raise Exception('No labels input.\n\n' +
                                'Add labels to individual items when adding them to' +
                                'the plotting object with the "label=" parameter.  ' +
                                'or enter them as the "labels" parameter.')

            legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels):
                legend.SetEntry(i, vtk_object, text, parse_color(color))

        else:
            legend.SetNumberOfEntries(len(labels))
            legendface = single_triangle()
            for i, (text, color) in enumerate(labels):
                legend.SetEntry(i, legendface, text, parse_color(color))

        if size:
            legend.SetPosition2(size[0], size[1])

        if bcolor is None:
            legend.UseBackgroundOff()
        else:
            legend.UseBackgroundOn()
            legend.SetBackgroundColor(bcolor)

        if border:
            legend.BorderOn()
        else:
            legend.BorderOff()

        # Add to renderer
        self.renderer.AddActor(legend)
        return legend

    def _plot(self, title=None, window_size=DEFAULT_WINDOW_SIZE, interactive=True,
              autoclose=True, interactive_update=False, full_screen=False):
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

        autoclose : bool, optional
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
        if title:
            self.ren_win.SetWindowName(title)

        # if full_screen:
        if full_screen:
            self.ren_win.SetFullScreen(True)
            self.ren_win.BordersOn()  # super buggy when disabled
        else:
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
                raise Exception('Install iPython to display image in a notebook')

            img = PIL.Image.fromarray(self.screenshot())
            disp = IPython.display.display(img)

        if autoclose:
            self.close()

        if self.notebook:
            return disp

        return cpos

    def plot(self, title=None, window_size=DEFAULT_WINDOW_SIZE, interactive=True,
             autoclose=True, in_background=False, interactive_update=False,
             full_screen=False):
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

        autoclose : bool, optional
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

        def PlotFun():
            return self._plot(title,
                              window_size=window_size,
                              interactive=interactive,
                              autoclose=autoclose,
                              interactive_update=interactive_update,
                              full_screen=full_screen)

        if in_background:
            log.debug('Starting process')
            process = Process(target=PlotFun)
            process.start()
            return process
        else:
            return PlotFun()

    def remove_actor(self, actor):
        """
        Removes an actor from the Plotter.

        Parameters
        ----------
        actor : vtk.vtkActor
            Actor that has previously added to the Plotter.
        """
        self.renderer.RemoveActor(actor)

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
        return self.marker_actor

    def add_axes(self):
        """ Add an interactive axes widget """
        if hasattr(self, 'axes_widget'):
            raise Exception('plotter already has axes widget')
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        if hasattr(self, 'iren'):
            self.axes_widget.SetInteractor(self.iren)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.InteractiveOn()

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
        >>> sphere = vtki.Sphere()
        >>> plotter.Plotter()
        >>> plotter.add_mesh(sphere)
        >>> plotter.screenshot('screenshot.png')
        """
        # remove_ren_win = False
        # if not hasattr(self, 'ren_win'):
            # self.plot(auto_close=False, interactive=False)
            # raise Exception('Render window has been closed.\n'
                            # 'Run again with plot(autoclose=False)')

        # configure image filter
        if transparent_background:
            self.ifilter.SetInputBufferTypeToRGBA()
        else:
            self.ifilter.SetInputBufferTypeToRGB()

        # this needs to be called twice for some reason,  debug later
        self.render()
        img = self.image
        img = self.image

        if not img.size:
            raise Exception('Empty image.  Have you run plot() first?')

        # write screenshot to file
        if filename:
            imageio.imwrite(filename, img)

        # if remove_ren_win:
        #     self.close()

        return img

    def render(self):
        self.ren_win.Render()

    def set_focus(self, point):
        """ sets focus to a point """
        self.camera.SetFocalPoint(point)

    # def __del__(self):
        # print('collected')
    #     log.debug('Object collected')


# def line_segments_actor(pdata):
#     # Create arrow object
#     lines_source = vtk.vtkLineSource()
#     lines_source.Update()
#     glyph3D = vtk.vtkGlyph3D()
#     glyph3D.SetSourceData(lines_source.GetOutput())
#     glyph3D.SetInputData(pdata)
#     glyph3D.SetVectorModeToUseVector()
#     glyph3D.Update()

#     # Create mapper
#     mapper = vtk.vtkDataSetMapper()
#     mapper.SetInputConnection(glyph3D.GetOutputPort())

#     # Create actor
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().LightingOff()

#     return actor, mapper


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
        return [1, 1, 1]
    elif isinstance(color, str):
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
