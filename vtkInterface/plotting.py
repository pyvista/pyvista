"""
vtk plotting module

"""
from multiprocessing import Process
import colorsys
import numpy as np
import vtkInterface
import imageio
import time
import logging

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# for readthedocs
try:
    import vtk
    from vtk.util import numpy_support as VN
    font_keys = {'arial': vtk.VTK_ARIAL,
                 'courier': vtk.VTK_COURIER,
                 'times': vtk.VTK_TIMES}
except ImportError:
    pass


def Plot(mesh, **args):
    """
    Convenience plotting function for a vtk object.

    Parameters
    ----------
    mesh : vtk object
        vtk object to plot.

    **args : various, optional
        See help(vtkInterface.PlotClass.AddMesh)

    screenshot : str, optional
        Saves screenshot to file when enabled.  See:
        help(vtkinterface.PlotClass.TakeScreenShot).  Default disabled.

    window_size : list, optional
        Window size in pixels.  Defaults to [1024, 768]

    show_bounds : bool, optional
        Shows mesh bounds when True.  Default False.

    Returns
    -------
    cpos : list
        List of camera position, focal point, and view up.

    """
    if 'off_screen' in args:
        off_screen = args['off_screen']
        del args['off_screen']
    else:
        off_screen = False

    if 'screenshot' in args:
        filename = args['screenshot']
        del args['screenshot']
    else:
        filename = None

    if 'interactive' in args:
        interactive = args['interactive']
        del args['interactive']
    else:
        interactive = True

    if 'cpos' in args:
        cpos = args['cpos']
        del args['cpos']
    else:
        cpos = None

    if 'window_size' in args:
        window_size = args['window_size']
        del args['window_size']
    else:
        window_size = [1024, 768]

    # add bounds
    if 'show_bounds' in args:
        show_bounds = True
        del args['show_bounds']
    else:
        show_bounds = False

    # create plotting object and add mesh
    plobj = PlotClass(off_screen=off_screen)

    if 'background' in args:
        plobj.SetBackground(args['background'])
        del args['background']

    if isinstance(mesh, np.ndarray):
        plobj.AddPoints(mesh, **args)
    else:
        plobj.AddMesh(mesh, **args)

    if 'text' in args:
        plobj.AddText(args['text'])

    if show_bounds:
        plobj.AddBoundsAxes()

    # Set camera
    if cpos:
        plobj.SetCameraPosition(cpos)

    cpos = plobj.Plot(window_size=window_size,
                      autoclose=False,
                      interactive=interactive)

    # take screenshot
    if filename:
        plobj.TakeScreenShot(filename)

    # close and return camera position
    plobj.Close()
    return cpos


def PlotArrows(cent, direction):
    """ Plots arrows """
    plotter = PlotClass()
    plotter.AddArrows(cent, direction)
    return plotter.Plot()


class PlotClass(object):
    """
    Plotting object to display vtk meshes or numpy arrays.

    Example
    -------
    plobj = PlotClass()
    plobj.AddMesh(mesh, color='red')
    plobj.AddMesh(another_mesh, color='blue')
    plobj.Plot()

    Parameters
    ----------
    off_screen : bool, optional
        Renders off screen when False.  Useful for automated screenshots.

    """
    last_update_time = 0.0
    q_pressed = False
    right_timer_id = -1

    def __init__(self, off_screen=False):

        def onTimer(iren, eventId):
            if 'TimerEvent' == eventId:
                # TODO: python binding didn't provide
                # third parameter, which indicate right timer id
                # timer_id = iren.GetCommand(44)
                # if timer_id != self.right_timer_id:
                #     return
                self.iren.TerminateApp()

        """
        Initialize a vtk plotting object
        """
        self.off_screen = off_screen

        # initialize render window
        self.renderer = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)

        if self.off_screen:
            self.renWin.SetOffScreenRendering(1)
        else:  # Allow user to interact
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetDesiredUpdateRate(30.0)
            self.iren.SetRenderWindow(self.renWin)
            istyle = vtk.vtkInteractorStyleTrackballCamera()
            self.iren.SetInteractorStyle(istyle)
            self.iren.AddObserver("KeyPressEvent", self.KeyPressEvent)

        # Set background
        self.renderer.SetBackground(0.3, 0.3, 0.3)

        # initialize image filter
        self.ifilter = vtk.vtkWindowToImageFilter()
        self.ifilter.SetInput(self.renWin)
        self.ifilter.SetInputBufferTypeToRGB()
        self.ifilter.ReadFrontBufferOff()

        # initialize movie type
        self.movietype = None

        # add timer event if interactive render exists
        if hasattr(self, 'iren'):
            self.iren.AddObserver(vtk.vtkCommand.TimerEvent, onTimer)

    def KeyPressEvent(self, obj, event):
        """ Listens for q key press """
        key = self.iren.GetKeySym()
        if key == 'q':
            self.q_pressed = True
        elif key == 'b':
            self.observer = self.iren.AddObserver('LeftButtonPressEvent',
                                                  self.OnLeftButtonDown)

    def OnLeftButtonDown(self, obj, eventType):

        # Get 2D click location on window
        clickPos = self.iren.GetEventPosition()

        # Get corresponding click location in the 3D plot
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
        self.pickpoint = np.asarray(picker.GetPickPosition()).reshape((-1, 3))

        self.iren.AddObserver('LeftButtonPressEvent',
                              self.OnLeftButtonDown)

    def Update(self, stime=1, force_redraw=True):
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
        if PlotClass.last_update_time > curr_time:
            PlotClass.last_update_time = curr_time

        update_rate = self.iren.GetDesiredUpdateRate()
        if (curr_time - PlotClass.last_update_time) > (1.0/update_rate):
            self.right_timer_id = self.iren.CreateRepeatingTimer(stime)
            self.iren.Start()
            self.iren.DestroyTimer(self.right_timer_id)
            self.Render()
            PlotClass.last_update_time = curr_time
        else:
            if force_redraw:
                self.iren.Render()

    def AddMesh(
            self,
            mesh,
            color=None,
            style=None,
            scalars=None,
            rng=None,
            stitle=None,
            showedges=True,
            psize=5.0,
            opacity=1,
            linethick=None,
            flipscalars=False,
            lighting=False,
            ncolors=256,
            interpolatebeforemap=False,
            colormap=None,
            **kwargs):
        """
        Adds a vtk unstructured, structured, or polymesh to the plotting object

        Parameters
        ----------
        mesh : vtk unstructured, structured, or polymesh
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
            Number of colors to use when displaying scalars.  Default 256.

        interpolatebeforemap : bool, default False
            Enabling makes for a smoother scalar display.  Default False

        colormap : str, optional
           Colormap string.  See available matplotlib colormaps.  Only applicable for
           when displaying scalars.  Defaults None (rainbow).  Requires matplotlib.

        Returns
        -------
        actor: vtk.vtkActor
            VTK actor of the mesh.
        """
        # set main values
        self.mesh = mesh
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(self.mesh)
        actor, prop = self.AddActor(self.mapper)

        # Scalar formatting ===================================================
        if scalars is not None:
            # convert to numpy array
            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            # ravel if not 1 dimentional
            if scalars.ndim != 1:
                scalars = scalars.ravel()

            # Scalar interpolation approach
            if scalars.size == mesh.GetNumberOfPoints():
                self.mesh.AddPointScalars(scalars, '', True)
                self.mapper.SetScalarModeToUsePointData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(ncolors)
                if interpolatebeforemap:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
            elif scalars.size == mesh.GetNumberOfCells():
                self.mesh.AddCellScalars(scalars, '')
                self.mapper.SetScalarModeToUseCellData()
            else:
                raise Exception('Number of scalars (%d) ' % scalars.size +
                                'must match either the number of points ' +
                                '(%d) ' % mesh.GetNumberOfPoints() +
                                'or the number of cells ' +
                                '(%d) ' % mesh.GetNumberOfCells())

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
        prop.SetColor(ParseColor(color))
        prop.SetOpacity(opacity)

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if linethick:
            prop.SetLineWidth(linethick)

        # Add scalar bar if available
        if stitle is not None:
            self.AddScalarBar(stitle)

        return actor

    def AddActor(self, uinput):
        """adds an actor to render window.  creates an actor if input is a
        mapper"""
        if isinstance(uinput, vtk.vtkMapper):
            actor = vtk.vtkActor()
            actor.SetMapper(uinput)
        else:
            actor = uinput
        self.renderer.AddActor(actor)

        return actor, actor.GetProperty()

    def AddBoundsAxes(self, mesh=None, bounds=None, show_xaxis=True,
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

        cubeAxesActor.SetCamera(self.renderer.GetActiveCamera())

        # set color
        color = ParseColor(color)
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
        font_family = ParseFontFamily(font_family)
        for i in range(3):
            cubeAxesActor.GetTitleTextProperty(i).SetFontSize(fontsize)
            cubeAxesActor.GetTitleTextProperty(i).SetColor(color)
            cubeAxesActor.GetTitleTextProperty(i).SetFontFamily(font_family)
            cubeAxesActor.GetTitleTextProperty(i).SetBold(bold)

            cubeAxesActor.GetLabelTextProperty(i).SetFontSize(fontsize)
            cubeAxesActor.GetLabelTextProperty(i).SetColor(color)
            cubeAxesActor.GetLabelTextProperty(i).SetFontFamily(font_family)
            cubeAxesActor.GetLabelTextProperty(i).SetBold(bold)

        self.AddActor(cubeAxesActor)
        self.cubeAxesActor = cubeAxesActor
        return cubeAxesActor

    def AddScalarBar(self, title=None, nlabels=5, italic=False, bold=True,
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
        color = ParseColor(color)

        # Create scalar bar
        self.scalarBar = vtk.vtkScalarBarActor()
        self.scalarBar.SetLookupTable(self.mapper.GetLookupTable())
        self.scalarBar.SetNumberOfLabels(nlabels)

        if label_fontsize or title_fontsize:
            self.scalarBar.UnconstrainedFontSizeOn()

        if nlabels:
            label_text = self.scalarBar.GetLabelTextProperty()
            label_text.SetColor(color)
            label_text.SetShadow(shadow)

            # Set font
            label_text.SetFontFamily(ParseFontFamily(font_family))
            label_text.SetItalic(italic)
            label_text.SetBold(bold)
            if label_fontsize:
                label_text.SetFontSize(label_fontsize)

        # Set properties
        if title:
            self.scalarBar.SetTitle(title)
            title_text = self.scalarBar.GetTitleTextProperty()

            title_text.SetItalic(italic)
            title_text.SetBold(bold)
            title_text.SetShadow(shadow)
            if title_fontsize:
                title_text.SetFontSize(title_fontsize)

            # Set font
            title_text.SetFontFamily(ParseFontFamily(font_family))

            # set color
            title_text.SetColor(color)

        self.renderer.AddActor(self.scalarBar)

    def UpdateScalars(self, scalars, mesh=None, render=True):
        """ updates scalars of object (point only for now)
        assumes last inputted mesh if mesh left empty
        """

        if mesh is None:
            mesh = self.mesh

        # get pointer to active point scalars
        if scalars.shape[0] == mesh.GetNumberOfPoints():
            s = VN.vtk_to_numpy(self.mesh.GetPointData().GetScalars())
            s[:] = scalars

        # get pointer to active cell scalars
        elif scalars.shape[0] == mesh.GetNumberOfCells():
            s = VN.vtk_to_numpy(self.mesh.GetCellData().GetScalars())
            s[:] = scalars

        if render:
            self.Render()

    def UpdatePointScalars(self, scalars, points=None, render=True):
        """ updates scalars of object (point only for now)
        assumes last inputted mesh if mesh left empty
        """

        if points is None:
            points = self.points

        # get pointer to active point scalars
        if scalars.shape[0] == points.GetNumberOfPoints():
            s = VN.vtk_to_numpy(points.GetPointData().GetScalars())
            s[:] = scalars

        if render:
            self.Render()

    def UpdateCoordinates(self, points, mesh=None, render=True):
        """
        Updates points of object (point only for now)
        assumes last inputted mesh if mesh left empty
        """
        if mesh is None:
            mesh = self.mesh

        mesh.points = points

        if render:
            self.Render()

    def Close(self):
        """ closes render window """

        if hasattr(self, 'renWin'):
            self.renWin.Finalize()
            del self.renWin

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

    def AddText(self, text, position=[10, 10], fontsize=50, color=None,
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
        self.textActor.GetTextProperty().SetColor(ParseColor(color))
        self.textActor.GetTextProperty().SetFontFamily(font_keys[font])
        self.textActor.GetTextProperty().SetShadow(shadow)
        self.textActor.SetInput(text)
        self.AddActor(self.textActor)

        return self.textActor

    def OpenMovie(self, filename, framerate=24, codec='libx264',
                  preset='medium'):
        """ Establishes a connection to the ffmpeg writer """

        # Attempt to load moviepy
        try:
            import moviepy.video.io.ffmpeg_writer as mwrite
        except BaseException:
            print('\n\nTo use this feature install moviepy and ffmpeg\n\n')
            import moviepy.video.io.ffmpeg_writer as mwrite

        # Create movie object and check if render window is active
        self.window_size = self.renWin.GetSize()
        if not self.window_size[0]:
            raise Exception('Run Plot first')

        self.mwriter = mwrite.FFMPEG_VideoWriter(filename, self.window_size,
                                                 framerate, codec=codec,
                                                 preset=preset)

        self.movietype = 'mp4'

    def OpenGif(self, filename):
        if filename[-3:] != 'gif':
            raise Exception('Unsupported filetype')
        self.mwriter = imageio.get_writer(filename, mode='I')

    def WriteFrame(self):
        """ Writes a single frame to the movie file """
        if self.movietype is 'mp4':
            self.mwriter.write_frame(self.GetImage())
        else:
            self.mwriter.append_data(self.GetImage())

    def GetImage(self):
        """ Returns an image array of current render window """
        window_size = self.renWin.GetSize()

        # Update filter and grab pixels
        self.ifilter.Modified()
        self.ifilter.Update()
        image = self.ifilter.GetOutput()
        img_array = vtkInterface.GetPointScalars(image, 'ImageScalars')

        # Reshape and write
        return img_array.reshape((window_size[1], window_size[0], -1))[::-1]

    def AddLines(self, lines, color=[1, 1, 1], width=5):
        """ Adds an actor to the renderwindow """

        if isinstance(lines, np.ndarray):
            lines = vtkInterface.MakeLine(lines)

        # Create mapper and add lines
        mapper = vtk.vtkDataSetMapper()
        vtkInterface.SetVTKInput(mapper, lines)

        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(color)
        actor.GetProperty().SetColor(ParseColor(color))
        actor.GetProperty().LightingOff()

        # Add to renderer
        self.renderer.AddActor(actor)

    def AddPointLabels(self, points, labels, bold=True, fontsize=16,
                       textcolor='k', font_family='courier', shadow=False,
                       showpoints=True, pointcolor='k', pointsize=5):
        """
        Creates a point actor with one label from list labels assigned to
        each point.

        Parameters
        ----------
        points : np.ndarray
            3 x n numpy array of points.

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

        vtkpoints = vtkInterface.MakePointMesh(points)

        vtklabels = vtk.vtkStringArray()
        vtklabels.SetName('labels')
        for item in labels:
            vtklabels.InsertNextValue(str(item))

        vtkpoints.GetPointData().AddArray(vtklabels)

        # create label mapper
        labelMapper = vtk.vtkLabeledDataMapper()
        labelMapper.SetInputData(vtkpoints)
        textprop = labelMapper.GetLabelTextProperty()
        textprop.SetBold(bold)
        textprop.SetFontSize(fontsize)
        textprop.SetFontFamily(ParseFontFamily(font_family))
        textprop.SetColor(ParseColor(textcolor))
        textprop.SetShadow(shadow)
        labelMapper.SetLabelModeToLabelFieldData()
        labelMapper.SetFieldDataName('labels')

        labelActor = vtk.vtkActor2D()
        labelActor.SetMapper(labelMapper)

        # add points
        if showpoints:
            self.AddMesh(vtkpoints, style='points', color=pointcolor,
                         psize=pointsize)
        else:
            self.AddMesh(vtkpoints)

        self.AddActor(labelActor)
        return labelMapper

    def AddPoints(self, points, color=None, psize=5, scalars=None,
                  rng=None, name='', opacity=1, stitle='', flipscalars=False,
                  colormap=None, ncolors=256):
        """ Adds a point actor or numpy points array to plotting object """

        # select color
        if color is None:
            color = [1, 1, 1]
        elif isinstance(color, str):
            color = vtkInterface.StringToRGB(color)

        # Convert to vtk points object if "points" is a numpy array
        if isinstance(points, np.ndarray):
            # check size of points
            if points.ndim != 2 or points.shape[1] != 3:
                try:
                    points = points.reshape((-1, 3))
                except:
                    raise Exception('Invalid point array shape'
                                    '%s' % str(points.shape))
            self.points = vtkInterface.MakeVTKPointsMesh(points)
        else:
            self.points = points

        # Create mapper and add lines
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.points)

        if np.any(scalars):
            self.points.AddPointScalars(scalars, name, True)
            mapper.SetScalarModeToUsePointData()
            mapper.GetLookupTable().SetNumberOfTableValues(ncolors)

            if not rng:
                rng = [np.min(scalars), np.max(scalars)]
            elif isinstance(rng, float):
                rng = [-rng, rng]

            if np.any(rng):
                mapper.SetScalarRange(rng[0], rng[1])

        # Flip if requested
        table = mapper.GetLookupTable()
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

        else:  # no colormap specifide
            mapper.GetLookupTable().SetHueRange(0.66667, 0.0)


        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(psize)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetOpacity(opacity)

        self.renderer.AddActor(actor)

        # Add scalar bar
        if stitle:
            self.scalarBar = vtk.vtkScalarBarActor()
            self.scalarBar.SetLookupTable(mapper.GetLookupTable())

            self.scalarBar.GetTitleTextProperty().SetFontFamilyToCourier()
            self.scalarBar.GetTitleTextProperty().ItalicOff()
            self.scalarBar.GetTitleTextProperty().BoldOn()
            self.scalarBar.GetLabelTextProperty().SetFontFamilyToCourier()
            self.scalarBar.GetLabelTextProperty().ItalicOff()
            self.scalarBar.GetLabelTextProperty().BoldOn()

            self.scalarBar.SetTitle(stitle)
            self.scalarBar.SetNumberOfLabels(5)

            self.renderer.AddActor(self.scalarBar)

    def AddArrows(self, cent, direction, mag=1):
        """ Adds arrows to plotting object """

        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        pdata = vtkInterface.CreateVectorPolyData(cent, direction * mag)
        arrows = CreateArrowsActor(pdata)
        self.AddActor(arrows)

        return arrows, pdata

    def AddLineSegments(self, points, edges, color=None, scalars=None,
                        ncolors=256):
        """ Adds arrows to plotting object """

        cent = (points[edges[:, 0]] + points[edges[:, 1]]) / 2
        direction = points[edges[:, 1]] - points[edges[:, 0]]
        # pdata = vtkInterface.CreateVectorPolyData(cent, direction)
        pdata = vtkInterface.CreateVectorPolyData(cent, direction)
        arrows, mapper = CreateLineSegmentsActor(pdata)

        # set color
        if isinstance(color, str):
            color = vtkInterface.StringToRGB(color)
            mapper.ScalarVisibilityOff()
            arrows.GetProperty().SetColor(color)

        if scalars is not None:
            if scalars.size == edges.shape[0]:
                pdata.AddCellScalars(scalars, '', True)
                mapper.SetScalarModeToUseCellData()
                mapper.GetLookupTable().SetNumberOfTableValues(ncolors)
                # if interpolatebeforemap:
                    # self.mapper.InterpolateScalarsBeforeMappingOn()
            else:
                raise Exception('Number of scalars must match number of edges')

        # add to rain class
        self.AddActor(arrows)
        return arrows

    def GetCameraPosition(self):
        """ Returns camera position of active render window """
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        fpt = camera.GetFocalPoint()
        vup = camera.GetViewUp()
        return [pos, fpt, vup]

    def SetCameraPosition(self, cameraloc):
        """ Set camera position of active render window """
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(cameraloc[0])
        camera.SetFocalPoint(cameraloc[1])
        camera.SetViewUp(cameraloc[2])

        # reset clipping range
        self.renderer.ResetCameraClippingRange()

    def SetBackground(self, color):
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
            color = [1, 1, 1]
        elif isinstance(color, str):
            color = vtkInterface.StringToRGB(color)

        self.renderer.SetBackground(color)

    def AddLegend(self, entries, bcolor=[0.5, 0.5, 0.5], border=False,
                  pos=None):
        """
        Adds a legend to render window.  Entries must be a list containing
        one string and color entry for each item

        pos : list
            Two float list, each float between 0 and 1.  For example
            [0.5, 0.5] would put the legend in the middle of the figure.

        Example
        -------

        legend_entries = []
        legend_entries.append(['Label', 'w'])
        plobj = PlotClass()
        plobj.AddMesh(mesh)
        plobj.AddLegend(legend_entries)
        plobj.Plot()

        """

        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(len(entries))
        if pos:
            legend.SetPosition2(pos[0], pos[1])

        legendface = MakeLegendPoly()
        for i, (text, color) in enumerate(entries):
            # try:
            legend.SetEntry(i, legendface, text, ParseColor(color))
            # except:
                # import pdb; pdb.set_trace()

        legend.UseBackgroundOn()
        legend.SetBackgroundColor(bcolor)
        if border:
            legend.BorderOn()
        else:
            legend.BorderOff()

        # Add to renderer
        self.renderer.AddActor(legend)
        return legend

    def _plot(self, title=None, window_size=[1024, 768], interactive=True,
              autoclose=True, interactive_update=False):
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

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up

        """

        if title:
            self.renWin.SetWindowName(title)

        # size window
        self.renWin.SetSize(window_size[0], window_size[1])

        # Render
        self.renWin.Render()
        if interactive and (not self.off_screen):
            self.iren.Initialize()

            if not interactive_update:
                # interrupts will be caught here
                try:
                    self.iren.Start()
                except KeyboardInterrupt:
                    self.Close()
                    raise KeyboardInterrupt

        # Get camera position before closing
        cpos = self.GetCameraPosition()

        if autoclose:
            self.Close()

        return cpos

    def Plot(self, title=None, window_size=[1024, 768], interactive=True,
             autoclose=True, in_background=False, interactive_update=False):
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

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up

        """
        def PlotFun():
            return self._plot(title, window_size, interactive, autoclose, interactive_update)

        if in_background:
            process = Process(target=PlotFun)
            process.start()
            return process
        else:
            return PlotFun()

    def RemoveActor(self, actor):
        self.renderer.RemoveActor(actor)

    def AddAxes(self):
        """ Add axes actor at origin """
        pass  # causes segfault
        # axes = vtk.vtkAxesActor()
        # self.marker = vtk.vtkOrientationMarkerWidget()
        # self.marker.SetInteractor(self.iren)
        # self.marker.SetOrientationMarker(axes)
        # self.marker.SetEnabled(1)

        # axes = vtk.vtkAxesActor()
        # widget = vtk.vtkOrientationMarkerWidget()

    def TakeScreenShot(self, filename=None, transparent_background=False):
        """
        Takes screenshot at current camera position

        Parameters
        ----------
        filename : str, optional
            Location to write image to.

        transparent_background : bool, optional
            Makes the background transparent.  Default False.

        Returns
        -------
        img :  numpy.ndarray
            Array containing pixel RGB and alpha.  Sized:
            [Window height x Window width x 3] for transparent_background=False
            [Window height x Window width x 4] for transparent_background=True

        """
        # check render window exists
        if not hasattr(self, 'renWin'):
            raise Exception('Render window has been closed.\n'
                            'Run again with Plot(autoclose=False)')

        # create image filter
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.renWin)
        if transparent_background:
            ifilter.SetInputBufferTypeToRGBA()
        else:
            ifilter.SetInputBufferTypeToRGB()
        ifilter.ReadFrontBufferOff()
        ifilter.Update()

        image = ifilter.GetOutput()
        origshape = image.GetDimensions()

        img_array = vtkInterface.GetPointScalars(image, 'ImageScalars')
        if not img_array.size:
            raise Exception('Empty image.  Have you run Plot first?')

        # write screenshot to file
        img = img_array.reshape((origshape[1], origshape[0], -1))[::-1, :, :]
        if filename:
            image = Image.fromarray(img)
            image.save(filename)

        return img

    def Render(self):
        self.renWin.Render()

    def SetFocus(self, point):
        """ sets focus to a point """
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(point)

    def __del__(self):
        log.debug('Object collected')


def CreateLineSegmentsActor(pdata):

    # Create arrow object
    lines_source = vtk.vtkLineSource()
    lines_source.Update()
    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceData(lines_source.GetOutput())
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

    return actor, mapper


def CreateArrowsActor(pdata):
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


def PlotGrids(grids, wFEM=False, background=[0, 0, 0], style='wireframe',
              legend_entries=None):
    """ Creates a plot of several grids as wireframes.  When wFEM is
    true, the first grid is a white solid """

    # Make grid colors
    N = len(grids)
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    pobj = PlotClass()
    for i in range(len(grids)):
        if not i and wFEM:  # Special plotting for first grid
            pobj.AddMesh(grids[i])
        else:
            pobj.AddMesh(grids[i], color=colors[i], style=style)

    # Render plot and delete when finished
    pobj.SetBackground(background)

    if legend_entries:
        legend = []
        if isinstance(legend_entries, range):
            for c, i in enumerate(legend_entries):
                legend.append([str(legend_entries[i]), colors[c]])
        else:
            for i in range(len(legend_entries)):
                legend.append([legend_entries[i], colors[i]])

        pobj.AddLegend(legend)

    pobj.Plot()


def PlotEdges(mesh, angle, width=10):
    """ Plots edges of a mesh """
    edges = vtkInterface.GetEdgePoints(mesh, angle, False)
    pobj = PlotClass()
    pobj.AddLines(edges, [0, 1, 1], width)
    pobj.AddMesh(mesh)
    pobj.Plot()


def PlotBoundaries(mesh, **args):
    """ Plots boundaries of a mesh """
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(mesh)
    featureEdges.FeatureEdgesOff()
    featureEdges.BoundaryEdgesOn()
    featureEdges.NonManifoldEdgesOn()
    featureEdges.ManifoldEdgesOff()
    featureEdges.Update()
    edges = vtkInterface.PolyData(featureEdges.GetOutput())

    plobj = PlotClass()
    plobj.AddMesh(edges, 'r', style='wireframe')
    plobj.AddMesh(mesh)
    plobj.Plot()


def MakeLegendPoly():
    """ Creates a legend polydata object """
    pts = np.zeros((4, 3))
    vtkpoints = vtkInterface.MakevtkPoints(pts)
    triangles = np.array([[4, 0, 1, 2, 3]], np.int64)
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(triangles.shape[0],
                      VN.numpy_to_vtkIdTypeArray(triangles, deep=True))

    # Create polydata object
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtkpoints)
    mesh.SetPolys(vtkcells)

    return mesh


def ParseColor(color):
    """ Parses color into a vtk friendly rgb list """
    if color is None:
        return [1, 1, 1]
    elif isinstance(color, str):
        return vtkInterface.StringToRGB(color)
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


def ParseFontFamily(font_family):
    """ checks font name """
    # check font name
    font_family = font_family.lower()
    if font_family not in ['courier', 'times', 'arial']:
        raise Exception('Font must be either "courier", "times" ' +
                        'or "arial"')

    return font_keys[font_family]
