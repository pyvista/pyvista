"""
vtk plotting module 

"""
import colorsys

import vtk
from vtk.util import numpy_support as VN
import numpy as np

import vtkInterface

font_keys = {'arial': vtk.VTK_ARIAL,
             'courier': vtk.VTK_COURIER,
             'times': vtk.VTK_TIMES}

#==============================================================================
# Functions
#==============================================================================
def Plot(mesh, **args):
    """
    Convenience plotting function for a vtk object
    
    Includes extra argument 'screenshot', otherwise see :
    help(vtkInterface.PlotClass.AddMesh)
    
    """
    
    if 'screenshot' in args:
        filename = args['screenshot']
        del args['screenshot']
    else:
        filename = None

    if 'cpos' in args:
        cpos = args['cpos']
        del args['cpos']
    else:
        cpos = None

    # create plotting object and add mesh
    plobj = PlotClass()
    
    if type(mesh) is np.ndarray:
        plobj.AddPoints(mesh, **args)
    else:
        plobj.AddMesh(mesh, **args)
    
    # Set camera
    if cpos:
        plobj.SetCameraPosition(cpos)
        
    cpos = plobj.Plot(autoclose=False)
    
    # take screenshot
    if filename:
        plobj.TakeScreenShot(filename)

    # close and return camera position
    plobj.Close()
    del plobj
    return cpos


#==============================================================================
# Classes
#==============================================================================
class PlotClass(object):
    """
    Plotting object to display vtk meshes or numpy arrays.

    Example
    -------
    plobj = PlotClass()
    plobj.AddMesh(mesh, color='red')
    plobj.AddMesh(another_mesh, color='blue')
    plobj.Plot()
    del plobj
    
    """
    
    def __init__(self, off_screen=False):
        """ 
        Initialize a vtk plotting object
        
        Parameters
        ----------
        off_screen : bool, optional
            Renders off screen when False.  Useful for automated screenshots.
            
        Returns
        -------
        None
        
        """

        # Store setting
        self.off_screen = off_screen

        # Add FEM Actor to renderer window
        self.ren = vtk.vtkRenderer()
        
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        
        if self.off_screen:
            self.renWin.SetOffScreenRendering(1)
            
        else:
            
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetRenderWindow(self.renWin)
            
            # Allow user to interact
            istyle = vtk.vtkInteractorStyleTrackballCamera()
            self.iren.SetInteractorStyle(istyle)

        # Set background
        self.ren.SetBackground(0.3, 0.3, 0.3)
        
        # track objects
        self.objects = []
        
        self.frames = []

        # initialize image filter
        self.ifilter = vtk.vtkWindowToImageFilter()
        self.ifilter.SetInput(self.renWin)
        self.ifilter.SetInputBufferTypeToRGB()
        self.ifilter.ReadFrontBufferOff()
        
        # initialize movie type
        self.movietype = None


    def AddMesh(self, meshin, color=None, style='surface', scalars=None, 
                rng=None, stitle=None, showedges=True, psize=5.0, opacity=1,
                linethick=None, flipscalars=False, lighting=False, ncolors=1000,
                interpolatebeforemap=False, no_copy=False):
        """ 
        Adds a vtk unstructured, structured, or polymesh to the plotting object
        
        By default, the input mesh is copied on load.


        Parameters
        ----------
        meshin : vtk unstructured, structured, or polymesh
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
            Flip scalar display approach.  Default is red is minimum and blue
            is maximum.
            
        lighting : bool, optional
            Enable or disable Z direction lighting.  True by default.
        
        ncolors : int, optional
            Number of colors to use when displaying scalars.
        
        interpolatebeforemap : bool, default False
            Enabling makes for a smoother scalar display.  Default False
        
        no_copy : bool, optional
            Enabling forces the mesh to not to copy.  Faster, but adds 
            possibly unwanted extra scalars to the mesh.
            

        Returns
        -------
        mesh : vtk object
            Pointer to added mesh (either copy or original)
            
        
        """
        
        # add convenience functions on load if not already loaded
        if not hasattr(meshin, 'Copy'):
            vtkInterface.AddFunctions(meshin)
        
        # Create mapper
        self.mapper = vtk.vtkDataSetMapper()

        # copy grid on import for display purposes        
        if not no_copy:
            self.mesh = meshin.Copy()
        else:
            self.mesh = meshin
        
        #======================================================================
        # Scalar formatting
        #======================================================================
        if scalars is not None:
            # convert to numpy array
            if type(scalars) != np.ndarray:
                scalars = np.asarray(scalars)

            # ravel if not 1 dimentional            
            if scalars.ndim != 1:
                scalars = scalars.ravel()
                
            # Scalar interpolation approach
            if scalars.size == meshin.GetNumberOfPoints():
                self.mesh.AddPointScalars(scalars, '', True)
                self.mapper.SetScalarModeToUsePointData()
                self.mapper.GetLookupTable().SetNumberOfTableValues(ncolors)               
                if interpolatebeforemap:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
    
            elif scalars.size == meshin.GetNumberOfCells():
                self.mesh.AddCellScalars(scalars, '')
                self.mapper.SetScalarModeToUseCellData()

            # Set scalar range
            if not rng:
                rng = [np.min(scalars), np.max(scalars)]
            elif type(rng) is float:
                rng = [-rng, rng]
                    
            if np.any(rng):
                self.mapper.SetScalarRange(rng[0], rng[1])
        
            # Flip if requested
            if flipscalars:
                self.mapper.GetLookupTable().SetHueRange(0.66667, 0.0)  
        else:
            self.mapper.SetScalarModeToUseFieldData()
        
        # Set mapper
        self.mapper.SetInputData(self.mesh)
        
        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)
        
        # select view style
        if style == 'wireframe':
            actor.GetProperty().SetRepresentationToWireframe()
        elif style == 'points':
            actor.GetProperty().SetRepresentationToPoints()
            actor.GetProperty().SetPointSize(psize)
        elif style =='surface':
            actor.GetProperty().SetRepresentationToSurface()
            
        # edge display style
        if showedges:
            actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetColor(ParseColor(color))
        actor.GetProperty().SetOpacity(opacity)
        
        # lighting display style
        if lighting is False:
            actor.GetProperty().LightingOff()
        
        # set line thickness
        if linethick:
            actor.GetProperty().SetLineWidth(linethick) 
        
        # Add to renderer
        self.ren.AddActor(actor)
        
        # Add scalar bar if available
        if stitle is not None:
            self.AddScalarBar(stitle)
            
        # return pointer to mesh
        return self.mesh
    
    
    def AddBoundsAxes(self, mesh=None, bounds=None, show_xaxis=True, 
                      show_yaxis=True, show_zaxis=True, show_xlabels=True,
                      show_ylabels=True, show_zlabels=True, italic=False,
                      bold=True, shadow=False, fontsize=16, 
                      font_family='courier', color='w', 
                      xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis'):
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
        cubeAxesActor.SetUse2DMode(True)

        # set bounds
        if not bounds:
            bounds = mesh.GetBounds()
        cubeAxesActor.SetBounds(mesh.GetBounds())
        
        # show or hide axes
        cubeAxesActor.SetXAxisVisibility(show_xaxis)
        cubeAxesActor.SetYAxisVisibility(show_yaxis)
        cubeAxesActor.SetZAxisVisibility(show_zaxis)
        
        # disable minor ticks
        cubeAxesActor.XAxisMinorTickVisibilityOff();
        cubeAxesActor.YAxisMinorTickVisibilityOff();
        cubeAxesActor.ZAxisMinorTickVisibilityOff();
        
        cubeAxesActor.SetCamera(self.ren.GetActiveCamera())
        
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
            
        Returns
        -------
        None
        
        Notes
        -----
        Setting title_fontsize, or label_fontsize disables automatic font 
        sizing for both the title and label.

        
        """
        # check if maper exists
        if not hasattr(self, 'mapper'):
            raise Exception('Mapper does not exist.  ' +\
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
            if title_fontsize:
                title_text.SetFontSize(title_fontsize)
    
            # Set font
            title_text.SetFontFamily(ParseFontFamily(font_family))
                
            # set color
            title_text.SetColor(color)

        
        self.ren.AddActor(self.scalarBar)
        
    
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
        """ updates points of object (point only for now) 
        assumes last inputted mesh if mesh left empty
        """
        if mesh is None:
            mesh = self.mesh
            
        self.mesh.SetNumpyPoints(points)
        
        if render:
            self.Render()
            
    
    def Close(self):
        """ closes render window """

        if hasattr(self, 'renWin'):
            del self.renWin
        
        if hasattr(self, 'iren'):
            del self.iren
            
        if hasattr(self, 'textActor'):
            del self.textActor
            
        # end movie
        if hasattr(self, 'mwriter'):
            try:
                self.mwriter.close()
            except:
                pass
            

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
        shadow : False
        
        """
        
            
        self.textActor = vtk.vtkTextActor()
        self.textActor.SetPosition(position)
        self.textActor.GetTextProperty().SetFontSize(fontsize)
        self.textActor.GetTextProperty().SetColor(ParseColor(color))
        self.textActor.GetTextProperty().SetFontFamily(font_keys[font])
        self.textActor.GetTextProperty().SetShadow(shadow)
        self.textActor.SetInput(text)
        self.AddActor(self.textActor)
        

    def OpenMovie(self, filename, framerate=24, codec='libx264', 
                  preset='medium'):
        """ Establishes a connection to the ffmpeg writer """
        
        # Attempt to load moviepy
        try:
            import moviepy.video.io.ffmpeg_writer as mwrite
        except:
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
        try:
            import imageio
        except:
            raise Exception('To use this feature, install imageio')
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
                
        if type(lines) is np.ndarray:
            lines = vtkInterface.MakeLine(lines)
        
        # Create mapper and add lines
        mapper = vtk.vtkDataSetMapper()
        vtkInterface.SetVTKInput(mapper, lines)
        
        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(width); 
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(color)
        actor.GetProperty().SetColor(ParseColor(color))
        actor.GetProperty().LightingOff()
        
        # Add to renderer
        self.ren.AddActor(actor)


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
                  rng=None, name='', opacity=1, stitle='', flipscalars=False):
        """ Adds a point actor or numpy points array to plotting object """
        
        # select color
        if color is None:
            color = [1, 1, 1]
        elif type(color) is str or type(color) is unicode:
            color = vtkInterface.StringToRGB(color)
            
        # Convert to vtk points object if "points" is a numpy array
        if type(points) == np.ndarray:
            self.points = MakeVTKPointsMesh(points)
        else:
            self.points = points
            
        # Create mapper and add lines
        mapper = vtk.vtkDataSetMapper()
        vtkInterface.SetVTKInput(mapper, self.points)

        if np.any(scalars):
            vtkInterface.AddPointScalars(self.points, scalars, name, True)
            mapper.SetScalarModeToUsePointData()
        
            if not rng:
                rng = [np.min(scalars), np.max(scalars)]
                    
            if np.any(rng):
                mapper.SetScalarRange(rng[0], rng[1])       
                
            # Flip if requested
            if flipscalars:
                mapper.GetLookupTable().SetHueRange(0.66667, 0.0)                   
                
        # Create Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(psize); 
        actor.GetProperty().SetColor(color)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetOpacity(opacity)

        self.ren.AddActor(actor)
        
        
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
            
            self.ren.AddActor(self.scalarBar)
                
            
    def AddArrows(self, cent, direction, mag=1):
        """ Adds arrows to plotting object """
        
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))
            
        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))
        
        pdata = vtkInterface.CreateVectorPolyData(cent, direction*mag)
        arrows = CreateArrowsActor(pdata)
        self.AddActor(arrows)

        return arrows
    
    
    def AddLineSegments(self, points, edges, color=None):
        """ Adds arrows to plotting object """
        
        cent = (points[edges[:, 0]] + points[edges[:, 1]])/2
        direction = points[edges[:, 1]] - points[edges[:, 0]]
        pdata = vtkInterface.CreateVectorPolyData(cent, direction)
        
        pdata = vtkInterface.CreateVectorPolyData(cent, direction)
        arrows, mapper = CreateLineSegmentsActor(pdata)
        
        # set color
        if type(color) is str or type(color) is unicode:
            color = vtkInterface.StringToRGB(color)
            mapper.ScalarVisibilityOff()
            arrows.GetProperty().SetColor(color)
#            print 'color', str(color)
            
        # add to mrain class
        self.AddActor(arrows)
        return arrows
    
        
    def GetCameraPosition(self):
        """ Returns camera position of active render window """
        camera = self.ren.GetActiveCamera()
        pos = camera.GetPosition()
        fpt = camera.GetFocalPoint()
        vup = camera.GetViewUp()
        return [pos, fpt, vup]
        

    def SetCameraPosition(self, cameraloc):
        """ Set camera position of active render window """
        camera = self.ren.GetActiveCamera()
        camera.SetPosition(cameraloc[0])
        camera.SetFocalPoint(cameraloc[1]) 
        camera.SetViewUp(cameraloc[2])

        # reset clipping range
        self.ren.ResetCameraClippingRange()


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
        elif type(color) is str or type(color) is unicode:
            color = vtkInterface.StringToRGB(color)
            
        self.ren.SetBackground(color)
        
        
    def AddLegend(self, entries, bcolor=[0.5, 0.5, 0.5], border=False):
        """
        Adds a legend to render window.  Entries must be a list containing
        one string and color entry for each item
        """
        
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(len(entries))
        
        c = 0
        legendface = MakeLegendPoly()
        for entry in entries:
            legend.SetEntry(c, legendface, entry[0], entry[1])
            c += 1
        
        legend.UseBackgroundOn()
        legend.SetBackgroundColor(bcolor)
        if border:
            legend.BorderOn()
        else:
            legend.BorderOff()
        
        # Add to renderer
        self.ren.AddActor(legend)
        
        
    def Plot(self, title=None, window_size=[1024, 768], interactive=True,
             autoclose=True):
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
        if interactive and (not self.off_screen):
            self.renWin.Render()
            self.iren.Initialize()
            self.iren.Start()
            
        else:
            self.renWin.Render()
        
        # Get camera position
        cpos = self.GetCameraPosition()
        
        if autoclose:
            self.Close()
            
        return cpos
    
        
    def AddActor(self, actor):
        """ Adds actor to render window """
        self.ren.AddActor(actor)
        
        
    def RemoveActor(self, actor):
        self.ren.RemoveActor(actor)
        
        
    def AddAxes(self):
        """ Add axes actor at origin """
        axesActor = vtk.vtkAxesActor()
        self.ren.AddActor(axesActor)
        
        # interactive axes appear broken as of version 7.0
#        # create interactive axes        
#        axes = vtk.vtkOrientationMarkerWidget()
#        axes.SetOrientationMarker(axesActor)
#        axes.SetInteractor(self.iren)
#        axes.SetViewport(0.0, 0.0, 0.4, 0.4)
##        axes.On()
#        axes.SetEnabled(1)
#        axes.InteractiveOn()
#        self.ren.ResetCamera()
#        self.Render()
        
        
    def TakeScreenShot(self, filename=None):
        """
        Takes screenshot at current camera position
        """
        
        # attempt to import imsave for saving screenshots from vtk
        try:
            from scipy.misc import imsave
        except:
            raise Exception('To use scipy.misc.imsave install pip install pillow')

        # create inage filter        
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.renWin)
        ifilter.SetInputBufferTypeToRGBA()
        ifilter.ReadFrontBufferOff()
        ifilter.Update()
        image = ifilter.GetOutput()
        origshape = image.GetDimensions()
        
        img_array = vtkInterface.GetPointScalars(image, 'ImageScalars')

        # overwrite background        
        background = self.ren.GetBackground()
        mask = img_array[:, -1] == 0
        img_array[mask, 0] = int(255*background[0])
        img_array[mask, 1] = int(255*background[1])
        img_array[mask, 2] = int(255*background[2])
        img_array[mask, -1] = 255
        
        mask = img_array[:, -1] != 255
        img_array[mask, -1] = 255
        
        img = img_array.reshape((origshape[1], origshape[0], -1))[::-1, :, :]
        if filename[-3:] == 'png':
            imsave(filename, img)
            
        elif filename[-3:] == 'jpg':
            imsave(filename, img[:, :, :-1])

        else:
            raise Exception('Only png and jpg supported')
            
        return img_array
            
            
    def Render(self):
        self.renWin.Render()

 
def MakeVTKPointsMesh(points):
    """ Creates a vtk polydata object from a numpy array """
    if points.ndim != 2:
        points = points.reshape((-1, 3))
        
    npoints = points.shape[0]
    
    # Make VTK cells array
    cells = np.hstack((np.ones((npoints, 1)), 
                       np.arange(npoints).reshape(-1, 1)))
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(npoints, VN.numpy_to_vtkIdTypeArray(cells, deep=True))
    
    # Convert points to vtk object
    vtkPoints = vtkInterface.MakevtkPoints(points)
    
    # Create polydata
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(vtkPoints)
    pdata.SetVerts(vtkcells)
    vtkInterface.AddFunctions(pdata)
    
    return pdata
            

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
    
        
def PlotCurvature(mesh, curvtype='Gaussian', rng=None):
    """
    Plots curvature
    Availble options for curvtype:
        'Mean'
        'Gaussian'
        'Maximum  '  
    
    """
    
    # Get curvature values and plot
    c = vtkInterface.GetCurvature(mesh, curvtype)
    cpos = Plot(mesh, scalars=c, rng=rng, 
                stitle='{:s}\nCurvature'.format(curvtype))

    # Return camera posision
    return cpos

    
def PlotGrids(grids, wFEM=False, background=[0, 0, 0], legend_entries=None):
    """
    Creates a plot of several grids as wireframes.  When wFEM is true, the first
    grid is a white solid
    """
    
    # Make grid colors
    N = len(grids)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    pobj = PlotClass()
    for i in range(len(grids)):
        if not i and wFEM: # Special plotting for first grid
            pobj.AddMesh(grids[i])
        else:
            pobj.AddMesh(grids[i], color=colors[i], style='wireframe')
    
    # Render plot and delete when finished
    pobj.SetBackground(background)
    
    if legend_entries:
        legend = []
        for i in range(len(legend_entries)):
            legend.append([legend_entries[i], colors[i]])
            
        pobj.AddLegend(legend)
    pobj.Plot(); del pobj


def PlotEdges(mesh, angle, width=10):
    """ Plots edges of a mesh """
    
    # Extract edge points from a mesh
    edges = vtkInterface.GetEdgePoints(mesh, angle, False)
        
    # Render
    pobj = PlotClass()
    pobj.AddLines(edges, [0, 1, 1], width)
    pobj.AddMesh(mesh)
    pobj.Plot(); del pobj
    
    
def PlotBoundaries(mesh):
    """ Plots boundaries of a mesh """
    featureEdges = vtk.vtkFeatureEdges()
    vtkInterface.SetVTKInput(featureEdges, mesh)
    
    featureEdges.FeatureEdgesOff()
    featureEdges.BoundaryEdgesOn()
    featureEdges.NonManifoldEdgesOn()
    featureEdges.ManifoldEdgesOff()
    
    edgeMapper = vtk.vtkPolyDataMapper();
    edgeMapper.SetInputConnection(featureEdges.GetOutputPort());
    
    edgeActor = vtk.vtkActor();
    edgeActor.GetProperty().SetLineWidth(5);
    edgeActor.SetMapper(edgeMapper)

    mapper = vtk.vtkDataSetMapper()
    vtkInterface.SetVTKInput(mapper, mesh)

    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()    
        
    # Render
    pobj = PlotClass()
    pobj.AddActor(actor)
    pobj.AddActor(edgeActor)
    pobj.Plot(); del pobj
    
    
def MakeLegendPoly():
    """ Creates a legend polydata object """
    pts = np.zeros((4, 3))
    vtkpoints = vtkInterface.MakevtkPoints(pts)
    triangles = np.array([[4, 0, 1, 2, 3]])
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
    elif type(color) is str or type(color) is unicode:
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
        raise Exception('Font must be either "courier", "times" ' +\
                        'or "arial"')
        
    return font_keys[font_family]