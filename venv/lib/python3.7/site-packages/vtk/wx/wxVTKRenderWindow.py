"""
A simple VTK widget for wxPython.

Find wxPython info at http://wxPython.org

Created by David Gobbi, December 2001
Based on vtkTkRenderWindget.py

Updated to new wx namespace and some cleaning by Andrea Gavana,
December 2006

"""

"""
Please see the example at the end of this file.

----------------------------------------
Creation:

wxVTKRenderWindow(parent, ID, stereo=0, [wx keywords]):

You should create a wx.App(False) or some other wx.App subclass
before creating the window.

----------------------------------------
Methods:

Render()
AddRenderer(ren)
GetRenderers()
GetRenderWindow()

----------------------------------------
Methods to override (all take a wx.Event):

OnButtonDown(event)  default: propagate event to Left, Right, Middle
OnLeftDown(event)    default: set _Mode to 'Rotate'
OnRightDown(event)   default: set _Mode to 'Zoom'
OnMiddleDown(event)  default: set _Mode to 'Pan'

OnButtonUp(event)    default: propagate event to L, R, M and unset _Mode
OnLeftUp(event)
OnRightUp(event)
OnMiddleUp(event)

OnMotion(event)      default: call appropriate handler for _Mode

OnEnterWindow(event) default: set focus to this window
OnLeaveWindow(event) default: release focus

OnKeyDown(event)     default: [R]eset, [W]irefreme, [S]olid, [P]ick
OnKeyUp(event)
OnChar(event)

OnSetFocus(event)
OnKillFocus(event)

OnSize(event)
OnMove(event)

OnPaint(event)       default: Render()

----------------------------------------
Protected Members:

_Mode:                 Current mode: 'Rotate', 'Zoom', 'Pan'
_LastX, _LastY:        The (x,y) coordinates of the previous event
_CurrentRenderer:      The renderer that was most recently clicked in
_CurrentCamera:        The camera for the current renderer

----------------------------------------
Private Members:

__Handle:              Handle to the window containing the vtkRenderWindow

"""

# import usual libraries
import math, os, sys
import wx
import vtk

# a few configuration items, see what works best on your system

# Use GLCanvas as base class instead of wx.Window.
# This is sometimes necessary under wxGTK or the image is blank.
# (in wxWindows 2.3.1 and earlier, the GLCanvas had scroll bars)
baseClass = wx.Window
if wx.Platform == "__WXGTK__":
    import wx.glcanvas
    baseClass = wx.glcanvas.GLCanvas

# Keep capturing mouse after mouse is dragged out of window
# (in wxGTK 2.3.2 there is a bug that keeps this from working,
# but it is only relevant in wxGTK if there are multiple windows)
_useCapture = (wx.Platform == "__WXMSW__")

# end of configuration items


class wxVTKRenderWindow(baseClass):
    """
    A wxRenderWindow for wxPython.
    Use GetRenderWindow() to get the vtkRenderWindow.
    Create with the keyword stereo=1 in order to
    generate a stereo-capable window.
    """

    def __init__(self, parent, ID, *args, **kw):
        """Default class constructor.
        @param parent: parent window
        @param ID: window id
        @param **kw: wxPython keywords (position, size, style) plus the
        'stereo' keyword
        """
        # miscellaneous protected variables
        self._CurrentRenderer = None
        self._CurrentCamera = None
        self._CurrentZoom = 1.0
        self._CurrentLight = None

        self._ViewportCenterX = 0
        self._ViewportCenterY = 0

        self._Picker = vtk.vtkCellPicker()
        self._PickedActor = None
        self._PickedProperty = vtk.vtkProperty()
        self._PickedProperty.SetColor(1,0,0)
        self._PrePickedProperty = None

        # these record the previous mouse position
        self._LastX = 0
        self._LastY = 0

        # the current interaction mode (Rotate, Pan, Zoom, etc)
        self._Mode = None
        self._ActiveButton = None

        # private attributes
        self.__OldFocus = None

        # used by the LOD actors
        self._DesiredUpdateRate = 15
        self._StillUpdateRate = 0.0001

        # First do special handling of some keywords:
        # stereo, position, size, width, height, style

        try:
            stereo = bool(kw['stereo'])
            del kw['stereo']
        except KeyError:
            stereo = False

        try:
            position = kw['position']
            del kw['position']
        except KeyError:
            position = wx.DefaultPosition

        try:
            size = kw['size']
            del kw['size']
        except KeyError:
            try:
                size = parent.GetSize()
            except AttributeError:
                size = wx.DefaultSize

        # wx.WANTS_CHARS says to give us e.g. TAB
        # wx.NO_FULL_REPAINT_ON_RESIZE cuts down resize flicker under GTK
        style = wx.WANTS_CHARS | wx.NO_FULL_REPAINT_ON_RESIZE

        try:
            style = style | kw['style']
            del kw['style']
        except KeyError:
            pass

        # the enclosing frame must be shown under GTK or the windows
        #  don't connect together properly
        l = []
        p = parent
        while p: # make a list of all parents
            l.append(p)
            p = p.GetParent()
        l.reverse() # sort list into descending order
        for p in l:
            p.Show(1)

        # initialize the wx.Window
        if baseClass.__name__ == 'GLCanvas':
            # Set the doublebuffer attribute of the GL canvas.
            baseClass.__init__(self, parent, ID, pos=position, size=size,
                               style=style,
                               attribList=[wx.glcanvas.WX_GL_DOUBLEBUFFER])
        else:
            baseClass.__init__(self, parent, ID, pos=position, size=size,
                               style=style)

        # create the RenderWindow and initialize it
        self._RenderWindow = vtk.vtkRenderWindow()
        self._RenderWindow.SetSize(size.width, size.height)

        if stereo:
            self._RenderWindow.StereoCapableWindowOn()
            self._RenderWindow.SetStereoTypeToCrystalEyes()

        self.__handle = None

        # refresh window by doing a Render
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        # turn off background erase to reduce flicker
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)

        # Bind the events to the event converters
        self.Bind(wx.EVT_RIGHT_DOWN, self._OnButtonDown)
        self.Bind(wx.EVT_LEFT_DOWN, self._OnButtonDown)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._OnButtonDown)
        self.Bind(wx.EVT_RIGHT_UP, self._OnButtonUp)
        self.Bind(wx.EVT_LEFT_UP, self._OnButtonUp)
        self.Bind(wx.EVT_MIDDLE_UP, self._OnButtonUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)

        self.Bind(wx.EVT_ENTER_WINDOW, self._OnEnterWindow)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._OnLeaveWindow)

        self.Bind(wx.EVT_CHAR, self.OnChar)

        # If we use EVT_KEY_DOWN instead of EVT_CHAR, capital versions
        # of all characters are always returned.  EVT_CHAR also performs
        # other necessary keyboard-dependent translations.
        self.Bind(wx.EVT_CHAR, self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)

        self.Bind(wx.EVT_SIZE, self._OnSize)
        self.Bind(wx.EVT_MOVE, self.OnMove)

        self.Bind(wx.EVT_SET_FOCUS, self.OnSetFocus)
        self.Bind(wx.EVT_KILL_FOCUS, self.OnKillFocus)

    def SetDesiredUpdateRate(self, rate):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor.
        """
        self._DesiredUpdateRate = rate

    def GetDesiredUpdateRate(self):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor.
        """
        return self._DesiredUpdateRate

    def SetStillUpdateRate(self, rate):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor.
        """
        self._StillUpdateRate = rate

    def GetStillUpdateRate(self):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor.
        """
        return self._StillUpdateRate

    def OnPaint(self, event):
        """Handles the wx.EVT_PAINT event for wxVTKRenderWindow.
        """
        dc = wx.PaintDC(self)
        self.Render()

    def _OnSize(self, event):
        """Handles the wx.EVT_SIZE event for wxVTKRenderWindow.
        """
        if wx.Platform != '__WXMSW__':
            width, height = event.GetSize()
            self._RenderWindow.SetSize(width, height)
        self.OnSize(event)
        self.Render()

    def OnSize(self, event):
        """Overridable event.
        """
        pass

    def OnMove(self, event):
        """Overridable event.
        """
        pass


    def _OnEnterWindow(self, event):
        """Handles the wx.EVT_ENTER_WINDOW event for
        wxVTKRenderWindow.
        """
        self.UpdateRenderer(event)
        self.OnEnterWindow(event)


    def OnEnterWindow(self, event):
        """Overridable event.
        """
        if self.__OldFocus == None:
            self.__OldFocus = wx.Window.FindFocus()
            self.SetFocus()

    def _OnLeaveWindow(self, event):
        """Handles the wx.EVT_LEAVE_WINDOW event for
        wxVTKRenderWindow.
        """
        self.OnLeaveWindow(event)

    def OnLeaveWindow(self, event):
        """Overridable event.
        """
        if self.__OldFocus:
            self.__OldFocus.SetFocus()
            self.__OldFocus = None

    def OnSetFocus(self, event):
        """Overridable event.
        """
        pass

    def OnKillFocus(self, event):
        """Overridable event.
        """
        pass

    def _OnButtonDown(self, event):
        """Handles the wx.EVT_LEFT/RIGHT/MIDDLE_DOWN events for
        wxVTKRenderWindow.
        """
        # helper function for capturing mouse until button released
        self._RenderWindow.SetDesiredUpdateRate(self._DesiredUpdateRate)

        if event.RightDown():
            button = "Right"
        elif event.LeftDown():
            button = "Left"
        elif event.MiddleDown():
            button = "Middle"
        else:
            button = None

        # save the button and capture mouse until the button is released
        if button and not self._ActiveButton:
            self._ActiveButton = button
            if _useCapture:
                self.CaptureMouse()

        self.OnButtonDown(event)

    def OnButtonDown(self, event):
        """Overridable event.
        """
        if not self._Mode:
            # figure out what renderer the mouse is over
            self.UpdateRenderer(event)

        if event.LeftDown():
            self.OnLeftDown(event)
        elif event.RightDown():
            self.OnRightDown(event)
        elif event.MiddleDown():
            self.OnMiddleDown(event)

    def OnLeftDown(self, event):
        """Overridable event.
        """
        if not self._Mode:
            if event.ControlDown():
                self._Mode = "Zoom"
            elif event.ShiftDown():
                self._Mode = "Pan"
            else:
                self._Mode = "Rotate"

    def OnRightDown(self, event):
        """Overridable event.
        """
        if not self._Mode:
            self._Mode = "Zoom"

    def OnMiddleDown(self, event):
        """Overridable event.
        """
        if not self._Mode:
            self._Mode = "Pan"

    def _OnButtonUp(self, event):
        """Handles the wx.EVT_LEFT/RIGHT/MIDDLE_UP events for
        wxVTKRenderWindow.
        """
        # helper function for releasing mouse capture
        self._RenderWindow.SetDesiredUpdateRate(self._StillUpdateRate)

        if event.RightUp():
            button = "Right"
        elif event.LeftUp():
            button = "Left"
        elif event.MiddleUp():
            button = "Middle"
        else:
            button = None

        # if the ActiveButton is realeased, then release mouse capture
        if self._ActiveButton and button == self._ActiveButton:
            if _useCapture:
                self.ReleaseMouse()
            self._ActiveButton = None

        self.OnButtonUp(event)

    def OnButtonUp(self, event):
        """Overridable event.
        """
        if event.LeftUp():
            self.OnLeftUp(event)
        elif event.RightUp():
            self.OnRightUp(event)
        elif event.MiddleUp():
            self.OnMiddleUp(event)

        # if not interacting, then do nothing more
        if self._Mode:
            if self._CurrentRenderer:
                self.Render()

        self._Mode = None

    def OnLeftUp(self, event):
        """Overridable event.
        """
        pass

    def OnRightUp(self, event):
        """Overridable event.
        """
        pass

    def OnMiddleUp(self, event):
        """Overridable event.
        """
        pass

    def OnMotion(self, event):
        """Overridable event.
        """
        if self._Mode == "Pan":
            self.Pan(event)
        elif self._Mode == "Rotate":
            self.Rotate(event)
        elif self._Mode == "Zoom":
            self.Zoom(event)

    def OnChar(self, event):
        """Overridable event.
        """
        pass

    def OnKeyDown(self, event):
        """Handles the wx.EVT_KEY_DOWN events for wxVTKRenderWindow.
        """
        if event.GetKeyCode() == ord('r'):
            self.Reset(event)
        if event.GetKeyCode() == ord('w'):
            self.Wireframe()
        if event.GetKeyCode() == ord('s'):
            self.Surface()
        if event.GetKeyCode() == ord('p'):
            self.PickActor(event)

        if event.GetKeyCode() < 256:
            self.OnChar(event)

    def OnKeyUp(self, event):
        """Overridable event.
        """
        pass

    def GetZoomFactor(self):
        """Returns the current zoom factor.
        """
        return self._CurrentZoom

    def GetRenderWindow(self):
        """Returns the render window (vtkRenderWindow).
        """
        return self._RenderWindow

    def GetPicker(self):
        """Returns the current picker (vtkCellPicker).
        """
        return self._Picker

    def Render(self):
        """Actually renders the VTK scene on screen.
        """
        if self._CurrentLight:
            light = self._CurrentLight
            light.SetPosition(self._CurrentCamera.GetPosition())
            light.SetFocalPoint(self._CurrentCamera.GetFocalPoint())

        if not self.GetUpdateRegion().IsEmpty() or self.__handle:
            if self.__handle and self.__handle == self.GetHandle():
                self._RenderWindow.Render()

            elif self.GetHandle():
                # this means the user has reparented us
                # let's adapt to the new situation by doing the WindowRemap
                # dance
                self._RenderWindow.SetNextWindowInfo(str(self.GetHandle()))
                self._RenderWindow.WindowRemap()
                # store the new situation
                self.__handle = self.GetHandle()

                self._RenderWindow.Render()

    def UpdateRenderer(self, event):
        """
        UpdateRenderer will identify the renderer under the mouse and set
        up _CurrentRenderer, _CurrentCamera, and _CurrentLight.
        """
        x = event.GetX()
        y = event.GetY()
        windowX, windowY = self._RenderWindow.GetSize()

        renderers = self._RenderWindow.GetRenderers()
        numRenderers = renderers.GetNumberOfItems()

        self._CurrentRenderer = None
        renderers.InitTraversal()
        for i in range(0,numRenderers):
            renderer = renderers.GetNextItem()
            vx,vy = (0,0)
            if (windowX > 1):
                vx = float(x)/(windowX-1)
            if (windowY > 1):
                vy = (windowY-float(y)-1)/(windowY-1)
            (vpxmin,vpymin,vpxmax,vpymax) = renderer.GetViewport()

            if (vx >= vpxmin and vx <= vpxmax and
                vy >= vpymin and vy <= vpymax):
                self._CurrentRenderer = renderer
                self._ViewportCenterX = float(windowX)*(vpxmax-vpxmin)/2.0\
                                        +vpxmin
                self._ViewportCenterY = float(windowY)*(vpymax-vpymin)/2.0\
                                        +vpymin
                self._CurrentCamera = self._CurrentRenderer.GetActiveCamera()
                lights = self._CurrentRenderer.GetLights()
                lights.InitTraversal()
                self._CurrentLight = lights.GetNextItem()
                break

        self._LastX = x
        self._LastY = y

    def GetCurrentRenderer(self):
        """Returns the current renderer.
        """
        return self._CurrentRenderer

    def Rotate(self, event):
        """Rotates the scene (camera).
        """
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            self._CurrentCamera.Azimuth(self._LastX - x)
            self._CurrentCamera.Elevation(y - self._LastY)
            self._CurrentCamera.OrthogonalizeViewUp()

            self._LastX = x
            self._LastY = y

            self._CurrentRenderer.ResetCameraClippingRange()
            self.Render()

    def Pan(self, event):
        """Pans the scene (camera).
        """
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            camera = self._CurrentCamera
            (pPoint0,pPoint1,pPoint2) = camera.GetPosition()
            (fPoint0,fPoint1,fPoint2) = camera.GetFocalPoint()

            if camera.GetParallelProjection():
                renderer.SetWorldPoint(fPoint0,fPoint1,fPoint2,1.0)
                renderer.WorldToDisplay()
                fx,fy,fz = renderer.GetDisplayPoint()
                renderer.SetDisplayPoint(fx-x+self._LastX,
                                         fy+y-self._LastY,
                                         fz)
                renderer.DisplayToWorld()
                fx,fy,fz,fw = renderer.GetWorldPoint()
                camera.SetFocalPoint(fx,fy,fz)

                renderer.SetWorldPoint(pPoint0,pPoint1,pPoint2,1.0)
                renderer.WorldToDisplay()
                fx,fy,fz = renderer.GetDisplayPoint()
                renderer.SetDisplayPoint(fx-x+self._LastX,
                                         fy+y-self._LastY,
                                         fz)
                renderer.DisplayToWorld()
                fx,fy,fz,fw = renderer.GetWorldPoint()
                camera.SetPosition(fx,fy,fz)

            else:
                (fPoint0,fPoint1,fPoint2) = camera.GetFocalPoint()
                # Specify a point location in world coordinates
                renderer.SetWorldPoint(fPoint0,fPoint1,fPoint2,1.0)
                renderer.WorldToDisplay()
                # Convert world point coordinates to display coordinates
                dPoint = renderer.GetDisplayPoint()
                focalDepth = dPoint[2]

                aPoint0 = self._ViewportCenterX + (x - self._LastX)
                aPoint1 = self._ViewportCenterY - (y - self._LastY)

                renderer.SetDisplayPoint(aPoint0,aPoint1,focalDepth)
                renderer.DisplayToWorld()

                (rPoint0,rPoint1,rPoint2,rPoint3) = renderer.GetWorldPoint()
                if (rPoint3 != 0.0):
                    rPoint0 = rPoint0/rPoint3
                    rPoint1 = rPoint1/rPoint3
                    rPoint2 = rPoint2/rPoint3

                camera.SetFocalPoint((fPoint0 - rPoint0) + fPoint0,
                                     (fPoint1 - rPoint1) + fPoint1,
                                     (fPoint2 - rPoint2) + fPoint2)

                camera.SetPosition((fPoint0 - rPoint0) + pPoint0,
                                   (fPoint1 - rPoint1) + pPoint1,
                                   (fPoint2 - rPoint2) + pPoint2)

            self._LastX = x
            self._LastY = y

            self.Render()

    def Zoom(self, event):
        """Zooms the scene (camera).
        """
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            camera = self._CurrentCamera

            zoomFactor = math.pow(1.02,(0.5*(self._LastY - y)))
            self._CurrentZoom = self._CurrentZoom * zoomFactor

            if camera.GetParallelProjection():
                parallelScale = camera.GetParallelScale()/zoomFactor
                camera.SetParallelScale(parallelScale)
            else:
                camera.Dolly(zoomFactor)
                renderer.ResetCameraClippingRange()

            self._LastX = x
            self._LastY = y

            self.Render()

    def Reset(self, event=None):
        """Resets the camera.
        """
        if self._CurrentRenderer:
            self._CurrentRenderer.ResetCamera()

        self.Render()

    def Wireframe(self):
        """Sets the current actor representation as wireframe.
        """
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for i in range(0,numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToWireframe()

        self.Render()

    def Surface(self):
        """Sets the current actor representation as surface.
        """
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for i in range(0,numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToSurface()

        self.Render()

    def PickActor(self, event):
        """Picks an actor.
        """
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            picker = self._Picker

            windowX, windowY = self._RenderWindow.GetSize()
            picker.Pick(x,(windowY - y - 1),0.0,renderer)
            actor = picker.GetActor()

            if (self._PickedActor != None and
                self._PrePickedProperty != None):
                self._PickedActor.SetProperty(self._PrePickedProperty)
                # release hold of the property
                self._PrePickedProperty.UnRegister(self._PrePickedProperty)
                self._PrePickedProperty = None

            if (actor != None):
                self._PickedActor = actor
                self._PrePickedProperty = self._PickedActor.GetProperty()
                # hold onto the property
                self._PrePickedProperty.Register(self._PrePickedProperty)
                self._PickedActor.SetProperty(self._PickedProperty)

            self.Render()


#----------------------------------------------------------------------------
def wxVTKRenderWindowConeExample():
    """Like it says, just a simple example.
    """
    # every wx app needs an app
    app = wx.App(False)

    # create the widget
    frame = wx.Frame(None, -1, "wxVTKRenderWindow", size=(400,400))
    widget = wxVTKRenderWindow(frame, -1)

    ren = vtk.vtkRenderer()
    widget.GetRenderWindow().AddRenderer(ren)

    cone = vtk.vtkConeSource()
    cone.SetResolution(8)

    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)

    ren.AddActor(coneActor)

    # show the window

    frame.Show()

    app.MainLoop()

if __name__ == "__main__":
    wxVTKRenderWindowConeExample()
