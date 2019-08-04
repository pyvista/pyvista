"""

A VTK RenderWindowInteractor widget for wxPython.

Find wxPython info at http://wxPython.org

Created by Prabhu Ramachandran, April 2002
Based on wxVTKRenderWindow.py

Fixes and updates by Charl P. Botha 2003-2008

Updated to new wx namespace and some cleaning up by Andrea Gavana,
December 2006
"""

"""
Please see the example at the end of this file.

----------------------------------------
Creation:

 wxVTKRenderWindowInteractor(parent, ID, stereo=0, [wx keywords]):

 You should create a wx.App(False) or some other wx.App subclass
 before creating the window.

Behaviour:

 Uses __getattr__ to make the wxVTKRenderWindowInteractor behave just
 like a vtkGenericRenderWindowInteractor.

----------------------------------------

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


class EventTimer(wx.Timer):
    """Simple wx.Timer class.
    """

    def __init__(self, iren):
        """Default class constructor.
        @param iren: current render window
        """
        wx.Timer.__init__(self)
        self.iren = iren


    def Notify(self):
        """ The timer has expired.
        """
        self.iren.TimerEvent()


class wxVTKRenderWindowInteractor(baseClass):
    """
    A wxRenderWindow for wxPython.
    Use GetRenderWindow() to get the vtkRenderWindow.
    Create with the keyword stereo=1 in order to
    generate a stereo-capable window.
    """

    # class variable that can also be used to request instances that use
    # stereo; this is overridden by the stereo=1/0 parameter.  If you set
    # it to True, the NEXT instantiated object will attempt to allocate a
    # stereo visual.  E.g.:
    # wxVTKRenderWindowInteractor.USE_STEREO = True
    # myRWI = wxVTKRenderWindowInteractor(parent, -1)
    USE_STEREO = False

    def __init__(self, parent, ID, *args, **kw):
        """Default class constructor.
        @param parent: parent window
        @param ID: window id
        @param **kw: wxPython keywords (position, size, style) plus the
        'stereo' keyword
        """
        # private attributes
        self.__RenderWhenDisabled = 0

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
        if wx.Platform != '__WXMSW__':
            l = []
            p = parent
            while p: # make a list of all parents
                l.append(p)
                p = p.GetParent()
            l.reverse() # sort list into descending order
            for p in l:
                p.Show(1)

        if baseClass.__name__ == 'GLCanvas':
            # code added by cpbotha to enable stereo and double
            # buffering correctly where the user requests this; remember
            # that the glXContext in this case is NOT allocated by VTK,
            # but by WX, hence all of this.

            # Initialize GLCanvas with correct attriblist
            attribList = [wx.glcanvas.WX_GL_RGBA,
                          wx.glcanvas.WX_GL_MIN_RED, 1,
                          wx.glcanvas.WX_GL_MIN_GREEN, 1,
                          wx.glcanvas.WX_GL_MIN_BLUE, 1,
                          wx.glcanvas.WX_GL_DEPTH_SIZE, 16,
                          wx.glcanvas.WX_GL_DOUBLEBUFFER]
            if stereo:
                attribList.append(wx.glcanvas.WX_GL_STEREO)

            try:
                baseClass.__init__(self, parent, ID, pos=position, size=size,
                                   style=style,
                                   attribList=attribList)
            except wx.PyAssertionError:
                # visual couldn't be allocated, so we go back to default
                baseClass.__init__(self, parent, ID, pos=position, size=size,
                                   style=style)
                if stereo:
                    # and make sure everyone knows that the stereo
                    # visual wasn't set.
                    stereo = 0

        else:
            baseClass.__init__(self, parent, ID, pos=position, size=size,
                               style=style)

        # create the RenderWindow and initialize it
        self._Iren = vtk.vtkGenericRenderWindowInteractor()
        self._Iren.SetRenderWindow( vtk.vtkRenderWindow() )
        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self._Iren.GetRenderWindow().AddObserver('CursorChangedEvent',
                                                 self.CursorChangedEvent)

        try:
            self._Iren.GetRenderWindow().SetSize(size.width, size.height)
        except AttributeError:
            self._Iren.GetRenderWindow().SetSize(size[0], size[1])

        if stereo:
            self._Iren.GetRenderWindow().StereoCapableWindowOn()
            self._Iren.GetRenderWindow().SetStereoTypeToCrystalEyes()

        self.__handle = None

        self.BindEvents()

        # with this, we can make sure that the reparenting logic in
        # Render() isn't called before the first OnPaint() has
        # successfully been run (and set up the VTK/WX display links)
        self.__has_painted = False

        # set when we have captured the mouse.
        self._own_mouse = False
        # used to store WHICH mouse button led to mouse capture
        self._mouse_capture_button = 0

        # A mapping for cursor changes.
        self._cursor_map = {0: wx.CURSOR_ARROW, # VTK_CURSOR_DEFAULT
                            1: wx.CURSOR_ARROW, # VTK_CURSOR_ARROW
                            2: wx.CURSOR_SIZENESW, # VTK_CURSOR_SIZENE
                            3: wx.CURSOR_SIZENWSE, # VTK_CURSOR_SIZENWSE
                            4: wx.CURSOR_SIZENESW, # VTK_CURSOR_SIZESW
                            5: wx.CURSOR_SIZENWSE, # VTK_CURSOR_SIZESE
                            6: wx.CURSOR_SIZENS, # VTK_CURSOR_SIZENS
                            7: wx.CURSOR_SIZEWE, # VTK_CURSOR_SIZEWE
                            8: wx.CURSOR_SIZING, # VTK_CURSOR_SIZEALL
                            9: wx.CURSOR_HAND, # VTK_CURSOR_HAND
                            10: wx.CURSOR_CROSS, # VTK_CURSOR_CROSSHAIR
                           }

    def BindEvents(self):
        """Binds all the necessary events for navigation, sizing,
        drawing.
        """
        # refresh window by doing a Render
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        # turn off background erase to reduce flicker
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)

        # Bind the events to the event converters
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnButtonDown)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnButtonDown)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.OnButtonDown)
        self.Bind(wx.EVT_RIGHT_UP, self.OnButtonUp)
        self.Bind(wx.EVT_LEFT_UP, self.OnButtonUp)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnButtonUp)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_MOTION, self.OnMotion)

        self.Bind(wx.EVT_ENTER_WINDOW, self.OnEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnLeave)

        # If we use EVT_KEY_DOWN instead of EVT_CHAR, capital versions
        # of all characters are always returned.  EVT_CHAR also performs
        # other necessary keyboard-dependent translations.
        self.Bind(wx.EVT_CHAR, self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)

        self.Bind(wx.EVT_SIZE, self.OnSize)

        # the wx 2.8.7.1 documentation states that you HAVE to handle
        # this event if you make use of CaptureMouse, which we do.
        if _useCapture and hasattr(wx, 'EVT_MOUSE_CAPTURE_LOST'):
            self.Bind(wx.EVT_MOUSE_CAPTURE_LOST,
                    self.OnMouseCaptureLost)


    def __getattr__(self, attr):
        """Makes the object behave like a
        vtkGenericRenderWindowInteractor.
        """
        if attr == '__vtk__':
            return lambda t=self._Iren: t
        elif hasattr(self._Iren, attr):
            return getattr(self._Iren, attr)
        else:
            raise AttributeError(self.__class__.__name__ +
                  " has no attribute named " + attr)

    def CreateTimer(self, obj, evt):
        """ Creates a timer.
        """
        self._timer = EventTimer(self)
        self._timer.Start(10, True)

    def DestroyTimer(self, obj, evt):
        """The timer is a one shot timer so will expire automatically.
        """
        return 1

    def _CursorChangedEvent(self, obj, evt):
        """Change the wx cursor if the renderwindow's cursor was
        changed.
        """
        cur = self._cursor_map[obj.GetCurrentCursor()]
        c = wx.StockCursor(cur)
        self.SetCursor(c)

    def CursorChangedEvent(self, obj, evt):
        """Called when the CursorChangedEvent fires on the render
        window."""
        # This indirection is needed since when the event fires, the
        # current cursor is not yet set so we defer this by which time
        # the current cursor should have been set.
        wx.CallAfter(self._CursorChangedEvent, obj, evt)

    def HideCursor(self):
        """Hides the cursor."""
        c = wx.StockCursor(wx.CURSOR_BLANK)
        self.SetCursor(c)

    def ShowCursor(self):
        """Shows the cursor."""
        rw = self._Iren.GetRenderWindow()
        cur = self._cursor_map[rw.GetCurrentCursor()]
        c = wx.StockCursor(cur)
        self.SetCursor(c)

    def GetDisplayId(self):
        """Function to get X11 Display ID from WX and return it in a format
        that can be used by VTK Python.

        We query the X11 Display with a new call that was added in wxPython
        2.6.0.1.  The call returns a SWIG object which we can query for the
        address and subsequently turn into an old-style SWIG-mangled string
        representation to pass to VTK.
        """
        d = None

        try:
            d = wx.GetXDisplay()

        except AttributeError:
            # wx.GetXDisplay was added by Robin Dunn in wxPython 2.6.0.1
            # if it's not available, we can't pass it.  In general,
            # things will still work; on some setups, it'll break.
            pass

        else:
            # wx returns None on platforms where wx.GetXDisplay is not relevant
            if d:
                d = hex(d)
                # On wxPython-2.6.3.2 and above there is no leading '0x'.
                if not d.startswith('0x'):
                    d = '0x' + d

                # VTK wants it as: _xxxxxxxx_p_void (SWIG pointer)
                d = '_%s_%s\0' % (d[2:], 'p_void')

        return d

    def OnMouseCaptureLost(self, event):
        """This is signalled when we lose mouse capture due to an
        external event, such as when a dialog box is shown.  See the
        wx documentation.
        """

        # the documentation seems to imply that by this time we've
        # already lost capture.  I have to assume that we don't need
        # to call ReleaseMouse ourselves.
        if _useCapture and self._own_mouse:
            self._own_mouse = False

    def OnPaint(self,event):
        """Handles the wx.EVT_PAINT event for
        wxVTKRenderWindowInteractor.
        """

        # wx should continue event processing after this handler.
        # We call this BEFORE Render(), so that if Render() raises
        # an exception, wx doesn't re-call OnPaint repeatedly.
        event.Skip()

        dc = wx.PaintDC(self)

        # make sure the RenderWindow is sized correctly
        self._Iren.GetRenderWindow().SetSize(self.GetSize())

        # Tell the RenderWindow to render inside the wx.Window.
        if not self.__handle:

            # on relevant platforms, set the X11 Display ID
            d = self.GetDisplayId()
            if d and self.__has_painted:
                self._Iren.GetRenderWindow().SetDisplayId(d)

            # store the handle
            self.__handle = self.GetHandle()
            # and give it to VTK
            self._Iren.GetRenderWindow().SetWindowInfo(str(self.__handle))

            # now that we've painted once, the Render() reparenting logic
            # is safe
            self.__has_painted = True

        self.Render()

    def OnSize(self,event):
        """Handles the wx.EVT_SIZE event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue (we call this before the
        # Render(), in case it raises an exception)
        event.Skip()

        try:
            width, height = event.GetSize()
        except:
            width = event.GetSize().width
            height = event.GetSize().height
        self._Iren.SetSize(width, height)
        self._Iren.ConfigureEvent()

        # this will check for __handle
        self.Render()

    def OnMotion(self,event):
        """Handles the wx.EVT_MOTION event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        # we call this early in case any of the VTK code raises an
        # exception.
        event.Skip()

        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            event.ControlDown(),
                                            event.ShiftDown(),
                                            chr(0), 0, None)
        self._Iren.MouseMoveEvent()

    def OnEnter(self,event):
        """Handles the wx.EVT_ENTER_WINDOW event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        event.Skip()

        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            event.ControlDown(),
              event.ShiftDown(),
              chr(0), 0, None)
        self._Iren.EnterEvent()


    def OnLeave(self,event):
        """Handles the wx.EVT_LEAVE_WINDOW event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        event.Skip()

        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            event.ControlDown(),
              event.ShiftDown(),
              chr(0), 0, None)
        self._Iren.LeaveEvent()


    def OnButtonDown(self,event):
        """Handles the wx.EVT_LEFT/RIGHT/MIDDLE_DOWN events for
        wxVTKRenderWindowInteractor.
        """

        # allow wx event processing to continue
        # on wxPython 2.6.0.1, omitting this will cause problems with
        # the initial focus, resulting in the wxVTKRWI ignoring keypresses
        # until we focus elsewhere and then refocus the wxVTKRWI frame
        # we do it this early in case any of the following VTK code
        # raises an exception.
        event.Skip()

        ctrl, shift = event.ControlDown(), event.ShiftDown()
        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            ctrl, shift, chr(0), 0, None)

        button = 0
        if event.RightDown():
            self._Iren.RightButtonPressEvent()
            button = 'Right'
        elif event.LeftDown():
            self._Iren.LeftButtonPressEvent()
            button = 'Left'
        elif event.MiddleDown():
            self._Iren.MiddleButtonPressEvent()
            button = 'Middle'

        # save the button and capture mouse until the button is released
        # we only capture the mouse if it hasn't already been captured
        if _useCapture and not self._own_mouse:
            self._own_mouse = True
            self._mouse_capture_button = button
            self.CaptureMouse()


    def OnButtonUp(self,event):
        """Handles the wx.EVT_LEFT/RIGHT/MIDDLE_UP events for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        event.Skip()

        button = 0
        if event.RightUp():
            button = 'Right'
        elif event.LeftUp():
            button = 'Left'
        elif event.MiddleUp():
            button = 'Middle'

        # if the same button is released that captured the mouse, and
        # we have the mouse, release it.
        # (we need to get rid of this as soon as possible; if we don't
        #  and one of the event handlers raises an exception, mouse
        #  is never released.)
        if _useCapture and self._own_mouse and \
                button==self._mouse_capture_button:
            self.ReleaseMouse()
            self._own_mouse = False

        ctrl, shift = event.ControlDown(), event.ShiftDown()
        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            ctrl, shift, chr(0), 0, None)

        if button == 'Right':
            self._Iren.RightButtonReleaseEvent()
        elif button == 'Left':
            self._Iren.LeftButtonReleaseEvent()
        elif button == 'Middle':
            self._Iren.MiddleButtonReleaseEvent()


    def OnMouseWheel(self,event):
        """Handles the wx.EVT_MOUSEWHEEL event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        event.Skip()

        ctrl, shift = event.ControlDown(), event.ShiftDown()
        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            ctrl, shift, chr(0), 0, None)
        if event.GetWheelRotation() > 0:
            self._Iren.MouseWheelForwardEvent()
        else:
            self._Iren.MouseWheelBackwardEvent()


    def OnKeyDown(self,event):
        """Handles the wx.EVT_KEY_DOWN event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        event.Skip()

        ctrl, shift = event.ControlDown(), event.ShiftDown()
        keycode, keysym = event.GetKeyCode(), None
        key = chr(0)
        if keycode < 256:
            key = chr(keycode)

        # wxPython 2.6.0.1 does not return a valid event.Get{X,Y}()
        # for this event, so we use the cached position.
        (x,y)= self._Iren.GetEventPosition()
        self._Iren.SetEventInformation(x, y,
                                       ctrl, shift, key, 0,
                                       keysym)

        self._Iren.KeyPressEvent()
        self._Iren.CharEvent()


    def OnKeyUp(self,event):
        """Handles the wx.EVT_KEY_UP event for
        wxVTKRenderWindowInteractor.
        """

        # event processing should continue
        event.Skip()

        ctrl, shift = event.ControlDown(), event.ShiftDown()
        keycode, keysym = event.GetKeyCode(), None
        key = chr(0)
        if keycode < 256:
            key = chr(keycode)

        self._Iren.SetEventInformationFlipY(event.GetX(), event.GetY(),
                                            ctrl, shift, key, 0,
                                            keysym)
        self._Iren.KeyReleaseEvent()


    def GetRenderWindow(self):
        """Returns the render window (vtkRenderWindow).
        """
        return self._Iren.GetRenderWindow()

    def Render(self):
        """Actually renders the VTK scene on screen.
        """
        RenderAllowed = 1

        if not self.__RenderWhenDisabled:
            # the user doesn't want us to render when the toplevel frame
            # is disabled - first find the top level parent
            topParent = wx.GetTopLevelParent(self)
            if topParent:
                # if it exists, check whether it's enabled
                # if it's not enabeld, RenderAllowed will be false
                RenderAllowed = topParent.IsEnabled()

        if RenderAllowed:
            if self.__handle and self.__handle == self.GetHandle():
                self._Iren.GetRenderWindow().Render()

            elif self.GetHandle() and self.__has_painted:
                # this means the user has reparented us; let's adapt to the
                # new situation by doing the WindowRemap dance
                self._Iren.GetRenderWindow().SetNextWindowInfo(
                    str(self.GetHandle()))

                # make sure the DisplayId is also set correctly
                d = self.GetDisplayId()
                if d:
                    self._Iren.GetRenderWindow().SetDisplayId(d)

                # do the actual remap with the new parent information
                self._Iren.GetRenderWindow().WindowRemap()

                # store the new situation
                self.__handle = self.GetHandle()
                self._Iren.GetRenderWindow().Render()

    def SetRenderWhenDisabled(self, newValue):
        """Change value of __RenderWhenDisabled ivar.

        If __RenderWhenDisabled is false (the default), this widget will not
        call Render() on the RenderWindow if the top level frame (i.e. the
        containing frame) has been disabled.

        This prevents recursive rendering during wx.SafeYield() calls.
        wx.SafeYield() can be called during the ProgressMethod() callback of
        a VTK object to have progress bars and other GUI elements updated -
        it does this by disabling all windows (disallowing user-input to
        prevent re-entrancy of code) and then handling all outstanding
        GUI events.

        However, this often triggers an OnPaint() method for wxVTKRWIs,
        resulting in a Render(), resulting in Update() being called whilst
        still in progress.
        """
        self.__RenderWhenDisabled = bool(newValue)


#--------------------------------------------------------------------
def wxVTKRenderWindowInteractorConeExample():
    """Like it says, just a simple example
    """
    # every wx app needs an app
    app = wx.App(False)

    # create the top-level frame, sizer and wxVTKRWI
    frame = wx.Frame(None, -1, "wxVTKRenderWindowInteractor", size=(400,400))
    widget = wxVTKRenderWindowInteractor(frame, -1)
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(widget, 1, wx.EXPAND)
    frame.SetSizer(sizer)
    frame.Layout()

    # It would be more correct (API-wise) to call widget.Initialize() and
    # widget.Start() here, but Initialize() calls RenderWindow.Render().
    # That Render() call will get through before we can setup the
    # RenderWindow() to render via the wxWidgets-created context; this
    # causes flashing on some platforms and downright breaks things on
    # other platforms.  Instead, we call widget.Enable().  This means
    # that the RWI::Initialized ivar is not set, but in THIS SPECIFIC CASE,
    # that doesn't matter.
    widget.Enable(1)

    widget.AddObserver("ExitEvent", lambda o,e,f=frame: f.Close())

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
    wxVTKRenderWindowInteractorConeExample()

