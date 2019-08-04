"""
Description:

  Provides a pyGtk vtkRenderWindowInteractor widget.  This embeds a
  vtkRenderWindow inside a GTK widget and uses the
  vtkGenericRenderWindowInteractor for the event handling.  This is
  similar to GtkVTKRenderWindowInteractor.py.

  The extensions here allow the use of gtkglext rather than gtkgl and
  pygtk-2 rather than pygtk-0.  It requires pygtk-2.0.0 or later.

  There is a working example at the bottom.

Credits:

  John Hunter <jdhunter@ace.bsd.uchicago.edu> developed and tested
  this code based on VTK's GtkVTKRenderWindow.py and extended it to
  work with pygtk-2.0.0.

License:

  VTK license.

"""

import sys
import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk
import gtk.gtkgl
import vtk

class GtkGLExtVTKRenderWindowInteractor(gtk.gtkgl.DrawingArea):

    """ Embeds a vtkRenderWindow into a pyGTK widget and uses
    vtkGenericRenderWindowInteractor for the event handling.  This
    class embeds the RenderWindow correctly.  A __getattr__ hook is
    provided that makes the class behave like a
    vtkGenericRenderWindowInteractor."""

    def __init__(self, *args):
        gtk.gtkgl.DrawingArea.__init__(self)

        self.set_double_buffered(gtk.FALSE)

        self._RenderWindow = vtk.vtkRenderWindow()

        # private attributes
        self.__Created = 0
        self._ActiveButton = 0

        self._Iren = vtk.vtkGenericRenderWindowInteractor()
        self._Iren.SetRenderWindow(self._RenderWindow)
        self._Iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self.ConnectSignals()

        # need this to be able to handle key_press events.
        self.set_flags(gtk.CAN_FOCUS)

    def set_size_request(self, w, h):
        gtk.gtkgl.DrawingArea.set_size_request(self, w, h)
        self._RenderWindow.SetSize(w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()

    def ConnectSignals(self):
        self.connect("realize", self.OnRealize)
        self.connect("expose_event", self.OnExpose)
        self.connect("configure_event", self.OnConfigure)
        self.connect("button_press_event", self.OnButtonDown)
        self.connect("button_release_event", self.OnButtonUp)
        self.connect("motion_notify_event", self.OnMouseMove)
        self.connect("enter_notify_event", self.OnEnter)
        self.connect("leave_notify_event", self.OnLeave)
        self.connect("key_press_event", self.OnKeyPress)
        self.connect("delete_event", self.OnDestroy)
        self.add_events(gdk.EXPOSURE_MASK| gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.KEY_PRESS_MASK |
                        gdk.POINTER_MOTION_MASK |
                        gdk.POINTER_MOTION_HINT_MASK |
                        gdk.ENTER_NOTIFY_MASK | gdk.LEAVE_NOTIFY_MASK)

    def __getattr__(self, attr):
        """Makes the object behave like a
        vtkGenericRenderWindowInteractor"""
        if attr == '__vtk__':
            return lambda t=self._Iren: t
        elif hasattr(self._Iren, attr):
            return getattr(self._Iren, attr)
        else:
            raise AttributeError(self.__class__.__name__ +
                  " has no attribute named " + attr)

    def CreateTimer(self, obj, event):
        gtk.timeout_add(10, self._Iren.TimerEvent)

    def DestroyTimer(self, obj, event):
        """The timer is a one shot timer so will expire automatically."""
        return 1

    def GetRenderWindow(self):
        return self._RenderWindow

    def Render(self):
        if self.__Created:
            self._RenderWindow.Render()

    def OnRealize(self, *args):
        if self.__Created == 0:
            # you can't get the xid without the window being realized.
            self.realize()
            if sys.platform=='win32':
                win_id = str(self.widget.window.handle)
            else:
                win_id = str(self.widget.window.xid)

            self._RenderWindow.SetWindowInfo(win_id)
            #self._Iren.Initialize()
            self.__Created = 1
        return gtk.TRUE

    def OnConfigure(self, widget, event):
        self.widget=widget
        self._Iren.SetSize(event.width, event.height)
        self._Iren.ConfigureEvent()
        self.Render()
        return gtk.TRUE

    def OnExpose(self, *args):
        self.Render()
        return gtk.TRUE

    def OnDestroy(self, event=None):
        self.hide()
        del self._RenderWindow
        self.destroy()
        return gtk.TRUE

    def _GetCtrlShift(self, event):
        ctrl, shift = 0, 0
        if ((event.state & gdk.CONTROL_MASK) == gdk.CONTROL_MASK):
            ctrl = 1
        if ((event.state & gdk.SHIFT_MASK) == gdk.SHIFT_MASK):
            shift = 1
        return ctrl, shift

    def OnButtonDown(self, wid, event):
        """Mouse button pressed."""
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            chr(0), 0, None)
        button = event.button
        if button == 3:
            self._Iren.RightButtonPressEvent()
            return gtk.TRUE
        elif button == 1:
            self._Iren.LeftButtonPressEvent()
            return gtk.TRUE
        elif button == 2:
            self._Iren.MiddleButtonPressEvent()
            return gtk.TRUE
        else:
            return gtk.FALSE

    def OnButtonUp(self, wid, event):
        """Mouse button released."""
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            chr(0), 0, None)
        button = event.button
        if button == 3:
            self._Iren.RightButtonReleaseEvent()
            return gtk.TRUE
        elif button == 1:
            self._Iren.LeftButtonReleaseEvent()
            return gtk.TRUE
        elif button == 2:
            self._Iren.MiddleButtonReleaseEvent()
            return gtk.TRUE

        return gtk.FALSE

    def OnMouseMove(self, wid, event):
        """Mouse has moved."""
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            chr(0), 0, None)
        self._Iren.MouseMoveEvent()
        return gtk.TRUE

    def OnEnter(self, wid, event):
        """Entering the vtkRenderWindow."""
        self.grab_focus()
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            chr(0), 0, None)
        self._Iren.EnterEvent()
        return gtk.TRUE

    def OnLeave(self, wid, event):
        """Leaving the vtkRenderWindow."""
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            chr(0), 0, None)
        self._Iren.LeaveEvent()
        return gtk.TRUE

    def OnKeyPress(self, wid, event):
        """Key pressed."""
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        keycode, keysym = event.keyval, event.string
        key = chr(0)
        if keycode < 256:
            key = chr(keycode)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            key, 0, keysym)
        self._Iren.KeyPressEvent()
        self._Iren.CharEvent()
        return gtk.TRUE

    def OnKeyRelease(self, wid, event):
        "Key released."
        m = self.get_pointer()
        ctrl, shift = self._GetCtrlShift(event)
        keycode, keysym = event.keyval, event.string
        key = chr(0)
        if keycode < 256:
            key = chr(keycode)
        self._Iren.SetEventInformationFlipY(m[0], m[1], ctrl, shift,
                                            key, 0, keysym)
        self._Iren.KeyReleaseEvent()
        return gtk.TRUE

    def Initialize(self):
        if self.__Created:
            self._Iren.Initialize()

    def SetPicker(self, picker):
        self._Iren.SetPicker(picker)

    def GetPicker(self, picker):
        return self._Iren.GetPicker()


def main():
    # The main window
    window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    window.set_title("A GtkVTKRenderWindow Demo!")
    window.connect("destroy", gtk.mainquit)
    window.connect("delete_event", gtk.mainquit)
    window.set_border_width(10)

    # A VBox into which widgets are packed.
    vbox = gtk.VBox(spacing=3)
    window.add(vbox)
    vbox.show()

    # The GtkVTKRenderWindow
    gvtk = GtkGLExtVTKRenderWindowInteractor()
    #gvtk.SetDesiredUpdateRate(1000)
    gvtk.set_size_request(400, 400)
    vbox.pack_start(gvtk)
    gvtk.show()
    gvtk.Initialize()
    gvtk.Start()
    # prevents 'q' from exiting the app.
    gvtk.AddObserver("ExitEvent", lambda o,e,x=None: x)

    # The VTK stuff.
    cone = vtk.vtkConeSource()
    cone.SetResolution(80)
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())
    #coneActor = vtk.vtkLODActor()
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetColor(0.5, 0.5, 1.0)
    ren = vtk.vtkRenderer()
    gvtk.GetRenderWindow().AddRenderer(ren)
    ren.AddActor(coneActor)

    # A simple quit button
    quit = gtk.Button("Quit!")
    quit.connect("clicked", gtk.mainquit)
    vbox.pack_start(quit)
    quit.show()

    # show the main window and start event processing.
    window.show()
    gtk.mainloop()


if __name__ == "__main__":
    main()
