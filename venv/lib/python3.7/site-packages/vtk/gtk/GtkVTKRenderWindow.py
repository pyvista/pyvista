"""
Description:

  Provides a simple VTK widget for pyGtk.  This embeds a
  vtkRenderWindow inside a GTK widget.  This is based on
  vtkTkRenderWidget.py.  The GtkVTKRenderWindowBase class provides the
  abstraction necessary for someone to use their own interaction
  behaviour.  The method names are similar to those in
  vtkInteractorStyle.h.

  The class uses the gtkgl.GtkGLArea widget (gtkglarea).  This avoids
  a lot of problems with flicker.

  There is a working example at the bottom.

Credits:

  Thanks to Dave Reed for testing the code under various platforms and
  for his suggestion to use the GtkGLArea widget to avoid flicker
  related issues.

Created by Prabhu Ramachandran, March 2001.

Using GtkGLArea, March, 2002.

Bugs:

  (*) There is a focus related problem.  Tkinter has a focus object
  that handles focus events.  I don't know of an equivalent object
  under GTK.  So, when an 'enter_notify_event' is received on the
  GtkVTKRenderWindow I grab the focus but I don't know what to do when
  I get a 'leave_notify_event'.

  (*) Will not work under Win32 because it uses the XID of a window in
  OnRealize.  Suggestions to fix this will be appreciated.

"""

import gtk, GDK, gtkgl
import vtk
import math


class GtkVTKRenderWindowBase(gtkgl.GtkGLArea):

    """ A base class that enables one to embed a vtkRenderWindow into
    a pyGTK widget.  This class embeds the RenderWindow correctly.
    Provided are some empty methods that can be overloaded to provide
    a user defined interaction behaviour.  The event handling
    functions have names that are somewhat similar to the ones in the
    vtkInteractorStyle class included with VTK. """

    def __init__(self, *args):
        l = list(args)
        attr = (gtkgl.RGBA, gtkgl.DOUBLEBUFFER)
        l.insert(0, self)
        l.insert(1, attr)
        apply(gtkgl.GtkGLArea.__init__, l)
        self._RenderWindow = vtk.vtkRenderWindow()

        # private attributes
        self.__Created = 0

        # used by the LOD actors
        self._DesiredUpdateRate = 15
        self._StillUpdateRate = 0.0001

        self.ConnectSignals()

        # need this to be able to handle key_press events.
        self.set_flags(gtk.CAN_FOCUS)
        # default size
        self.set_usize(300, 300)

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
        self.add_events(GDK.EXPOSURE_MASK| GDK.BUTTON_PRESS_MASK |
                        GDK.BUTTON_RELEASE_MASK |
                        GDK.KEY_PRESS_MASK |
                        GDK.POINTER_MOTION_MASK |
                        GDK.POINTER_MOTION_HINT_MASK |
                        GDK.ENTER_NOTIFY_MASK | GDK.LEAVE_NOTIFY_MASK)

    def GetRenderWindow(self):
        return self._RenderWindow

    def GetRenderer(self):
        self._RenderWindow.GetRenderers().InitTraversal()
        return self._RenderWindow.GetRenderers().GetNextItem()

    def SetDesiredUpdateRate(self, rate):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        self._DesiredUpdateRate = rate

    def GetDesiredUpdateRate(self):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        return self._DesiredUpdateRate

    def SetStillUpdateRate(self, rate):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        self._StillUpdateRate = rate

    def GetStillUpdateRate(self):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        return self._StillUpdateRate

    def Render(self):
        if self.__Created:
            self._RenderWindow.Render()

    def OnRealize(self, *args):
        if self.__Created == 0:
            # you can't get the xid without the window being realized.
            self.realize()
            win_id = str(self.get_window().xid)
            self._RenderWindow.SetWindowInfo(win_id)
            self.__Created = 1
        return gtk.TRUE

    def OnConfigure(self, wid, event=None):
        sz = self._RenderWindow.GetSize()
        if (event.width != sz[0]) or (event.height != sz[1]):
            self._RenderWindow.SetSize(event.width, event.height)
        return gtk.TRUE

    def OnExpose(self, *args):
        self.Render()
        return gtk.TRUE

    def OnDestroy(self, event=None):
        self.hide()
        del self._RenderWindow
        self.destroy()
        return gtk.TRUE

    def OnButtonDown(self, wid, event):
        """Mouse button pressed."""
        self._RenderWindow.SetDesiredUpdateRate(self._DesiredUpdateRate)
        return gtk.TRUE

    def OnButtonUp(self, wid, event):
        """Mouse button released."""
        self._RenderWindow.SetDesiredUpdateRate(self._StillUpdateRate)
        return gtk.TRUE

    def OnMouseMove(self, wid, event):
        """Mouse has moved."""
        return gtk.TRUE

    def OnEnter(self, wid, event):
        """Entering the vtkRenderWindow."""
        return gtk.TRUE

    def OnLeave(self, wid, event):
        """Leaving the vtkRenderWindow."""
        return gtk.TRUE

    def OnKeyPress(self, wid, event):
        """Key pressed."""
        return gtk.TRUE

    def OnKeyRelease(self, wid, event):
        "Key released."
        return gtk.TRUE


class GtkVTKRenderWindow(GtkVTKRenderWindowBase):

    """ An example of a fully functional GtkVTKRenderWindow that is
    based on the vtkRenderWidget.py provided with the VTK sources."""

    def __init__(self, *args):
        l = list(args)
        l.insert(0, self)
        apply(GtkVTKRenderWindowBase.__init__, l)

        self._CurrentRenderer = None
        self._CurrentCamera = None
        self._CurrentZoom = 1.0
        self._CurrentLight = None

        self._ViewportCenterX = 0
        self._ViewportCenterY = 0

        self._Picker = vtk.vtkCellPicker()
        self._PickedAssembly = None
        self._PickedProperty = vtk.vtkProperty()
        self._PickedProperty.SetColor(1, 0, 0)
        self._PrePickedProperty = None

        self._OldFocus = None

        # these record the previous mouse position
        self._LastX = 0
        self._LastY = 0

    def OnButtonDown(self, wid, event):
        self._RenderWindow.SetDesiredUpdateRate(self._DesiredUpdateRate)
        return self.StartMotion(wid, event)

    def OnButtonUp(self, wid, event):
        self._RenderWindow.SetDesiredUpdateRate(self._StillUpdateRate)
        return self.EndMotion(wid, event)

    def OnMouseMove(self, wid, event=None):
        if ((event.state & GDK.BUTTON1_MASK) == GDK.BUTTON1_MASK):
            if ((event.state & GDK.SHIFT_MASK) == GDK.SHIFT_MASK):
                m = self.get_pointer()
                self.Pan(m[0], m[1])
                return gtk.TRUE
            else:
                m = self.get_pointer()
                self.Rotate(m[0], m[1])
                return gtk.TRUE
        elif ((event.state & GDK.BUTTON2_MASK) == GDK.BUTTON2_MASK):
            m = self.get_pointer()
            self.Pan(m[0], m[1])
            return gtk.TRUE
        elif ((event.state & GDK.BUTTON3_MASK) == GDK.BUTTON3_MASK):
            m = self.get_pointer()
            self.Zoom(m[0], m[1])
            return gtk.TRUE
        else:
            return gtk.FALSE

    def OnEnter(self, wid, event=None):
        self.grab_focus()
        w = self.get_pointer()
        self.UpdateRenderer(w[0], w[1])
        return gtk.TRUE

    def OnLeave(self, wid, event):
        return gtk.TRUE

    def OnKeyPress(self, wid, event=None):
        if (event.keyval == GDK.r) or (event.keyval == GDK.R):
            self.Reset()
            return gtk.TRUE
        elif (event.keyval == GDK.w) or (event.keyval == GDK.W):
            self.Wireframe()
            return gtk.TRUE
        elif (event.keyval == GDK.s) or (event.keyval == GDK.S):
            self.Surface()
            return gtk.TRUE
        elif (event.keyval == GDK.p) or (event.keyval == GDK.P):
            m = self.get_pointer()
            self.PickActor(m[0], m[1])
            return gtk.TRUE
        else:
            return gtk.FALSE

    def GetZoomFactor(self):
        return self._CurrentZoom

    def SetZoomFactor(self, zf):
        self._CurrentZoom = zf

    def GetPicker(self):
        return self._Picker

    def Render(self):
        if (self._CurrentLight):
            light = self._CurrentLight
            light.SetPosition(self._CurrentCamera.GetPosition())
            light.SetFocalPoint(self._CurrentCamera.GetFocalPoint())

        GtkVTKRenderWindowBase.Render(self)

    def UpdateRenderer(self,x,y):
        """
        UpdateRenderer will identify the renderer under the mouse and set
        up _CurrentRenderer, _CurrentCamera, and _CurrentLight.
        """
        windowX = self.get_window().width
        windowY = self.get_window().height

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
        return self._CurrentRenderer

    def StartMotion(self, wid, event=None):
        x = event.x
        y = event.y
        self.UpdateRenderer(x,y)
        return gtk.TRUE

    def EndMotion(self, wid, event=None):
        if self._CurrentRenderer:
            self.Render()
        return gtk.TRUE

    def Rotate(self,x,y):
        if self._CurrentRenderer:

            self._CurrentCamera.Azimuth(self._LastX - x)
            self._CurrentCamera.Elevation(y - self._LastY)
            self._CurrentCamera.OrthogonalizeViewUp()

            self._LastX = x
            self._LastY = y

            self._CurrentRenderer.ResetCameraClippingRange()
            self.Render()

    def Pan(self,x,y):
        if self._CurrentRenderer:

            renderer = self._CurrentRenderer
            camera = self._CurrentCamera
            (pPoint0,pPoint1,pPoint2) = camera.GetPosition()
            (fPoint0,fPoint1,fPoint2) = camera.GetFocalPoint()

            if (camera.GetParallelProjection()):
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

    def Zoom(self,x,y):
        if self._CurrentRenderer:

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

    def Reset(self):
        if self._CurrentRenderer:
            self._CurrentRenderer.ResetCamera()

        self.Render()

    def Wireframe(self):
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for i in range(0,numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToWireframe()

        self.Render()

    def Surface(self):
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for i in range(0,numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToSurface()

        self.Render()

    def PickActor(self,x,y):
        if self._CurrentRenderer:

            renderer = self._CurrentRenderer
            picker = self._Picker

            windowY = self.get_window().height
            picker.Pick(x,(windowY - y - 1),0.0,renderer)
            assembly = picker.GetAssembly()

            if (self._PickedAssembly != None and
                self._PrePickedProperty != None):
                self._PickedAssembly.SetProperty(self._PrePickedProperty)
                # release hold of the property
                self._PrePickedProperty.UnRegister(self._PrePickedProperty)
                self._PrePickedProperty = None

            if (assembly != None):
                self._PickedAssembly = assembly
                self._PrePickedProperty = self._PickedAssembly.GetProperty()
                # hold onto the property
                self._PrePickedProperty.Register(self._PrePickedProperty)
                self._PickedAssembly.SetProperty(self._PickedProperty)

            self.Render()


def main():
    # The main window
    window = gtk.GtkWindow(gtk.WINDOW_TOPLEVEL)
    window.set_title("A GtkVTKRenderWindow Demo!")
    window.connect("destroy", gtk.mainquit)
    window.connect("delete_event", gtk.mainquit)
    window.set_border_width(10)

    # A VBox into which widgets are packed.
    vbox = gtk.GtkVBox(spacing=3)
    window.add(vbox)
    vbox.show()

    # The GtkVTKRenderWindow
    gvtk = GtkVTKRenderWindow()
    #gvtk.SetDesiredUpdateRate(1000)
    gvtk.set_usize(400, 400)
    vbox.pack_start(gvtk)
    gvtk.show()

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
    quit = gtk.GtkButton("Quit!")
    quit.connect("clicked", gtk.mainquit)
    vbox.pack_start(quit)
    quit.show()

    # show the main window and start event processing.
    window.show()
    gtk.mainloop()


if __name__ == "__main__":
    main()
