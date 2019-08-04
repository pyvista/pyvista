"""
A simple vtkTkRenderWidget for tkinter.

Created by David Gobbi, April 1999

May ??, 1999  - Modifications performed by Heather Drury,
                to rewrite _pan to match method in TkInteractor.tcl
May 11, 1999  - Major rewrite by David Gobbi to make the
                interactor bindings identical to the TkInteractor.tcl
                bindings.
July 14, 1999 - Added modification by Ken Martin for VTK 2.4, to
                use vtk widgets instead of Togl.
Aug 29, 1999  - Renamed file to vtkRenderWidget.py
Nov 14, 1999  - Added support for keyword 'rw'
Mar 23, 2000  - Extensive but backwards compatible changes,
                improved documentation

A few important notes:

This class is meant to be used as a base-class widget for
doing VTK rendering in Python.

In VTK (and C++) there is a very important distinction between
public ivars (attributes in pythonspeak), protected ivars, and
private ivars.  When you write a python class that you want
to 'look and feel' like a VTK class, you should follow these rules.

1) Attributes should never be public.  Attributes should always be
   either protected (prefixed with a single underscore) or private
   (prefixed with a double underscore).  You can provide access to
   attributes through public Set/Get methods (same as VTK).

2) Use a single underscore to denote a protected attribute, e.g.
   self._RenderWindow is protected (can be accessed from this
   class or a derived class).

3) Use a double underscore to denote a private attribute, e.g.
   self.__InExpose cannot be accessed outside of this class.

All attributes should be 'declared' in the __init__() function
i.e. set to some initial value.  Don't forget that 'None' means
'NULL' - the python/vtk wrappers guarantee their equivalence.
"""

from __future__ import absolute_import
import math, os, sys
import vtk

if sys.hexversion < 0x03000000:
    # for Python2
    import Tkinter as tkinter
else:
    # for Python3
    import tkinter

from .vtkLoadPythonTkWidgets import vtkLoadPythonTkWidgets

class vtkTkRenderWidget(tkinter.Widget):
    """
    A vtkTkRenderWidget for Python.

    Use GetRenderWindow() to get the vtkRenderWindow.

    Create with the keyword stereo=1 in order to generate a
    stereo-capable window.

    Create with the keyword focus_on_enter=1 to enable
    focus-follows-mouse.  The default is for a click-to-focus mode.
    """
    def __init__(self, master, cnf={}, **kw):
        """
        Constructor.

        Keyword arguments:

          rw -- Use passed render window instead of creating a new one.

          stereo -- If True, generate a stereo-capable window.
          Defaults to False.

          focus_on_enter -- If True, use a focus-follows-mouse mode.
          Defaults to False where the widget will use a click-to-focus
          mode.
        """
        # load the necessary extensions into tk
        vtkLoadPythonTkWidgets(master.tk)

        try: # check to see if a render window was specified
            renderWindow = kw['rw']
        except KeyError:
            renderWindow = vtk.vtkRenderWindow()

        try:  # was a stereo rendering context requested?
            if kw['stereo']:
               renderWindow.StereoCapableWindowOn()
               del kw['stereo']
        except KeyError:
            pass

        # check if focus should follow mouse
        if kw.get('focus_on_enter'):
            self._FocusOnEnter = 1
            del kw['focus_on_enter']
        else:
            self._FocusOnEnter = 0

        kw['rw'] = renderWindow.GetAddressAsString("vtkRenderWindow")
        tkinter.Widget.__init__(self, master, 'vtkTkRenderWidget', cnf, kw)

        self._CurrentRenderer = None
        self._CurrentCamera = None
        self._CurrentZoom = 1.0
        self._CurrentLight = None

        self._ViewportCenterX = 0
        self._ViewportCenterY = 0

        self._Picker = vtk.vtkCellPicker()
        self._PickedAssembly = None
        self._PickedProperty = vtk.vtkProperty()
        self._PickedProperty.SetColor(1,0,0)
        self._PrePickedProperty = None

        self._OldFocus = None

        # used by the LOD actors
        self._DesiredUpdateRate = 15
        self._StillUpdateRate = 0.0001

        # these record the previous mouse position
        self._LastX = 0
        self._LastY = 0

        # private attributes
        self.__InExpose = 0

        # create the Tk bindings
        self.BindTkRenderWidget()

    def __getattr__(self,attr):
        # because the tk part of vtkTkRenderWidget must have
        # the only remaining reference to the RenderWindow when
        # it is destroyed, we can't actually store the RenderWindow
        # as an attribute but instead have to get it from the tk-side
        if attr == '_RenderWindow':
            return self.GetRenderWindow()
        raise AttributeError(self.__class__.__name__ +
              " has no attribute named " + attr)

    def BindTkRenderWidget(self):
        """
        Bind some default actions.
        """
        self.bind("<ButtonPress>",
                  lambda e,s=self: s.StartMotion(e.x,e.y))
        self.bind("<ButtonRelease>",
                  lambda e,s=self: s.EndMotion(e.x,e.y))
        self.bind("<B1-Motion>",
                  lambda e,s=self: s.Rotate(e.x,e.y))
        self.bind("<B2-Motion>",
                  lambda e,s=self: s.Pan(e.x,e.y))
        self.bind("<B3-Motion>",
                  lambda e,s=self: s.Zoom(e.x,e.y))
        self.bind("<Shift-B1-Motion>",
                  lambda e,s=self: s.Pan(e.x,e.y))
        self.bind("<KeyPress-r>",
                  lambda e,s=self: s.Reset(e.x,e.y))
        self.bind("<KeyPress-u>",
                  lambda e,s=self: s.deiconify())
        self.bind("<KeyPress-w>",
                  lambda e,s=self: s.Wireframe())
        self.bind("<KeyPress-s>",
                  lambda e,s=self: s.Surface())
        self.bind("<KeyPress-p>",
                  lambda e,s=self: s.PickActor(e.x,e.y))
        if self._FocusOnEnter:
            self.bind("<Enter>",
                      lambda e,s=self: s.Enter(e.x,e.y))
            self.bind("<Leave>",
                      lambda e,s=self: s.Leave(e.x,e.y))
        else:
            self.bind("<ButtonPress>",
                      lambda e,s=self: s.Enter(e.x,e.y))
        self.bind("<Expose>",
                  lambda e,s=self: s.Expose())

    def GetZoomFactor(self):
        return self._CurrentZoom

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

    def GetRenderWindow(self):
        addr = self.tk.call(self._w, 'GetRenderWindow')[5:]
        return vtk.vtkRenderWindow('_%s_vtkRenderWindow_p' % addr)

    def GetPicker(self):
        return self._Picker

    def Expose(self):
        if (not self.__InExpose):
            self.__InExpose = 1
            if (not self._RenderWindow.IsA('vtkCocoaRenderWindow')):
                self.update()
            self._RenderWindow.Render()
            self.__InExpose = 0

    def Render(self):
        if (self._CurrentLight):
            light = self._CurrentLight
            light.SetPosition(self._CurrentCamera.GetPosition())
            light.SetFocalPoint(self._CurrentCamera.GetFocalPoint())

        self._RenderWindow.Render()

    def UpdateRenderer(self,x,y):
        """
        UpdateRenderer will identify the renderer under the mouse and set
        up _CurrentRenderer, _CurrentCamera, and _CurrentLight.
        """
        windowX = self.winfo_width()
        windowY = self.winfo_height()

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

    def Enter(self,x,y):
        self._OldFocus=self.focus_get()
        self.focus()
        self.StartMotion(x, y)

    def Leave(self,x,y):
        if (self._OldFocus != None):
            self._OldFocus.focus()

    def StartMotion(self,x,y):
        self.GetRenderWindow().SetDesiredUpdateRate(self._DesiredUpdateRate)
        self.UpdateRenderer(x,y)

    def EndMotion(self,x,y):
        self.GetRenderWindow().SetDesiredUpdateRate(self._StillUpdateRate)
        if self._CurrentRenderer:
            self.Render()

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

    def Reset(self,x,y):
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

            windowY = self.winfo_height()
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

#----------------------------------------------------------------------------
def vtkRenderWidgetConeExample():
    """Like it says, just a simple example
    """
    # create root window
    root = tkinter.Tk()

    # create vtkTkRenderWidget
    pane = vtkTkRenderWidget(root,width=300,height=300)

    ren = vtk.vtkRenderer()
    pane.GetRenderWindow().AddRenderer(ren)

    cone = vtk.vtkConeSource()
    cone.SetResolution(8)

    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)

    ren.AddActor(coneActor)

    # pack the pane into the tk root
    pane.pack()

    # start the tk mainloop
    root.mainloop()

if __name__ == "__main__":
    vtkRenderWidgetConeExample()

