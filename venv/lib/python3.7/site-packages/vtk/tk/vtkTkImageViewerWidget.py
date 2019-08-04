"""
A vtkTkImageViewerWidget for python, which is based on the
vtkTkImageWindowWidget.

Specify double=1 to get a double-buffered window.

Created by David Gobbi, Nov 1999
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

class vtkTkImageViewerWidget(tkinter.Widget):
    """
    A vtkTkImageViewerWidget for Python.

    Use GetImageViewer() to get the vtkImageViewer.

    Create with the keyword double=1 in order to generate a
    double-buffered viewer.

    Create with the keyword focus_on_enter=1 to enable
    focus-follows-mouse.  The default is for a click-to-focus mode.
    """
    def __init__(self, master, cnf={}, **kw):
        """
        Constructor.

        Keyword arguments:

          iv -- Use passed image viewer instead of creating a new one.

          double -- If True, generate a double-buffered viewer.
          Defaults to False.

          focus_on_enter -- If True, use a focus-follows-mouse mode.
          Defaults to False where the widget will use a click-to-focus
          mode.
        """
        # load the necessary extensions into tk
        vtkLoadPythonTkWidgets(master.tk)

        try: # use specified vtkImageViewer
            imageViewer = kw['iv']
        except KeyError: # or create one if none specified
            imageViewer = vtk.vtkImageViewer()

        doubleBuffer = 0
        try:
            if kw['double']:
                doubleBuffer = 1
            del kw['double']
        except:
            pass

        # check if focus should follow mouse
        if kw.get('focus_on_enter'):
            self._FocusOnEnter = 1
            del kw['focus_on_enter']
        else:
            self._FocusOnEnter = 0

        kw['iv'] = imageViewer.GetAddressAsString("vtkImageViewer")
        tkinter.Widget.__init__(self, master, 'vtkTkImageViewerWidget',
                                cnf, kw)
        if doubleBuffer:
            imageViewer.GetRenderWindow().DoubleBufferOn()

        self.BindTkImageViewer()

    def __getattr__(self,attr):
        # because the tk part of vtkTkImageViewerWidget must have
        # the only remaining reference to the ImageViewer when
        # it is destroyed, we can't actually store the ImageViewer
        # as an attribute but instead have to get it from the tk-side
        if attr == '_ImageViewer':
            addr = self.tk.call(self._w, 'GetImageViewer')[5:]
            return vtk.vtkImageViewer('_%s_vtkImageViewer_p' % addr)
        raise AttributeError(self.__class__.__name__ +
              " has no attribute named " + attr)

    def GetImageViewer(self):
        return self._ImageViewer

    def Render(self):
        self._ImageViewer.Render()

    def BindTkImageViewer(self):
        imager = self._ImageViewer.GetRenderer()

        # stuff for window level text.
        mapper = vtk.vtkTextMapper()
        mapper.SetInput("none")
        t_prop = mapper.GetTextProperty()
        t_prop.SetFontFamilyToTimes()
        t_prop.SetFontSize(18)
        t_prop.BoldOn()
        t_prop.ShadowOn()

        self._LevelMapper = mapper

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.SetLayerNumber(1)
        actor.GetPositionCoordinate().SetValue(4,22)
        actor.GetProperty().SetColor(1,1,0.5)
        actor.SetVisibility(0)
        imager.AddActor2D(actor)

        self._LevelActor = actor

        mapper = vtk.vtkTextMapper()
        mapper.SetInput("none")
        t_prop = mapper.GetTextProperty()
        t_prop.SetFontFamilyToTimes()
        t_prop.SetFontSize(18)
        t_prop.BoldOn()
        t_prop.ShadowOn()

        self._WindowMapper = mapper

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.SetLayerNumber(1)
        actor.GetPositionCoordinate().SetValue(4,4)
        actor.GetProperty().SetColor(1,1,0.5)
        actor.SetVisibility(0)
        imager.AddActor2D(actor)

        self._WindowActor = actor

        self._LastX = 0
        self._LastY = 0
        self._OldFocus = 0
        self._InExpose = 0

        # bindings
        # window level
        self.bind("<ButtonPress-1>",
                  lambda e,s=self: s.StartWindowLevelInteraction(e.x,e.y))
        self.bind("<B1-Motion>",
                  lambda e,s=self: s.UpdateWindowLevelInteraction(e.x,e.y))
        self.bind("<ButtonRelease-1>",
                  lambda e,s=self: s.EndWindowLevelInteraction())

        # Get the value
        self.bind("<ButtonPress-3>",
                  lambda e,s=self: s.StartQueryInteraction(e.x,e.y))
        self.bind("<B3-Motion>",
                  lambda e,s=self: s.UpdateQueryInteraction(e.x,e.y))
        self.bind("<ButtonRelease-3>",
                  lambda e,s=self: s.EndQueryInteraction())

        self.bind("<Expose>",
                  lambda e,s=self: s.ExposeTkImageViewer())
        self.bind("<Enter>",
                  lambda e,s=self: s.EnterTkViewer())
        self.bind("<Leave>",
                  lambda e,s=self: s.LeaveTkViewer())
        self.bind("<KeyPress-e>",
                  lambda e,s=self: s.quit())
        self.bind("<KeyPress-r>",
                  lambda e,s=self: s.ResetTkImageViewer())

    def GetImageViewer(self):
        return self._ImageViewer

    def Render(self):
        self._ImageViewer.Render()

    def _GrabFocus(self):
        self._OldFocus=self.focus_get()
        self.focus()

    def EnterTkViewer(self):
        if self._FocusOnEnter:
            self._GrabFocus()

    def LeaveTkViewer(self):
        if self._FocusOnEnter and (self._OldFocus != None):
            self._OldFocus.focus()

    def ExposeTkImageViewer(self):
        if (self._InExpose == 0):
            self._InExpose = 1
            if (not self._ImageViewer.GetRenderWindow().
                IsA('vtkCocoaRenderWindow')):
                self.update()
            self._ImageViewer.Render()
            self._InExpose = 0

    def StartWindowLevelInteraction(self,x,y):
        if not self._FocusOnEnter:
            self._GrabFocus()
        viewer = self._ImageViewer
        self._LastX = x
        self._LastY = y
        self._Window = float(viewer.GetColorWindow())
        self._Level = float(viewer.GetColorLevel())

        # make the window level text visible
        self._LevelActor.SetVisibility(1)
        self._WindowActor.SetVisibility(1)

        self.UpdateWindowLevelInteraction(x,y)

    def EndWindowLevelInteraction(self):
        # make the window level text invisible
        self._LevelActor.SetVisibility(0)
        self._WindowActor.SetVisibility(0)
        self.Render()

    def UpdateWindowLevelInteraction(self,x,y):
        # compute normalized delta
        dx = 4.0*(x - self._LastX)/self.winfo_width()*self._Window
        dy = 4.0*(self._LastY - y)/self.winfo_height()*self._Level

        # abs so that direction does not flip
        if (self._Window < 0.0):
            dx = -dx
        if (self._Level < 0.0):
            dy = -dy

        # compute new window level
        window = self._Window + dx
        if (window < 0.0):
            level = self._Level + dy
        else:
            level = self._Level - dy

        viewer = self._ImageViewer
        viewer.SetColorWindow(window)
        viewer.SetColorLevel(level)

        self._WindowMapper.SetInput("Window: %g" % window)
        self._LevelMapper.SetInput("Level: %g" % level)

        self.Render()


    def ResetTkImageViewer(self):
        # Reset: Set window level to show all values
        viewer = self._ImageViewer
        input = viewer.GetInput()
        if (input == None):
            return

        # Get the extent in viewer
        z = viewer.GetZSlice()

        input.UpdateInformation()
        info = input.GetOutputInformation(0)
        ext = info.Get(vtk.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT())
        ext[4] = z
        ext[5] = z
        input.Update(0, 1, 0, ext)

        (low,high) = input.GetScalarRange()

        viewer.SetColorWindow(high - low)
        viewer.SetColorLevel((high + low) * 0.5)

        self.Render()

    def StartQueryInteraction(self,x,y):
        if not self._FocusOnEnter:
            self._GrabFocus()
        # Query PixleValue stuff
        self._WindowActor.SetVisibility(1)
        self.UpdateQueryInteraction(x,y)

    def EndQueryInteraction(self):
        self._WindowActor.SetVisibility(0)
        self.Render()

    def UpdateQueryInteraction(self,x,y):
        viewer = self._ImageViewer
        input = viewer.GetInput()
        z = viewer.GetZSlice()

        # y is flipped upside down
        y = self.winfo_height() - y

        # make sure point is in the extent of the image.
        (xMin,xMax,yMin,yMax,zMin,zMax) = input.GetExtent()
        if (x < xMin or x > xMax or y < yMin or \
            y > yMax or z < zMin or z > zMax):
            return

        numComps = input.GetNumberOfScalarComponents()
        text = ""
        for i in xrange(numComps):
            val = input.GetScalarComponentAsDouble(x,y,z,i)
            text = "%s  %.1f" % (text,val)

        self._WindowMapper.SetInput("(%d, %d): %s" % (x,y,text))

        self.Render()

#-----------------------------------------------------------------------------
# an example of how to use this widget
if __name__ == "__main__":
    canvas = vtk.vtkImageCanvasSource2D()
    canvas.SetNumberOfScalarComponents(3)
    canvas.SetScalarType(3)
    canvas.SetExtent(0,511,0,511,0,0)
    canvas.SetDrawColor(100,100,0)
    canvas.FillBox(0,511,0,511)
    canvas.SetDrawColor(200,0,200)
    canvas.FillBox(32,511,100,500)
    canvas.SetDrawColor(100,0,0)
    canvas.FillTube(550,20,30,400,5)
    canvas.SetDrawColor(255,255,255)
    canvas.DrawSegment3D(10,20,0,90,510,0)
    canvas.SetDrawColor(200,50,50)
    canvas.DrawSegment3D(510,90,0,10,20,0)

    # Check segment clipping
    canvas.SetDrawColor(0,200,0)
    canvas.DrawSegment(-10,30,30,-10)
    canvas.DrawSegment(-10,481,30,521)
    canvas.DrawSegment(481,-10,521,30)
    canvas.DrawSegment(481,521,521,481)

    # Check Filling a triangle
    canvas.SetDrawColor(20,200,200)
    canvas.FillTriangle(-100,100,190,150,40,300)

    # Check drawing a circle
    canvas.SetDrawColor(250,250,10)
    canvas.DrawCircle(350,350,200.0)

    # Check drawing a point
    canvas.SetDrawColor(250,250,250)
    canvas.DrawPoint(350,350)
    canvas.DrawPoint(350,550)

    # Test filling functionality
    canvas.SetDrawColor(55,0,0)
    canvas.DrawCircle(450,350,80.0)
    canvas.SetDrawColor(100,255,100)
    canvas.FillPixel(450,350)

    # Create the GUI: two renderer widgets and a quit button

    frame = tkinter.Frame()

    widget = vtkTkImageViewerWidget(frame,width=512,height=512,double=1)
    viewer = widget.GetImageViewer()
    viewer.SetInputConnection(canvas.GetOutputPort())
    viewer.SetColorWindow(256)
    viewer.SetColorLevel(127.5)

    button = tkinter.Button(frame,text="Quit",command=frame.quit)

    widget.pack(side='top',padx=3,pady=3,fill='both',expand='t')
    frame.pack(fill='both',expand='t')
    button.pack(fill='x')

    frame.mainloop()

