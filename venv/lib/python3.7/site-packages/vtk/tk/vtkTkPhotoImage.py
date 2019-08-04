"""
A subclass of tkinter.PhotoImage that connects a
vtkImageData to a photo widget.

Created by Daniel Blezek, August 2002
"""

from __future__ import absolute_import
import sys

if sys.hexversion < 0x03000000:
    # for Python2
    import Tkinter as tkinter
else:
    # for Python3
    import tkinter

from .vtkLoadPythonTkWidgets import vtkLoadPythonTkWidgets

class vtkTkPhotoImage ( tkinter.PhotoImage ):
    """
    A subclass of PhotoImage with helper functions
    for displaying vtkImageData
    """
    def __init__ ( self, **kw ):
        # Caller the superclass
        tkinter.PhotoImage.__init__ ( self, kw )
        vtkLoadPythonTkWidgets ( self.tk )

    def PutImageSlice ( self, image, z, orientation='transverse', window=256, level=128 ):
        t = str ( image.__this__ )
        s = 'vtkImageDataToTkPhoto %s %s %d %s %d %d' % ( t, self.name, z, orientation, window, level )
        self.tk.eval ( s )
