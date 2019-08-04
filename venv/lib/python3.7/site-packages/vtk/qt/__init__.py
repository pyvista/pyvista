"""Qt module for VTK/Python.

Example usage:

    import sys
    import PyQt5
    from PyQt5.QtWidgets import QApplication
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    app = QApplication(sys.argv)

    widget = QVTKRenderWindowInteractor()
    widget.Initialize()
    widget.Start()

    renwin = widget.GetRenderWindow()

For more information, see QVTKRenderWidgetConeExample() in the file
QVTKRenderWindowInteractor.py.
"""

import sys

# PyQtImpl can be set by the user
PyQtImpl = None

# Has an implementation has been imported yet?
for impl in ["PyQt5", "PyQt4", "PySide"]:
    if impl in sys.modules:
        PyQtImpl = impl
        break

# QVTKRWIBase, base class for QVTKRenderWindowInteractor,
# can be altered by the user to "QGLWidget" in case
# of rendering errors (e.g. depth check problems, readGLBuffer
# warnings...)
QVTKRWIBase = "QWidget"

__all__ = ['QVTKRenderWindowInteractor']
