import vtk.qt

try:
    import PyQt4
    vtk.qt.PyQtImpl = "PyQt4"
except ImportError:
    try:
        import PySide
        vtk.qt.PyQtImpl = "PySide"
    except ImportError:
        raise ImportError("Cannot load either PyQt or PySide")

from vtk.qt.QVTKRenderWindowInteractor import *

if __name__ == "__main__":
    print(PyQtImpl)
    QVTKRenderWidgetConeExample()
