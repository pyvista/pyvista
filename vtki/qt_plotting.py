from threading import Thread
import time
import logging

import vtk
import vtk.qt

from vtki.plotting import BasePlotter, plotParams, run_from_ipython

# for display bugs due to older intel integrated GPUs
vtk.qt.QVTKRWIBase = 'QGLWidget'

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


# dummy reference for when PyQt5 is not installed
has_pyqt = False
class QVTKRenderWindowInteractor(object):
    pass


def pyqtSignal():
    return


try:
    from PyQt5.QtCore import pyqtSignal
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    has_pyqt = True
except:
    pass


class QtInteractor(QVTKRenderWindowInteractor, BasePlotter):
    """
    Extends QVTKRenderWindowInteractor class by adding the methods
    available to vtki.Plotter.

    """
    render_trigger = pyqtSignal()

    def __init__(self, parent=None):
        """ Initialize Qt interactor """
        assert has_pyqt, 'Requires PyQt5'
        QVTKRenderWindowInteractor.__init__(self, parent)

        # Create and start the interactive renderer
        self.ren_win = self.GetRenderWindow()
        self.ren_win.AddRenderer(self.renderer)
        self.iren = self.ren_win.GetInteractor()
        
        self.background_color = plotParams['background']

        self.iren.RemoveObservers('MouseMoveEvent')  # slows window update?
        self.iren.Initialize()
        # self.iren.Start()

        # Enter trackball camera mode
        istyle = vtk.vtkInteractorStyleTrackballCamera()
        self.SetInteractorStyle(istyle)
        self.add_axes()

        def quit_q(obj, event):
            key = self.iren.GetKeySym().lower()
            if key == 'q':
                self.iren.TerminateApp()
                self.close()

        # QVTKRenderWindowInteractor doesn't have a "q" event
        self.iren.AddObserver("KeyPressEvent", quit_q)

    def closeEvent(self, event):
        self.close()

class BackgroundPlotter(QtInteractor):

    def __init__(self, show=True, app=None):
        assert has_pyqt, 'Requires PyQt5'
        self.active = True

        # ipython magic
        if run_from_ipython():
            # breakpoint()
            from IPython import get_ipython
            ipython = get_ipython()
            ipython.magic('gui qt')

            from IPython.external.qt_for_kernel import QtGui
            QtGui.QApplication.instance()

        # run within python
        if app is None:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if not app:
                app = QApplication([''])

        self.app = app
        QtInteractor.__init__(self)
        if show:
            self.show()

        self._spawn_background_rendering()

    def _spawn_background_rendering(self, rate=5.0):
        """
        Spawns a thread that updates the render window.

        Sometimes directly modifiying object data doesn't trigger
        Modified() and upstream objects won't be updated.  This
        ensures the render window stays updated without consuming too
        many resources.
        """
        self.render_trigger.connect(self.ren_win.Render)
        twait = rate**-1

        def render():
            while self.active:
                time.sleep(twait)
                self._render()

        self.render_thread = Thread(target=render)
        self.render_thread.start()

    def closeEvent(self, event):
        self.active = False
        self.app.quit()
        self.close()

    def add_actor(self, actor, resetcam=None):
        actor, prop = super(BackgroundPlotter, self).add_actor(actor, resetcam)
        self.reset_camera()
        return actor, prop

    def __del__(self):
        self.close()
