from threading import Thread
import time
import logging
import numpy as np

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



def resample_image(arr, max_size=400):
    """Resamples a square image to an image that will fit inside max size"""
    dim = np.max(arr.shape[0:2])
    if dim < max_size:
        max_size = dim
    x, y, _ = arr.shape
    sx = int(np.ceil(x / max_size))
    sy = int(np.ceil(y / max_size))
    return arr[0:-1:sx, 0:-1:sy, :]


def pad_image(arr, max_size=400):
    """Pads an image to a square then resamples to max_size"""
    dim = np.max(arr.shape)
    img = np.zeros((dim,dim,3), dtype=arr.dtype)
    img[0:arr.shape[0], 0:arr.shape[1], :] = arr
    return resample_image(img, max_size=max_size)


class QtInteractor(QVTKRenderWindowInteractor, BasePlotter):
    """
    Extends QVTKRenderWindowInteractor class by adding the methods
    available to vtki.Plotter.

    Parameters
    ----------
    title : string, optional
        Title of plotting window.

    """
    render_trigger = pyqtSignal()
    allow_quit_keypress = True
    def __init__(self, parent=None, title=None):
        """ Initialize Qt interactor """
        assert has_pyqt, 'Requires PyQt5'
        QVTKRenderWindowInteractor.__init__(self, parent)

        # Create and start the interactive renderer
        self.ren_win = self.GetRenderWindow()
        self.ren_win.AddRenderer(self.renderer)
        self.iren = self.ren_win.GetInteractor()

        self.background_color = plotParams['background']

        if title:
            self.setWindowTitle(title)

        self.iren.RemoveObservers('MouseMoveEvent')  # slows window update?
        self.iren.Initialize()
        # self.iren.Start()

        # Enter trackball camera mode
        istyle = vtk.vtkInteractorStyleTrackballCamera()
        self.SetInteractorStyle(istyle)
        self.add_axes()

        def quit_q(obj, event):
            key = self.iren.GetKeySym().lower()
            if key == 'q' and self.allow_quit_keypress:
                self.iren.TerminateApp()
                self.close()

        # QVTKRenderWindowInteractor doesn't have a "q" event
        self.iren.AddObserver("KeyPressEvent", quit_q)

    def closeEvent(self, event):
        self.close()


class BackgroundPlotter(QtInteractor):

    def __init__(self, show=True, app=None, **kwargs):
        assert has_pyqt, 'Requires PyQt5'
        self.active = True
        self._last_update_time = time.time()

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
        QtInteractor.__init__(self, **kwargs)
        if show:
            self.show()

        self._spawn_background_rendering()

        self._last_window_size = self.window_size

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
        if resetcam:
            self.reset_camera()
        return actor, prop

    def _render(self):
        super(BackgroundPlotter, self)._render()
        # Now update the app icon if its been at least one second and the user
        #   is not trying to resize the window
        cur_time = time.time()
        if (cur_time - self._last_update_time > 1.0) and (self._last_window_size == self.window_size):
            from PyQt5 import QtGui
            # Update app icon as preview of the window
            img = pad_image(self.image)
            qimage = QtGui.QImage(img.copy(), img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(qimage))
            self.app.setWindowIcon(icon)
            # Update trackers
            self._last_update_time = cur_time
        # Update trackers
        self._last_window_size = self.window_size
        return

    def __del__(self):
        self.close()
