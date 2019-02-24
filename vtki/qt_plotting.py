from threading import Thread
import time
import logging
import numpy as np
import os

import vtk
import vtk.qt

from vtki.plotting import BasePlotter, rcParams, run_from_ipython

# for display bugs due to older intel integrated GPUs
vtk.qt.QVTKRWIBase = 'QGLWidget'

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


# dummy reference for when PyQt5 is not installed (i.e. readthedocs)
has_pyqt = False
class QVTKRenderWindowInteractor(object):
    pass


class QDialog(object):
    pass


class QSlider(object):
    pass


def pyqtSignal(*args, **kwargs):
    pass

try:
    from PyQt5.QtCore import pyqtSignal
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PyQt5 import QtGui
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import (QVBoxLayout, QFrame, QMainWindow, QSlider,
                                 QDialog, QFormLayout, QFileDialog)
    has_pyqt = True
except:
    pass


class DoubleSlider(QSlider):
    """
    Double precision slider from:
    https://gist.github.com/dennis-tra/994a65d6165a328d4eabaadbaedac2cc
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 20.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value


class ScaleAxesDialog(QDialog):
    """ Dialog to control axes scaling """
    accepted = pyqtSignal(float)
    signal_close = pyqtSignal()

    def __init__(self, parent, plotter, show=True):
        super(ScaleAxesDialog, self).__init__(parent)
        self.setGeometry(300, 300, 50, 50)
        self.setMinimumWidth(500)
        self.signal_close.connect(self.close)
        self.plotter = plotter

        # setup sliders
        def make_slider():
            """Makes a double slider"""
            slider = DoubleSlider(QtCore.Qt.Horizontal)
            slider.setTickInterval(0.1)
            slider.setMinimum(0)
            slider.setMaximum(20)
            slider.setValue(1)
            slider.valueChanged.connect(self.update_scale)
            return slider

        self.x_slider = make_slider()
        self.y_slider = make_slider()
        self.z_slider = make_slider()

        form_layout = QFormLayout(self)
        form_layout.addRow('X Scale', self.x_slider)
        form_layout.addRow('Y Scale', self.y_slider)
        form_layout.addRow('Z Scale', self.z_slider)

        self.setLayout(form_layout)

        if show:
            self.show()

    def update_scale(self, value):
        """ updates the scale of all actors in the plotter """
        self.plotter.set_scale(self.x_slider.value(),
                               self.y_slider.value(),
                               self.z_slider.value())


def resample_image(arr, max_size=400):
    """Resamples a square image to an image of max_size"""
    dim = np.max(arr.shape[0:2])
    if dim < max_size:
        max_size = dim
    x, y, _ = arr.shape
    sx = int(np.ceil(x / max_size))
    sy = int(np.ceil(y / max_size))
    img = np.zeros((max_size, max_size, 3), dtype=arr.dtype)
    arr = arr[0:-1:sx, 0:-1:sy, :]
    xl = (max_size - arr.shape[0]) // 2
    yl = (max_size - arr.shape[1]) // 2
    img[xl:arr.shape[0]+xl, yl:arr.shape[1]+yl, :] = arr
    return img


def pad_image(arr, max_size=400):
    """Pads an image to a square then resamples to max_size"""
    dim = np.max(arr.shape)
    img = np.zeros((dim, dim, 3), dtype=arr.dtype)
    xl = (dim - arr.shape[0]) // 2
    yl = (dim - arr.shape[1]) // 2
    img[xl:arr.shape[0]+xl, yl:arr.shape[1]+yl, :] = arr
    return resample_image(img, max_size=max_size)


class QtInteractor(QVTKRenderWindowInteractor, BasePlotter):
    """
    Extends QVTKRenderWindowInteractor class by adding the methods
    available to vtki.Plotter.

    Parameters
    ----------
    parent :

    title : string, optional
        Title of plotting window.

    """
    render_trigger = pyqtSignal()
    allow_quit_keypress = True
    signal_close = pyqtSignal()

    def __init__(self, parent=None, title=None):
        """ Initialize Qt interactor """
        assert has_pyqt, 'Requires PyQt5'
        QVTKRenderWindowInteractor.__init__(self, parent)
        self.parent = parent

        # Create and start the interactive renderer
        self.ren_win = self.GetRenderWindow()
        self.ren_win.AddRenderer(self.renderer)
        self.iren = self.ren_win.GetInteractor()

        self.background_color = rcParams['background']

        if title:
            self.setWindowTitle(title)

        self.iren.RemoveObservers('MouseMoveEvent')  # slows window update?
        self.iren.Initialize()

        # Enter trackball camera mode
        istyle = vtk.vtkInteractorStyleTrackballCamera()
        self.SetInteractorStyle(istyle)
        self.add_axes()

        # QVTKRenderWindowInteractor doesn't have a "q" quit event
        self.iren.AddObserver("KeyPressEvent", self.quit)

    def quit(self, obj=None, event=None):
        try:
            key = self.iren.GetKeySym().lower()

            if key == 'q' and self.allow_quit_keypress:
                self.iren.TerminateApp()
                self.close()
                self.signal_close.emit()

        except:
            pass


class BackgroundPlotter(QtInteractor):

    ICON_TIME_STEP = 5.0

    def __init__(self, show=True, app=None, **kwargs):
        assert has_pyqt, 'Requires PyQt5'
        self.active = True
        self.saved_camera_positions = []

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
        self.app_window = QMainWindow()

        self.frame = QFrame()
        self.frame.setFrameStyle(QFrame.NoFrame)

        QtInteractor.__init__(self, parent=self.frame, **kwargs)
        self.signal_close.connect(self.app_window.close)

        # build main menu
        main_menu = self.app_window.menuBar()

        fileMenu = main_menu.addMenu('File')
        fileMenu.addAction('Exit', self.quit)
        fileMenu.addAction('Screenshot', self._qt_screenshot)
        fileMenu.addAction('Export as VTKjs', self._qt_export_vtkjs)

        view_menu = main_menu.addMenu('View')
        view_menu.addAction('Scale Axes', self.scale_axes_dialog)
        view_menu.addAction('Clear All', self.clear)

        cam_menu = view_menu.addMenu('Camera')
        cam_menu.addAction('Reset Camera', self.reset_camera)
        cam_menu.addAction('Isometric View', self.isometric_view)
        cam_menu.addSeparator()
        cam_menu.addAction('Save Current Camera Position', self.save_camera_position)
        cam_menu.addAction('Clear Saved Positions', self.clear_camera_positions)

        view_menu.addSeparator()
        # Orientation marker
        orien_menu = view_menu.addMenu('Orientation Marker')
        orien_menu.addAction('Show', self.show_axes)
        orien_menu.addAction('Hide', self.hide_axes)
        # Bounds axes
        axes_menu = view_menu.addMenu('Bounds Axes')
        axes_menu.addAction('Add Bounds Axes (front)', self.add_bounds_axes)
        axes_menu.addAction('Add Bounds Grid (back)', self.show_grid)
        axes_menu.addAction('Add Bounding Box', self.add_bounding_box)
        axes_menu.addSeparator()
        axes_menu.addAction('Remove Bounding Box', self.remove_bounding_box)
        axes_menu.addAction('Remove Bounds', self.remove_bounds_axes)

        # A final separator to seperate OS options
        view_menu.addSeparator()

        self.saved_camera_menu = main_menu.addMenu('Camera Positions')

        vlayout = QVBoxLayout()
        vlayout.addWidget(self)

        self.frame.setLayout(vlayout)
        self.app_window.setCentralWidget(self.frame)

        if show:
            self.app_window.show()
            self.show()

        self._last_update_time = time.time() - BackgroundPlotter.ICON_TIME_STEP / 2
        self._last_window_size = self.window_size
        self._last_camera_pos = self.camera_position

        self._spawn_background_rendering()

    def scale_axes_dialog(self):
        ScaleAxesDialog(self.app_window, self)

    def clear_camera_positions(self):
        """ clears all camera positions """
        for action in self.saved_camera_menu.actions():
            self.saved_camera_menu.removeAction(action)

    def save_camera_position(self):
        """ Saves camera position to saved camera menu for recall """
        self.saved_camera_positions.append(self.camera_position)
        ncam = len(self.saved_camera_positions)
        camera_position = self.camera_position[:]  # py2.7 copy compatibility

        def load_camera_position():
            self.camera_position = camera_position

        self.saved_camera_menu.addAction('Camera Position %2d' % ncam,
                                         load_camera_position)

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

    def add_actor(self, actor, reset_camera=None, name=None):
        actor, prop = super(BackgroundPlotter, self).add_actor(actor, reset_camera, name)
        if reset_camera:
            self.reset_camera()
        self.update_app_icon()
        return actor, prop

    def update_app_icon(self):
        """
        Update the app icon if the user is not trying to resize the window.
        """
        if os.name == 'nt':
            # DO NOT EVEN ATTEMPT TO UPDATE ICON ON WINDOWS
            return
        cur_time = time.time()
        if self._last_window_size != self.window_size:
            # Window size hasn't remained constant since last render.
            # This means the user is resizing it so ignore update.
            pass
        elif ((cur_time - self._last_update_time > BackgroundPlotter.ICON_TIME_STEP)
                and self._last_camera_pos != self.camera_position):
            # its been a while since last update OR
            #   the camera position has changed and its been at leat one second

            # Update app icon as preview of the window
            img = pad_image(self.image)
            qimage = QtGui.QImage(img.copy(), img.shape[1],
                                  img.shape[0], QtGui.QImage.Format_RGB888)
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(qimage))

            self.app.setWindowIcon(icon)

            # Update trackers
            self._last_update_time = cur_time
            self._last_camera_pos = self.camera_position
        # Update trackers
        self._last_window_size = self.window_size

    def _qt_screenshot(self):
        filename = QFileDialog.getSaveFileName(self.app_window,
                        caption='Save Screenshot...', directory=os.getcwd(), filter='*.png')
        filename = filename[0]
        if not os.path.isdir(os.path.dirname(filename)):
            return
        return self.screenshot(filename, return_img=False)

    def _qt_export_vtkjs(self):
        filename = QFileDialog.getSaveFileName(self.app_window,
                        caption='Save VTKjs...', directory=os.getcwd())
        filename = filename[0]
        if not os.path.isdir(os.path.dirname(filename)):
            return
        return self.export_vtkjs(filename)

    def _render(self):
        super(BackgroundPlotter, self)._render()
        self.update_app_icon()
        return

    def __del__(self):
        self.close()
