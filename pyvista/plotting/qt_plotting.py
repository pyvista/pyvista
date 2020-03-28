"""Qt interactive plotter."""
import logging
import os
import platform
import time
import warnings
from functools import wraps

import numpy as np
import vtk

import pyvista
from pyvista.utilities import conditional_decorator, threaded
import scooby

from .plotting import BasePlotter
from .theme import rcParams

# for display bugs due to older intel integrated GPUs
vtk_major_version = vtk.vtkVersion.GetVTKMajorVersion()
vtk_minor_version = vtk.vtkVersion.GetVTKMinorVersion()
if vtk_major_version == 8 and vtk_minor_version < 2:
    import vtk.qt
    vtk.qt.QVTKRWIBase = 'QGLWidget'
else:
    import vtkmodules.qt
    vtkmodules.qt.QVTKRWIBase = 'QGLWidget'

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


SAVE_CAM_BUTTON_TEXT = 'Save Camera'
CLEAR_CAMS_BUTTON_TEXT = 'Clear Cameras'


# dummy reference for when PyQt5 is not installed (i.e. readthedocs)
class QDialog(object):
    """Dummy QFileDialog class."""

    pass


class QSlider(object):
    """Dummy QSlider class."""

    pass


def pyqtSignal(*args, **kwargs):  # pragma: no cover
    """Declare dummy pyqtSignal function."""
    pass


class QHBoxLayout(object):
    """Dummy QHBoxLayout class."""

    pass


class QFileDialog(object):
    """Dummy QFileDialog class."""

    pass


def pyqtSlot(*args, **kwargs):
    """Declare dummy function for environments without pyqt5."""
    return lambda *x: None


class QMainWindow(object):
    """Dummy QMainWindow class."""

    pass


class QObject(object):
    """Dummy QObject class."""

    pass


has_pyqt = False
try:
    from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QTimer
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PyQt5 import QtGui
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import (QMenuBar, QVBoxLayout, QHBoxLayout, QDoubleSpinBox,
                                 QFrame, QMainWindow, QSlider, QAction,
                                 QHBoxLayout, QDialog,
                                 QFormLayout, QFileDialog)
    has_pyqt = True
except ImportError:  # pragma: no cover
    pass


class FileDialog(QFileDialog):
    """Generic file query.

    It emits a signal when a file is selected and
    the dialog was property closed.
    """

    dlg_accepted = pyqtSignal(str)

    def __init__(self, parent=None, filefilter=None, save_mode=True, show=True,
                 callback=None, directory=False):
        """Initialize the file dialog."""
        super(FileDialog, self).__init__(parent)

        if filefilter is not None:
            self.setNameFilters(filefilter)

        self.setOption(QFileDialog.DontUseNativeDialog)
        self.accepted.connect(self.emit_accepted)

        if directory:
            self.FileMode(QFileDialog.DirectoryOnly)
            self.setOption(QFileDialog.ShowDirsOnly, True)

        if save_mode:
            self.setAcceptMode(QFileDialog.AcceptSave)

        if callback is not None:
            self.dlg_accepted.connect(callback)

        if show:  # pragma: no cover
            self.show()

    def emit_accepted(self):
        """Send signal that the file dialog was closed properly.

        Sends:
        filename

        """
        if self.result():
            filename = self.selectedFiles()[0]
            if os.path.isdir(os.path.dirname(filename)):
                self.dlg_accepted.emit(filename)


class DoubleSlider(QSlider):
    """Double precision slider.

    Reference:
    https://gist.github.com/dennis-tra/994a65d6165a328d4eabaadbaedac2cc

    """

    def __init__(self, *args, **kwargs):
        """Initialize the double slider."""
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 20.0

    @property
    def _value_range(self):
        """Return the value range of the slider."""
        return self._max_value - self._min_value

    def value(self):
        """Return the value of the slider."""
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        """Set the value of the slider."""
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        """Set the minimum value of the slider."""
        if value > self._max_value:  # pragma: no cover
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        """Set the maximum value of the slider."""
        if value < self._min_value:  # pragma: no cover
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())


# this is redefined from above because the above object is a dummy object
# we use dummy objects to allow the module to import when PyQt5 isn't installed
class RangeGroup(QHBoxLayout):
    """Range group box widget."""

    def __init__(self, parent, callback, minimum=0.0, maximum=20.0,
                 value=1.0):
        """Initialize the range widget."""
        super(RangeGroup, self).__init__(parent)
        self.slider = DoubleSlider(QtCore.Qt.Horizontal)
        self.slider.setTickInterval(0.1)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(value)

        self.minimum = minimum
        self.maximum = maximum

        self.spinbox = QDoubleSpinBox(value=value, minimum=minimum,
                                      maximum=maximum, decimals=4)

        self.addWidget(self.slider)
        self.addWidget(self.spinbox)

        # Connect slider to spinbox
        self.slider.valueChanged.connect(self.update_spinbox)
        self.spinbox.valueChanged.connect(self.update_value)
        self.spinbox.valueChanged.connect(callback)

    def update_spinbox(self, value):
        """Set the value of the internal spinbox."""
        self.spinbox.setValue(self.slider.value())

    def update_value(self, value):
        """Update the value of the internal slider."""
        # if self.spinbox.value() < self.minimum:
        #     self.spinbox.setValue(self.minimum)
        # elif self.spinbox.value() > self.maximum:
        #     self.spinbox.setValue(self.maximum)

        self.slider.blockSignals(True)
        self.slider.setValue(self.spinbox.value())
        self.slider.blockSignals(False)

    @property
    def value(self):
        """Return the value of the internal spinbox."""
        return self.spinbox.value()

    @value.setter
    def value(self, new_value):
        """Set the value of the internal slider."""
        self.slider.setValue(new_value)


class ScaleAxesDialog(QDialog):
    """Dialog to control axes scaling."""

    accepted = pyqtSignal(float)
    signal_close = pyqtSignal()

    def __init__(self, parent, plotter, show=True):
        """Initialize the scaling dialog."""
        super(ScaleAxesDialog, self).__init__(parent)
        self.setGeometry(300, 300, 50, 50)
        self.setMinimumWidth(500)
        self.signal_close.connect(self.close)
        self.plotter = plotter
        self.plotter.app_window.signal_close.connect(self.close)

        self.x_slider_group = RangeGroup(parent, self.update_scale,
                                         value=plotter.scale[0])
        self.y_slider_group = RangeGroup(parent, self.update_scale,
                                         value=plotter.scale[1])
        self.z_slider_group = RangeGroup(parent, self.update_scale,
                                         value=plotter.scale[2])

        form_layout = QFormLayout(self)
        form_layout.addRow('X Scale', self.x_slider_group)
        form_layout.addRow('Y Scale', self.y_slider_group)
        form_layout.addRow('Z Scale', self.z_slider_group)

        self.setLayout(form_layout)

        if show:  # pragma: no cover
            self.show()

    def update_scale(self, value):
        """Update the scale of all actors in the plotter."""
        self.plotter.set_scale(self.x_slider_group.value,
                               self.y_slider_group.value,
                               self.z_slider_group.value)


def resample_image(arr, max_size=400):
    """Resample a square image to an image of max_size."""
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
    """Pad an image to a square then resamples to max_size."""
    dim = np.max(arr.shape)
    img = np.zeros((dim, dim, 3), dtype=arr.dtype)
    xl = (dim - arr.shape[0]) // 2
    yl = (dim - arr.shape[1]) // 2
    img[xl:arr.shape[0]+xl, yl:arr.shape[1]+yl, :] = arr
    return resample_image(img, max_size=max_size)


class QVTKRenderWindowInteractorAdapter(QObject):
    """Adapter class for QVTKRenderWindowInteractor that uses super()."""

    def __init__(self, parent, **kwargs):
        """Initialize the internal interactor."""
        self.interactor = QVTKRenderWindowInteractor(parent=parent)
        self.interactor.dragEnterEvent = self.dragEnterEvent
        self.interactor.dropEvent = self.dropEvent
        super(QVTKRenderWindowInteractorAdapter, self).__init__(**kwargs)

    def GetRenderWindow(self):
        """Get the render window."""
        return self.interactor.GetRenderWindow()

    def setWindowTitle(self, title):
        """Set the window title."""
        return self.interactor.setWindowTitle(title)

    def SetInteractorStyle(self, style):
        """Set the interactor style."""
        return self.interactor.SetInteractorStyle(style)

    def show(self):
        """Show the window."""
        return self.interactor.show()

    def close(self):
        """Close the window."""
        return self.interactor.close()

    def setAcceptDrops(self, state):
        """Enable drop event or not."""
        self.interactor.setAcceptDrops(state)

    def dragEnterEvent(self, event):
        """Manage drag event."""
        pass

    def dropEvent(self, event):
        """Manage drop event."""
        pass


class QtInteractor(QVTKRenderWindowInteractorAdapter, BasePlotter):
    """Extend QVTKRenderWindowInteractor class.

    This adds the methods available to pyvista.Plotter.

    Parameters
    ----------
    parent :
        Qt parent.

    title : str, optional
        Title of plotting window.

    multi_samples : int, optional
        The number of multi-samples used to mitigate aliasing. 4 is a
        good default but 8 will have better results with a potential
        impact on performance.

    line_smoothing : bool, optional
        If True, enable line smothing

    point_smoothing : bool, optional
        If True, enable point smothing

    polygon_smoothing : bool, optional
        If True, enable polygon smothing

    auto_update : float, bool, optional
        Automatic update rate in seconds.  Useful for automatically
        updating the render window when actors are change without
        being automatically ``Modified``.
    """

    # Signals must be class attributes
    render_signal = pyqtSignal()
    key_press_event_signal = pyqtSignal(vtk.vtkGenericRenderWindowInteractor, str)

    def __init__(self, parent=None, title=None, off_screen=None,
                 multi_samples=None, line_smoothing=False,
                 point_smoothing=False, polygon_smoothing=False,
                 splitting_position=None, auto_update=5.0, **kwargs):
        """Initialize Qt interactor."""
        if not has_pyqt:
            raise AssertionError('Requires PyQt5')
        super(QtInteractor, self).__init__(parent=parent, **kwargs)
        self.parent = parent

        if multi_samples is None:
            multi_samples = rcParams['multi_samples']

        self.setAcceptDrops(True)

        # Create and start the interactive renderer
        self.ren_win = self.GetRenderWindow()
        self.ren_win.SetMultiSamples(multi_samples)
        if line_smoothing:
            self.ren_win.LineSmoothingOn()
        if point_smoothing:
            self.ren_win.PointSmoothingOn()
        if polygon_smoothing:
            self.ren_win.PolygonSmoothingOn()

        for renderer in self.renderers:
            self.ren_win.AddRenderer(renderer)

        self.render_signal.connect(self._render)
        self.key_press_event_signal.connect(super(QtInteractor, self).key_press_event)

        self.background_color = rcParams['background']
        if self.title:
            self.setWindowTitle(title)

        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN

        if off_screen:
            self.ren_win.SetOffScreenRendering(1)
        else:
            self.iren = self.ren_win.GetInteractor()
            self.iren.RemoveObservers('MouseMoveEvent')  # slows window update?

            # Enter trackball camera mode
            istyle = vtk.vtkInteractorStyleTrackballCamera()
            self.SetInteractorStyle(istyle)

            self.iren.Initialize()

            self._observers = {}    # Map of events to observers of self.iren
            self._add_observer("KeyPressEvent", self.key_press_event)

        # Make the render timer but only activate if using auto update
        self.render_timer = QTimer(parent=parent)
        if float(auto_update) > 0.0:  # Can be False as well
            # Spawn a thread that updates the render window.
            # Sometimes directly modifiying object data doesn't trigger
            # Modified() and upstream objects won't be updated.  This
            # ensures the render window stays updated without consuming too
            # many resources.
            twait = (auto_update**-1) * 1000.0
            self.render_timer.timeout.connect(self.render)
            self.render_timer.start(twait)

        if rcParams["depth_peeling"]["enabled"]:
            if self.enable_depth_peeling():
                for renderer in self.renderers:
                    renderer.enable_depth_peeling()

        self._first_time = False # Crucial!
        self.view_isometric()

    def key_press_event(self, obj, event):
        """Call `key_press_event` using a signal."""
        self.key_press_event_signal.emit(obj, event)

    @wraps(BasePlotter.render)
    def _render(self, *args, **kwargs):
        """Wrap ``BasePlotter.render``."""
        return BasePlotter.render(self, *args, **kwargs)

    @conditional_decorator(threaded, platform.system() == 'Darwin')
    def render(self):
        """Override the ``render`` method to handle threading issues."""
        return self.render_signal.emit()

    def dragEnterEvent(self, event):
        """Event is called when something is dropped onto the vtk window.

        Only triggers event when event contains file paths that
        exist.  User can drop anything in this window and we only want
        to allow files.
        """
        try:
            for url in event.mimeData().urls():
                if os.path.isfile(url.path()):
                    # only call accept on files
                    event.accept()
        except Exception as e:
            warnings.warn('Exception when dropping files: %s' % str(e))

    def dropEvent(self, event):
        """Event is called after dragEnterEvent."""
        for url in event.mimeData().urls():
            self.url = url
            filename = self.url.path()
            if os.path.isfile(filename):
                try:
                    self.add_mesh(pyvista.read(filename))
                except Exception as e:
                    print(str(e))
                    pass

    def add_toolbars(self, main_window):
        """Add the toolbars."""
        def _add_action(tool_bar, key, method):
            action = QAction(key, main_window)
            action.triggered.connect(method)
            tool_bar.addAction(action)
            return

        # Camera toolbar
        self.default_camera_tool_bar = main_window.addToolBar('Camera Position')
        _view_vector = lambda *args: self.view_vector(*args)
        cvec_setters = {
            # Viewing vector then view up vector
            'Top (-Z)': lambda: _view_vector((0, 0, 1),  (0, 1, 0)),
            'Bottom (+Z)': lambda: _view_vector((0, 0, -1),  (0, 1, 0)),
            'Front (-Y)': lambda: _view_vector((0, 1, 0),  (0, 0, 1)),
            'Back (+Y)': lambda: _view_vector((0, -1, 0),  (0, 0, 1)),
            'Left (-X)': lambda: _view_vector((1, 0, 0),  (0, 0, 1)),
            'Right (+X)': lambda: _view_vector((-1, 0, 0),  (0, 0, 1)),
            'Isometric': lambda: _view_vector((1, 1, 1),  (0, 0, 1))
        }
        for key, method in cvec_setters.items():
            _add_action(self.default_camera_tool_bar, key, method)
        _add_action(self.default_camera_tool_bar, 'Reset', lambda: self.reset_camera())

        # Saved camera locations toolbar
        self.saved_camera_positions = []
        self.saved_cameras_tool_bar = main_window.addToolBar('Saved Camera Positions')

        _add_action(self.saved_cameras_tool_bar, SAVE_CAM_BUTTON_TEXT, self.save_camera_position)
        _add_action(self.saved_cameras_tool_bar, CLEAR_CAMS_BUTTON_TEXT, self.clear_camera_positions)

        return

    def save_camera_position(self):
        """Save camera position to saved camera menu for recall."""
        self.saved_camera_positions.append(self.camera_position)
        ncam = len(self.saved_camera_positions)
        camera_position = self.camera_position[:]  # py2.7 copy compatibility

        if hasattr(self, "saved_cameras_tool_bar"):
            def load_camera_position():
                self.camera_position = camera_position

            self.saved_cameras_tool_bar.addAction('Cam %2d' % ncam,
                                                  load_camera_position)
            if ncam < 10:
                self.add_key_event(str(ncam), load_camera_position)
        return

    def clear_camera_positions(self):
        """Clear all camera positions."""
        if hasattr(self, "saved_cameras_tool_bar"):
            for action in self.saved_cameras_tool_bar.actions():
                if action.text() not in [SAVE_CAM_BUTTON_TEXT, CLEAR_CAMS_BUTTON_TEXT]:
                    self.saved_cameras_tool_bar.removeAction(action)
        self.saved_camera_positions = []
        return

    def close(self):
        """Quit application."""
        if self._closed:
            return
        if hasattr(self, "render_timer"):
            self.render_timer.stop()
        BasePlotter.close(self)
        QVTKRenderWindowInteractorAdapter.close(self)


class BackgroundPlotter(QtInteractor):
    """Qt interactive plotter.

    Background plotter for pyvista that allows you to maintain an
    interactive plotting window without blocking the main python
    thread.

    Parameters
    ----------
    show : bool, optional
        Show the plotting window.  If ``False``, show this window by
        running ``show()``

    app : PyQt5.QtWidgets.QApplication, optional
        Creates a `QApplication` if left as `None`.

    window_size : list, optional
        Window size in pixels.  Defaults to ``[1024, 768]``

    off_screen : bool, optional
        Renders off screen when True.  Useful for automated
        screenshots or debug testing.

    allow_quit_keypress : bool, optional
        Allow user to exit by pressing ``"q"``.

    title : str, optional
        Title of plotting window.

    multi_samples : int, optional
        The number of multi-samples used to mitigate aliasing. 4 is a
        good default but 8 will have better results with a potential
        impact on performance.

    line_smoothing : bool, optional
        If True, enable line smothing

    point_smoothing : bool, optional
        If True, enable point smothing

    polygon_smoothing : bool, optional
        If True, enable polygon smothing

    auto_update : float, bool, optional
        Automatic update rate in seconds.  Useful for automatically
        updating the render window when actors are change without
        being automatically ``Modified``.  If set to ``True``, update
        rate will be 1 second.

    Examples
    --------
    >>> import pyvista as pv
    >>> plotter = pv.BackgroundPlotter()
    >>> actor = plotter.add_mesh(pv.Sphere())
    """

    ICON_TIME_STEP = 5.0

    def __init__(self, show=True, app=None, window_size=None,
                 off_screen=None, allow_quit_keypress=True, **kwargs):
        """Initialize the qt plotter."""
        if not has_pyqt:
            raise AssertionError('Requires PyQt5')
        self.active = True
        self.counters = []
        self.allow_quit_keypress = allow_quit_keypress

        if window_size is None:
            window_size = rcParams['window_size']

        # Remove notebook argument in case user passed it
        kwargs.pop('notebook', None)

        # ipython magic
        if scooby.in_ipython():  # pragma: no cover
            from IPython import get_ipython
            ipython = get_ipython()
            ipython.magic('gui qt')

            from IPython.external.qt_for_kernel import QtGui
            QtGui.QApplication.instance()
        else:
            ipython = None

        # run within python
        if app is None:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if not app:  # pragma: no cover
                app = QApplication([''])

        self.app = app
        self.app_window = MainWindow()
        self.app_window.setWindowTitle(kwargs.get('title', rcParams['title']))

        self.frame = QFrame()
        self.frame.setFrameStyle(QFrame.NoFrame)
        super(BackgroundPlotter, self).__init__(parent=self.frame,
                                                off_screen=off_screen,
                                                **kwargs)
        self.app_window.signal_close.connect(self._close)
        self.add_toolbars(self.app_window)

        # build main menu
        self.main_menu = _create_menu_bar(parent=self.app_window)
        self.app_window.signal_close.connect(self.main_menu.clear)

        file_menu = self.main_menu.addMenu('File')
        file_menu.addAction('Take Screenshot', self._qt_screenshot)
        file_menu.addAction('Export as VTKjs', self._qt_export_vtkjs)
        file_menu.addSeparator()
        # member variable for testing only
        self._menu_close_action = file_menu.addAction('Exit', self.app_window.close)

        view_menu = self.main_menu.addMenu('View')
        view_menu.addAction('Toggle Eye Dome Lighting', self._toggle_edl)
        view_menu.addAction('Scale Axes', self.scale_axes_dialog)
        view_menu.addAction('Clear All', self.clear)

        tool_menu = self.main_menu.addMenu('Tools')
        tool_menu.addAction('Enable Cell Picking (through)', self.enable_cell_picking)
        tool_menu.addAction('Enable Cell Picking (visible)', lambda: self.enable_cell_picking(through=False))

        cam_menu = view_menu.addMenu('Camera')
        cam_menu.addAction('Toggle Parallel Projection', self._toggle_parallel_projection)

        view_menu.addSeparator()
        # Orientation marker
        orien_menu = view_menu.addMenu('Orientation Marker')
        orien_menu.addAction('Show All', self.show_axes_all)
        orien_menu.addAction('Hide All', self.hide_axes_all)
        # Bounds axes
        axes_menu = view_menu.addMenu('Bounds Axes')
        axes_menu.addAction('Add Bounds Axes (front)', self.show_bounds)
        axes_menu.addAction('Add Bounds Grid (back)', self.show_grid)
        axes_menu.addAction('Add Bounding Box', self.add_bounding_box)
        axes_menu.addSeparator()
        axes_menu.addAction('Remove Bounding Box', self.remove_bounding_box)
        axes_menu.addAction('Remove Bounds', self.remove_bounds_axes)

        # A final separator to separate OS options
        view_menu.addSeparator()

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.interactor)

        self.frame.setLayout(vlayout)
        self.app_window.setCentralWidget(self.frame)

        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN

        if show and not off_screen:  # pragma: no cover
            self.app_window.show()
            self.show()

        self.window_size = window_size
        self._last_update_time = time.time() - BackgroundPlotter.ICON_TIME_STEP / 2
        self._last_window_size = self.window_size
        self._last_camera_pos = self.camera_position

        self.add_callback(self.update_app_icon)

        # Keypress events
        self.add_key_event("S", self._qt_screenshot)  # shift + s

    def reset_key_events(self):
        """Reset all of the key press events to their defaults.

        Handles closing configuration for q-key.
        """
        super(BackgroundPlotter, self).reset_key_events()
        if self.allow_quit_keypress:
            self.add_key_event("q", lambda: self.close())

    def scale_axes_dialog(self, show=True):
        """Open scale axes dialog."""
        return ScaleAxesDialog(self.app_window, self, show=show)

    def close(self):
        """Close the plotter.

        This function closes the window which in turn will
        close the plotter through `signal_close`.

        """
        if not self._closed:
            self.app_window.close()

    def _close(self):
        super().close()

    def update_app_icon(self):
        """Update the app icon if the user is not trying to resize the window."""
        if os.name == 'nt' or not hasattr(self, '_last_window_size'):  # pragma: no cover
            # DO NOT EVEN ATTEMPT TO UPDATE ICON ON WINDOWS
            return
        cur_time = time.time()
        if self._last_window_size != self.window_size:  # pragma: no cover
            # Window size hasn't remained constant since last render.
            # This means the user is resizing it so ignore update.
            pass
        elif ((cur_time - self._last_update_time > BackgroundPlotter.ICON_TIME_STEP)
                and self._last_camera_pos != self.camera_position):
            # its been a while since last update OR
            # the camera position has changed and its been at least one second

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

    def _qt_screenshot(self, show=True):
        return FileDialog(self.app_window,
                          filefilter=['Image File (*.png)',
                                      'JPEG (*.jpeg)'],
                          show=show,
                          directory=os.getcwd(),
                          callback=self.screenshot)

    def _qt_export_vtkjs(self, show=True):
        """Spawn an save file dialog to export a vtkjs file."""
        return FileDialog(self.app_window,
                          filefilter=['VTK JS File(*.vtkjs)'],
                          show=show,
                          directory=os.getcwd(),
                          callback=self.export_vtkjs)

    def _toggle_edl(self):
        if hasattr(self.renderer, 'edl_pass'):
            return self.renderer.disable_eye_dome_lighting()
        return self.renderer.enable_eye_dome_lighting()

    def _toggle_parallel_projection(self):
        if self.camera.GetParallelProjection():
            return self.disable_parallel_projection()
        return self.enable_parallel_projection()

    @property
    def window_size(self):
        """Return render window size."""
        the_size = self.app_window.baseSize()
        return the_size.width(), the_size.height()

    @window_size.setter
    def window_size(self, window_size):
        """Set the render window size."""
        self.app_window.setBaseSize(*window_size)
        self.app_window.resize(*window_size)
        # NOTE: setting BasePlotter is unnecessary and Segfaults CI
        # BasePlotter.window_size.fset(self, window_size)

    def __del__(self):  # pragma: no cover
        """Delete the qt plotter."""
        if not self._closed:
            self.app_window.close()

    def add_callback(self, func, interval=1000, count=None):
        """Add a function that can update the scene in the background.

        Parameters
        ----------
        func : callable
            Function to be called with no arguments.
        interval : int
            Time interval between calls to `func` in milliseconds.
        count : int, optional
            Number of times `func` will be called. If None,
            `func` will be called until the main window is closed.

        """
        self._callback_timer = QTimer(parent=self.app_window)
        self._callback_timer.timeout.connect(func)
        self._callback_timer.start(interval)
        self.app_window.signal_close.connect(self._callback_timer.stop)
        if count is not None:
            counter = Counter(count)
            counter.signal_finished.connect(self._callback_timer.stop)
            self._callback_timer.timeout.connect(counter.decrease)
            self.counters.append(counter)


class MainWindow(QMainWindow):
    """Convenience MainWindow that manages the application."""

    signal_close = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the main window."""
        super(MainWindow, self).__init__(parent)

    def closeEvent(self, event):
        """Manage the close event."""
        self.signal_close.emit()
        event.accept()


class Counter(QObject):
    """Counter augmented with a Qt timer."""

    signal_finished = pyqtSignal()

    def __init__(self, count):
        """Initialize the counter."""
        super(Counter, self).__init__()
        if isinstance(count, int) and count > 0:
            self.count = count
        elif count > 0:
            raise TypeError('Expected `count` to be'
                            '`int` but got: {}'.format(type(count)))
        else:
            raise ValueError('count is not strictly positive.')

    @pyqtSlot()
    def decrease(self):
        """Decrease the count."""
        self.count -= 1
        if self.count <= 0:
            self.signal_finished.emit()


def _create_menu_bar(parent):
    """Create a menu bar.

    The menu bar is expected to behave consistently
    for every operating system since `setNativeMenuBar(False)`
    is called by default and therefore lifetime and ownership can
    be tested.
    """
    menu_bar = QMenuBar(parent=parent)
    menu_bar.setNativeMenuBar(False)
    if parent is not None:
        parent.setMenuBar(menu_bar)
    return menu_bar
