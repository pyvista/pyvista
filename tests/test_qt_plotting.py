import os

import numpy as np
import pytest

import pyvista
from pyvista import QtInteractor, MainWindow
from pyvista.plotting import system_supports_plotting


NO_PLOTTING = not system_supports_plotting()


try:
    from PyQt5.Qt import (QMainWindow, QFrame, QVBoxLayout, QAction)
    has_pyqt5 = True
except:
    has_pyqt5 = False
    class QMainWindow(object):
        pass


class TstWindow(MainWindow):
    def __init__(self, parent=None, show=True):
        MainWindow.__init__(self, parent)

        self.frame = QFrame()
        vlayout = QVBoxLayout()
        self.vtk_widget = QtInteractor(self.frame)
        vlayout.addWidget(self.vtk_widget.interactor)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu('File')
        self.exit_action = QAction('Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.triggered.connect(self.close)
        fileMenu.addAction(self.exit_action)

        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QAction('Add Sphere', self)
        self.exit_action.setShortcut('Ctrl+A')
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        self.signal_close.connect(self.vtk_widget.interactor.close)

        if show:
            self.show()

    def add_sphere(self):
        sphere = pyvista.Sphere()
        self.vtk_widget.add_mesh(sphere)
        self.vtk_widget.reset_camera()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_qt_interactor(qtbot):
    pytest.skip()
    window = TstWindow(show=False)
    qtbot.addWidget(window)
    window.add_sphere()
    assert np.any(window.vtk_widget.mesh.points)
    window.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_axes_scale(qtbot):
    sphere = pyvista.Sphere()
    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(sphere)
    assert np.any(plotter.mesh.points)

    dlg = plotter.scale_axes_dialog(show=False)

    value = 2.0
    dlg.x_slider_group.value = value
    assert plotter.scale[0] == value

    dlg.x_slider_group.spinbox.setValue(-1)
    assert dlg.x_slider_group.value == 0
    dlg.x_slider_group.spinbox.setValue(1000.0)
    assert dlg.x_slider_group.value < 100

    plotter._last_update_time = 0.0
    plotter.update_app_icon()
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_camera(qtbot):
    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())

    cpos = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera_position = cpos
    plotter.save_camera_position()
    plotter.camera_position = [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    # load existing position
    # NOTE: 2 because first two (0 and 1) buttons save and clear positions
    plotter.saved_cameras_tool_bar.actions()[2].trigger()
    assert plotter.camera_position == cpos

    plotter.clear_camera_positions()
    # 2 because the first two buttons are save and clear
    assert len(plotter.saved_cameras_tool_bar.actions()) == 2
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotter_export_files(qtbot, tmpdir):
    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())

    filename = str(tmpdir.mkdir("tmpdir").join('tmp.png'))
    dlg = plotter._qt_screenshot(show=False)
    dlg.selectFile(filename)
    dlg.accept()
    plotter.close()

    assert os.path.isfile(filename)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotter_export_vtkjs(qtbot, tmpdir):
    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())

    filename = str(tmpdir.mkdir("tmpdir").join('tmp'))
    dlg = plotter._qt_export_vtkjs(show=False)
    dlg.selectFile(filename)
    dlg.accept()
    plotter.close()

    assert os.path.isfile(filename + '.vtkjs')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_orbit(qtbot):
    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())
    # perform the orbit:
    plotter.orbit_on_path(bkg=False, step=0.0)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_add_callback(qtbot):
    class CallBack(object):
        def __init__(self, sphere):
            self.sphere = sphere

        def __call__(self):
            self.sphere.points *= 0.5

    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    sphere = pyvista.Sphere()
    mycallback = CallBack(sphere)
    plotter.add_mesh(sphere)
    plotter.add_callback(mycallback, interval=1000, count=3)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
@pytest.mark.parametrize('close_event', [
    "plotter_close",
    "window_close",
    "q_key_press",
    # "menu_exit"
    ])
def test_background_plotting_close(qtbot, close_event):
    from pyvista.plotting.plotting import close_all, _ALL_PLOTTERS
    close_all()  # this is necessary to test _ALL_PLOTTERS
    assert len(_ALL_PLOTTERS) == 0

    sphere = pyvista.Sphere()
    plotter = pyvista.BackgroundPlotter(show=False, off_screen=False)
    plotter.add_mesh(sphere)
    plotter.enable_cell_picking()

    # check that BackgroundPlotter.__init__() is called
    assert hasattr(plotter, "app_window")
    # check that BasePlotter.__init__() is called
    assert hasattr(plotter, "_style")
    # check that QtInteractor.__init__() is called
    assert hasattr(plotter, "iren")
    assert hasattr(plotter, "render_timer")
    # check that QVTKRenderWindowInteractorAdapter._init__() is called
    assert hasattr(plotter, "interactor")

    window = plotter.app_window  # MainWindow
    interactor = plotter.interactor  # QVTKRenderWindowInteractor
    render_timer = plotter.render_timer  # QTimer

    # ensure that self.render is called by the timer
    render_blocker = qtbot.wait_signals([render_timer.timeout], timeout=500)
    render_blocker.wait()

    # ensure that the widgets are showed
    with qtbot.wait_exposed(window, timeout=500):
        window.show()
    with qtbot.wait_exposed(interactor, timeout=500):
        interactor.show()

    # check that the widgets are showed properly
    assert render_timer.isActive()
    assert window.isVisible()
    assert interactor.isVisible()

    with qtbot.wait_signals([window.signal_close_test], timeout=500):
        if close_event == "plotter_close":
            plotter.close()
        elif close_event == "window_close":
            window.close()
        elif close_event == "q_key_press":
            qtbot.keyClick(interactor, "q")

    # check that the widgets are closed
    assert not window.isVisible()
    assert not interactor.isVisible()
    assert not render_timer.isActive()

    # check that BasePlotter.close() is called
    assert not hasattr(plotter, "_style")
    assert not hasattr(plotter, "iren")

    # check that BasePlotter.__init__() is called only once
    assert len(_ALL_PLOTTERS) == 1
