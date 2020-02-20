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
    plotter.update()
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
    plotter.update()
    dlg = plotter._qt_screenshot(show=False)
    dlg.selectFile(filename)
    dlg.accept()
    plotter.close()

    assert os.path.isfile(filename)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotter_export_files_again(qtbot, tmpdir):
    plotter = pyvista.BackgroundPlotter(off_screen=False, show=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())

    filename = str(tmpdir.mkdir("tmpdir").join('tmp.png'))
    # plotter.update()
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
    plotter = pyvista.BackgroundPlotter(show=False, title='Testing Window')
    sphere = pyvista.Sphere()
    plotter.add_mesh(sphere)

    def mycallback():
        sphere.points *= 0.5
    plotter.add_callback(mycallback, interval=1000, count=3)
    plotter.close()
