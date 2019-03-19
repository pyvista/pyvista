import os
import sys
import time

import numpy as np
import pytest

import vtki
from vtki import QtInteractor
from vtki.plotting import running_xserver


NO_PLOTTING = not running_xserver()
try:
    if os.environ['ALLOW_PLOTTING'] == 'True':
        NO_PLOTTING = False
except KeyError:
    pass



# dummy class to allow module init
class QMainWindow(object):
    pass


try:
    import PyQt5
    from PyQt5.Qt import (QMainWindow, QFrame, QVBoxLayout, QAction)
    has_pyqt5 = True
except:
    has_pyqt5 = False
    class QMainWindow(object):
        pass


class MainWindow(QMainWindow):

    def __init__(self, parent=None, show=True):
        QMainWindow.__init__(self, parent)

        self.frame = QFrame()
        vlayout = QVBoxLayout()
        self.vtk_widget = QtInteractor(self.frame)
        vlayout.addWidget(self.vtk_widget)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        exitButton = QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QAction('Add Sphere', self)

        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        if show:
            self.show()

    def add_sphere(self):
        sphere = vtki.Sphere()
        self.vtk_widget.add_mesh(sphere)
        self.vtk_widget.reset_camera()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_qt_interactor(qtbot):
    window = MainWindow(show=False)
    qtbot.addWidget(window)
    window.add_sphere()
    assert np.any(window.vtk_widget.mesh.points)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_axes_scale(qtbot):
    sphere = vtki.Sphere()
    plotter = vtki.BackgroundPlotter(show=False, title='Testing Window')
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

    assert plotter.quit() is None


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_camera(qtbot):
    plotter = vtki.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(vtki.Sphere())

    cpos = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera_position = cpos
    plotter.save_camera_position()
    plotter.camera_position = [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    # load existing position
    plotter.saved_camera_menu.actions()[0].trigger()
    assert plotter.camera_position == cpos

    plotter.clear_camera_positions()
    assert not len(plotter.saved_camera_menu.actions())
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotter_export_files(qtbot, tmpdir):
    plotter = vtki.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(vtki.Sphere())

    filename = str(tmpdir.mkdir("tmpdir").join('tmp.png'))
    dlg = plotter._qt_screenshot(show=False)
    dlg.selectFile(filename)
    dlg.accept()
    plotter.close()

    assert os.path.isfile(filename)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotter_export_vtkjs(qtbot, tmpdir):
    plotter = vtki.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(vtki.Sphere())

    filename = str(tmpdir.mkdir("tmpdir").join('tmp'))
    dlg = plotter._qt_export_vtkjs(show=False)
    dlg.selectFile(filename)
    dlg.accept()
    plotter.close()

    assert os.path.isfile(filename + '.vtkjs')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_orbit(qtbot):
    plotter = vtki.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(vtki.Sphere())
    # perfrom the orbit:
    plotter.orbit_on_path(bkg=False, step=0.0)
    plotter.close()
