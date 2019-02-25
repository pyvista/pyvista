import sys

import pytest
import numpy as np

import vtki
from vtki import QtInteractor
from vtki.plotting import running_xserver


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


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_qt_interactor(qtbot):
    window = MainWindow(show=False)
    qtbot.addWidget(window)
    window.add_sphere()
    assert np.any(window.vtk_widget.mesh.points)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting(qtbot):
    sphere = vtki.Sphere()
    plotter = vtki.BackgroundPlotter(show=False, title='Testing Window')
    plotter.add_mesh(sphere)
    assert np.any(plotter.mesh.points)
    # now test some of the features
    plotter.save_camera_position()
    plotter.clear_camera_positions()
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

    assert plotter.close()
