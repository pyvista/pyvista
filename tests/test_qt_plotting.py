import sys

import pytest
from PyQt5 import Qt
import numpy as np

import vtki
from vtki import QtInteractor
from vtki.plotting import running_xserver

try:
    import PyQt5
    has_pyqt5 = True
except:
    has_pyqt5 = False


class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)

        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()
        self.vtk_widget = QtInteractor(self.frame)
        vlayout.addWidget(self.vtk_widget)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = Qt.QAction('Add Sphere', self)

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
    


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
