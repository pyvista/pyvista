.. _qt_ref:

vtki PyQt Interface
===================
``vtki`` has an interface for placing plots in ``PyQt5`` that extends the functionality of the ``QVTKRenderWindowInteractor`` class.  The ``vtki.QtInteractor`` class allows you to have the same functionality of the ``Plotter`` class within a ``PyQt5`` application.  This simplifies adding meshes, updating, and controlling them when using ``PyQt5``.


Example PyQt5 vtki QtInteractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following example shows how to create a simple application that adds a sphere to an empty plotting window.

.. code:: python

    import sys

    from PyQt5 import Qt
    import numpy as np

    import vtki


    class MainWindow(Qt.QMainWindow):

        def __init__(self, parent=None, show=True):
            Qt.QMainWindow.__init__(self, parent)

	    # create the frame
            self.frame = Qt.QFrame()
            vlayout = Qt.QVBoxLayout()

	    # add the vtki interactor object
            self.vtk_widget = vtki.QtInteractor(self.frame)
            vlayout.addWidget(self.vtk_widget)

            self.frame.setLayout(vlayout)
            self.setCentralWidget(self.frame)

	    # simple menu to demo functions
            mainMenu = self.menuBar()
            fileMenu = mainMenu.addMenu('File')
            exitButton = Qt.QAction('Exit', self)
            exitButton.setShortcut('Ctrl+Q')
            exitButton.triggered.connect(self.close)
            fileMenu.addAction(exitButton)

	    # allow adding a sphere
            meshMenu = mainMenu.addMenu('Mesh')
            self.add_sphere_action = Qt.QAction('Add Sphere', self)
            self.add_sphere_action.triggered.connect(self.add_sphere)
            meshMenu.addAction(self.add_sphere_action)

            if show:
                self.show()

        def add_sphere(self):
	    """ add a sphere to the pyqt frame """
            sphere = vtki.Sphere()
            self.vtk_widget.add_mesh(sphere)
            self.vtk_widget.reset_camera()


    if __name__ == '__main__':
        app = Qt.QApplication(sys.argv)
        window = MainWindow()
        sys.exit(app.exec_())


.. figure:: ../images/user-generated/qt_plotting_sphere.png
    :width: 600pt

    PyQt5 vtki QtInteractor


Background Plotting
~~~~~~~~~~~~~~~~~~~
Normal ``vtki`` plotting windows exhibit blocking behavior, but it is possible to plot in the background and update the plotter in real-time using the ``BackgroundPlotter`` object.  This requires ``PyQt5``, but otherwise appears and functions like a normal ``vtki`` ``Plotter`` instance.  For example:

.. code:: python

    import vtki

    sphere = vtki.Sphere()

    plotter = vtki.BackgroundPlotter()
    plotter.add_mesh(sphere)

    # can now operate on the sphere and have it updated in the background
    sphere.points *= 0.5



.. rubric:: Attributes

.. autoautosummary:: vtki.BackgroundPlotter
   :attributes:

.. rubric:: Methods

.. autoautosummary:: vtki.BackgroundPlotter
   :methods:

.. autoclass:: vtki.BackgroundPlotter
   :members:
   :undoc-members:
   :show-inheritance:



.. autoclass:: vtki.ScaledPlotter
  :members:
  :undoc-members:
  :show-inheritance:
