from qtpy.QtWidgets import QApplication

import pyvista as pv

pl = pv.Plotter()
pl.iren.initialize()
pl.app = QApplication.instance()
pl.add_mesh(pv.Cone())
pl.camera_position = 'xy'

pl = pv.Plotter()
pl.add_mesh(pv.Cone())
