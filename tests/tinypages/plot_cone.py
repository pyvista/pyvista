from qtpy.QtWidgets import QApplication

import pyvista as pv

pv.Cone().plot()

pl = pv.Plotter()
pl.iren.initialize()
pl.app = QApplication.instance()
pl.add_mesh(pv.Cone())
pl.camera_position = 'xy'

pl2 = pv.Plotter()
pl2.add_mesh(pv.Cone())
