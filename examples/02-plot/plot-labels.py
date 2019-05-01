"""
Label Points
~~~~~~~~~~~~

Use string arrays in a point set to label points
"""
from vtki import examples
import vtki
import numpy as np

###############################################################################
# Label Array
# +++++++++++

poly = vtki.PolyData(np.random.rand(10, 3))

print(poly)


###############################################################################
# Add string labels to the point data - this associates a label with every node:

poly['My Labels'] = ['Label {}'.format(i) for i in range(poly.n_points)]

###############################################################################
# Now plot the points with labels:

plotter = vtki.Plotter()
plotter.add_point_labels(poly, 'My Labels', point_size=20, font_size=36)
plotter.show()


###############################################################################
# Label Node Locations
# ++++++++++++++++++++

# Load example beam file
grid = vtki.UnstructuredGrid(examples.hexbeamfile)

# Create plotting class and add the unstructured grid
plotter = vtki.Plotter()
plotter.add_mesh(grid, show_edges=True, color='tan')

# Add labels to points on the yz plane (where x == 0)
points = grid.points
mask = points[:, 0] == 0
plotter.add_point_labels(points[mask], points[mask].tolist())

plotter.camera_position = [
                (-1.4643015810492384, 1.5603923627830638, 3.16318236536270),
                (0.05268120500967251, 0.639442034364944, 1.204095304165153),
                (0.2364061044392675, 0.9369426029156169, -0.25739213784721)]

plotter.show()
