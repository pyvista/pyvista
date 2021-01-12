"""
.. _ref_create_explicit_structured_grid:

Creating an Explicit Structured Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an explicit structured grid from NumPy arrays:

"""
import numpy as np
import pyvista as pv

"""
Algorithm for corners ordering:

>>> for ? in {x, y, z} do {
>>>     for k from 1 until nk do {
>>>         for j from 1 until nj do {
>>>             for i from 1 until ni do {
>>>                 write (0,0,0) and (1,0,0) ?-coordinate values
>>>                 for block (i,j,k)
>>>             }
>>>             for i from 1 until ni do {
>>>                 write (1,0,0) and (1,1,0) ?-coordinate values
>>>                 for block (i,j,k)
>>>             }
>>>         }
>>>         for j from 1 until nj do {
>>>             for i from 1 until ni do {
>>>                 write (0,0,1) and (0,1,1) ?-coordinate values
>>>                 for block (i,j,k)
>>>             }
>>>             for i from 1 until ni do {
>>>                 write (1,0,1) and (1,1,1) ?-coordinate values
>>>                 for block (i,j,k)
>>>             }
>>>         }
>>>     }
>>> }

where (ni, nj, nk) is the number of grid cells in the I, J and K directions
respectively.

"""
corners = '''
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400
0 100 100 200 200 300 300 400

0 0 0 0 0 0 0 0
200 200 200 200 200 200 200 200
200 200 200 200 200 200 200 200
400 400 400 400 400 400 400 400
0 0 0 0 0 0 0 0
200 200 200 200 200 200 200 200
200 200 200 200 200 200 200 200
400 400 400 400 400 400 400 400

2000 2001 2001 2002 2002 2003 2003 2004
2000 2001 2001 2002 2002 2003 2003 2004
2000 2001 2001 2002 2002 2003 2003 2004
2000 2001 2001 2002 2002 2003 2003 2004
2010 2011 2011 2012 2012 2013 2013 2014
2010 2011 2011 2012 2012 2013 2013 2014
2010 2011 2011 2012 2012 2013 2013 2014
2010 2011 2011 2012 2012 2013 2013 2014
'''
corners = corners.split()

points = np.asarray(corners, dtype=np.int)
points = points.reshape((-1, 3), order='F')

dims = np.asarray((5, 3, 2))
grid = pv.ExplicitStructuredGrid(dims, points)
grid.hide_cells((0, 7))
grid.plot(color='w', show_edges=True)
