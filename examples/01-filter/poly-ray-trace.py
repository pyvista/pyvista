"""
Ray Tracing
~~~~~~~~~~~

Single line segment ray tracing for PolyData objects.
"""

import vtki

# Create source to ray trace
sphere = vtki.Sphere(radius=0.85)

# Define line segment
start = [0, 0, 0]
stop = [0.25, 1, 0.5]

# Perfrom ray trace
points, ind = sphere.ray_trace(start, stop)

# Create geometry to represent ray trace
ray = vtki.Line(start, stop)
intersection = vtki.PolyData(points)

# Render the result
p = vtki.Plotter()
p.add_mesh(sphere, show_edges=True, opacity=0.5, color='w', lighting=False, label='Test Mesh')
p.add_mesh(ray, color='blue', line_width=5, label='Ray Segment')
p.add_mesh(intersection, color='maroon', point_size=10, label='Intersection Points')
p.add_legend()
p.show()
