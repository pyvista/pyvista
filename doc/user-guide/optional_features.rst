Optional Features
=================
Due to its usage of `numpy`, the `pyvista` module plays well with
other modules, including `matplotlib`, `trimesh`, `rtree`, and
`pyembree`.  The following examples show some optional features
included within `pyvista` that use or combine several modules to
perform advanced analyses not normally included within `VTK`.

Vectorised Ray Tracing
~~~~~~~~~~~~~~~~~~~~~~
Perform many ray traces simultaneously with a PolyData Object
(requires optional dependencies trimesh, rtree and pyembree)

.. code-block:: python

    from math import sin, cos, radians
    import pyvista as pv

    # Create source to ray trace
    sphere = pv.Sphere(radius=0.85)

    # Define a list of origin points and a list of direction vectors for each ray
    vectors = [ [cos(radians(x)), sin(radians(x)), 0] for x in range(0, 360, 5)]
    origins = [[0, 0, 0]] * len(vectors)

    # Perform ray trace
    points, ind_ray, ind_tri = sphere.multi_ray_trace(origins, vectors)

    # Create geometry to represent ray trace
    rays = [pv.Line(o, v) for o, v in zip(origins, vectors)]
    intersections = pv.PolyData(points)

    # Render the result
    p = pv.Plotter()
    p.add_mesh(sphere,
               show_edges=True, opacity=0.5, color="w",
               lighting=False, label="Test Mesh")
    p.add_mesh(rays[0], color="blue", line_width=5, label="Ray Segments")
    for ray in rays[1:]:
        p.add_mesh(ray, color="blue", line_width=5)
    p.add_mesh(intersections, color="maroon",
               point_size=25, label="Intersection Points")
    p.add_legend()
    p.show()


.. image:: ../images/user-generated/ray_trace.png
