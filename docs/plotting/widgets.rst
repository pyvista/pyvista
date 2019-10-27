.. _widgets:

Widgets
-------

PyVista has several widgets that can be added to the rendering scene to control
filters like clipping, slicing, and thresholding - specifically there are
widgets to control the positions of boxes, planes, and lines or slider bars
which can all be highly customized through the use of custom callback
functions.

Here we'll take a look a the various widgets, some helper methods that leverage
those widgets to do common tasks, and demonstrate how to leverage the widgets
for user defined tasks and processing routines.

The :class:`pyvista.BasePlotter` class inherits all of the widget methods in
:class:`pyvista.WidgetHelper` so, all of the following methods
are available from any PyVista plotter.

.. rubric:: Attributes

.. autoautosummary:: pyvista.WidgetHelper
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.WidgetHelper
   :methods:


.. autoclass:: pyvista.WidgetHelper
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Box Widget
~~~~~~~~~~

The box widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_box_widget` and
:func:`pyvista.WidgetHelper.clear_box_widgets` methods respectively.
When enabling the box widget, you must provide a custom callback function
otherwise the box would appear and do nothing - the callback functions are
what allow us to leverage the widget to perfrom a task like clipping/cropping.

Considering that using a box to clip/crop a mesh is one of the most common use
cases, we have included a helper method that will allow you to add a mesh to a
scene with a box widget that controls its extent, the
:func:`pyvista.WidgetHelper.add_mesh_clip_box` method.

.. code-block:: python

    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_nefertiti()

    p = pv.Plotter(notebook=False)
    p.add_mesh_clip_box(mesh, color='white')
    p.show()


.. image:: ../images/gifs/box-clip.gif



Plane Widget
~~~~~~~~~~~~

The plane widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_plane_widget` and
:func:`pyvista.WidgetHelper.clear_plane_widgets` methods respectively.
As with all widgets, you must provide a custom callback method to utilize that
plane. Considering that planes are most commonly used for clipping and slicing
meshes, we have included two helper methods for doing those tasks!

Let's use a plane to clip a mesh:

.. code-block:: python

    import pyvista as pv
    from pyvista import examples

    vol = examples.download_brain()

    p = pv.Plotter(notebook=False)
    p.add_mesh_clip_plane(vol)
    p.show()


.. image:: ../images/gifs/plane-clip.gif


Or you could slice a mesh using the plane widget:

.. code-block:: python

    p = pv.Plotter(notebook=False)
    p.add_mesh_slice(vol)
    p.show()


.. image:: ../images/gifs/plane-slice.gif

Or you could leverage the plane widget for some custom task like glyphing a
vector field along that plane.

.. code-block:: python

    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_carotid()

    p = pv.Plotter(notebook=False)
    p.add_mesh(mesh.contour(8).extract_largest(), opacity=0.5)

    def my_plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        arrows = slc.glyph(orient='vectors', scale="scalars", factor=0.01)
        p.add_mesh(arrows, name='arrows')

    p.add_plane_widget(my_plane_func)
    p.show_grid()
    p.add_axes()
    p.show()


.. image:: ../images/gifs/plane-glyph.gif


Line Widget
~~~~~~~~~~~

The line widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_line_widget` and
:func:`pyvista.WidgetHelper.clear_line_widgets` methods respectively.
Unfortunately, PyVista does not have any helper methods to utilize this
widget, so it is necessary to pas a custom callback method.

One particularly fun example is to use the line widget to create source for
the :func:`pyvista.DataSetFilters.streamlines` filter.

.. code-block:: python

    import pyvista as pv
    from pyvista import examples
    import numpy as np

    pv.set_plot_theme('doc')

    mesh = examples.download_kitchen()
    furniture = examples.download_kitchen(split=True)

    arr = np.linalg.norm(mesh['velocity'], axis=1)
    clim = [arr.min(), arr.max()]

    p = pv.Plotter(notebook=False)
    p.add_mesh(furniture, name='furniture', color=True)
    p.add_mesh(mesh.outline(), color='black')
    p.add_axes()

    def simulate(pointa, pointb):
        streamlines = mesh.streamlines(n_points=10, max_steps=100,
                                       pointa=pointa, pointb=pointb,
                                       integration_direction='forward')
        p.add_mesh(streamlines, name='streamlines', line_width=5,
                   render_lines_as_tubes=True, clim=clim)

    p.add_line_widget(callback=simulate, use_vertices=True)
    p.show()


.. image:: ../images/gifs/line-widget-streamlines.gif



Slider Bar Widget
~~~~~~~~~~~~~~~~~

The slider widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_slider_widget` and
:func:`pyvista.WidgetHelper.clear_slider_widgets` methods respectively.
This is one of the most versatile widgets as it can control a value that can
be used for just about anything.

One helper method we've add is the
:func:`pyvista.WidgetHelper.add_mesh_threshold` method which leverages the
slider widget to control a thresholding value.


.. code-block:: python

    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_knee_full()

    p = pv.Plotter(notebook=False)
    p.add_mesh_threshold(mesh)
    p.show()


.. image:: ../images/gifs/slider-widget-threshold.gif


Or you could leverage a custom callback function that takes a single value
from the slider as its argument to do something like control the resolution
of a mesh:

.. code-block:: python

    p = pv.Plotter(notebook=False)

    def create_mesh(value):
        res = int(value)
        sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
        p.add_mesh(sphere, name='sphere', show_edges=True)
        return

    p.add_slider_widget(create_mesh, [5, 100], title='Resolution')
    p.show()


.. image:: ../images/gifs/slider-widget-resolution.gif


Sphere Widget
~~~~~~~~~~~~~

The slider widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_sphere_widget` and
:func:`pyvista.WidgetHelper.clear_sphere_widgets` methods respectively.
This is a very versatile widgets as it can control vertex location that can
be used to control or update the location of just about anything.

We don't have any convenient helper methods that utilize this widget out of
the box, but we have added a lot of ways to use this widget so that you can
easily add several widgets to a scene.

Let's look at a few use cases that all update a surface mesh.

Example A
+++++++++

Use a single sphere widget

.. code-block:: python

    import pyvista as pv
    import numpy as np

    # Create a triangle surface
    surf = pv.PolyData()
    surf.points = np.array([[-10,-10,-10],
                        [10,10,-10],
                        [-10,10,0],])
    surf.faces = np.array([3, 0, 1, 2])

    p = pv.Plotter(notebook=False)

    def callback(point):
        surf.points[0] = point

    p.add_sphere_widget(callback)
    p.add_mesh(surf, color=True)

    p.show_grid()
    p.show()



.. image:: ../images/gifs/sphere-widget-a.gif



Example B
+++++++++

Use several sphere widgets at once

.. code-block:: python

    import pyvista as pv
    import numpy as np

    # Create a triangle surface
    surf = pv.PolyData()
    surf.points = np.array([[-10,-10,-10],
                            [10,10,-10],
                            [-10,10,0],])
    surf.faces = np.array([3, 0, 1, 2])


    p = pv.Plotter(notebook=False)

    def callback(point, i):
        surf.points[i] = point

    p.add_sphere_widget(callback, center=surf.points)
    p.add_mesh(surf, color=True)

    p.show_grid()
    p.show()



.. image:: ../images/gifs/sphere-widget-b.gif


Example C
+++++++++

This one is the coolest - use three sphere widgets to update perturbations on
a surface and interpolate between them with some boundary conditions

.. code-block:: python

    from scipy.interpolate import griddata
    import numpy as np
    import pyvista as pv

    def get_colors(n):
        """A helper function to get n colors"""
        from itertools import cycle
        import matplotlib
        cycler = matplotlib.rcParams['axes.prop_cycle']
        colors = cycle(cycler)
        colors = [next(colors)['color'] for i in range(n)]
        return colors

    # Create a grid to interpolate to
    xmin, xmax, ymin, ymax = 0, 100, 0, 100
    x = np.linspace(xmin, xmax, num=25)
    y = np.linspace(ymin, ymax, num=25)
    xx, yy, zz = np.meshgrid(x, y, [0])

    # Make sure boundary conditions exist
    boundaries = np.array([[xmin,ymin,0],
                       [xmin,ymax,0],
                       [xmax,ymin,0],
                       [xmax,ymax,0]])

    # Create the PyVista mesh to hold this grid
    surf = pv.StructuredGrid(xx, yy, zz)

    # Create some intial perturbations
    # - this array will be updated inplace
    points = np.array([[33,25,45],
                   [70,80,13],
                   [51,57,10],
                   [25,69,20]])

    # Create an interpolation function to update that surface mesh
    def update_surface(point, i):
        points[i] = point
        tp = np.vstack((points, boundaries))
        zz = griddata(tp[:,0:2], tp[:,2], (xx[:,:,0], yy[:,:,0]), method='cubic')
        surf.points[:,-1] = zz.ravel(order='F')
        return

    # Get a list of unique colors for each widget
    colors = get_colors(len(points))

    # Begin the plotting routine
    p = pv.Plotter(notebook=False)

    # Add the surface to the scene
    p.add_mesh(surf, color=True)

    # Add the widgets which will update the surface
    p.add_sphere_widget(update_surface, center=points,
                           color=colors, radius=3)
    # Add axes grid
    p.show_grid()

    # Show it!
    p.show()


.. image:: ../images/gifs/sphere-widget-c.gif


Spline Widget
~~~~~~~~~~~~~


A spline widget can be added to the scenee by the
:func:`pyvista.WidgetHelper.add_spline_widget` and
:func:`pyvista.WidgetHelper.clear_spline_widgets` methods respectively.
This widget allows users to interactively create a poly line (spline) through
a scene and use that spline.

A common task with splines is to slice a volumetric dataset using an irregular
path. To do this, we have added a convenient helper method which leverages the
:func:`pyvista.DataSetFilters.slice_along_line` filter named
:func:`pyvosta.WidgetHelper.add_mesh_slice_spline`.
