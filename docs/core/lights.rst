Lights
======
The :class:`pyvista.Light` class adds additional functionality and a pythonic API
to the ``vtk.vtkLight`` class. :class:`pyvista.Plotter` objects come with a default
set of lights that work well in most cases, but in many situations a more hands-on
access to lighting is necessary.


Brief Example
-------------

Ceate a red spotlight that shines on the origin.

.. # TODO: code or testcode directives?
.. testcode:: python

    import pyvista as pv
    light = pv.Light(position=(-1, 1, 1), color='red')
    light.positional = True

Create a scene without lighting and add our light to it manually.

.. testcode:: python

    from pyvista import examples
    plotter = pv.Plotter(lighting='none')
    mesh = examples.download_bunny()
    mesh.rotate_x(90)
    mesh.rotate_z(180)
    plotter.add_mesh(mesh, specular=1.0, diffuse=0.7, smooth_shading=True)
    plotter.add_light(light)
    plotter.show(screenshot='shiny_bunny.png')

.. image:: ../images/auto-generated/shiny_bunny.png

Light API
---------
Description of the ``pyvista.Light`` class.

.. autoclass:: pyvista.Light
   :members:
