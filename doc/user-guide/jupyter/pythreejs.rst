.. _pythreejs_ref:

Using ``pythreejs`` with PyVista
--------------------------------
The `pythreejs <https://github.com/jupyter-widgets/pythreejs>`_ jupyterlab
plotting backend is a powerful module that enables pure plotting that
leverages `threejs <https://threejs.org/>`_.  It allows
for embedded html documentation (as shown here).

The biggest advantage to using the ``pythreejs`` backend is it
accurately recreates the VTK scene into a ``threejs`` scene including:

* Mesh edges
* Lighting
* Physically based rendering
* Face and point scalars

.. note::
   This backend has better support and features than the ``ipygany``
   backend, but is still missing support for vtk widgets and textures.


PyVista Wrapping
~~~~~~~~~~~~~~~~
Plotting scenes from pyvista are automatically serialized to a
three.js scene when using the ``pythreejs`` backend.  This can be
enabled globally with :func:`pyvista.set_jupyter_backend` or by
setting it in :func:`pyvista.Plotter.plot`.  

.. jupyter-execute::
   :hide-code:

   import pyvista
   pyvista.global_theme.background = 'white'
   pyvista.global_theme.window_size = [600, 600]
   pyvista.global_theme.antialiasing = True
   pyvista.global_theme.jupyter_backend = 'pythreejs'

.. jupyter-execute::

    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_bunny()
    mesh.flip_normals()
    
    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightgrey')
    pl.camera_position = 'xy'
    pl.show(jupyter_backend='pythreejs')


Note how the mesh color, background color, and camera position are all
mapped over to the ``three.js`` scene, meaning that you can reuse
existing code and change the backend depending on the type of plotting
backend you wish to use.

Note that there are many missing features, including all vtk widgets,
but many of these can be replaced with jupyterlab widgets.  If you
wish to assemble your own scene, change the jupyter_backend while
returning the "viewer" with:

.. code:: python

    >>> pl = pv.Plotter()
    >>> pl.add_mesh(mesh, color='lightgrey')
    >>> pl.background_color = 'white'
    >>> pl.camera_position = 'xy'
    >>> renderer = pl.show(jupyter_backend='pythreejs', return_viewer=True)
    >>> type(scene)
    pythreejs.core.Renderer.Renderer


This renderer can then be added to any number of jupyterlab widgets and
then shown as a complete widget.  For example, you could even display
two side by side using ``ipywidgets.AppLayout``.


Plotting Representation and Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The pyvista plotting scenes are faithfully serialized the same plotting
scene within three.js using the same lighting, camera projection, and
materials.

.. jupyter-execute::

   # set the global theme to use pythreejs
   pyvista.global_theme.jupyter_backend = 'pythreejs'

   pl = pyvista.Plotter()

   # lower left, using physically based rendering
   pl.add_mesh(pyvista.Sphere(center=(-1, 0, -1)),
               show_edges=False, pbr=True, color='white', roughness=0.2,
               metallic=0.5)

   # upper right, matches default pyvista plotting
   pl.add_mesh(pyvista.Sphere(center=(1, 0, 1)))

   # Upper left, mesh displayed as points
   pl.add_mesh(pyvista.Sphere(center=(-1, 0, 1)),
               color='k', style='points', point_size=10)

   # mesh in lower right with flat shading
   pl.add_mesh(pyvista.Sphere(center=(1, 0, -1)), lighting=False,
               show_edges=True)

   # show mesh in the center with a red wireframe
   pl.add_mesh(pyvista.Sphere(), lighting=True, show_edges=False,
               color='red', line_width=0.5, style='wireframe',
               opacity=0.99)

   pl.camera_position = 'xz'
   pl.show()


Scalars Support
~~~~~~~~~~~~~~~
The ``pythreejs`` backend supports plotting scalars for faces and
points for point, wireframe, and surface representations.

.. jupyter-execute::

   import numpy as np

   def make_cube(center=(0, 0, 0), resolution=1):
       cube = pyvista.Cube(center=center)
       return cube.clean().triangulate().subdivide(resolution)

   pl = pyvista.Plotter()

   # test face scalars with no lighting
   mesh = make_cube(center=(-1, 0, -1))
   pl.add_mesh(mesh, lighting=False, cmap='jet',
               scalars=np.arange(mesh.n_faces),
               show_edges=True)

   # test point scalars on a surface mesh
   mesh = make_cube(center=(1, 0, 1))
   pl.add_mesh(mesh,
               lighting=True,
               scalars=mesh.points[:, 2]*mesh.points[:, 0], cmap='bwr',
               line_width=1)

   mesh = make_cube(center=(-1, 0, 1))
   pl.add_mesh(mesh, style='points',
               scalars=mesh.points[:, 2],
               point_size=30)

   # test wireframe
   mesh = make_cube(center=(1, 0, -1))
   pl.add_mesh(mesh, lighting=True, show_edges=False,
               scalars=mesh.points[:, 2],
               line_width=3, style='wireframe',
               cmap='inferno')

   pl.camera_position = 'xz'
   pl.show()



Point Cloud Example
~~~~~~~~~~~~~~~~~~~
Plot a sample point cloud with pyvista using the ``pythreejs`` backend
while giving the scalars random values.

.. jupyter-execute::

   pc = pyvista.PolyData(np.random.random((100, 3)))
   pc['scalars'] = np.random.random(100)
   pc.plot(jupyter_backend='pythreejs', style='points', point_size=10, cmap='jet')


Textures
~~~~~~~~
The ``pythreejs`` backend also supports :attr:`textures <pyvista.DataSet.textures>`.

.. jupyter-execute::

   import pyvista
   globe = examples.load_globe()
   globe.plot(jupyter_backend='pythreejs', smooth_shading=True)

See the :ref:`ref_texture_example` example for more details regarding textures.


RGB and RGBA Coloring
~~~~~~~~~~~~~~~~~~~~~
The ``pythreejs`` supports rgba plotting.  See the ``rgba`` parameter
within :func:`add_mesh() <pyvista.Plotting.add_mesh>`.

.. jupyter-execute::

   import numpy as np
   import pyvista

   mesh = pyvista.Sphere()

   # treat the points as RGB coordinates to make a colorful mesh
   pts = mesh.points.copy()
   pts -= pts.min()
   rgba_sphere = (255*pts).astype(np.uint8)

   # plot the corners for fun
   corners = mesh.outline_corners()
   pts = corners.points.copy()
   pts -= pts.min()
   pts = 255*(pts/pts.max())
   corners['rgba_values'] = pts.astype(np.uint8)
   edges = corners.tube(radius=0.01).triangulate()

   pl = pyvista.Plotter(window_size=(600, 600))
   pl.add_mesh(mesh, scalars=rgba_sphere, rgba=True, smooth_shading=True)
   pl.add_mesh(edges, rgba=True)
   pl.show(jupyter_backend='pythreejs')


Large Models and Physically Based Rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example shows a large mesh and demonstrates how even fairly large
meshes, like the carburetor example which contains 500,000 faces and
250,000 points, can be quickly loaded.  This is, of course, bandwidth
dependent, as this mesh is around 6 MB.

Note that here we enable physically based rendering using ``pbr=True``.

.. jupyter-execute::

   import pyvista as pv
   from pyvista import examples

   pv.set_jupyter_backend('pythreejs')

   # download an example and reduce the mesh density
   mesh = examples.download_carburator()
   mesh.decimate(0.5, inplace=True)

   # Plot it on a white background with a lightgrey mesh color.  Enable
   # physically based rendering and give the mesh a metallic look.
   mesh.plot(window_size=(600, 600), background='w', color='lightgrey',
             pbr=True, metallic=0.5)


Create Interactive Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the documentation on this page was generated using a combination
of ``pythreejs``, ``pyvista`` and ``jupyter_sphinx``.

For example, in a sample ``*.rst`` file, add the following::

  .. jupyter-execute::

     import pyvista as pv
     from pyvista import examples
     pv.set_jupyter_backend('pythreejs')
     mesh = pv.Cube()
     mesh.plot(show_edges=True)

To generate:

.. jupyter-execute::
   :hide-code:

   import pyvista as pv
   from pyvista import examples
   pv.set_jupyter_backend('pythreejs')
   
   mesh = pv.Cube()
   mesh.plot(show_edges=True)

You can also use the ``:hide-code:`` option to hide the code and only
display the plot.

You should also consider changing the global theme when plotting to improve the look of your plots:

.. code:: python

   import pyvista
   pyvista.global_theme.background = 'white'
   pyvista.global_theme.window_size = [600, 600]
   pyvista.global_theme.antialiasing = True

You will need the following packages:

* ``pyvista``
* ``pythreejs``
* ``jupyter_sphinx``

In your ``conf.py``, add the following:

.. code:: python

   extensions = [
       "jupyter_sphinx",
       # all your other extensions
   ]

