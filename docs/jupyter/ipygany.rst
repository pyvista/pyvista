ipygany Viewer
==============


Example
-------

.. jupyter-execute::

   import pyvista as pv
   from pyvista import examples

   pv.set_jupyter_backend('ipygany')

   # download an example and reduce the mesh density
   mesh = examples.download_carburator()
   mesh.decimate(0.5, inplace=True)

   # plot it on a white background with a lightgrey mesh color
   mesh.plot(background='w', color='lightgrey')


.. jupyter-execute::

    import pyvista as pv

    sphere = pv.Sphere()
    pl = pv.Plotter(window_size=(600, 600))
    pl.add_mesh(sphere, color='grey')
    pl.background_color = 'white'
    pl.show(jupyter_backend='ipygany')


.. jupyter-execute::

    import pyvista as pv

    sphere = pv.Sphere()
    sphere['Z-Points'] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=(600, 600))
    pl.add_mesh(sphere, scalars='Z-Points')
    pl.background_color = 'white'
    pl.show(jupyter_backend='ipygany')
