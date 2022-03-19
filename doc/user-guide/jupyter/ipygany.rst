.. _ipygany_ref:

Using ``ipygany`` with PyVista
------------------------------
.. warning::
   Currently, this backend has inferior support and features than the
   ``pythreejs``.  If you would like accurate recreations of VTK
   scenes in three.js, please see :ref:`pythreejs_ref`.

The `ipygany <https://github.com/QuantStack/ipygany>`_ jupyterlab
plotting backend is a powerful module that enables pure plotting that
leverages `threejs <https://threejs.org/>`_ through the `pythreejs
widget <https://github.com/jupyter-widgets/pythreejs>`_.  It allows
for embedded html documentation (as shown here), rapid plotting (as
compared to the other client jupyterlab plotting modules like
``panel`` or ``itkwidgets``).

There is an excellent block post at `ipygany: Jupyter into the third dimension <https://blog.jupyter.org/ipygany-jupyter-into-the-third-dimension-29a97597fc33>`_
and without repeating too much here, ``ipygany`` includes the
following features:

- IsoColor: apply color-mapping to your mesh.
- Warp: deform your mesh given a 3-D input data (e.g. displacement
  data on a beam)
- WarpByScalar: deform your mesh given a 1-D input data (e.g. terrain
  elevation)
- Threshold: only visualize mesh parts inside a range of data
  (e.g. 222 K≤ temperature ≤ 240 K)
- IsoSurface: only visualize the surface where the mesh respects a
  data value (e.g. pressure == 3 bar)
- Glyph effects like PointCloud
- Water visualization


PyVista Wrapping
~~~~~~~~~~~~~~~~
There are two approaches for plotting using ipygany with ``pyvista``.
First, you can convert between pyvista meshes ``ipygany`` PolyMesh
objects using the ``from_pyvista`` method from ``ipygany`` to enable a
variety of advanced ``ipygany`` methods and follow their examples
outlined in the `ipygany Documentation
<https://ipygany.readthedocs.io/en/latest/>`_, or you can simply use
an existing ``Plotter`` class and set ``jupyter_backend='ipygany'``.

Perhaps best of all, the resulting widgets can be embedded within
sphinx documentation:

.. jupyter-execute::

    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_bunny()
    
    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightgrey')
    pl.background_color = 'white'
    pl.camera_position = 'xy'
    pl.show(jupyter_backend='ipygany')

Note how the mesh color, background color, and camera position are all
mapped over to the ``ipygany`` scene, meaning that you can reuse
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
    >>> scene = pl.show(jupyter_backend='ipygany', return_viewer=True)
    >>> type(scene)
    ipygany.ipygany.Scene

This scene can then be added to any number of jupyterlab widgets and
then shown as a complete widget.  For example, you could even display
two side by side using ``ipywidgets.AppLayout``.


Examples: Large Models
~~~~~~~~~~~~~~~~~~~~~~
This example shows a large mesh and demonstrates how even fairly large
meshes, like the carburetor example which contains 500,000 faces and
250,000 points, can be quickly loaded.  This is, of course, bandwidth
dependent, as this mesh is around 6 MB.

.. jupyter-execute::

   import pyvista as pv
   from pyvista import examples

   pv.set_jupyter_backend('ipygany')

   # download an example and reduce the mesh density
   mesh = examples.download_carburator()
   mesh.decimate(0.5, inplace=True)

   # plot it on a white background with a lightgrey mesh color
   mesh.plot(background='w', color='lightgrey')


Returning Scenes
~~~~~~~~~~~~~~~~
Show several widgets simultaneously using
``ipywidgets.TwoByTwoLayout``.  This is similar to the
:ref:`ref_parametric_example`, except with interactive widgets.

.. jupyter-execute::

    from ipywidgets import TwoByTwoLayout

    import pyvista as pv


    # consistent view options for all plotters
    plot_kwargs = {'color': 'tan', 'jupyter_backend': 'ipygany',
                   'return_viewer': True, 'background': 'white'}

    supertoroid = pv.ParametricSuperToroid(n1=0.5)
    scene_0 = supertoroid.plot(**plot_kwargs)

    ellipsoid = pv.ParametricEllipsoid(10, 5, 5)
    scene_1 = ellipsoid.plot(**plot_kwargs)

    pseudosphere = pv.ParametricPseudosphere()
    scene_2 = pseudosphere.plot(**plot_kwargs)

    conicspiral = pv.ParametricConicSpiral()
    scene_3 = conicspiral.plot(**plot_kwargs)

    TwoByTwoLayout(top_left=scene_0,
                   top_right=scene_1,
                   bottom_left=scene_2,
                   bottom_right=scene_3)


Scalar Bars
~~~~~~~~~~~
Scalar bars are automatically shown when a plot has active scalars.
For example, the St. Helens ``mesh`` from ``active_scalar_name`` is
``'Elevation'``.  Scalar bars, scalar bar title, and the colormap
dropdown menu are automatically added to the scene.

.. jupyter-execute::

    # Load St Helens DEM and warp the topography
    mesh = examples.download_st_helens().warp_by_scalar()

    pl = pv.Plotter()
    pl.background_color = 'white'
    pl.add_mesh(mesh)
    pl.show()
