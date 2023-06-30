API Reference
*************

.. toctree::
   :caption: API Reference
   :hidden:

   core/index
   plotting/index
   utilities/index
   readers/index
   examples/index

In this section, you can learn more about how PyVista wraps different VTK mesh
types and how you can leverage powerful 3D plotting and mesh analysis tools.
Highlights of the API include:

* Pythonic interface to VTK's Python bindings
* Filtering/plotting tools built for interactivity (see :ref:`widgets`)
* Direct access to common VTK filters (see :ref:`filters_ref`)
* Intuitive plotting routines with `matplotlib`_ similar syntax (see
  :ref:`plotting_ref`)


.. card:: Core API
   :link: core-api-index
   :link-type: ref
   :class-title: pyvista-card-title

   Learn more about PyVista's different mesh types and direct access to common
   VTK filters.

   .. jupyter-execute::

      >>> import pyvista as pv
      >>> mesh = pv.Sphere()
      >>> sliced = mesh.slice()
      >>> sliced.length


.. card:: Plotting API
   :link: plotting-api-index
   :link-type: ref
   :class-title: pyvista-card-title

   Explore PyVista's robust plotting interface for visualizing the core data
   structures.

   .. jupyter-execute::

      >>> import pyvista as pv
      >>> mesh = pv.Cube()
      >>> pl = pv.Plotter()
      >>> actor = pl.add_mesh(mesh, scalars=mesh.points)
      >>> actor.prop

.. card:: Readers
   :link: reader_api
   :link-type: ref
   :class-title: pyvista-card-title

   Use PyVista's Reader classes to read data files using
   :func:`pyvista.get_reader`.

   .. jupyter-execute::

      >>> import pyvista as pv
      >>> from pyvista import examples
      >>> reader = pv.get_reader(examples.hexbeamfile)
      >>> reader


.. card:: Utilities
   :link: utilities-api-index
   :link-type: ref
   :class-title: pyvista-card-title

   Utilize PyVista's helper modules, conversion tools, and geometric object
   creation routines.

   .. jupyter-execute::

      >>> import pyvista as pv
      >>> mesh = pv.ParametricSuperEllipsoid(xradius=0.1)
      >>> mesh


.. _matplotlib: https://matplotlib.org/
