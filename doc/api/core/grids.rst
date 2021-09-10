.. jupyter-execute::
   :hide-code:

   # jupyterlab boiler plate setup
   import pyvista
   pyvista.set_plot_theme('document')
   pyvista.set_jupyter_backend('pythreejs')
   pyvista.global_theme.window_size = [600, 400]
   pyvista.global_theme.axes.show = False
   pyvista.global_theme.antialiasing = True
   pyvista.global_theme.show_scalar_bar = False


Gridded Data
============

Gridded datasets in PyVista are datasets with topologically regular
point spacing.  These are less flexible than :ref:`point_sets_api`,
but are much more memory efficient as they can be described with a
handful of parameters rather than having to explicitly describe the
points and geometry of the dataset.

PyVista gridded data is composed of the
:class:`pyvista.RectilinearGrid` and :class:`pyvista.UniformGrid`
classes.  These classes inherit from the `vtkRectilinearGrid`_ and
`vtkImageData`_ classes and are commonly used to model images or
volumetric data.

A :class:`pyvista.RectilinearGrid` is used for modeling datasets with
variable spacing in the three coordinate directions.

.. jupyter-execute::
   :hide-code:

   from pyvista import demos
   demos.plot_datasets('RectilinearGrid')


A :class:`pyvista.UniformGrid` is used for modeling datasets with
uniform spacing in the three coordinate directions.

.. jupyter-execute::
   :hide-code:

   from pyvista import demos
   demos.plot_datasets('UniformGrid')


**Class Descriptions**

The following table describes PyVista's grid set classes.  These
classes inherit all methods from their corresponding VTK
`vtkRectilinearGrid`_ and `vtkImageData`_ superclasses.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   pyvista.RectilinearGrid
   pyvista.UniformGrid

.. _vtkRectilinearGrid: https://www.vtk.org/doc/nightly/html/classvtkRectilinearGrid.html
.. _vtkImageData: https://www.vtk.org/doc/nightly/html/classvtkImageData.html
