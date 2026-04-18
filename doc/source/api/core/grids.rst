.. jupyter-execute::
   :hide-code:

   # jupyterlab boiler plate setup
   import pyvista as pv
   pv.set_plot_theme('document')
   pv.set_jupyter_backend('static')
   pv.global_theme.window_size = [600, 400]
   pv.global_theme.axes.show = False
   pv.global_theme.anti_aliasing = 'fxaa'
   pv.global_theme.show_scalar_bar = False


Gridded Data
============

Gridded datasets in PyVista are datasets with topologically regular
point spacing. These are less flexible than :ref:`point_sets_api`,
but are much more memory efficient as they can be described with a
handful of parameters rather than having to explicitly describe the
points and geometry of the dataset.

PyVista gridded data is composed of the
:class:`pyvista.RectilinearGrid` and :class:`pyvista.ImageData`
classes. These classes inherit from the :vtk:`vtkRectilinearGrid` and
:vtk:`vtkImageData` classes and are commonly used to model images or
volumetric data.

A :class:`pyvista.RectilinearGrid` is used for modeling datasets with
variable spacing in the three coordinate directions.

.. jupyter-execute::
   :hide-code:

   from pyvista import demos
   demos.plot_datasets('RectilinearGrid')


A :class:`pyvista.ImageData` is used for modeling datasets with
uniform spacing in the three coordinate directions.

.. jupyter-execute::
   :hide-code:

   from pyvista import demos
   demos.plot_datasets('ImageData')


**Class Descriptions**

The following table describes PyVista's grid set classes. These
classes inherit all methods from their corresponding VTK
:vtk:`vtkRectilinearGrid` and :vtk:`vtkImageData` superclasses.

.. autosummary::
   :toctree: _autosummary

   pyvista.RectilinearGrid
   pyvista.ImageData
