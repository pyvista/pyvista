.. jupyter-execute::
   :hide-code:

   import pyvista as pv
   pv.set_plot_theme('document')
   pv.set_jupyter_backend('static')
   pv.global_theme.window_size = [600, 400]
   pv.global_theme.axes.show = False
   pv.global_theme.anti_aliasing = 'fxaa'
   pv.global_theme.show_scalar_bar = False


.. _grids_api:

Gridded Data
============

Gridded datasets have topologically regular point spacing. They are more
memory efficient than :ref:`point sets <point_sets_api>` because their
geometry can be described with a few parameters rather than explicit point
arrays.


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: RectilinearGrid
      :link: _autosummary/pyvista.RectilinearGrid
      :link-type: doc
      :class-title: pyvista-card-title

      Variable spacing along the three coordinate directions.
      Extension of :vtk:`vtkRectilinearGrid`.

      .. jupyter-execute::
         :hide-code:

         from pyvista import demos
         demos.plot_datasets('RectilinearGrid')

   .. grid-item-card:: ImageData
      :link: _autosummary/pyvista.ImageData
      :link-type: doc
      :class-title: pyvista-card-title

      Uniform spacing along the three coordinate directions.
      Commonly used for images and volumetric data.
      Extension of :vtk:`vtkImageData`.

      .. jupyter-execute::
         :hide-code:

         from pyvista import demos
         demos.plot_datasets('ImageData')


Class Reference
---------------

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   RectilinearGrid
   ImageData
