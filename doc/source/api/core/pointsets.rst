.. jupyter-execute::
   :hide-code:

   import pyvista as pv
   pv.set_plot_theme('document')
   pv.set_jupyter_backend('static')
   pv.global_theme.window_size = [600, 400]
   pv.global_theme.axes.show = False
   pv.global_theme.anti_aliasing = 'fxaa'
   pv.global_theme.show_scalar_bar = False


.. _point_sets_api:

Point Sets
==========

Point sets are datasets with explicit geometry where the point and cell
topology are specified and not inferred.


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: PolyData
      :link: _autosummary/pyvista.PolyData
      :link-type: doc
      :class-title: pyvista-card-title

      Surface geometry: vertices, lines, polygons, and triangles.
      Extension of :vtk:`vtkPolyData`.

      .. jupyter-execute::
         :hide-code:

         from pyvista import demos
         demos.plot_datasets('PolyData')

   .. grid-item-card:: UnstructuredGrid
      :link: _autosummary/pyvista.UnstructuredGrid
      :link-type: doc
      :class-title: pyvista-card-title

      Arbitrary combinations of all cell types for volumetric and
      surface data. Extension of :vtk:`vtkUnstructuredGrid`.

      .. jupyter-execute::
         :hide-code:

         from pyvista import demos
         demos.plot_datasets('UnstructuredGrid')

   .. grid-item-card:: StructuredGrid
      :link: _autosummary/pyvista.StructuredGrid
      :link-type: doc
      :class-title: pyvista-card-title

      Regular lattice of points with connectivity defined by grid
      ordering. Extension of :vtk:`vtkStructuredGrid`.

      .. jupyter-execute::
         :hide-code:

         from pyvista import demos
         demos.plot_datasets('StructuredGrid')

   .. grid-item-card:: PointSet
      :link: _autosummary/pyvista.PointSet
      :link-type: doc
      :class-title: pyvista-card-title

      A concrete class for storing a set of points with no cell
      connectivity. Extension of :vtk:`vtkPointSet`.

      .. jupyter-execute::
         :hide-code:

         import numpy as np
         import pyvista as pv
         rng = np.random.default_rng(0)
         pv.PointSet(rng.random((10, 3))).plot(color='red')


Class Reference
---------------

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   PointSet
   PolyData
   UnstructuredGrid
   StructuredGrid
   ExplicitStructuredGrid
