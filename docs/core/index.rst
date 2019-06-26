Core API
========

The `Visualization Toolkit`_ (VTK), developed by Kitware_, has many mesh data
types that PyVista wraps.
This chapter is intended to describe these different mesh types and how we wrap
and implement each of those mesh types in VTK. This chapter also highlights
how all PyVista types have direct access to common VTK filters
(see :ref:`filters_ref`).

.. _Visualization Toolkit: https://vtk.org
.. _Kitware: https://www.kitware.com

.. toctree::
   :maxdepth: 2
   :hidden:

   common
   points
   point-grids
   grids
   composite
   filters
