
.. dropdown:: Quad Cell Info

   Info about :attr:`~pyvista.CellType.QUAD` cell quality measures.
   See :func:`~pyvista.examples.cells.Quadrilateral` for an example unit cell.

   .. list-table::
       :widths: 20 20 20 20 20
       :header-rows: 1

       * - Measure
         - Acceptable
           Range
         - Normal
           Range
         - Full
           Range
         - Unit Cell
           Value

       * - ``area``
         - [0.0, inf]
         - [0.0, inf]
         - [-inf, inf]
         - 1.0

       * - ``aspect_ratio``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``condition``
         - [1.0, 4.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``distortion``
         - [0.5, 1.0]
         - [0.0, 1.0]
         - [-inf, inf]
         - 1.0

       * - ``jacobian``
         - [0.0, inf]
         - [0.0, inf]
         - [-inf, inf]
         - 1.0

       * - ``max_aspect_frobenius``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``max_angle``
         - [90.0, 135.0]
         - [90.0, 360.0]
         - [0.0, 360.0]
         - 90.0

       * - ``max_edge_ratio``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``med_aspect_frobenius``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``min_angle``
         - [45.0, 90.0]
         - [0.0, 90.0]
         - [0.0, 360.0]
         - 90.0

       * - ``oddy``
         - [0.0, 0.5]
         - [0.0, inf]
         - [0.0, inf]
         - 0.0

       * - ``radius_ratio``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``relative_size_squared``
         - [0.3, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``scaled_jacobian``
         - [0.3, 1.0]
         - [-1.0, 1.0]
         - [-1.0, 1.0]
         - 1.0

       * - ``shape``
         - [0.3, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``shape_and_size``
         - [0.2, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``shear``
         - [0.3, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``shear_and_size``
         - [0.2, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``skew``
         - [0.0, 0.7]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 0.0

       * - ``stretch``
         - [0.25, 1.0]
         - [0.0, 1.0]
         - [0.0, inf]
         - 1.0

       * - ``taper``
         - [0.0, 0.7]
         - [0.0, inf]
         - [0.0, inf]
         - 0.0

       * - ``warpage``
         - [0.5, 1.0]
         - [0.0, 2.0]
         - [0.0, inf]
         - 1.0
