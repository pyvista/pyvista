
.. dropdown:: Triangle Cell Info

   Info about :attr:`~pyvista.CellType.TRIANGLE` cell quality measures.
   See :func:`~pyvista.examples.cells.Triangle` for an example unit cell.

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
         - [0.0, inf]
         - 0.433

       * - ``aspect_ratio``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``aspect_frobenius``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``condition``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``distortion``
         - [0.5, 1.0]
         - [0.0, 1.0]
         - [-inf, inf]
         - 1.0

       * - ``max_angle``
         - [60.0, 90.0]
         - [60.0, 180.0]
         - [0.0, 180.0]
         - 60.0

       * - ``min_angle``
         - [30.0, 60.0]
         - [0.0, 60.0]
         - [0.0, 360.0]
         - 60.0

       * - ``scaled_jacobian``
         - [0.5, 1.15]
         - [-1.15, 1.15]
         - [-inf, inf]
         - 1.0

       * - ``radius_ratio``
         - [1.0, 3.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``shape``
         - [0.25, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``shape_and_size``
         - [0.25, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0
