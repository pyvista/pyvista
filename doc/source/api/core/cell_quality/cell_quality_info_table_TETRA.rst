
.. dropdown:: Tetra Cell Info

   Info about :attr:`~pyvista.CellType.TETRA` cell quality measures.
   See :func:`~pyvista.examples.cells.Tetrahedron` for an example unit cell.

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

       * - ``aspect_frobenius``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``aspect_gamma``
         - [1.0, 3.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``aspect_ratio``
         - [1.0, 3.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``collapse_ratio``
         - [0.1, inf]
         - [0.0, inf]
         - [0.0, inf]
         - 0.816

       * - ``condition``
         - [1.0, 3.0]
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
         - 0.707

       * - ``min_angle``
         - [40.0, 70.5]
         - [0.0, 70.5]
         - [0.0, 360.0]
         - 70.5

       * - ``radius_ratio``
         - [1.0, 3.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``relative_size_squared``
         - [0.3, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``scaled_jacobian``
         - [0.5, 1.0]
         - [-1.0, 1.0]
         - [-inf, inf]
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

       * - ``volume``
         - [0.0, inf]
         - [-inf, inf]
         - [-inf, inf]
         - 0.118
