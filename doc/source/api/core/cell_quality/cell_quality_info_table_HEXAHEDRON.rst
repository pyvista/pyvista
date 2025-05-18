
.. dropdown:: Hexahedron Cell Info

   Info about :attr:`~pyvista.CellType.HEXAHEDRON` cell quality measures.
   See :func:`~pyvista.examples.cells.Hexahedron` for an example unit cell.

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

       * - ``diagonal``
         - [0.65, 1.0]
         - [0.0, 1.0]
         - [0.0, inf]
         - 1.0

       * - ``dimension``
         - [0.0, inf]
         - [0.0, inf]
         - [0.0, inf]
         - 0.577

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

       * - ``max_edge_ratio``
         - [1.0, 1.3]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``max_aspect_frobenius``
         - [1.0, 3.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``med_aspect_frobenius``
         - [1.0, 3.0]
         - [1.0, inf]
         - [1.0, inf]
         - 1.0

       * - ``oddy``
         - [0.0, 0.5]
         - [0.0, inf]
         - [0.0, inf]
         - 0.0

       * - ``relative_size_squared``
         - [0.5, 1.0]
         - [0.0, 1.0]
         - [0.0, 1.0]
         - 1.0

       * - ``scaled_jacobian``
         - [0.5, 1.0]
         - [-1.0, 1.0]
         - [-1.0, inf]
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
         - [0.0, 0.5]
         - [0.0, 1.0]
         - [0.0, inf]
         - 0.0

       * - ``stretch``
         - [0.25, 1.0]
         - [0.0, 1.0]
         - [0.0, inf]
         - 1.0

       * - ``taper``
         - [0.0, 0.5]
         - [0.0, inf]
         - [0.0, inf]
         - 0.0

       * - ``volume``
         - [0.0, inf]
         - [0.0, inf]
         - [-inf, inf]
         - 1.0
