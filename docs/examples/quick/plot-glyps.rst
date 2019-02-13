Plotting Glyphs
===============

Using vectors in a dataset to plot and orient glyphs can be done via the
:func:`vtki.DataSetFilters.glyph` filter:


.. testcode:: python

    import vtki
    import numpy as np

    # Make a grid
    x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
                      np.linspace(-5, 5, 20),
                      np.linspace(-5, 5, 5))

    grid = vtki.StructuredGrid(x, y, z)

    vectors = np.sin(grid.points)**3


    # Compute a direction for the vector field
    grid.point_arrays['mag'] = np.linalg.norm(vectors, axis=1)
    grid.point_arrays['vec'] = vectors

    # plot using the plotting class
    p = vtki.Plotter()
    p.add_mesh(grid.glyph(orient='vec', scale='mag', factor=1), cmap='Greens')
    p.show(auto_close=False)
    p.screenshot('./images/vectorfield.png')
    p.close()


.. image:: ../../images/vectorfield.png
