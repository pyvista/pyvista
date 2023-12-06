##########
Some Plots
##########

Plot 1 does not use doctest syntax

.. pyvista-plot::

    import pyvista
    pyvista.Sphere().plot()


Plot 2 uses doctest syntax:

.. pyvista-plot::
    :format: doctest

    >>> import pyvista
    >>> pyvista.Cube().plot()


Plot 3 shows that a new block with context does not see the variable defined
in the no-context block:

.. pyvista-plot::
    :context:

    assert 'a' not in globals()


Plot 4 defines ``a`` in a context block:

.. pyvista-plot::
    :context:

    a = 10
    import pyvista
    pyvista.Plane().plot()


Plot 5 shows that a block with context sees the new variable. It also uses
``:nofigs:``:

.. pyvista-plot::
    :context:
    :nofigs:

    assert a == 10


Plot 6 shows that a non-context block still doesn't have ``a``:

.. pyvista-plot::

    assert 'a' not in globals()


Plot 7 uses ``include-source``:

.. pyvista-plot::
    :include-source: True

    # Only a comment


Plot _ uses an external file with the plot commands and a caption:

.. pyvista-plot:: plot_cone.py

   This is the caption for plot 8.


Plot _ uses a specific function in a file with plot commands:

.. pyvista-plot:: plot_polygon.py plot_poly


Plot 8 gets a caption specified by the :caption: option:

.. pyvista-plot::
   :caption: Plot 10 uses the caption option.

   import pyvista
   pyvista.Disc().plot()


Plot __ uses an external file with the plot commands and a caption
using the :caption: option:

.. pyvista-plot:: plot_cone.py
   :caption: This is the caption for plot 11.


Plot 9 shows that the default template correctly prints the multi-image
scenario:

.. pyvista-plot::
   :caption: This caption applies to both plots.

   import pyvista
   pyvista.Text3D('hello').plot()

   pyvista.Text3D('world').plot()


Plot 10 uses the skip directive and should not generate a plot.

.. pyvista-plot::

   import pyvista
   pyvista.Sphere().plot()  # doctest:+SKIP

Plot 11 uses ``include-source`` False:

.. pyvista-plot::
    :include-source: False

    # you should not be reading this right now

Plot 12 uses ``include-source`` with no args:

.. pyvista-plot::
    :include-source:

    # should be printed: include-source with no args

Plot 13 should create two plots and be able to plot while skipping
lines, even in two sections.

.. pyvista-plot::

    >>> import pyvista
    >>> pyvista.Sphere().plot(color='blue',
    ...                       cpos='xy')

    >>> pyvista.Sphere().plot(color='red',
    ...                       cpos='xy')
