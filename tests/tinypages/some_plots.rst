##########
Some Plots
##########

**Plot 1** Does not use doctest syntax:

.. pyvista-plot::

    import pyvista
    pyvista.Sphere().plot()


**Plot 2** Uses doctest syntax:

.. pyvista-plot::
    :format: doctest

    >>> import pyvista
    >>> pyvista.Cube().plot()


**Plot 3** Shows that a new block with context does not see the variable defined
in the no-context block:

.. pyvista-plot::
    :context:

    assert 'a' not in globals()


**Plot 4** Defines ``a`` in a context block:

.. pyvista-plot::
    :context:

    a = 10
    import pyvista
    pyvista.Plane().plot()


**Plot 5** Shows that a block with context sees the new variable. It also uses
``:nofigs:``:

.. pyvista-plot::
    :context:
    :nofigs:

    assert a == 10


**Plot 6** Shows that a non-context block still doesn't have ``a``:

.. pyvista-plot::

    assert 'a' not in globals()


**Plot 7** Uses ``include-source``:

.. pyvista-plot::
    :include-source: True

    # Only a comment


Plot _ Uses an external file with the plot commands and a caption:

.. pyvista-plot:: plot_cone.py
   :force_static:

   This is the caption for plot 8.


Plot _ Uses a specific function in a file with plot commands:

.. pyvista-plot:: plot_polygon.py plot_poly


**Plot 8** Gets a caption specified by the ``:caption:`` option:

.. pyvista-plot::
   :force_static:
   :caption: Plot 8 uses the caption option.

   import pyvista
   pyvista.Disc().plot()


Plot __ Uses an external file with the plot commands and a caption
using the ``:caption:`` option:

.. pyvista-plot:: plot_cone.py
   :force_static:
   :caption: This is the caption for plot_cone.py


**Plot 9** Shows that the default template correctly prints the multi-image
scenario:

.. pyvista-plot::
   :force_static:
   :caption: This caption applies to both plots.

   import pyvista
   pyvista.Text3D('hello').plot()

   pyvista.Text3D('world').plot()


**Plot 10** Uses the skip directive and should not generate a plot:

.. pyvista-plot::

   import pyvista
   pyvista.Sphere().plot()  # doctest:+SKIP


**Plot 11** Uses ``:include-source: False``:

.. pyvista-plot::
    :include-source: False

    # you should not be reading this right now


**Plot 12** Uses ``:include-source:`` with no args:

.. pyvista-plot::
    :include-source:

    # should be printed: include-source with no args


**Plot 13** Should create two plots and be able to plot while skipping
lines, even in two sections:

.. pyvista-plot::

    >>> import pyvista
    >>> pyvista.Sphere().plot(color='blue',
    ...                       cpos='xy')

    >>> pyvista.Sphere().plot(color='red',
    ...                       cpos='xy')


**Plot 14** Forces two static images instead of interactive scenes:

.. pyvista-plot::
   :force_static:

   >>> import pyvista
   >>> pyvista.Sphere().plot(color='blue',
   ...                       cpos='xy')

   >>> pyvista.Sphere().plot(color='red',
   ...                       cpos='xy')


**Plot 15** Uses caption with tabbed UI:

.. pyvista-plot::
   :caption: Plot 15 uses the caption option with tabbed UI.

   import pyvista
   pyvista.Disc().plot()


**Plot 16** Should never be skipped, using the ``:skip: no`` option:

.. pyvista-plot::
   :skip: no
   :caption: Plot 16 will never be skipped

   import pyvista
   pyvista.Cube().plot()


This plot will always be skipped, using the ``:skip: yes`` option,
but the source will always be included but with no caption:

.. pyvista-plot::
   :skip: yes
   :caption: This plot will always be skipped with no caption

   # should be printed: skip is enforced


**Plot 18** Conditional plot execution using ``:optional:`` option,
but the source will always be included with a conditional caption:

.. pyvista-plot::
   :optional:
   :caption: This plot may be skipped with no caption

   import pyvista
   pyvista.Cube().plot()
