API Reference
*************

In this section, you can learn more about how PyVista wraps different VTK mesh
types and how you can leverage powerful 3D plotting and mesh analysis tools.
Highlights of the API include:

* Pythonic interface to VTK's Python bindings
* Filtering/plotting tools built for interactivity (see :ref:`widgets`)
* Direct access to common VTK filters (see :ref:`filters_ref`)
* Intuitive plotting routines with `matplotlib`_ similar syntax (see
  :ref:`plotting_ref`)

.. toctree::
   :caption: API Reference
   :hidden:

   core/index
   plotting/index
   utilities/index
   readers/index
   examples/index

.. grid:: 1 2 2 2
    :gutter: 2

    Core API
    ^^^^^^^^

    Learn more anout PyVista's different mesh types and direct access to common VTK filters.

    .. button-ref:: core/index
        :text: Core API
        :classes: btn-outline-primary btn-block stretched-link

    ---

    Plotting API
    ^^^^^^^^^^^^

    Explore PyVista's robust plotting interface for visualizing the core data structures.

    .. button-ref:: plotting/index
        :text: Plotting API
        :classes: btn-outline-primary btn-block stretched-link

    ---

    File API
    ^^^^^^^^

    Use PyVista's Reader classes to read data files.

    .. button-ref:: readers/index
        :text: Readers
        :classes: btn-outline-primary btn-block stretched-link

    ---

    Utilities
    ^^^^^^^^^

    Utilize PyVista's helper modules, conversion tools, and geometric object creation routines.

    .. button-ref:: utilities/index
        :text: Utilities
        :classes: btn-outline-primary btn-block stretched-link

.. _matplotlib: https://matplotlib.org/
