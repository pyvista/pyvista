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

.. panels::
    :column: col-12 p-3

    Core API
    ^^^^^^^^

    Learn more anout PyVista's different mesh types and direct access to common VTK filters.

    .. link-button:: core/index
        :type: ref
        :text: Core API
        :classes: btn-outline-primary btn-block stretched-link

    ---

    Plotting API
    ^^^^^^^^^^^^

    Explore PyVista's robust plotting interface for visualizing the core data structures.

    .. link-button:: plotting/index
        :type: ref
        :text: Plotting API
        :classes: btn-outline-primary btn-block stretched-link

    ---

    File API
    ^^^^^^^^

    Use PyVista's Reader classes to read data files.

    .. link-button:: readers/index
        :type: ref
        :text: Readers
        :classes: btn-outline-primary btn-block stretched-link

    ---

    Utilities
    ^^^^^^^^^

    Utilize PyVista's helper modules, conversion tools, and geometric object creation routines.

    .. link-button:: utilities/index
        :type: ref
        :text: Utilities
        :classes: btn-outline-primary btn-block stretched-link

.. _matplotlib: https://matplotlib.org/
