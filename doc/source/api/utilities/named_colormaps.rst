.. _named_colormaps:

Named Colormaps
===============

Named colormaps supported by plotting-related classes such as
:class:`~pyvista.Plotter`.

The colormaps on this page come from various sources, and include:

- :bdg-secondary:`matplotlib` - colormaps from
  `matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_.
- :bdg-success:`colorcet` - colormaps from
  `colorcet <https://colorcet.holoviz.org>`_.
- :bdg-primary:`cmocean` - colormaps from
  `cmocean <https://matplotlib.org/cmocean/>`_.

.. note::

    Some colormaps such as ``gray`` and ``rainbow`` are duplicated across
    packages. If installed, colormaps from ``colorcet`` have priority, followed
    by ``cmocean``, followed by ``matplotlib``.

.. note::

    Install PyVista with ``pyvista[colormaps]`` or ``pyvista[all]`` to also
    install the ``colorcet`` and ``cmocean`` packages.

.. dropdown:: Sequential
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_SEQUENTIAL.rst

.. dropdown:: Diverging
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_DIVERGING.rst

.. dropdown:: Cyclic
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CYCLIC.rst

.. dropdown:: Categorical
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CATEGORICAL.rst

.. dropdown:: Misc
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_MISC.rst
