.. _named_colormaps:

Named Colormaps
===============

Named colormaps supported by :class:`~pyvista.LookupTable`, :class:`~pyvista.Plotter`,
and other plotting-related methods.

The colormaps on this page are from multiple packages:

- :bdg-secondary:`matplotlib` -
  `matplotlib colormaps <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.
- :bdg-success:`colorcet` -
  `continuous <https://colorcet.holoviz.org/user_guide/Continuous.html#named-colormaps>`_
  and `categorical <https://colorcet.holoviz.org/user_guide/Categorical.html#categorical>`_
  colormaps from ``colorcet``.
- :bdg-primary:`cmocean` - `cmocean colormaps <https://matplotlib.org/cmocean/>`_.

The type of the colormap is indicated as:

- :bdg-muted:`LS` if it is a :class:`mpl.colors.LinearSegmentedColormap`
- :bdg-muted:`L` if it is a :class:`mpl.colors.ListedColormap`

.. note::

    Some colormap names such as ``gray`` and ``rainbow`` are duplicated across
    packages, but have slight variations in the mapped colors. Colormaps from
    ``colorcet`` have priority and are used first if available, followed
    by ``cmocean``, followed by the stock colormaps from ``matplotlib``.

.. note::

    Install PyVista with ``pyvista[colormaps]`` or ``pyvista[all]`` to also
    install the ``colorcet`` and ``cmocean`` packages.

.. seealso::

    :ref:`colormap_example`
        Example using colormaps from different sources.

    :ref:`named_colors`
        Similar reference for named colors.

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
