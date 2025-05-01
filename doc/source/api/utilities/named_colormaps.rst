.. _named_colormaps:

Named Colormaps
===============

Named colormaps supported by :class:`~pyvista.LookupTable`, :class:`~pyvista.Plotter`,
and other plotting-related methods.

The colormaps on this page are from multiple packages:

- :bdg-secondary:`matplotlib` -
  `matplotlib colormaps <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.
- :bdg-success:`colorcet` -
  `Continuous <https://colorcet.holoviz.org/user_guide/Continuous.html#named-colormaps>`_
  and `categorical <https://colorcet.holoviz.org/user_guide/Categorical.html#categorical>`_
  colormaps from ``colorcet``.
- :bdg-primary:`cmocean` - `cmocean colormaps <https://matplotlib.org/cmocean/>`_.

.. note::

    Some colormaps such as ``gray`` and ``rainbow`` are duplicated across
    packages. If installed, colormaps from ``colorcet`` have priority, followed
    by ``cmocean``, followed by ``matplotlib``.

.. note::

    Install PyVista with ``pyvista[colormaps]`` or ``pyvista[all]`` to also
    install the ``colorcet`` and ``cmocean`` packages.

.. seealso::

    :ref:`colormap_example`
        Example using colormaps from different sources.

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
