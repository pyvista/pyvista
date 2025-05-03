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

- :bdg-muted:`LS` if it is a :class:`~matplotlib.colors.LinearSegmentedColormap`
- :bdg-muted:`L` if it is a :class:`~matplotlib.colors.ListedColormap`

.. warning::

    Many of the ``matplotlib`` colormaps such as ``jet`` are not perceptually
    uniform and should be avoided where possible, since these colormaps
    can generate misleading visualizations. Colormaps from ``colorcet`` and
    ``cmocean`` are therefore generally recommended over those from ``matplotlib``.
    See `testing-perceptual-uniformity <https://colorcet.holoviz.org/user_guide/Continuous.html#testing-perceptual-uniformity>`_

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

Linear (Sequential)
-------------------

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_SEQUENTIAL.rst

Diverging
---------

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_DIVERGING.rst

Cyclic
------

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CYCLIC.rst

Categorical (Qualitative)
-------------------------

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CATEGORICAL.rst

Miscellaneous
-------------

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_MISC.rst

CET Colormaps
=============

This table includes all ``colorcet`` colormaps which have a named ``CET``
alias. Most of the ``colorcet`` colormaps presented above are duplicated
here (using their ``CET`` alias).

.. dropdown::

    .. include:: /api/utilities/colormap_table/colormap_table_CET.rst
