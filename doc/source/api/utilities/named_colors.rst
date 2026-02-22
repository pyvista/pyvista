.. _named_colors:

Named Colors
============

Named colors supported by :class:`~pyvista.Color` and plotting-related classes such as
:class:`~pyvista.Plotter`.

The colors on this page come from various sources, and include:

- :bdg-primary:`CSS` - all standard `named web colors <https://www.w3.org/TR/css-color-4/#named-colors>`_
- :bdg-secondary:`VTK` - colors from `vtkNamedColors <https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKNamedColorPatches.html#VTKColorNames>`_
- :bdg-success:`TAB` - palette of 10 colors used by ``Tableau``
- :bdg-danger:`PV` - default colors used by ``ParaView``

See also ``matplotlib``'s `list of named colors <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
for additional information about the ``CSS`` and ``TAB`` colors.

.. note::

    Many colors have multiple names which refer to the same color (``'gray'`` and ``'grey'``,
    for example). These alternate names are listed where applicable. Upper-case letters, and underscores, spaces,
    and hyphens between words are also acceptable inputs. See :attr:`Color.name <pyvista.Color.name>`
    for examples.

.. warning::

    Some color names are not internally consistent. For example,

    - ``'dark_gray'`` is lighter than ``'gray'``,
    - ``'light_pink'`` is darker than ``'pink'``, and
    - ``'cadmium_yellow'`` is classified as an ``Orange`` color.

.. seealso::

    :ref:`named_colormaps`
        Similar reference for named colormaps.

Sorted by Name
--------------

.. dropdown:: All Colors

    .. include:: /api/utilities/color_table/color_table.rst

Sorted by Color
---------------

.. dropdown:: Blacks
    :open:

    .. include:: /api/utilities/color_table/color_table_BLACK.rst

.. dropdown:: Grays
    :open:

    .. include:: /api/utilities/color_table/color_table_GRAY.rst

.. dropdown:: Whites
    :open:

    .. include:: /api/utilities/color_table/color_table_WHITE.rst

.. dropdown:: Reds
    :open:

    .. include:: /api/utilities/color_table/color_table_RED.rst

.. dropdown:: Oranges
    :open:

    .. include:: /api/utilities/color_table/color_table_ORANGE.rst

.. dropdown:: Browns
    :open:

    .. include:: /api/utilities/color_table/color_table_BROWN.rst

.. dropdown:: Yellows
    :open:

    .. include:: /api/utilities/color_table/color_table_YELLOW.rst

.. dropdown:: Greens
    :open:

    .. include:: /api/utilities/color_table/color_table_GREEN.rst

.. dropdown:: Cyans
    :open:

    .. include:: /api/utilities/color_table/color_table_CYAN.rst

.. dropdown:: Blues
    :open:

    .. include:: /api/utilities/color_table/color_table_BLUE.rst

.. dropdown:: Violets
    :open:

    .. include:: /api/utilities/color_table/color_table_VIOLET.rst

.. dropdown:: Magentas
    :open:

    .. include:: /api/utilities/color_table/color_table_MAGENTA.rst
