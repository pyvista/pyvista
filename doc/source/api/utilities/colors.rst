.. _colors_api:

Colors
------
.. currentmodule:: pyvista

.. seealso::

   :ref:`colormap_example`
      Choose a colormap when plotting scalars.

   :ref:`color_cycler_example`
      Cycle through colors when adding meshes.

   :ref:`lookup_table_example`
      Build a :class:`~pyvista.LookupTable` from a colormap or colors.

   :ref:`theme_api`
      Set default colors through a theme.

.. autosummary::
   :toctree: _autosummary

   Color
   ColorLike
   get_cmap_safe

Named colors supported by :class:`~pyvista.Color`, :class:`~pyvista.Plotter`,
and other plotting-related methods:

.. toctree::
   :maxdepth: 3

   /api/utilities/named_colors


Named colormaps supported by :class:`~pyvista.LookupTable`, :class:`~pyvista.Plotter`,
and other plotting-related methods:

.. toctree::
   :maxdepth: 3

   /api/utilities/named_colormaps
