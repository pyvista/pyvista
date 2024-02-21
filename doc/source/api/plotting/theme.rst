.. _theme_api:

Themes
~~~~~~
PyVista plotting parameters can be controlled on a plot by plot basis
or through a global theme, making it possible to control mesh colors
and styles through one global configuration.

The :func:`~pyvista.plotting.themes.Theme.document_theme` is the default theme for PyVista.
Directly using :class:`~pyvista.plotting.themes.Theme` like ``Theme()`` or
:func:`~pyvista.plotting.themes.Theme.vtk_theme`
provides a theme that is similar to the default styling of VTK.

See :ref:`themes_example` for an example on how to use themes within
PyVista.

.. currentmodule:: pyvista.plotting

.. autosummary::
   :toctree: _autosummary

   themes.set_plot_theme
   themes.Theme
   themes._AxesConfig
   themes._CameraConfig
   themes._ColorbarConfig
   themes._DepthPeelingConfig
   themes._Font
   themes._LightingConfig
   themes._SilhouetteConfig
   themes._SliderConfig
   themes._SliderStyleConfig
   themes._TrameConfig
   themes.load_theme
