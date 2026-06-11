.. _theme_api:

Themes
~~~~~~
PyVista plotting parameters can be controlled on a plot by plot basis
or through a global theme, making it possible to control mesh colors
and styles through one global configuration.

The ``DocumentTheme`` is the default theme for PyVista. ``Theme``
provides a theme that is similar to the default styling of VTK.

See :ref:`themes_example` for an example on how to use themes within
PyVista.

.. currentmodule:: pyvista.plotting

.. autosummary::
   :toctree: _autosummary

   themes.DarkTheme
   themes.Theme
   themes.DocumentTheme
   themes.ParaViewTheme
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

Custom Interactor Styles
~~~~~~~~~~~~~~~~~~~~~~~~

Themes can also choose the default interactor style by name through
``Theme.interactor_style``. Downstream packages can register
additional style names programmatically or through the
``pyvista.interactor_styles`` entry-point group.

.. autofunction:: pyvista.register_interactor_style
