.. _configuration:

Global Configuration
====================

This page is the central reference for every process-wide setting in
PyVista:

* :ref:`config_objects` -- runtime settings on ``pv.global_theme``
  (plotting) and ``pv.global_config`` (everything else).
* :ref:`config_flags` -- module-level attributes such as
  ``pv.OFF_SCREEN``.
* :ref:`config_env` -- environment variables read when PyVista is
  imported.
* :ref:`config_vtk` -- runtime controls for the VTK interface.
* :ref:`config_registries` -- registration functions and entry points
  for extending PyVista.
* :ref:`config_report` -- inspecting the active configuration with
  :class:`pyvista.Report`.

.. _config_objects:

Configuration Objects
---------------------

Two singleton objects hold PyVista's runtime settings. Plotting
defaults live on ``pv.global_theme`` and all other settings live on
``pv.global_config``. Both share the same machinery: attribute
access, dict-style item access, and ``to_dict`` / ``from_dict``
round-tripping.

.. _config_theme:

Plotting: The Global Theme
~~~~~~~~~~~~~~~~~~~~~~~~~~

``pv.global_theme`` is a :class:`~pyvista.plotting.themes.Theme`
instance holding every plotting default: colors, fonts, window size,
camera behavior, the Jupyter backend, and more. Assign to its
attributes to change the defaults for all later plots:

.. code-block:: python

    import pyvista as pv

    pv.global_theme.color = 'lightblue'
    pv.global_theme.window_size = [600, 400]
    pv.global_theme.smooth_shading = True

Swap the entire theme with :func:`pyvista.set_plot_theme` or the
:envvar:`PYVISTA_PLOT_THEME` environment variable, list the available
names with :func:`pyvista.registered_themes`, and save or restore a
customized theme with :meth:`~pyvista.plotting.themes.Theme.save` and
:func:`pyvista.load_theme`. A theme can also be applied to a single
plotter with ``pv.Plotter(theme=my_theme)``.

.. seealso::

   :ref:`userguide_themes`
      User guide for customizing and applying themes.

   :ref:`theme_api`
      API reference for every theme class.

.. _config_core:

Core: The Global Config
~~~~~~~~~~~~~~~~~~~~~~~

``pv.global_config`` holds the non-plotting settings and is the core
counterpart of ``pv.global_theme``:

.. code-block:: python

    import pyvista as pv

    pv.global_config.validate_on_wrap = False

.. autoclass:: pyvista.core.config.Config
   :members:

The warning emitted when
:attr:`~pyvista.core.config.Config.validate_on_wrap` finds an invalid
data array:

.. autoclass:: pyvista.InvalidMeshWarning

.. _config_flags:

Module-Level Flags
------------------

These attributes are plain module globals. Set them at runtime to
change the behavior of the whole process:

.. code-block:: python

    import pyvista as pv

    pv.OFF_SCREEN = True

``pv.OFF_SCREEN`` (default: ``False``)
    Render all plots off screen, without opening a window.
    Initialized from :envvar:`PYVISTA_OFF_SCREEN`.

``pv.BUILDING_GALLERY`` (default: ``False``)
    Enable behavior needed when Sphinx-Gallery builds the example
    gallery. Initialized from :envvar:`PYVISTA_BUILDING_GALLERY`.

``pv.FIGURE_PATH`` (default: ``None``)
    Directory where screenshots are saved when a relative file name
    is given. Initialized from :envvar:`PYVISTA_FIGURE_PATH`.

``pv.ON_SCREENSHOT`` (default: ``False``)
    Render off screen and save a screenshot with a unique file name
    each time a plot is shown. Initialized from
    :envvar:`PYVISTA_ON_SCREENSHOT`.

``pv.PLOT_DIRECTIVE_THEME`` (default: ``None``)
    Theme applied by the ``pyvista-plot`` Sphinx directive when
    building documentation.

``pv.FLOAT_FORMAT`` (default: ``'{:.3e}'``)
    Format string used to print floats in dataset representations.

``pv.PICKLE_FORMAT`` (default: ``'vtk'``)
    In-memory serialization format used when pickling a
    :class:`~pyvista.DataObject`. Set it with
    :func:`pyvista.set_pickle_format`.

``pv.DEFAULT_SCALARS_NAME`` (default: ``'Data'``)
    Name given to data arrays added without a name.

``pv.MAX_N_COLOR_BARS`` (default: ``10``)
    Maximum number of color bars a plotter can show at once.

.. _config_env:

Environment Variables
---------------------

Most environment variables are read once, when PyVista (or the module
that uses them) is first imported. The theme-related variables are
re-read each time a new :class:`~pyvista.plotting.themes.Theme` is
created. Use the runtime equivalent listed with each variable to
change behavior in a running process. Boolean variables accept
``true`` or ``false`` (case-insensitive).

Rendering
~~~~~~~~~

.. envvar:: PYVISTA_OFF_SCREEN

   Render all plots off screen, without opening a window. Sets
   ``pv.OFF_SCREEN``; a single plotter can opt in with
   ``pv.Plotter(off_screen=True)``.

.. envvar:: PYVISTA_MULTI_SAMPLES

   Number of multi-samples used for anti-aliasing. Sets the default
   of :attr:`pyvista.plotting.themes.Theme.multi_samples`.

.. envvar:: PYVISTA_AUTO_CLOSE

   Set to ``false`` to stop plotters from closing automatically after
   showing. Sets the default of
   :attr:`pyvista.plotting.themes.Theme.auto_close`.

.. note::

   * VTK's own ``VTK_DEFAULT_OPENGL_WINDOW`` environment variable
     selects the render window class VTK creates, such as an EGL
     window for headless rendering; see the `VTK runtime settings
     <https://docs.vtk.org/en/latest/advanced/runtime_settings.html#opengl>`_.
   * ``PYVISTA_VIRTUAL_DISPLAY``, mentioned in some older issues, is
     not a PyVista setting.
   * :attr:`~pyvista.plotting.themes.Theme.interactive` controls
     whether shown plots accept user interaction, not off-screen
     rendering.

Theme and Jupyter
~~~~~~~~~~~~~~~~~

.. envvar:: PYVISTA_PLOT_THEME

   Theme to apply when the plotting module is first loaded. Any name
   reported by :func:`pyvista.registered_themes` is accepted, as is a
   ``"package.module:ClassName"`` dotted path to a
   :class:`~pyvista.plotting.themes.Theme` subclass. An invalid value
   emits a warning. Equivalent to calling
   :func:`pyvista.set_plot_theme`.

.. envvar:: PYVISTA_JUPYTER_BACKEND

   Default Jupyter plotting backend. Sets the default of
   :attr:`pyvista.plotting.themes.Theme.jupyter_backend`. See
   :ref:`jupyter_plotting`.

.. envvar:: PYVISTA_TRAME_SERVER_PROXY_PREFIX

   URL prefix for a Jupyter server proxy. Setting it also enables the
   proxy. See :ref:`trame_jupyter`.

.. envvar:: PYVISTA_TRAME_JUPYTER_MODE

   How Trame communicates with Jupyter: ``extension``, ``proxy``, or
   ``native``. See :ref:`trame_jupyter`.

VTK
~~~

.. envvar:: PYVISTA_VTK_BACKEND

   Which VTK build PyVista imports: ``vtk`` or ``vtkmodules`` for
   stock VTK, or the package name of an alternative build. Query the
   active backend with :func:`pyvista.vtk_backend`.

Example Data
~~~~~~~~~~~~

.. envvar:: PYVISTA_USERDATA_PATH

   Writable directory where downloaded example data is cached. See
   :ref:`examples_api`.

.. envvar:: PYVISTA_VTK_DATA

   Path to a local clone of `pyvista/data
   <https://github.com/pyvista/data>`_ to use instead of downloading
   example files. See :ref:`examples_api`.

Both example-data variables are included in the output of
``pv.Report(downloads=True)``.

Documentation Building
~~~~~~~~~~~~~~~~~~~~~~

.. envvar:: PYVISTA_FIGURE_PATH

   Directory where screenshots are saved when a relative file name is
   given. Sets ``pv.FIGURE_PATH``.

.. envvar:: PYVISTA_BUILDING_GALLERY

   Enable Sphinx-Gallery build behavior. Sets ``pv.BUILDING_GALLERY``.

.. envvar:: PYVISTA_ON_SCREENSHOT

   Save a screenshot each time a plot is shown. Sets
   ``pv.ON_SCREENSHOT``.

.. note::

   ``PYVISTA_GALLERY_FORCE_STATIC`` and
   ``PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT`` are not environment
   variables: they are Python variables assigned inside a
   Sphinx-Gallery example script to force static images for one plot
   or for a whole document.

.. note::

   ``PYVISTA_KILL_DISPLAY`` is no longer used and has no effect.

.. _config_vtk:

VTK Interface Controls
----------------------

These settings control how PyVista interacts with VTK at runtime. The
state managers ``pv.vtk_verbosity``, ``pv.vtk_snake_case``, and
``pv.allow_new_attributes``, along with
:func:`pyvista.enable_smp_tools`, apply globally when called and
temporarily when used as context managers:

.. code-block:: python

    import pyvista as pv

    pv.vtk_verbosity('off')  # applies globally

    with pv.vtk_verbosity('info'):  # applies within the context
        ...

.. autosummary::

   ~pyvista.vtk_verbosity
   ~pyvista.vtk_snake_case
   ~pyvista.allow_new_attributes
   ~pyvista.enable_smp_tools
   ~pyvista.vtk_backend

Related settings: :attr:`~pyvista.core.config.Config.show_vtk_api` on
``pv.global_config`` controls whether the VTK-inherited API appears in
:func:`dir` and tab completion, and ``pv.vtk_version_info`` reports
the version of VTK in use.

.. seealso::

   :ref:`vtk_to_pyvista_docs`
      How PyVista's interface relates to VTK's.

.. _config_registries:

Extension Registries
--------------------

Third-party packages extend PyVista through registries. Each registry
has a function for registering at runtime and an entry-point group
for registering from a package's ``pyproject.toml`` so the extension
is discovered without an explicit import.

.. list-table::
   :header-rows: 1
   :widths: 24 30 26 20

   * - Extension
     - Register
     - List
     - Entry-point group
   * - :ref:`File readers <reader_api>`
     - :func:`~pyvista.register_reader`
     - :func:`~pyvista.registered_readers`
     - ``pyvista.readers``
   * - :ref:`File writers <reader_api>`
     - :func:`~pyvista.register_writer`
     - :func:`~pyvista.registered_writers`
     - ``pyvista.writers``
   * - :ref:`Dataset accessors <accessor-api>`
     - :func:`~pyvista.register_dataset_accessor`
     - :func:`~pyvista.registered_accessors`
     - ``pyvista.accessors``
   * - :ref:`Plotter components <plotter-component-api>`
     - :func:`~pyvista.register_plotter_component`
     - :func:`~pyvista.registered_plotter_components`
     - ``pyvista.plotter_components``
   * - :ref:`Jupyter backends <jupyter_plotting>`
     - :func:`~pyvista.register_jupyter_backend`
     - :func:`~pyvista.registered_jupyter_backends`
     - ``pyvista.jupyter_backends``
   * - :ref:`Themes <theme_api>`
     - Subclass :class:`~pyvista.plotting.themes.Theme`
     - :func:`~pyvista.registered_themes`
     - ``pyvista.themes``
   * - :ref:`Interactor styles <theme_api>`
     - :func:`~pyvista.register_interactor_style`
     -
     - ``pyvista.interactor_styles``

.. seealso::

   :ref:`extending-pyvista`
      Guide to writing a plugin package, with a worked accessor
      example.

.. _config_report:

Inspecting the Environment
--------------------------

:class:`pyvista.Report` summarizes the running environment: package
versions, GPU information, and, with ``pv.Report(downloads=True)``,
the example-data configuration.
