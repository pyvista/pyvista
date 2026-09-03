.. _plotting-api-index:

Plotting
========

* Intuitive plotting routines with `matplotlib`_ like syntax. See
  :ref:`plotting` for using the interactive window.
* Plotting tools built for interactivity (see :ref:`widgets`).

.. toctree::
   :hidden:

   camera
   lights
   theme
   qt_plotting
   trame
   components

.. _matplotlib: https://matplotlib.org/


Plotting Functions
------------------
These functions provide a simplified interface to the plotting classes below.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   close_all
   plot
   plot_arrows
   plot_compare
   plot_compare_four


Plotter
-------
Cameras and lights are documented on their own pages: :ref:`cameras_api` and
:ref:`lights_api`.

.. autosummary::
   :toctree: _autosummary

   Plotter
   CameraPosition
   Renderer
   RenderWindowInteractor
   Timer

.. seealso::

   :ref:`multi_window_example`
      Lay out several renderers in one window.

   :ref:`screenshot_example`
      Save screenshots from a plotter.

   :ref:`movie_example`
      Write frames to a movie.


Actors and Mappers
------------------
.. autosummary::
   :toctree: _autosummary

   Actor
   DataSetMapper
   Follower
   plotting.mapper.PointGaussianMapper
   Prop3D
   Property

.. seealso::

   :ref:`backface_prop_example`
      Set properties for the back faces of an actor.

   :ref:`lighting_mesh_example`
      Control how an actor's surface reflects light.


Volume Rendering
----------------
.. autosummary::
   :toctree: _autosummary

   plotting.volume.Volume
   plotting.mapper.FixedPointVolumeRayCastMapper
   plotting.mapper.GPUVolumeRayCastMapper
   plotting.mapper.OpenGLGPUVolumeRayCastMapper
   plotting.mapper.SmartVolumeMapper
   plotting.mapper.UnstructuredGridVolumeRayCastMapper
   plotting.volume_property.VolumeProperty
   opacity_transfer_function

.. seealso::

   :ref:`volume_rendering_example`
      Render a volume with the different mappers.

   :ref:`opacity_example`
      Build an opacity transfer function.


Text and Labels
---------------
.. autosummary::
   :toctree: _autosummary

   CornerAnnotation
   Label
   Text
   TextProperty

.. seealso::

   :ref:`point_labels_example`
      Label points in a scene.


Axes and Orientation
--------------------
.. autosummary::
   :toctree: _autosummary

   Axes
   AxesActor
   AxesAssembly
   AxesAssemblySymmetric
   CubeAxesActor
   PlanesAssembly
   create_axes_marker
   create_axes_orientation_box

.. seealso::

   :ref:`axes_objects_example`
      Compare the axes objects and add them to a scene.


Lookup Tables
-------------
Colors, colormaps, and the :class:`~pyvista.Color` class are documented under
:ref:`colors_api`.

.. autosummary::
   :toctree: _autosummary

   LookupTable

.. seealso::

   :ref:`lookup_table_example`
      Build a lookup table from a colormap or a list of colors.


Enumerations
------------
.. autosummary::
   :toctree: _autosummary

   plotting.opts.ElementType
   plotting.opts.InterpolationType
   plotting.opts.PointSpriteShape
   plotting.opts.RepresentationType
   plotting.opts.ShaderType


Composite Plotting
------------------
These classes are used when plotting :class:`pyvista.MultiBlock` datasets.

.. autosummary::
   :toctree: _autosummary

   BlockAttributes
   CompositeAttributes
   CompositePolyDataMapper


Charts API
----------
Charts API reference. These dedicated classes can be used to embed
charts in plotting windows.

.. toctree::
   :maxdepth: 2

   charts/index


Widget API
----------
Every interactive widget on the plotter (box, plane, line, slider,
sphere, spline, button, radio button, measurement, logo, camera,
camera3d) lives on the :class:`~pyvista.plotting.widgets.WidgetComponent`
plotter component, accessible as ``plotter.widgets``. The top-level
plotter exposes ``add_*_widget`` and ``clear_*_widgets`` methods as
forwarding shims; both surfaces produce identical results. For additional
details, see the :ref:`widgets` examples.

.. autosummary::
   :toctree: _autosummary

   plotting.widgets.AffineWidget3D
   plotting.widgets.WidgetComponent


Picking API
-----------
Picking lives on the :class:`~pyvista.plotting.picking.PickingComponent`
plotter component, accessible as ``plotter.picking``. The top-level
plotter exposes ``enable_*_picking``, ``disable_picking``, and
``picked_*`` properties as forwarding shims.

.. autosummary::
   :toctree: _autosummary

   plotting.picking.PickingComponent


Interactor Style API
--------------------
An interactor style sets mouse and key bindings to interact with
the plot. Most often methods like :func:`pyvista.Plotter.enable_trackball_style`
would be used, but this API can be used as a starting point for customizing the
interaction styles.

.. autosummary::
   :toctree: _autosummary

   plotting.render_window_interactor.InteractorStyleCaptureMixin
   plotting.render_window_interactor.InteractorStyleImage
   plotting.render_window_interactor.InteractorStyleJoystickActor
   plotting.render_window_interactor.InteractorStyleJoystickCamera
   plotting.render_window_interactor.InteractorStyleRubberBand2D
   plotting.render_window_interactor.InteractorStyleRubberBandPick
   plotting.render_window_interactor.InteractorStyleTerrain
   plotting.render_window_interactor.InteractorStyleTrackballActor
   plotting.render_window_interactor.InteractorStyleTrackballCamera
   plotting.render_window_interactor.InteractorStyleZoom


VTK Algorithm Utilities
-----------------------
These functions create VTK algorithm pipeline nodes for use with PyVista's
pipeline-based rendering. They are useful for advanced users who need
fine-grained control over the VTK pipeline.

.. autosummary::
   :toctree: _autosummary

   plotting.utilities.active_scalars_algorithm
   plotting.utilities.add_ids_algorithm
   plotting.utilities.algorithm_to_mesh_handler
   plotting.utilities.callback_algorithm
   plotting.utilities.cell_data_to_point_data_algorithm
   plotting.utilities.crinkle_algorithm
   plotting.utilities.decimation_algorithm
   plotting.utilities.extract_surface_algorithm
   plotting.utilities.outline_algorithm
   plotting.utilities.point_data_to_cell_data_algorithm
   plotting.utilities.pointset_to_polydata_algorithm
   plotting.utilities.set_algorithm_input
   plotting.utilities.smooth_shading_algorithm
   plotting.utilities.source_algorithm
   plotting.utilities.triangulate_algorithm


Jupyter Backends
----------------
The backend is selected with :func:`~pyvista.set_jupyter_backend`, and
third-party packages can register additional backends. See
:ref:`jupyter_plotting` for plotting in a notebook.

.. autosummary::
   :toctree: _autosummary

   JupyterBackendRegistration
   register_jupyter_backend
   registered_jupyter_backends
   set_jupyter_backend


Shared Base Classes
-------------------
These classes are not used directly. They are documented because they define
members that several of the classes above share, so that each of those members is
documented once, here, and linked from every class that inherits it.

.. autosummary::
   :toctree: _autosummary

   plotting.mapper._BaseDataSetMapper
   plotting.mapper._BaseMapper
   plotting.mapper._BaseVolumeMapper
   core.utilities.misc._NameMixin
   plotting.prop3d._Prop3DMixin
   plotting.axes_assembly._XYZAssembly
