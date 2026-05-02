.. _plotting-api-index:

Plotting
========

* Intuitive plotting routines with `matplotlib`_ like syntax (see :ref:`plotting`).
* Plotting tools built for interactivity (see :ref:`widgets`).

.. toctree::
   :hidden:

   plotting
   qt_plotting
   theme
   trame
   components

.. _matplotlib: https://matplotlib.org/


Plotting API Reference
----------------------
Plotting module API reference. These plotting modules are the basis for
all plotting functionality in PyVista.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   Actor
   Axes
   AxesActor
   AxesAssembly
   AxesAssemblySymmetric
   CameraPosition
   CornerAnnotation
   CubeAxesActor
   DataSetMapper
   Follower
   Label
   LookupTable
   PlanesAssembly
   Plotter
   Prop3D
   Property
   Renderer
   RenderWindowInteractor
   Text
   TextProperty
   Timer
   plotting.mapper._BaseMapper
   plotting.mapper.FixedPointVolumeRayCastMapper
   plotting.mapper.GPUVolumeRayCastMapper
   plotting.mapper.OpenGLGPUVolumeRayCastMapper
   plotting.mapper.PointGaussianMapper
   plotting.mapper.SmartVolumeMapper
   plotting.mapper.UnstructuredGridVolumeRayCastMapper
   plotting.opts.ElementType
   plotting.opts.InterpolationType
   plotting.opts.PointSpriteShape
   plotting.opts.RepresentationType
   plotting.opts.ShaderType
   plotting.volume.Volume
   plotting.volume_property.VolumeProperty


Composite Plotting Reference
----------------------------
These classes are used when plotting :class:`pyvista.MultiBlock` datasets.

.. autosummary::
   :toctree: _autosummary

   CompositePolyDataMapper
   CompositeAttributes
   BlockAttributes


Charts API
----------
Charts API reference. These dedicated classes can be used to embed
charts in plotting windows. Note that full charts functionality
requires a VTK version of at least 9.3. Most components work fine
in older versions though.

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

   plotting.widgets.WidgetComponent
   plotting.widgets.AffineWidget3D


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
   plotting.render_window_interactor.InteractorStyleTrackballActor
   plotting.render_window_interactor.InteractorStyleTrackballCamera
   plotting.render_window_interactor.InteractorStyleTerrain
   plotting.render_window_interactor.InteractorStyleZoom


VTK Algorithm Utilities
~~~~~~~~~~~~~~~~~~~~~~~
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


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~
These functions provide a simplified interface to the various plotting
routines in PyVista.

.. toctree::
   :maxdepth: 2

   conv_func
