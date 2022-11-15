"""Module dedicated to widgets."""

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import (
    NORMALS,
    generate_plane,
    get_array,
    get_array_association,
    try_callback,
)

from .colors import Color


class WidgetHelper:
    """An internal class to manage widgets.

    It also manages and other helper methods involving widgets.

    """

    def __init__(self, *args, **kwargs):
        """Initialize widget helper."""
        super().__init__(*args, **kwargs)
        self.camera_widgets = []
        self.box_widgets = []
        self.box_clipped_meshes = []
        self.plane_widgets = []
        self.plane_clipped_meshes = []
        self.plane_sliced_meshes = []
        self.line_widgets = []
        self.slider_widgets = []
        self.threshold_meshes = []
        self.isovalue_meshes = []
        self.spline_widgets = []
        self.spline_sliced_meshes = []
        self.sphere_widgets = []
        self.button_widgets = []

    def add_box_widget(
        self,
        callback,
        bounds=None,
        factor=1.25,
        rotation_enabled=True,
        color=None,
        use_planes=False,
        outline_translation=True,
        pass_widget=False,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
    ):
        """Add a box widget to the scene.

        This is useless without a callback function. You can pass a
        callable function that takes a single argument, the PolyData
        box output from this widget, and performs a task with that
        box.

        Parameters
        ----------
        callback : callable
            The method called every time the box is updated. This has
            two options: Take a single argument, the ``PolyData`` box
            (default) or if ``use_planes=True``, then it takes a
            single argument of the plane collection as a ``vtkPlanes``
            object.

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is
            placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing.

        rotation_enabled : bool, optional
            If ``False``, the box widget cannot be rotated and is
            strictly orthogonal to the cartesian axes.

        color : color_like, optional
            Either a string, rgb sequence, or hex color string.
            Defaults to :attr:`pyvista.global_theme.font.color
            <pyvista.themes._Font.color>`.

        use_planes : bool, optional
            Changes the arguments passed to the callback to the planes
            that make up the box.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is
            strictly placed at the given bounds.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        Returns
        -------
        vtk.vtkBoxWidget
            Box widget.

        Examples
        --------
        Shows an interactive clip box.

        >>> import pyvista as pv
        >>> mesh = pv.ParametricConicSpiral()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh_clip_box(mesh, color='white')
        >>> pl.show()

        For a full example see :ref:`box_widget_example`.

        """
        if bounds is None:
            bounds = self.bounds

        def _the_callback(box_widget, event_id):
            the_box = pyvista.PolyData()
            box_widget.GetPolyData(the_box)
            planes = _vtk.vtkPlanes()
            box_widget.GetPlanes(planes)
            if callable(callback):
                if use_planes:
                    args = [planes]
                else:
                    args = [the_box]
                if pass_widget:
                    args.append(box_widget)
                try_callback(callback, *args)
            return

        box_widget = _vtk.vtkBoxWidget()
        box_widget.GetOutlineProperty().SetColor(
            Color(color, default_color=pyvista.global_theme.font.color).float_rgb
        )
        box_widget.SetInteractor(self.iren.interactor)
        box_widget.SetCurrentRenderer(self.renderer)
        box_widget.SetPlaceFactor(factor)
        box_widget.SetRotationEnabled(rotation_enabled)
        box_widget.SetTranslationEnabled(outline_translation)
        box_widget.PlaceWidget(bounds)
        box_widget.On()
        box_widget.AddObserver(interaction_event, _the_callback)
        _the_callback(box_widget, None)

        self.box_widgets.append(box_widget)
        return box_widget

    def clear_box_widgets(self):
        """Remove all of the box widgets."""
        self.box_widgets.clear()

    def add_mesh_clip_box(
        self,
        mesh,
        invert=False,
        rotation_enabled=True,
        widget_color=None,
        outline_translation=True,
        merge_points=True,
        crinkle=False,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
        **kwargs,
    ):
        """Clip a mesh using a box widget.

        Add a mesh to the scene with a box widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.box_clipped_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and clip.

        invert : bool, optional
            Flag on whether to flip/invert the clip.

        rotation_enabled : bool, optional
            If ``False``, the box widget cannot be rotated and is strictly
            orthogonal to the cartesian axes.

        widget_color : color_like, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        outline_translation : bool, optional
            If ``False``, the plane widget cannot be translated and is
            strictly placed at the given bounds.

        merge_points : bool, optional
            If ``True`` (default), coinciding points of independently
            defined mesh elements will be merged.

        crinkle : bool, optional
            Crinkle the clip by extracting the entire cells along the clip.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`BasePlotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the mesh.

        """
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        port = 1 if invert else 0

        if crinkle:
            mesh.cell_data['cell_ids'] = np.arange(mesh.n_cells)

        alg = _vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            alg.SetLocator(_vtk.vtkNonMergingPointLocator())
        alg.SetInputDataObject(mesh)
        alg.GenerateClippedOutputOn()

        box_clipped_mesh = pyvista.wrap(alg.GetOutput(port))
        self.box_clipped_meshes.append(box_clipped_mesh)

        def callback(planes):
            bounds = []
            for i in range(planes.GetNumberOfPlanes()):
                plane = planes.GetPlane(i)
                bounds.append(plane.GetNormal())
                bounds.append(plane.GetOrigin())

            alg.SetBoxClip(*bounds)
            alg.Update()
            clipped = pyvista.wrap(alg.GetOutput(port))
            if crinkle:
                clipped = mesh.extract_cells(np.unique(clipped.cell_data['cell_ids']))
            box_clipped_mesh.shallow_copy(clipped)

        self.add_box_widget(
            callback=callback,
            bounds=mesh.bounds,
            factor=1.25,
            rotation_enabled=rotation_enabled,
            use_planes=True,
            color=widget_color,
            outline_translation=outline_translation,
            interaction_event=interaction_event,
        )

        return self.add_mesh(box_clipped_mesh, reset_camera=False, **kwargs)

    def add_plane_widget(
        self,
        callback,
        normal='x',
        origin=None,
        bounds=None,
        factor=1.25,
        color=None,
        assign_to_axis=None,
        tubing=False,
        outline_translation=False,
        origin_translation=True,
        implicit=True,
        pass_widget=False,
        test_callback=True,
        normal_rotation=True,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
    ):
        """Add a plane widget to the scene.

        This is useless without a callback function. You can pass a
        callable function that takes two arguments, the normal and
        origin of the plane in that order output from this widget, and
        performs a task with that plane.

        Parameters
        ----------
        callback : callable
            The method called every time the plane is updated. Takes
            two arguments, the normal and origin of the plane in that
            order.

        normal : str or tuple(float)
            The starting normal vector of the plane.

        origin : tuple(float)
            The starting coordinate of the center of the plane.

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing.

        color : color_like, optional
            Either a string, rgb list, or hex color string.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given
            axis: options are ``(0, 'x')``, ``(1, 'y')``, or ``(2,
            'z')``.

        tubing : bool, optional
            When using an implicit plane wiget, this controls whether
            or not tubing is shown around the plane's boundaries.

        outline_translation : bool, optional
            If ``False``, the plane widget cannot be translated and is
            strictly placed at the given bounds. Only valid when using
            an implicit plane.

        origin_translation : bool, optional
            If ``False``, the plane widget cannot be translated by its
            origin and is strictly placed at the given origin. Only
            valid when using an implicit plane.

        implicit : bool, optional
            When ``True``, a ``vtkImplicitPlaneWidget`` is used and
            when ``False``, a ``vtkPlaneWidget`` is used.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        test_callback : bool, optional
            If ``True``, run the callback function after the widget is
            created.

        normal_rotation : bool, optional
            Set the opacity of the normal vector arrow to 0 such that
            it is effectively disabled. This prevents the user from
            rotating the normal. This is forced to ``False`` when
            ``assign_to_axis`` is set.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        Returns
        -------
        vtk.vtkImplicitPlaneWidget or vtk.vtkPlaneWidget
            Plane widget.

        """
        if origin is None:
            origin = self.center
        if bounds is None:
            bounds = self.bounds

        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]

        color = Color(color, default_color=pyvista.global_theme.font.color)

        if assign_to_axis:
            normal_rotation = False

        def _the_callback(widget, event_id):
            the_plane = _vtk.vtkPlane()
            widget.GetPlane(the_plane)
            normal = the_plane.GetNormal()
            origin = the_plane.GetOrigin()
            if callable(callback):
                if pass_widget:
                    try_callback(callback, normal, origin, widget)
                else:
                    try_callback(callback, normal, origin)
            return

        if implicit:
            plane_widget = _vtk.vtkImplicitPlaneWidget()
            plane_widget.GetNormalProperty().SetColor(color.float_rgb)
            plane_widget.GetOutlineProperty().SetColor(color.float_rgb)
            plane_widget.GetOutlineProperty().SetColor(color.float_rgb)
            plane_widget.SetTubing(tubing)
            plane_widget.SetOutlineTranslation(outline_translation)
            plane_widget.SetOriginTranslation(origin_translation)

            _start_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(True)
            _stop_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(False)

            plane_widget.SetDrawPlane(False)
            plane_widget.AddObserver(_vtk.vtkCommand.StartInteractionEvent, _start_interact)
            plane_widget.AddObserver(interaction_event, _stop_interact)
            plane_widget.SetPlaceFactor(factor)
            plane_widget.PlaceWidget(bounds)
            plane_widget.SetOrigin(origin)

            if not normal_rotation:
                plane_widget.GetNormalProperty().SetOpacity(0)

        else:
            # Position of the small plane
            source = _vtk.vtkPlaneSource()
            source.SetNormal(normal)
            source.SetCenter(origin)
            source.SetPoint1(
                origin[0] + (bounds[1] - bounds[0]) * 0.01,
                origin[1] - (bounds[3] - bounds[2]) * 0.01,
                origin[2],
            )
            source.SetPoint2(
                origin[0] - (bounds[1] - bounds[0]) * 0.01,
                origin[1] + (bounds[3] - bounds[2]) * 0.01,
                origin[2],
            )
            source.Update()
            plane_widget = _vtk.vtkPlaneWidget()
            plane_widget.SetHandleSize(0.01)
            # Position of the widget
            plane_widget.SetInputData(source.GetOutput())
            plane_widget.SetRepresentationToOutline()
            plane_widget.SetPlaceFactor(factor)
            plane_widget.PlaceWidget(bounds)
            plane_widget.SetCenter(origin)  # Necessary
            plane_widget.GetPlaneProperty().SetColor(color.float_rgb)  # self.C_LOT[fn])
            plane_widget.GetHandleProperty().SetColor(color.float_rgb)

            if not normal_rotation:
                plane_widget.GetHandleProperty().SetOpacity(0)

        plane_widget.GetPlaneProperty().SetOpacity(0.5)
        plane_widget.SetInteractor(self.iren.interactor)
        plane_widget.SetCurrentRenderer(self.renderer)

        if assign_to_axis:
            # Note that normal_rotation was forced to False
            if assign_to_axis in [0, "x", "X"]:
                plane_widget.NormalToXAxisOn()
                plane_widget.SetNormal(NORMALS["x"])
            elif assign_to_axis in [1, "y", "Y"]:
                plane_widget.NormalToYAxisOn()
                plane_widget.SetNormal(NORMALS["y"])
            elif assign_to_axis in [2, "z", "Z"]:
                plane_widget.NormalToZAxisOn()
                plane_widget.SetNormal(NORMALS["z"])
            else:
                raise RuntimeError("assign_to_axis not understood")
        else:
            plane_widget.SetNormal(normal)

        plane_widget.Modified()
        plane_widget.UpdatePlacement()
        plane_widget.On()
        plane_widget.AddObserver(interaction_event, _the_callback)
        if test_callback:
            _the_callback(plane_widget, None)  # Trigger immediate update

        self.plane_widgets.append(plane_widget)
        return plane_widget

    def clear_plane_widgets(self):
        """Remove all of the plane widgets."""
        self.plane_widgets.clear()

    def add_mesh_clip_plane(
        self,
        mesh,
        normal='x',
        invert=False,
        widget_color=None,
        value=0.0,
        assign_to_axis=None,
        tubing=False,
        origin_translation=True,
        outline_translation=False,
        implicit=True,
        normal_rotation=True,
        crinkle=False,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
        origin=None,
        **kwargs,
    ):
        """Clip a mesh using a plane widget.

        Add a mesh to the scene with a plane widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.plane_clipped_meshes``
        attribute on the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and clip.

        normal : str or tuple(float), optional
            The starting normal vector of the plane.

        invert : bool, optional
            Flag on whether to flip/invert the clip.

        widget_color : color_like, optional
            Either a string, RGB list, or hex color string.

        value : float, optional
            Set the clipping value along the normal direction.
            The default value is 0.0.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given
            axis.  Options are ``(0, 'x')``, ``(1, 'y')``, or ``(2,
            'z')``.

        tubing : bool, optional
            When using an implicit plane wiget, this controls whether
            or not tubing is shown around the plane's boundaries.

        origin_translation : bool, optional
            If ``False``, the plane widget cannot be translated by its
            origin and is strictly placed at the given origin. Only
            valid when using an implicit plane.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is
            strictly placed at the given bounds.

        implicit : bool, optional
            When ``True``, a ``vtkImplicitPlaneWidget`` is used and
            when ``False``, a ``vtkPlaneWidget`` is used.

        normal_rotation : bool, optional
            Set the opacity of the normal vector arrow to 0 such that
            it is effectively disabled. This prevents the user from
            rotating the normal. This is forced to ``False`` when
            ``assign_to_axis`` is set.

        crinkle : bool, optional
            Crinkle the clip by extracting the entire cells along the clip.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        origin : tuple(float), optional
            The starting coordinate of the center of the plane.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`BasePlotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the mesh.

        """
        from pyvista.core.filters import _get_output  # avoids circular import

        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))
        if origin is None:
            origin = mesh.center

        self.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        if crinkle:
            mesh.cell_data['cell_ids'] = np.arange(0, mesh.n_cells, dtype=int)

        if isinstance(mesh, _vtk.vtkPolyData):
            alg = _vtk.vtkClipPolyData()
        # elif isinstance(mesh, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = _vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(mesh)  # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetInsideOut(invert)  # invert the clip if needed

        plane_clipped_mesh = _get_output(alg)
        self.plane_clipped_meshes.append(plane_clipped_mesh)

        def callback(normal, loc):
            function = generate_plane(normal, loc)
            alg.SetClipFunction(function)  # the implicit function
            alg.Update()  # Perform the Cut
            clipped = pyvista.wrap(alg.GetOutput())
            if crinkle:
                clipped = mesh.extract_cells(np.unique(clipped.cell_data['cell_ids']))
            plane_clipped_mesh.shallow_copy(clipped)

        self.add_plane_widget(
            callback=callback,
            bounds=mesh.bounds,
            factor=1.25,
            normal=normal,
            color=widget_color,
            tubing=tubing,
            assign_to_axis=assign_to_axis,
            origin_translation=origin_translation,
            outline_translation=outline_translation,
            implicit=implicit,
            origin=origin,
            normal_rotation=normal_rotation,
            interaction_event=interaction_event,
        )

        return self.add_mesh(plane_clipped_mesh, **kwargs)

    def add_mesh_slice(
        self,
        mesh,
        normal='x',
        generate_triangles=False,
        widget_color=None,
        assign_to_axis=None,
        tubing=False,
        origin_translation=True,
        outline_translation=False,
        implicit=True,
        normal_rotation=True,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
        origin=None,
        **kwargs,
    ):
        """Slice a mesh using a plane widget.

        Add a mesh to the scene with a plane widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.plane_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and slice.

        normal : str or tuple(float), optional
            The starting normal vector of the plane.

        generate_triangles : bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        widget_color : color_like, optional
            Either a string, RGB sequence, or hex color string.  Defaults
            to ``'white'``.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given axis:
            options are (0, 'x'), (1, 'y'), or (2, 'z').

        tubing : bool, optional
            When using an implicit plane wiget, this controls whether or not
            tubing is shown around the plane's boundaries.

        origin_translation : bool, optional
            If ``False``, the plane widget cannot be translated by its origin
            and is strictly placed at the given origin. Only valid when using
            an implicit plane.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is strictly
            placed at the given bounds.

        implicit : bool, optional
            When ``True``, a ``vtkImplicitPlaneWidget`` is used and when
            ``False``, a ``vtkPlaneWidget`` is used.

        normal_rotation : bool, optional
            Set the opacity of the normal vector arrow to 0 such that it is
            effectively disabled. This prevents the user from rotating the
            normal. This is forced to ``False`` when ``assign_to_axis`` is set.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        origin : tuple(float), optional
            The starting coordinate of the center of the plane.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`BasePlotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the mesh.

        """
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))
        if origin is None:
            origin = mesh.center

        self.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(mesh)  # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        plane_sliced_mesh = pyvista.wrap(alg.GetOutput())
        self.plane_sliced_meshes.append(plane_sliced_mesh)

        def callback(normal, origin):
            # create the plane for clipping
            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane)  # the cutter to use the plane we made
            alg.Update()  # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())

        self.add_plane_widget(
            callback=callback,
            bounds=mesh.bounds,
            factor=1.25,
            normal=normal,
            color=widget_color,
            tubing=tubing,
            assign_to_axis=assign_to_axis,
            origin_translation=origin_translation,
            outline_translation=outline_translation,
            implicit=implicit,
            origin=origin,
            normal_rotation=normal_rotation,
            interaction_event=interaction_event,
        )

        return self.add_mesh(plane_sliced_mesh, **kwargs)

    def add_mesh_slice_orthogonal(
        self,
        mesh,
        generate_triangles=False,
        widget_color=None,
        tubing=False,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
        **kwargs,
    ):
        """Slice a mesh with three interactive planes.

        Adds three interactive plane slicing widgets for orthogonal slicing
        along each cartesian axis.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and threshold.

        generate_triangles : bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        widget_color : color_like, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        tubing : bool, optional
            When using an implicit plane wiget, this controls whether or not
            tubing is shown around the plane's boundaries.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`BasePlotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        list
            List of vtk.vtkActor(s).

        """
        actors = []
        name = kwargs.pop("name", None)
        for ax in ["x", "y", "z"]:
            axkwargs = kwargs.copy()
            if name:
                axkwargs["name"] = f"{name}-{ax}"
            a = self.add_mesh_slice(
                mesh,
                assign_to_axis=ax,
                origin_translation=False,
                outline_translation=False,
                generate_triangles=generate_triangles,
                widget_color=widget_color,
                tubing=tubing,
                interaction_event=interaction_event,
                **axkwargs,
            )
            actors.append(a)

        return actors

    def add_line_widget(
        self,
        callback,
        bounds=None,
        factor=1.25,
        resolution=100,
        color=None,
        use_vertices=False,
        pass_widget=False,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
    ):
        """Add a line widget to the scene.

        This is useless without a callback function. You can pass a
        callable function that takes a single argument, the PolyData
        line output from this widget, and performs a task with that
        line.

        Parameters
        ----------
        callback : callable
            The method called every time the line is updated. This has
            two options: Take a single argument, the ``PolyData`` line
            (default) or if ``use_vertices=True``, then it can take
            two arguments of the coordinates of the line's end points.

        bounds : tuple(float), optional
            Length 6 tuple of the bounding box where the widget is
            placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing.

        resolution : int, optional
            The number of points in the line created.

        color : color_like, optional, defaults to white
            Either a string, rgb sequence, or hex color string.

        use_vertices : bool, optional
            Changes the arguments of the callback method to take the end
            points of the line instead of a PolyData object.

        pass_widget : boollist
            If ``True``, the widget will be passed as the last
            argument of the callback.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        Returns
        -------
        vtk.vtkLineWidget
            Created line widget.

        """
        if bounds is None:
            bounds = self.bounds

        color = Color(color, default_color=pyvista.global_theme.font.color)

        def _the_callback(widget, event_id):
            pointa = widget.GetPoint1()
            pointb = widget.GetPoint2()
            if callable(callback):
                if use_vertices:
                    args = [pointa, pointb]
                else:
                    the_line = pyvista.Line(pointa, pointb, resolution=resolution)
                    args = [the_line]
                if pass_widget:
                    args.append(widget)
                try_callback(callback, *args)

        line_widget = _vtk.vtkLineWidget()
        line_widget.GetLineProperty().SetColor(color.float_rgb)
        line_widget.SetInteractor(self.iren.interactor)
        line_widget.SetCurrentRenderer(self.renderer)
        line_widget.SetPlaceFactor(factor)
        line_widget.PlaceWidget(bounds)
        line_widget.SetResolution(resolution)
        line_widget.Modified()
        line_widget.On()
        line_widget.AddObserver(interaction_event, _the_callback)
        _the_callback(line_widget, None)

        self.line_widgets.append(line_widget)
        return line_widget

    def clear_line_widgets(self):
        """Remove all of the line widgets."""
        self.line_widgets.clear()

    def add_text_slider_widget(
        self,
        callback,
        data,
        value=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        color=None,
        event_type='end',
        style=None,
    ):
        """Add a text slider bar widget.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the value of this slider widget,
        and performs a task with that value.

        Parameters
        ----------
        callback : callable
            The method called every time the slider is updated. This should take
            a single parameter: the float value of the slider.

        data : list
            The list of possible values displayed on the slider bar.

        value : float, optional
            The starting value of the slider.

        pointa : tuple(float), optional
            The relative coordinates of the left point of the slider on the
            display port.

        pointb : tuple(float), optional
            The relative coordinates of the right point of the slider on the
            display port.

        color : color_like, optional
            Either a string, RGB list, or hex color string.  Defaults
            to :attr:`pyvista.global_theme.font.color
            <pyvista.themes._Font.color>`.

        event_type : str, optional
            Either ``'start'``, ``'end'`` or ``'always'``, this
            defines how often the slider interacts with the callback.

        style : str, optional
            The name of the slider style. The list of available styles
            are in ``pyvista.global_theme.slider_styles``. Defaults to
            ``None``.

        Returns
        -------
        vtk.vtkSliderWidget
            The VTK slider widget configured to display text.

        """
        if not isinstance(data, list):
            raise TypeError(
                f"The `data` parameter must be a list but {type(data).__name__} was passed instead"
            )
        n_states = len(data)
        if n_states == 0:
            raise ValueError("The input list of values is empty")
        delta = (n_states - 1) / float(n_states)
        # avoid division by zero in case there is only one element
        delta = 1 if delta == 0 else delta

        def _the_callback(value):
            if isinstance(value, float):
                idx = int(value / delta)
                # handle limit index
                if idx == n_states:
                    idx = n_states - 1
                if callable(callback):
                    try_callback(callback, data[idx])
            return

        slider_widget = self.add_slider_widget(
            callback=_the_callback,
            rng=[0, n_states - 1],
            value=value,
            pointa=pointa,
            pointb=pointb,
            color=color,
            event_type=event_type,
            style=style,
        )
        slider_rep = slider_widget.GetRepresentation()
        slider_rep.ShowSliderLabelOff()

        def title_callback(widget, event):
            value = widget.GetRepresentation().GetValue()
            idx = int(value / delta)
            # handle limit index
            if idx == n_states:
                idx = n_states - 1
            slider_rep.SetTitleText(data[idx])

        if event_type == 'start':
            slider_widget.AddObserver(_vtk.vtkCommand.StartInteractionEvent, title_callback)
        elif event_type == 'end':
            slider_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, title_callback)
        elif event_type == 'always':
            slider_widget.AddObserver(_vtk.vtkCommand.InteractionEvent, title_callback)
        else:
            raise ValueError(
                "Expected value for `event_type` is 'start',"
                f" 'end' or 'always': {event_type} was given."
            )
        title_callback(slider_widget, None)
        return slider_widget

    def add_slider_widget(
        self,
        callback,
        rng,
        value=None,
        title=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        color=None,
        pass_widget=False,
        event_type='end',
        style=None,
        title_height=0.03,
        title_opacity=1.0,
        title_color=None,
        fmt=None,
        slider_width=None,
        tube_width=None,
    ):
        """Add a slider bar widget.

        This is useless without a callback function. You can pass a
        callable function that takes a single argument, the value of
        this slider widget, and performs a task with that value.

        Parameters
        ----------
        callback : callable
            The method called every time the slider is updated. This
            should take a single parameter: the float value of the
            slider.

        rng : tuple(float)
            Length two tuple of the minimum and maximum ranges of the
            slider.

        value : float, optional
            The starting value of the slider.

        title : str, optional
            The string label of the slider widget.

        pointa : tuple(float), optional
            The relative coordinates of the left point of the slider
            on the display port.

        pointb : tuple(float), optional
            The relative coordinates of the right point of the slider
            on the display port.

        color : color_like, optional
            Either a string, RGB list, or hex color string.  Defaults
            to :attr:`pyvista.global_theme.font.color
            <pyvista.themes._Font.color>`.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        event_type : str, optional
            Either ``'start'``, ``'end'`` or ``'always'``, this
            defines how often the slider interacts with the callback.

        style : str, optional
            The name of the slider style. The list of available styles
            are in ``pyvista.global_theme.slider_styles``. Defaults to
            ``None``.

        title_height : float, optional
            Relative height of the title as compared to the length of
            the slider.

        title_opacity : float, optional
            Opacity of title. Defaults to 1.0.

        title_color : color_like, optional
            Either a string, RGB sequence, or hex color string.  Defaults
            to the value given in ``color``.

        fmt : str, optional
            String formatter used to format numerical data. Defaults
            to ``None``.

        slider_width : float, optional
            Normalized width of the slider. Defaults to the theme's slider width.

        tube_width : float, optional
            Normalized width of the tube. Defaults to the theme's tube width.

        Returns
        -------
        vtk.vtkSliderWidget
            Slider widget.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> def create_mesh(value):
        ...     res = int(value)
        ...     sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
        ...     pl.add_mesh(sphere, name="sphere", show_edges=True)
        >>> slider = pl.add_slider_widget(
        ...     create_mesh,
        ...     [5, 100],
        ...     title="Resolution",
        ...     title_opacity=0.5,
        ...     title_color="red",
        ...     fmt="%0.9f",
        ...     title_height=0.08,
        ... )
        >>> pl.show()
        """
        if self.iren is None:
            raise RuntimeError('Cannot add a widget to a closed plotter.')

        if value is None:
            value = ((rng[1] - rng[0]) / 2) + rng[0]

        color = Color(color, default_color=pyvista.global_theme.font.color)
        title_color = Color(title_color, default_color=color)

        if fmt is None:
            fmt = pyvista.global_theme.font.fmt

        def normalize(point, viewport):
            return (point[0] * (viewport[2] - viewport[0]), point[1] * (viewport[3] - viewport[1]))

        pointa = normalize(pointa, self.renderer.GetViewport())
        pointb = normalize(pointb, self.renderer.GetViewport())

        slider_rep = _vtk.vtkSliderRepresentation2D()
        slider_rep.SetPickable(False)
        slider_rep.SetMinimumValue(rng[0])
        slider_rep.SetMaximumValue(rng[1])
        slider_rep.SetValue(value)
        slider_rep.SetTitleText(title)
        slider_rep.GetTitleProperty().SetColor(color.float_rgb)
        slider_rep.GetSliderProperty().SetColor(color.float_rgb)
        slider_rep.GetCapProperty().SetColor(color.float_rgb)
        slider_rep.GetLabelProperty().SetColor(color.float_rgb)
        slider_rep.GetTubeProperty().SetColor(color.float_rgb)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(pointa[0], pointa[1])
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(pointb[0], pointb[1])
        slider_rep.SetSliderLength(0.05)
        slider_rep.SetSliderWidth(0.05)
        slider_rep.SetEndCapLength(0.01)

        if style is not None:
            if not isinstance(style, str):
                raise TypeError(
                    f"Expected type for ``style`` is str but {type(style).__name__} was given."
                )
            slider_style = getattr(pyvista.global_theme.slider_styles, style)
            slider_rep.SetSliderLength(slider_style.slider_length)
            slider_rep.SetSliderWidth(slider_style.slider_width)
            slider_rep.GetSliderProperty().SetColor(slider_style.slider_color.float_rgb)
            slider_rep.SetTubeWidth(slider_style.tube_width)
            slider_rep.GetTubeProperty().SetColor(slider_style.tube_color.float_rgb)
            slider_rep.GetCapProperty().SetOpacity(slider_style.cap_opacity)
            slider_rep.SetEndCapLength(slider_style.cap_length)
            slider_rep.SetEndCapWidth(slider_style.cap_width)

        if slider_width is not None:
            slider_rep.SetSliderWidth(slider_width)
        if tube_width is not None:
            slider_rep.SetTubeWidth(tube_width)

        def _the_callback(widget, event):
            value = widget.GetRepresentation().GetValue()
            if callable(callback):
                if pass_widget:
                    try_callback(callback, value, widget)
                else:
                    try_callback(callback, value)
            return

        slider_widget = _vtk.vtkSliderWidget()
        slider_widget.SetInteractor(self.iren.interactor)
        slider_widget.SetCurrentRenderer(self.renderer)
        slider_widget.SetRepresentation(slider_rep)
        slider_widget.GetRepresentation().SetTitleHeight(title_height)
        slider_widget.GetRepresentation().GetTitleProperty().SetOpacity(title_opacity)
        slider_widget.GetRepresentation().GetTitleProperty().SetColor(title_color.float_rgb)
        if fmt is not None:
            slider_widget.GetRepresentation().SetLabelFormat(fmt)
        slider_widget.On()
        if not isinstance(event_type, str):
            raise TypeError(f"Expected type for `event_type` is str: {type(event_type)} was given.")
        if event_type == 'start':
            slider_widget.AddObserver(_vtk.vtkCommand.StartInteractionEvent, _the_callback)
        elif event_type == 'end':
            slider_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
        elif event_type == 'always':
            slider_widget.AddObserver(_vtk.vtkCommand.InteractionEvent, _the_callback)
        else:
            raise ValueError(
                "Expected value for `event_type` is 'start',"
                f" 'end' or 'always': {event_type} was given."
            )
        _the_callback(slider_widget, None)

        self.slider_widgets.append(slider_widget)
        return slider_widget

    def clear_slider_widgets(self):
        """Remove all of the slider widgets."""
        self.slider_widgets.clear()

    def add_mesh_threshold(
        self,
        mesh,
        scalars=None,
        invert=False,
        widget_color=None,
        preference='cell',
        title=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        continuous=False,
        **kwargs,
    ):
        """Apply a threshold on a mesh with a slider.

        Add a mesh to the scene with a slider widget that is used to
        threshold the mesh interactively.

        The threshold mesh is saved to the ``.threshold_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and threshold.

        scalars : str, optional
            The string name of the scalars on the mesh to threshold and display.

        invert : bool, optional
            Invert (flip) the threshold.

        widget_color : color_like, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        preference : str, optional
            When ``mesh.n_points == mesh.n_cells`` and setting
            scalars, this parameter sets how the scalars will be
            mapped to the mesh.  Default ``'cell'``, causes the
            scalars to be associated with the mesh cells.  Can be
            either ``'point'`` or ``'cell'``.

        title : str, optional
            The string label of the slider widget.

        pointa : sequence, optional
            The relative coordinates of the left point of the slider
            on the display port.

        pointb : sequence, optional
            The relative coordinates of the right point of the slider
            on the display port.

        continuous : bool, optional
            If this is enabled (default is ``False``), use the continuous
            interval ``[minimum cell scalar, maximum cell scalar]``
            to intersect the threshold bound, rather than the set of
            discrete scalar values from the vertices.

        **kwargs : dict, optional
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the mesh.

        """
        # avoid circular import
        from ..core.filters.data_set import _set_threshold_limit

        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for threshold widget.')
        name = kwargs.get('name', mesh.memory_address)
        if scalars is None:
            field, scalars = mesh.active_scalars_info
        arr = get_array(mesh, scalars, preference=preference)
        if arr is None:
            raise ValueError('No arrays present to threshold.')
        field = get_array_association(mesh, scalars, preference=preference)

        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars
        mesh.set_active_scalars(scalars)

        self.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        alg = _vtk.vtkThreshold()
        alg.SetInputDataObject(mesh)
        alg.SetInputArrayToProcess(
            0, 0, 0, field.value, scalars
        )  # args: (idx, port, connection, field, name)
        alg.SetUseContinuousCellRange(continuous)

        threshold_mesh = pyvista.wrap(alg.GetOutput())
        self.threshold_meshes.append(threshold_mesh)

        def callback(value):
            _set_threshold_limit(alg, value, invert)
            alg.Update()
            threshold_mesh.shallow_copy(alg.GetOutput())

        self.add_slider_widget(
            callback=callback,
            rng=rng,
            title=title,
            color=widget_color,
            pointa=pointa,
            pointb=pointb,
        )

        kwargs.setdefault("reset_camera", False)
        return self.add_mesh(threshold_mesh, scalars=scalars, **kwargs)

    def add_mesh_isovalue(
        self,
        mesh,
        scalars=None,
        compute_normals=False,
        compute_gradients=False,
        compute_scalars=True,
        preference='point',
        title=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        widget_color=None,
        **kwargs,
    ):
        """Create a contour of a mesh with a slider.

        Add a mesh to the scene with a slider widget that is used to
        contour at an isovalue of the *point* data on the mesh
        interactively.

        The isovalue mesh is saved to the ``.isovalue_meshes``
        attribute on the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and contour.

        scalars : str, optional
            The string name of the scalars on the mesh to contour and display.

        compute_normals : bool, optional
            Enable or disable the computation of normals.  If the
            output data will be processed by filters that modify
            topology or geometry, it may be wise to disable computing
            normals.

        compute_gradients : bool, optional
            Enable or disable the computation of gradients.  If the
            output data will be processed by filters that modify
            topology or geometry, it may be wise to disable computing
            gradients.

        compute_scalars : bool, optional
            Enable or disable the computation of scalars.

        preference : str, optional
            When ``mesh.n_points == mesh.n_cells`` and setting
            scalars, this parameter sets how the scalars will be
            mapped to the mesh.  Default ``'point'``, causes the
            scalars will be associated with the mesh points.  Can be
            either ``'point'`` or ``'cell'``.

        title : str, optional
            The string label of the slider widget.

        pointa : sequence, optional
            The relative coordinates of the left point of the slider
            on the display port.

        pointb : sequence
            The relative coordinates of the right point of the slider
            on the display port.

        widget_color : color_like, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`BasePlotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the mesh.

        """
        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for this widget.')
        name = kwargs.get('name', mesh.memory_address)
        # set the array to contour on
        if mesh.n_arrays < 1:
            raise ValueError('Input dataset for the contour filter must have data arrays.')
        if scalars is None:
            field, scalars = mesh.active_scalars_info
        else:
            field = get_array_association(mesh, scalars, preference=preference)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != pyvista.FieldAssociation.POINT:
            raise TypeError(
                f'Contour filter only works on Point data. Array ({scalars}) is in the Cell data.'
            )

        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars
        mesh.set_active_scalars(scalars)

        alg = _vtk.vtkContourFilter()
        alg.SetInputDataObject(mesh)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
        alg.SetNumberOfContours(1)  # Only one contour level

        self.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        isovalue_mesh = pyvista.wrap(alg.GetOutput())
        self.isovalue_meshes.append(isovalue_mesh)

        def callback(value):
            alg.SetValue(0, value)
            alg.Update()
            isovalue_mesh.shallow_copy(alg.GetOutput())

        self.add_slider_widget(
            callback=callback,
            rng=rng,
            title=title,
            color=widget_color,
            pointa=pointa,
            pointb=pointb,
        )

        kwargs.setdefault("reset_camera", False)
        return self.add_mesh(isovalue_mesh, scalars=scalars, **kwargs)

    def add_spline_widget(
        self,
        callback,
        bounds=None,
        factor=1.25,
        n_handles=5,
        resolution=25,
        color="yellow",
        show_ribbon=False,
        ribbon_color="pink",
        ribbon_opacity=0.5,
        pass_widget=False,
        closed=False,
        initial_points=None,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
    ):
        """Create and add a spline widget to the scene.

        Use the bounds argument to place this widget. Several "handles" are
        used to control a parametric function for building this spline. Click
        directly on the line to translate the widget.

        Parameters
        ----------
        callback : callable
            The method called every time the spline is updated. This passes a
            :class:`pyvista.PolyData` object to the callback function of the
            generated spline.

        bounds : tuple(float), optional
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing.

        n_handles : int, optional
            The number of interactive spheres to control the spline's
            parametric function.

        resolution : int, optional
            The number of points in the spline created between all the handles.

        color : color_like, optional
            Either a string, RGB sequence, or hex color string.

        show_ribbon : bool, optional
            If ``True``, the poly plane used for slicing will also be shown.

        ribbon_color : color_like, optional
            Color of the ribbon.  Either a string, RGB sequence, or
            hex color string.

        ribbon_opacity : float, optional
            Opacity of ribbon. Defaults to 1.0 and must be between
            ``[0, 1]``.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last argument of the
            callback.

        closed : bool, optional
            Make the spline a closed loop.

        initial_points : sequence, optional
            The points to initialize the widget placement. Must have
            same number of elements as ``n_handles``. If the first and
            last point are the same, this will be a closed loop
            spline.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        Returns
        -------
        vtk.vtkSplineWidget
            The newly created spline widget.

        Notes
        -----
        This widget has trouble displaying certain colors. Use only simple
        colors (white, black, yellow).

        """
        if initial_points is not None and len(initial_points) != n_handles:
            raise ValueError("`initial_points` must be length `n_handles`.")

        color = Color(color, default_color=pyvista.global_theme.color)

        if bounds is None:
            bounds = self.bounds

        ribbon = pyvista.PolyData()

        def _the_callback(widget, event_id):
            para_source = _vtk.vtkParametricFunctionSource()
            para_source.SetParametricFunction(widget.GetParametricSpline())
            para_source.Update()
            polyline = pyvista.wrap(para_source.GetOutput())
            ribbon.shallow_copy(polyline.ribbon(normal=(0, 0, 1), angle=90.0))
            if callable(callback):
                if pass_widget:
                    try_callback(callback, polyline, widget)
                else:
                    try_callback(callback, polyline)
            return

        spline_widget = _vtk.vtkSplineWidget()
        spline_widget.GetLineProperty().SetColor(color.float_rgb)
        spline_widget.SetNumberOfHandles(n_handles)
        spline_widget.SetInteractor(self.iren.interactor)
        spline_widget.SetCurrentRenderer(self.renderer)
        spline_widget.SetPlaceFactor(factor)
        spline_widget.PlaceWidget(bounds)
        spline_widget.SetResolution(resolution)
        if initial_points is not None:
            spline_widget.InitializeHandles(pyvista.vtk_points((initial_points)))
        else:
            spline_widget.SetClosed(closed)
        spline_widget.Modified()
        spline_widget.On()
        spline_widget.AddObserver(interaction_event, _the_callback)
        _the_callback(spline_widget, None)

        if show_ribbon:
            self.add_mesh(ribbon, color=ribbon_color, opacity=ribbon_opacity)

        self.spline_widgets.append(spline_widget)
        return spline_widget

    def clear_spline_widgets(self):
        """Remove all of the spline widgets."""
        self.spline_widgets.clear()

    def add_mesh_slice_spline(
        self,
        mesh,
        generate_triangles=False,
        n_handles=5,
        resolution=25,
        widget_color=None,
        show_ribbon=False,
        ribbon_color="pink",
        ribbon_opacity=0.5,
        initial_points=None,
        closed=False,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
        **kwargs,
    ):
        """Slice a mesh with a spline widget.

        Add a mesh to the scene with a spline widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.spline_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and slice along the spline.

        generate_triangles : bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        n_handles : int, optional
            The number of interactive spheres to control the spline's
            parametric function.

        resolution : int, optional
            The number of points to generate on the spline.

        widget_color : color_like, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        show_ribbon : bool, optional
            If ``True``, the poly plane used for slicing will also be shown.

        ribbon_color : color_like, optional
            Color of the ribbon.  Either a string, RGB sequence, or
            hex color string.

        ribbon_opacity : float, optional
            Opacity of ribbon. Defaults to 1.0 and must be between
            ``[0, 1]``.

        initial_points : sequence, optional
            The points to initialize the widget placement. Must have same
            number of elements as ``n_handles``. If the first and last point
            are the same, this will be a closed loop spline.

        closed : bool, optional
            Make the spline a closed loop.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`BasePlotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the mesh.

        """
        name = kwargs.get('name', None)
        if name is None:
            name = mesh.memory_address
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(mesh)  # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        spline_sliced_mesh = pyvista.wrap(alg.GetOutput())
        self.spline_sliced_meshes.append(spline_sliced_mesh)

        def callback(spline):
            polyline = spline.GetCell(0)
            # create the plane for clipping
            polyplane = _vtk.vtkPolyPlane()
            polyplane.SetPolyLine(polyline)
            alg.SetCutFunction(polyplane)  # the cutter to use the poly planes
            alg.Update()  # Perform the Cut
            spline_sliced_mesh.shallow_copy(alg.GetOutput())

        self.add_spline_widget(
            callback=callback,
            bounds=mesh.bounds,
            factor=1.25,
            color=widget_color,
            n_handles=n_handles,
            resolution=resolution,
            show_ribbon=show_ribbon,
            ribbon_color=ribbon_color,
            ribbon_opacity=ribbon_opacity,
            initial_points=initial_points,
            closed=closed,
            interaction_event=interaction_event,
        )

        return self.add_mesh(spline_sliced_mesh, **kwargs)

    def add_sphere_widget(
        self,
        callback,
        center=(0, 0, 0),
        radius=0.5,
        theta_resolution=30,
        phi_resolution=30,
        color=None,
        style="surface",
        selected_color="pink",
        indices=None,
        pass_widget=False,
        test_callback=True,
        interaction_event=_vtk.vtkCommand.EndInteractionEvent,
    ):
        """Add one or many sphere widgets to a scene.

        Use a sphere widget to control a vertex location.

        Parameters
        ----------
        callback : callable
            The function to call back when the widget is modified. It
            takes a single argument: the center of the sphere as an
            XYZ coordinate (a 3-length sequence).  If multiple centers
            are passed in the ``center`` parameter, the callback must
            also accept an index of that widget.

        center : tuple(float), optional
            Length 3 array for the XYZ coordinate of the sphere's
            center when placing it in the scene. If more than one
            location is passed, then that many widgets will be added
            and the callback will also be passed the integer index of
            that widget.

        radius : float, optional
            The radius of the sphere.

        theta_resolution : int, optional
            Set the number of points in the longitude direction.

        phi_resolution : int, optional
            Set the number of points in the latitude direction.

        color : color_like, optional
            The color of the sphere's surface.  If multiple centers
            are passed, then this must be a list of colors.  Each
            color is either a string, rgb list, or hex color string.
            For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        style : str, optional
            Representation style: ``'surface'`` or ``'wireframe'``.

        selected_color : color_like, optional
            Color of the widget when selected during interaction.

        indices : sequence, optional
            Indices to assign the sphere widgets.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        test_callback : bool, optional
            If ``True``, run the callback function after the widget is
            created.

        interaction_event : vtk.vtkCommand.EventIds, optional
            The VTK interaction event to use for triggering the callback.

        Returns
        -------
        vtk.vtkSphereWidget
            The sphere widget.

        """
        if color is None:
            color = pyvista.global_theme.color.float_rgb
        selected_color = Color(selected_color)

        center = np.array(center)
        num = 1
        if center.ndim > 1:
            num = len(center)

        if isinstance(color, (list, tuple, np.ndarray)):
            if len(color) == num and not isinstance(color[0], float):
                colors = color
            else:
                colors = [color] * num
        else:
            colors = [color] * num

        def _the_callback(widget, event_id):
            point = widget.GetCenter()
            index = widget.WIDGET_INDEX
            if callable(callback):
                if num > 1:
                    args = [point, index]
                else:
                    args = [point]
                if pass_widget:
                    args.append(widget)
                try_callback(callback, *args)
            return

        if indices is None:
            indices = [x for x in range(num)]

        for i in range(num):
            if center.ndim > 1:
                loc = center[i]
            else:
                loc = center
            sphere_widget = _vtk.vtkSphereWidget()
            sphere_widget.WIDGET_INDEX = indices[i]  # Monkey patch the index
            if style in "wireframe":
                sphere_widget.SetRepresentationToWireframe()
            else:
                sphere_widget.SetRepresentationToSurface()
            sphere_widget.GetSphereProperty().SetColor(Color(colors[i]).float_rgb)
            sphere_widget.GetSelectedSphereProperty().SetColor(selected_color.float_rgb)
            sphere_widget.SetInteractor(self.iren.interactor)
            sphere_widget.SetCurrentRenderer(self.renderer)
            sphere_widget.SetRadius(radius)
            sphere_widget.SetCenter(loc)
            sphere_widget.SetThetaResolution(theta_resolution)
            sphere_widget.SetPhiResolution(phi_resolution)
            sphere_widget.Modified()
            sphere_widget.On()
            sphere_widget.AddObserver(interaction_event, _the_callback)
            self.sphere_widgets.append(sphere_widget)

        if test_callback is True:
            # Test call back in the last
            _the_callback(sphere_widget, None)
        if num > 1:
            return self.sphere_widgets

        return sphere_widget

    def clear_sphere_widgets(self):
        """Remove all of the sphere widgets."""
        self.sphere_widgets.clear()

    def add_checkbox_button_widget(
        self,
        callback,
        value=False,
        position=(10.0, 10.0),
        size=50,
        border_size=5,
        color_on='blue',
        color_off='grey',
        background_color='white',
    ):
        """Add a checkbox button widget to the scene.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the state of this button widget
        and performs a task with that value.

        Parameters
        ----------
        callback : callable
            The method called every time the button is clicked. This should take
            a single parameter: the bool value of the button.

        value : bool, optional
            The default state of the button.

        position : tuple(float), optional
            The absolute coordinates of the bottom left point of the button.

        size : int, optional
            The size of the button in number of pixels.

        border_size : int, optional
            The size of the borders of the button in pixels.

        color_on : color_like, optional
            The color used when the button is checked. Default is ``'blue'``.

        color_off : color_like, optional
            The color used when the button is not checked. Default is ``'grey'``.

        background_color : color_like, optional
            The background color of the button. Default is ``'white'``.

        Returns
        -------
        vtk.vtkButtonWidget
            The VTK button widget configured as a checkbox button.

        Examples
        --------
        The following example generates a static image of the widget.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> p = pv.Plotter()
        >>> actor = p.add_mesh(mesh)
        >>> def toggle_vis(flag):
        ...     actor.SetVisibility(flag)
        >>> _ = p.add_checkbox_button_widget(toggle_vis, value=True)
        >>> p.show()

        Download the interactive example at :ref:`checkbox_widget_example`.

        """
        if self.iren is None:  # pragma: no cover
            raise RuntimeError('Cannot add a widget to a closed plotter.')

        def create_button(color1, color2, color3, dims=(size, size, 1)):
            color1 = np.array(Color(color1).int_rgb)
            color2 = np.array(Color(color2).int_rgb)
            color3 = np.array(Color(color3).int_rgb)

            n_points = dims[0] * dims[1]
            button = pyvista.UniformGrid(dimensions=dims)
            arr = np.array([color1] * n_points).reshape(dims[0], dims[1], 3)  # fill with color1
            arr[1 : dims[0] - 1, 1 : dims[1] - 1] = color2  # apply color2
            arr[
                border_size : dims[0] - border_size, border_size : dims[1] - border_size
            ] = color3  # apply color3
            button.point_data['texture'] = arr.reshape(n_points, 3).astype(np.uint8)
            return button

        button_on = create_button(color_on, background_color, color_on)
        button_off = create_button(color_on, background_color, color_off)

        bounds = [position[0], position[0] + size, position[1], position[1] + size, 0.0, 0.0]

        button_rep = _vtk.vtkTexturedButtonRepresentation2D()
        button_rep.SetNumberOfStates(2)
        button_rep.SetState(value)
        button_rep.SetButtonTexture(0, button_off)
        button_rep.SetButtonTexture(1, button_on)
        button_rep.SetPlaceFactor(1)
        button_rep.PlaceWidget(bounds)

        button_widget = _vtk.vtkButtonWidget()
        button_widget.SetInteractor(self.iren.interactor)
        button_widget.SetRepresentation(button_rep)
        button_widget.SetCurrentRenderer(self.renderer)
        button_widget.On()

        def _the_callback(widget, event):
            state = widget.GetRepresentation().GetState()
            if callable(callback):
                try_callback(callback, bool(state))

        button_widget.AddObserver(_vtk.vtkCommand.StateChangedEvent, _the_callback)
        self.button_widgets.append(button_widget)
        return button_widget

    def add_camera_orientation_widget(self, animate=True, n_frames=20):
        """Add a camera orientation widget to the active renderer.

        .. note::
           This widget requires ``vtk>=9.1.0``.

        Parameters
        ----------
        animate : bool, optional
            Enable or disable jump-to-axis-view animation.
        n_frames : int, optional
            The number of frames to animate the jump-to-axis-viewpoint feature.

        Returns
        -------
        vtkCameraOrientationWidget
            Camera orientation widget.

        Examples
        --------
        Add a camera orientation widget to the scene.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(mesh, scalars=range(6), show_scalar_bar=False)
        >>> _ = plotter.add_camera_orientation_widget()
        >>> plotter.show()

        """
        widget = _vtk.lazy_vtkCameraOrientationWidget()
        widget.SetParentRenderer(self.renderer)
        widget.SetAnimate(animate)
        widget.SetAnimatorTotalFrames(n_frames)
        widget.On()
        self.camera_widgets.append(widget)
        return widget

    def clear_camera_widgets(self):
        """Remove all of the camera widgets."""
        self.camera_widgets.clear()

    def clear_button_widgets(self):
        """Remove all of the button widgets."""
        self.button_widgets.clear()

    def close(self):
        """Close the widgets."""
        self.clear_box_widgets()
        self.clear_plane_widgets()
        self.clear_line_widgets()
        self.clear_slider_widgets()
        self.clear_sphere_widgets()
        self.clear_spline_widgets()
        self.clear_button_widgets()
        self.clear_camera_widgets()
