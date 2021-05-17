"""Module dedicated to widgets."""

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import NORMALS, generate_plane, get_array, try_callback
from .tools import parse_color


class WidgetHelper:
    """An internal class to manage widgets.

    It also manages and other helper methods involving widgets.

    """

    def add_box_widget(self, callback, bounds=None, factor=1.25,
                       rotation_enabled=True, color=None, use_planes=False,
                       outline_translation=True, pass_widget=False):
        """Add a box widget to the scene.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the PolyData box output from
        this widget, and performs a task with that box.

        Parameters
        ----------
        callback : callable
            The method called every time the box is updated. This has two
            options: Take a single argument, the ``PolyData`` box (default) or
            if ``use_planes=True``, then it takes a single argument of the
            plane collection as a ``vtkPlanes`` object.

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing

        rotation_enabled : bool
            If ``False``, the box widget cannot be rotated and is strictly
            orthogonal to the cartesian axes.

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        use_planes : bool, optional
            Changes the arguments passed to the callback to the planes that
            make up the box.

        outline_translation : bool
            If ``False``, the box widget cannot be translated and is strictly
            placed at the given bounds.

        pass_widget : bool
            If true, the widget will be passed as the last argument of the
            callback

        """
        if not hasattr(self, "box_widgets"):
            self.box_widgets = []

        if bounds is None:
            bounds = self.bounds

        if color is None:
            color = pyvista.global_theme.font.color

        def _the_callback(box_widget, event_id):
            the_box = pyvista.PolyData()
            box_widget.GetPolyData(the_box)
            planes = _vtk.vtkPlanes()
            box_widget.GetPlanes(planes)
            if hasattr(callback, '__call__'):
                if use_planes:
                    args = [planes]
                else:
                    args = [the_box]
                if pass_widget:
                    args.append(box_widget)
                try_callback(callback, *args)
            return

        box_widget = _vtk.vtkBoxWidget()
        box_widget.GetOutlineProperty().SetColor(parse_color(color))
        box_widget.SetInteractor(self.iren.interactor)
        box_widget.SetCurrentRenderer(self.renderer)
        box_widget.SetPlaceFactor(factor)
        box_widget.SetRotationEnabled(rotation_enabled)
        box_widget.SetTranslationEnabled(outline_translation)
        box_widget.PlaceWidget(bounds)
        box_widget.On()
        box_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(box_widget, None)

        self.box_widgets.append(box_widget)
        return box_widget

    def clear_box_widgets(self):
        """Disable all of the box widgets."""
        if hasattr(self, 'box_widgets'):
            for widget in self.box_widgets:
                widget.Off()
            del self.box_widgets
        return

    def add_mesh_clip_box(self, mesh, invert=False, rotation_enabled=True,
                          widget_color=None, outline_translation=True,
                          **kwargs):
        """Clip a mesh using a box widget.

        Add a mesh to the scene with a box widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.box_clipped_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and clip

        invert : bool
            Flag on whether to flip/invert the clip

        rotation_enabled : bool
            If ``False``, the box widget cannot be rotated and is strictly
            orthogonal to the cartesian axes.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        """
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        port = 1 if invert else 0

        alg = _vtk.vtkBoxClipDataSet()
        alg.SetInputDataObject(mesh)
        alg.GenerateClippedOutputOn()

        if not hasattr(self, "box_clipped_meshes"):
            self.box_clipped_meshes = []
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
            box_clipped_mesh.shallow_copy(alg.GetOutput(port))

        self.add_box_widget(callback=callback, bounds=mesh.bounds,
                            factor=1.25, rotation_enabled=rotation_enabled,
                            use_planes=True, color=widget_color,
                            outline_translation=outline_translation)

        actor = self.add_mesh(box_clipped_mesh, reset_camera=False,
                              **kwargs)

        return actor

    def add_plane_widget(self, callback, normal='x', origin=None,
                         bounds=None, factor=1.25, color=None,
                         assign_to_axis=None, tubing=False,
                         outline_translation=False,
                         origin_translation=True, implicit=True,
                         pass_widget=False, test_callback=True,
                         normal_rotation=True):
        """Add a plane widget to the scene.

        This is useless without a callback function. You can pass a callable
        function that takes two arguments, the normal and origin of the plane
        in that order output from this widget, and performs a task with that
        plane.

        Parameters
        ----------
        callback : callable
            The method called every time the plane is updated. Takes two
            arguments, the normal and origin of the plane in that order.

        normal : str or tuple(float)
            The starting normal vector of the plane

        origin : tuple(float)
            The starting coordinate of the center of the place

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        assign_to_axis : str or int
            Assign the normal of the plane to be parallel with a given axis:
            options are (0, 'x'), (1, 'y'), or (2, 'z').

        tubing : bool
            When using an implicit plane wiget, this controls whether or not
            tubing is shown around the plane's boundaries.

        outline_translation : bool
            If ``False``, the plane widget cannot be translated and is strictly
            placed at the given bounds. Only valid when using an implicit
            plane.

        origin_translation : bool
            If ``False``, the plane widget cannot be translated by its origin
            and is strictly placed at the given origin. Only valid when using
            an implicit plane.

        implicit : bool
            When ``True``, a ``vtkImplicitPlaneWidget`` is used and when
            ``False``, a ``vtkPlaneWidget`` is used.

        pass_widget : bool
            If true, the widget will be passed as the last argument of the
            callback

        test_callback : bool
            If true, run the callback function after the widget is created.

        normal_rotation : bool
            Set the opacity of the normal vector arrow to 0 such that it is
            effectively disabled. This prevents the user from rotating the
            normal. This is forced to ``False`` when ``assign_to_axis`` is set.

        """
        if not hasattr(self, "plane_widgets"):
            self.plane_widgets = []

        if origin is None:
            origin = self.center
        if bounds is None:
            bounds = self.bounds

        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]

        if color is None:
            color = pyvista.global_theme.font.color

        if assign_to_axis:
            normal_rotation = False

        def _the_callback(widget, event_id):
            the_plane = _vtk.vtkPlane()
            widget.GetPlane(the_plane)
            normal = the_plane.GetNormal()
            origin = the_plane.GetOrigin()
            if hasattr(callback, '__call__'):
                if pass_widget:
                    try_callback(callback, normal, origin, widget)
                else:
                    try_callback(callback, normal, origin)
            return

        if implicit:
            plane_widget = _vtk.vtkImplicitPlaneWidget()
            plane_widget.GetNormalProperty().SetColor(parse_color(color))
            plane_widget.GetOutlineProperty().SetColor(parse_color(color))
            plane_widget.GetOutlineProperty().SetColor(parse_color(color))
            plane_widget.SetTubing(tubing)
            plane_widget.SetOutlineTranslation(outline_translation)
            plane_widget.SetOriginTranslation(origin_translation)

            _start_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(True)
            _stop_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(False)

            plane_widget.SetDrawPlane(False)
            plane_widget.AddObserver(_vtk.vtkCommand.StartInteractionEvent, _start_interact)
            plane_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _stop_interact)
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
            source.SetPoint1(origin[0] + (bounds[1] - bounds[0]) * 0.01,
                             origin[1] - (bounds[3] - bounds[2]) * 0.01,
                             origin[2])
            source.SetPoint2(origin[0] - (bounds[1] - bounds[0]) * 0.01,
                             origin[1] + (bounds[3] - bounds[2]) * 0.01,
                             origin[2])
            source.Update()
            plane_widget = _vtk.vtkPlaneWidget()
            plane_widget.SetHandleSize(.01)
            # Position of the widget
            plane_widget.SetInputData(source.GetOutput())
            plane_widget.SetRepresentationToOutline()
            plane_widget.SetPlaceFactor(factor)
            plane_widget.PlaceWidget(bounds)
            plane_widget.SetCenter(origin) # Necessary
            plane_widget.GetPlaneProperty().SetColor(parse_color(color))  # self.C_LOT[fn])
            plane_widget.GetHandleProperty().SetColor(parse_color(color))

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
        plane_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
        if test_callback:
            _the_callback(plane_widget, None) # Trigger immediate update

        self.plane_widgets.append(plane_widget)
        return plane_widget

    def clear_plane_widgets(self):
        """Disable all of the plane widgets."""
        if hasattr(self, 'plane_widgets'):
            for widget in self.plane_widgets:
                widget.Off()
            del self.plane_widgets
        return

    def add_mesh_clip_plane(self, mesh, normal='x', invert=False,
                            widget_color=None, value=0.0, assign_to_axis=None,
                            tubing=False, origin_translation=True,
                            outline_translation=False, implicit=True,
                            normal_rotation=True, **kwargs):
        """Clip a mesh using a plane widget.

        Add a mesh to the scene with a plane widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.plane_clipped_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and clip

        normal : str or tuple(float)
            The starting normal vector of the plane

        invert : bool
            Flag on whether to flip/invert the clip

        widget_color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        value : float, optional
            Set the clipping value along the normal direction.
            The default value is 0.0.

        assign_to_axis : str or int
            Assign the normal of the plane to be parallel with a given axis:
            options are (0, 'x'), (1, 'y'), or (2, 'z').

        tubing : bool
            When using an implicit plane wiget, this controls whether or not
            tubing is shown around the plane's boundaries.

        outline_translation : bool
            If ``False``, the plane widget cannot be translated and is strictly
            placed at the given bounds. Only valid when using an implicit
            plane.

        origin_translation : bool
            If ``False``, the plane widget cannot be translated by its origin
            and is strictly placed at the given origin. Only valid when using
            an implicit plane.

        implicit : bool
            When ``True``, a ``vtkImplicitPlaneWidget`` is used and when
            ``False``, a ``vtkPlaneWidget`` is used.

        normal_rotation : bool
            Set the opacity of the normal vector arrow to 0 such that it is
            effectively disabled. This prevents the user from rotating the
            normal. This is forced to ``False`` when ``assign_to_axis`` is set.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        """
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        if isinstance(mesh, _vtk.vtkPolyData):
            alg = _vtk.vtkClipPolyData()
        # elif isinstance(mesh, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = _vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetInsideOut(invert) # invert the clip if needed

        if not hasattr(self, "plane_clipped_meshes"):
            self.plane_clipped_meshes = []
        plane_clipped_mesh = pyvista.wrap(alg.GetOutput())
        self.plane_clipped_meshes.append(plane_clipped_mesh)

        def callback(normal, origin):
            function = generate_plane(normal, origin)
            alg.SetClipFunction(function) # the implicit function
            alg.Update() # Perform the Cut
            plane_clipped_mesh.shallow_copy(alg.GetOutput())

        self.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal=normal,
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=mesh.center,
                              normal_rotation=normal_rotation)

        actor = self.add_mesh(plane_clipped_mesh, **kwargs)

        return actor

    def add_mesh_slice(self, mesh, normal='x', generate_triangles=False,
                       widget_color=None, assign_to_axis=None,
                       tubing=False, origin_translation=True,
                       outline_translation=False, implicit=True,
                       normal_rotation=True, **kwargs):
        """Slice a mesh using a plane widget.

        Add a mesh to the scene with a plane widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.plane_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and slice

        normal : str or tuple(float)
            The starting normal vector of the plane

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        widget_color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        assign_to_axis : str or int
            Assign the normal of the plane to be parallel with a given axis:
            options are (0, 'x'), (1, 'y'), or (2, 'z').

        tubing : bool
            When using an implicit plane wiget, this controls whether or not
            tubing is shown around the plane's boundaries.

        outline_translation : bool
            If ``False``, the plane widget cannot be translated and is strictly
            placed at the given bounds. Only valid when using an implicit
            plane.

        origin_translation : bool
            If ``False``, the plane widget cannot be translated by its origin
            and is strictly placed at the given origin. Only valid when using
            an implicit plane.

        implicit : bool
            When ``True``, a ``vtkImplicitPlaneWidget`` is used and when
            ``False``, a ``vtkPlaneWidget`` is used.

        normal_rotation : bool
            Set the opacity of the normal vector arrow to 0 such that it is
            effectively disabled. This prevents the user from rotating the
            normal. This is forced to ``False`` when ``assign_to_axis`` is set.


        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        """
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = _vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        if not hasattr(self, "plane_sliced_meshes"):
            self.plane_sliced_meshes = []
        plane_sliced_mesh = pyvista.wrap(alg.GetOutput())
        self.plane_sliced_meshes.append(plane_sliced_mesh)

        def callback(normal, origin):
            # create the plane for clipping
            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())

        self.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal=normal,
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=mesh.center,
                              normal_rotation=normal_rotation)

        actor = self.add_mesh(plane_sliced_mesh, **kwargs)

        return actor

    def add_mesh_slice_orthogonal(self, mesh, generate_triangles=False,
                                  widget_color=None, tubing=False, **kwargs):
        """Slice a mesh with three interactive planes.

        Adds three interactive plane slicing widgets for orthogonal slicing
        along each cartesian axis.

        """
        actors = []
        for ax in ["x", "y", "z"]:
            a = self.add_mesh_slice(mesh, assign_to_axis=ax,
                                    origin_translation=False,
                                    outline_translation=False,
                                    generate_triangles=generate_triangles,
                                    widget_color=widget_color,
                                    tubing=tubing, **kwargs)
            actors.append(a)

        return actors

    def add_line_widget(self, callback, bounds=None, factor=1.25,
                        resolution=100, color=None, use_vertices=False,
                        pass_widget=False):
        """Add a line widget to the scene.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the PolyData line output from this
        widget, and performs a task with that line.

        Parameters
        ----------
        callback : callable
            The method called every time the line is updated. This has two
            options: Take a single argument, the ``PolyData`` line (default) or
            if ``use_vertices=True``, then it can take two arguments of the
            coordinates of the line's end points.

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing

        resolution : int
            The number of points in the line created

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        use_vertices : bool, optional
            Changess the arguments of the callback method to take the end
            points of the line instead of a PolyData object.

        pass_widget : bool
            If true, the widget will be passed as the last argument of the
            callback

        """
        if not hasattr(self, "line_widgets"):
            self.line_widgets = []

        if bounds is None:
            bounds = self.bounds

        if color is None:
            color = pyvista.global_theme.font.color

        def _the_callback(widget, event_id):
            pointa = widget.GetPoint1()
            pointb = widget.GetPoint2()
            if hasattr(callback, '__call__'):
                if use_vertices:
                    args = [pointa, pointb]
                else:
                    the_line = pyvista.Line(pointa, pointb, resolution=resolution)
                    args = [the_line]
                if pass_widget:
                    args.append(widget)
                try_callback(callback, *args)
            return

        line_widget = _vtk.vtkLineWidget()
        line_widget.GetLineProperty().SetColor(parse_color(color))
        line_widget.SetInteractor(self.iren.interactor)
        line_widget.SetCurrentRenderer(self.renderer)
        line_widget.SetPlaceFactor(factor)
        line_widget.PlaceWidget(bounds)
        line_widget.SetResolution(resolution)
        line_widget.Modified()
        line_widget.On()
        line_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(line_widget, None)

        self.line_widgets.append(line_widget)
        return line_widget

    def clear_line_widgets(self):
        """Disable all of the line widgets."""
        if hasattr(self, 'line_widgets'):
            for widget in self.line_widgets:
                widget.Off()
            del self.line_widgets
        return

    def add_text_slider_widget(self, callback, data, value=None,
                              pointa=(.4, .9), pointb=(.9, .9),
                              color=None, event_type='end',
                              style=None):
        """Add a text slider bar widget.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the value of this slider widget,
        and performs a task with that value.

        Parameters
        ----------
        callback : callable
            The method called every time the slider is updated. This should take
            a single parameter: the float value of the slider

        data: list
            The list of possible values displayed on the slider bar

        value : float, optional
            The starting value of the slider

        pointa : tuple(float)
            The relative coordinates of the left point of the slider on the
            display port

        pointb : tuple(float)
            The relative coordinates of the right point of the slider on the
            display port

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        event_type: str
            Either 'start', 'end' or 'always', this defines how often the
            slider interacts with the callback.

        style : str, optional
            The name of the slider style. The list of available styles
            are in ``pyvista.global_theme.slider_styles``. Defaults to
            ``None``.

        Returns
        -------
        slider_widget: vtk.vtkSliderWidget
            The VTK slider widget configured to display text.

        """
        if not isinstance(data, list):
            raise TypeError("The `data` parameter must be a list "
                            "but {} was given : ", type(data))
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
                if hasattr(callback, '__call__'):
                    try_callback(callback, data[idx])
            return

        slider_widget = self.add_slider_widget(callback=_the_callback, rng=[0, n_states - 1],
                                               value=value,
                                               pointa=pointa, pointb=pointb,
                                               color=color, event_type=event_type,
                                               style=style)
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
            raise ValueError("Expected value for `event_type` is 'start',"
                             f" 'end' or 'always': {event_type} was given.")
        title_callback(slider_widget, None)
        return slider_widget

    def add_slider_widget(self, callback, rng, value=None, title=None,
                          pointa=(.4, .9), pointb=(.9, .9),
                          color=None, pass_widget=False,
                          event_type='end', style=None,
                          title_height=0.03, title_opacity=1.0, title_color=None, fmt=None):
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

        pointb : tuple(float)
            The relative coordinates of the right point of the slider
            on the display port

        color : string or 3 item list, optional
            Either a string, rgb list, or hex color string.

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

        title_height: float, optional
            Relative height of the title as compared to the length of
            the slider.

        title_opacity: str, optional
            Opacity of title. Defaults to 1.0.

        title_color : string or 3 item list, optional
            Either a string, rgb list, or hex color string.  Defaults
            to the value given in ``color``.

        fmt : str, optional
            String formatter used to format numerical data. Defaults
            to ``None``.

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
        >>> cpos = pl.show()
        """
        if not hasattr(self, "slider_widgets"):
            self.slider_widgets = []

        if value is None:
            value = ((rng[1] - rng[0]) / 2) + rng[0]

        if color is None:
            color = pyvista.global_theme.font.color

        if title_color is None:
            title_color = color

        if fmt is None:
            fmt = pyvista.global_theme.font.fmt

        def normalize(point, viewport):
            return (point[0]*(viewport[2]-viewport[0]), point[1]*(viewport[3]-viewport[1]))

        pointa = normalize(pointa, self.renderer.GetViewport())
        pointb = normalize(pointb, self.renderer.GetViewport())

        slider_rep = _vtk.vtkSliderRepresentation2D()
        slider_rep.SetPickable(False)
        slider_rep.SetMinimumValue(rng[0])
        slider_rep.SetMaximumValue(rng[1])
        slider_rep.SetValue(value)
        slider_rep.SetTitleText(title)
        slider_rep.GetTitleProperty().SetColor(parse_color(color))
        slider_rep.GetSliderProperty().SetColor(parse_color(color))
        slider_rep.GetCapProperty().SetColor(parse_color(color))
        slider_rep.GetLabelProperty().SetColor(parse_color(color))
        slider_rep.GetTubeProperty().SetColor(parse_color(color))
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(pointa[0], pointa[1])
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(pointb[0], pointb[1])
        slider_rep.SetSliderLength(0.05)
        slider_rep.SetSliderWidth(0.05)
        slider_rep.SetEndCapLength(0.01)

        if style is not None:
            if not isinstance(style, str):
                raise TypeError("Expected type for ``style`` is str but"
                                f" {type(style)} was given.")
            slider_style = getattr(pyvista.global_theme.slider_styles, style)
            slider_rep.SetSliderLength(slider_style.slider_length)
            slider_rep.SetSliderWidth(slider_style.slider_width)
            slider_rep.GetSliderProperty().SetColor(slider_style.slider_color)
            slider_rep.SetTubeWidth(slider_style.tube_width)
            slider_rep.GetTubeProperty().SetColor(slider_style.tube_color)
            slider_rep.GetCapProperty().SetOpacity(slider_style.cap_opacity)
            slider_rep.SetEndCapLength(slider_style.cap_length)
            slider_rep.SetEndCapWidth(slider_style.cap_width)

        def _the_callback(widget, event):
            value = widget.GetRepresentation().GetValue()
            if hasattr(callback, '__call__'):
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
        slider_widget.GetRepresentation().GetTitleProperty().SetColor(parse_color(title_color))
        if fmt is not None:
            slider_widget.GetRepresentation().SetLabelFormat(fmt)
        slider_widget.On()
        if not isinstance(event_type, str):
            raise TypeError("Expected type for `event_type` is str: "
                            f"{type(event_type)} was given.")
        if event_type == 'start':
            slider_widget.AddObserver(_vtk.vtkCommand.StartInteractionEvent, _the_callback)
        elif event_type == 'end':
            slider_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
        elif event_type == 'always':
            slider_widget.AddObserver(_vtk.vtkCommand.InteractionEvent, _the_callback)
        else:
            raise ValueError("Expected value for `event_type` is 'start',"
                             f" 'end' or 'always': {event_type} was given.")
        _the_callback(slider_widget, None)

        self.slider_widgets.append(slider_widget)
        return slider_widget

    def clear_slider_widgets(self):
        """Disable all of the slider widgets."""
        if hasattr(self, 'slider_widgets'):
            for widget in self.slider_widgets:
                widget.Off()
            del self.slider_widgets
        return

    def add_mesh_threshold(self, mesh, scalars=None, invert=False,
                           widget_color=None, preference='cell',
                           title=None, pointa=(.4, .9), pointb=(.9, .9),
                           continuous=False, **kwargs):
        """Apply a threshold on a mesh with a slider.

        Add a mesh to the scene with a slider widget that is used to
        threshold the mesh interactively.

        The threshold mesh is saved to the ``.threshold_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and threshold

        scalars : str
            The string name of the scalars on the mesh to threshold and display

        invert : bool
            Invert/flip the threshold

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        """
        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for threshold widget.')
        name = kwargs.get('name', mesh.memory_address)
        if scalars is None:
            field, scalars = mesh.active_scalars_info
        arr, field = get_array(mesh, scalars, preference=preference, info=True)
        if arr is None:
            raise ValueError('No arrays present to threshold.')
        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars
        mesh.set_active_scalars(scalars)

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = _vtk.vtkThreshold()
        alg.SetInputDataObject(mesh)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars) # args: (idx, port, connection, field, name)
        alg.SetUseContinuousCellRange(continuous)

        if not hasattr(self, "threshold_meshes"):
            self.threshold_meshes = []
        threshold_mesh = pyvista.wrap(alg.GetOutput())
        self.threshold_meshes.append(threshold_mesh)

        def callback(value):
            if invert:
                alg.ThresholdByLower(value)
            else:
                alg.ThresholdByUpper(value)
            alg.Update()
            threshold_mesh.shallow_copy(alg.GetOutput())

        self.add_slider_widget(callback=callback, rng=rng, title=title,
                               color=widget_color, pointa=pointa,
                               pointb=pointb)

        kwargs.setdefault("reset_camera", False)
        actor = self.add_mesh(threshold_mesh, scalars=scalars, **kwargs)

        return actor

    def add_mesh_isovalue(self, mesh, scalars=None, compute_normals=False,
                          compute_gradients=False, compute_scalars=True,
                          preference='point', title=None, pointa=(.4, .9),
                          pointb=(.9, .9), widget_color=None, **kwargs):
        """Create a contour of a mesh with a slider.

        Add a mesh to the scene with a slider widget that is used to
        contour at an isovalue of the *point* data on the mesh interactively.

        The isovalue mesh is saved to the ``.isovalue_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and contour

        scalars : str
            The string name of the scalars on the mesh to contour and display

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

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
            _, field = get_array(mesh, scalars, preference=preference, info=True)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != pyvista.FieldAssociation.POINT:
            raise TypeError(f'Contour filter only works on Point data. Array ({scalars}) is in the Cell data.')

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
        alg.SetNumberOfContours(1) # Only one contour level

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        if not hasattr(self, "isovalue_meshes"):
            self.isovalue_meshes = []
        isovalue_mesh = pyvista.wrap(alg.GetOutput())
        self.isovalue_meshes.append(isovalue_mesh)

        def callback(value):
            alg.SetValue(0, value)
            alg.Update()
            isovalue_mesh.shallow_copy(alg.GetOutput())

        self.add_slider_widget(callback=callback, rng=rng, title=title,
                               color=widget_color, pointa=pointa,
                               pointb=pointb)

        kwargs.setdefault("reset_camera", False)
        actor = self.add_mesh(isovalue_mesh, scalars=scalars, **kwargs)

        return actor

    def add_spline_widget(self, callback, bounds=None, factor=1.25,
                          n_handles=5, resolution=25, color="yellow",
                          show_ribbon=False, ribbon_color="pink",
                          ribbon_opacity=0.5, pass_widget=False,
                          closed=False, initial_points=None):
        """Create and add a spline widget to the scene.

        Use the bounds argument to place this widget. Several "handles" are
        used to control a parametric function for building this spline. Click
        directly on the line to translate the widget.

        Note
        ----
        This widget has trouble displaying certain colors. Use only simple
        colors (white, black, yellow).

        Parameters
        ----------
        callback : callable
            The method called every time the spline is updated. This passes a
            :class:`pyvista.PolyData` object to the callback function of the
            generated spline.

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing

        n_handles : int
            The number of interactive spheres to control the spline's
            parametric function.

        resolution : int
            The number of points in the spline created between all the handles

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        show_ribbon : bool
            If ``True``, the poly plane used for slicing will also be shown.

        pass_widget : bool
            If true, the widget will be passed as the last argument of the
            callback

        closed : bool
            Make the spline a closed loop.

        initial_points : np.ndarray
            The points to initialize the widget placement. Must have same
            number of elements as ``n_handles``. If the first and last point
            are the same, this will be a closed loop spline.

        """
        if initial_points is not None and len(initial_points) != n_handles:
            raise ValueError("`initial_points` must be length `n_handles`.")

        if not hasattr(self, "spline_widgets"):
            self.spline_widgets = []

        if color is None:
            color = pyvista.global_theme.color

        if bounds is None:
            bounds = self.bounds

        ribbon = pyvista.PolyData()

        def _the_callback(widget, event_id):
            para_source = _vtk.vtkParametricFunctionSource()
            para_source.SetParametricFunction(widget.GetParametricSpline())
            para_source.Update()
            polyline = pyvista.wrap(para_source.GetOutput())
            ribbon.shallow_copy(polyline.ribbon(normal=(0,0,1), angle=90.0))
            if hasattr(callback, '__call__'):
                if pass_widget:
                    try_callback(callback, polyline, widget)
                else:
                    try_callback(callback, polyline)
            return

        spline_widget = _vtk.vtkSplineWidget()
        spline_widget.GetLineProperty().SetColor(parse_color(color))
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
        spline_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(spline_widget, None)

        if show_ribbon:
            self.add_mesh(ribbon, color=ribbon_color, opacity=ribbon_opacity)

        self.spline_widgets.append(spline_widget)
        return spline_widget

    def clear_spline_widgets(self):
        """Disable all of the spline widgets."""
        if hasattr(self, 'spline_widgets'):
            for widget in self.spline_widgets:
                widget.Off()
            del self.spline_widgets

    def add_mesh_slice_spline(self, mesh, generate_triangles=False,
                              n_handles=5, resolution=25,
                              widget_color=None, show_ribbon=False,
                              ribbon_color="pink", ribbon_opacity=0.5,
                              initial_points=None, closed=False,
                              **kwargs):
        """Slice a mesh with a spline widget.

        Add a mesh to the scene with a spline widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.spline_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The input dataset to add to the scene and slice along the spline

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        """
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = _vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        if not hasattr(self, "spline_sliced_meshes"):
            self.spline_sliced_meshes = []
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

        self.add_spline_widget(callback=callback, bounds=mesh.bounds,
                               factor=1.25, color=widget_color,
                               n_handles=n_handles, resolution=resolution,
                               show_ribbon=show_ribbon,
                               ribbon_color=ribbon_color,
                               ribbon_opacity=ribbon_opacity,
                               initial_points=initial_points,
                               closed=closed)

        return self.add_mesh(spline_sliced_mesh, **kwargs)

    def add_sphere_widget(self, callback, center=(0, 0, 0), radius=0.5,
                          theta_resolution=30, phi_resolution=30,
                          color=None, style="surface",
                          selected_color="pink", indices=None,
                          pass_widget=False, test_callback=True):
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

        theta_resolution: int, optional
            Set the number of points in the longitude direction.

        phi_resolution : int, optional
            Set the number of points in the latitude direction.

        color : string or 3 item iterable, optional
            The color of the sphere's surface.  If multiple centers
            are passed, then this must be a list of colors.  Each
            color is either a string, rgb list, or hex color string.
            For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        style : str, optional
            Representation style: ``'surface'`` or ``'wireframe'``.

        selected_color : str, optional
            Color of the widget when selected during interaction.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        test_callback: bool, optional
            if ``True``, run the callback function after the widget is
            created.

        """
        if not hasattr(self, "sphere_widgets"):
            self.sphere_widgets = []

        if color is None:
            color = pyvista.global_theme.color

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
            if hasattr(callback, '__call__'):
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
            sphere_widget.WIDGET_INDEX = indices[i] # Monkey patch the index
            if style in "wireframe":
                sphere_widget.SetRepresentationToWireframe()
            else:
                sphere_widget.SetRepresentationToSurface()
            sphere_widget.GetSphereProperty().SetColor(parse_color(colors[i]))
            sphere_widget.GetSelectedSphereProperty().SetColor(parse_color(selected_color))
            sphere_widget.SetInteractor(self.iren.interactor)
            sphere_widget.SetCurrentRenderer(self.renderer)
            sphere_widget.SetRadius(radius)
            sphere_widget.SetCenter(loc)
            sphere_widget.SetThetaResolution(theta_resolution)
            sphere_widget.SetPhiResolution(phi_resolution)
            sphere_widget.Modified()
            sphere_widget.On()
            sphere_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _the_callback)
            self.sphere_widgets.append(sphere_widget)

        if test_callback is True:
            # Test call back in the last
            _the_callback(sphere_widget, None)
        if num > 1:
            return self.sphere_widgets

        return sphere_widget

    def clear_sphere_widgets(self):
        """Disable all of the sphere widgets."""
        if hasattr(self, 'sphere_widgets'):
            for widget in self.sphere_widgets:
                widget.Off()
            del self.sphere_widgets
        return

    def add_checkbox_button_widget(self, callback, value=False,
                                   position=(10., 10.), size=50, border_size=5,
                                   color_on='blue', color_off='grey',
                                   background_color='white'):
        """Add a checkbox button widget to the scene.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the state of this button widget
        and performs a task with that value.

        Parameters
        ----------
        callback : callable
            The method called every time the button is clicked. This should take
            a single parameter: the bool value of the button

        value : bool
            The default state of the button

        position: tuple(float)
            The absolute coordinates of the bottom left point of the button

        size : int
            The size of the button in number of pixels

        border_size : int
            The size of the borders of the button in pixels

        color_on : string or 3 item list, optional
            The color used when the button is checked. Default is 'blue'

        color_off : string or 3 item list, optional
            The color used when the button is not checked. Default is 'grey'

        background_color : string or 3 item list, optional
            The background color of the button. Default is 'white'

        Returns
        -------
        button_widget: vtk.vtkButtonWidget
            The VTK button widget configured as a checkbox button.

        """
        if not hasattr(self, "button_widgets"):
            self.button_widgets = []

        def create_button(color1, color2, color3, dims=[size, size, 1]):
            color1 = np.array(parse_color(color1)) * 255
            color2 = np.array(parse_color(color2)) * 255
            color3 = np.array(parse_color(color3)) * 255

            n_points = dims[0] * dims[1]
            button = pyvista.UniformGrid(dims)
            arr = np.array([color1] * n_points).reshape(dims[0], dims[1], 3)  # fill with color1
            arr[1:dims[0]-1, 1:dims[1]-1] = color2  # apply color2
            arr[
                border_size:dims[0]-border_size,
                border_size:dims[1]-border_size
            ] = color3  # apply color3
            button.point_arrays['texture'] = arr.reshape(n_points, 3).astype(np.uint8)
            return button

        button_on = create_button(color_on, background_color, color_on)
        button_off = create_button(color_on, background_color, color_off)

        bounds = [
            position[0], position[0] + size,
            position[1], position[1] + size,
            0., 0.
        ]

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
            if hasattr(callback, '__call__'):
                try_callback(callback, bool(state))

        button_widget.AddObserver(_vtk.vtkCommand.StateChangedEvent, _the_callback)
        self.button_widgets.append(button_widget)
        return button_widget

    def clear_button_widgets(self):
        """Disable all of the button widgets."""
        if hasattr(self, 'button_widgets'):
            for widget in self.button_widgets:
                widget.Off()
            del self.button_widgets
        return

    def close(self):
        """Close the widgets."""
        self.clear_box_widgets()
        self.clear_plane_widgets()
        self.clear_line_widgets()
        self.clear_slider_widgets()
        self.clear_sphere_widgets()
        self.clear_spline_widgets()
        self.clear_button_widgets()
