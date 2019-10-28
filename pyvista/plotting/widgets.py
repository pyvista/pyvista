import numpy as np
import vtk

import pyvista
from pyvista.utilities import NORMALS, generate_plane, get_array, try_callback

from .theme import rcParams, parse_color


class WidgetHelper(object):
    """An internal class to manage widgets and other helper methods involving
    widgets"""

    def add_box_widget(self, callback, bounds=None, factor=1.25,
                       rotation_enabled=True, color=None, use_planes=False,
                       outline_translation=True, **kwargs):
        """Add a box widget to the scene. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the PolyData box output from this widget, and performs a
        task with that box.

        Parameters
        ----------
        callback : callable
            The method called everytime the box is updated. This has two
            options: Take a single argument, the ``PolyData`` box (defualt) or
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

        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Box widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')
        if not hasattr(self, "box_widgets"):
            self.box_widgets = []

        if bounds is None:
            bounds = self.bounds

        if color is None:
            color = rcParams['font']['color']

        def _the_callback(box_widget, event_id):
            the_box = pyvista.PolyData()
            box_widget.GetPolyData(the_box)
            planes = vtk.vtkPlanes()
            box_widget.GetPlanes(planes)
            if hasattr(callback, '__call__'):
                if use_planes:
                    try_callback(callback, planes)
                else:
                    try_callback(callback, the_box)
            return

        box_widget = vtk.vtkBoxWidget()
        box_widget.GetOutlineProperty().SetColor(parse_color(color))
        box_widget.SetInteractor(self.iren)
        box_widget.SetCurrentRenderer(self.renderer)
        box_widget.SetPlaceFactor(factor)
        box_widget.SetRotationEnabled(rotation_enabled)
        box_widget.SetTranslationEnabled(outline_translation)
        box_widget.PlaceWidget(bounds)
        box_widget.On()
        box_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(box_widget, None)

        self.box_widgets.append(box_widget)
        return box_widget


    def clear_box_widgets(self):
        """ Disables all of the box widgets """
        if hasattr(self, 'box_widgets'):
            for widget in self.box_widgets:
                widget.Off()
            del self.box_widgets
        return


    def add_mesh_clip_box(self, mesh, invert=False, rotation_enabled=True,
                          widget_color=None, outline_translation=True,
                          **kwargs):
        """Add a mesh to the scene with a box widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.box_clipped_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
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
        name = kwargs.get('name', str(hex(id(mesh))))
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        port = 1 if invert else 0

        alg = vtk.vtkBoxClipDataSet()
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
                         origin_translation=True, **kwargs):
        """Add a plane widget to the scene. This is useless without a callback
        function. You can pass a callable function that takes two
        arguments, the normal and origin of the plane in that order output
        from this widget, and performs a task with that plane.

        Parameters
        ----------
        callback : callable
            The method called everytime the plane is updated. Takes two
            arguments, the normal and origin of the plane in that order.

        noraml : str or tuple(flaot)
            The starting normal vector of the plane

        origin : tuple(float)
            The starting coordinate of the center of the place

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.

        translation_enabled : bool
            If ``False``, the box widget cannot be translated and is strictly
            placed at the given bounds.

        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Plane widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')
        if not hasattr(self, "plane_widgets"):
            self.plane_widgets = []

        if origin is None:
            origin = self.center
        if bounds is None:
            bounds = self.bounds

        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]

        if color is None:
            color = rcParams['font']['color']

        def _the_callback(plane_widget, event_id):
            the_plane = vtk.vtkPlane()
            plane_widget.GetPlane(the_plane)
            normal = the_plane.GetNormal()
            origin = the_plane.GetOrigin()
            if hasattr(callback, '__call__'):
                try_callback(callback, normal, origin)
            return

        plane_widget = vtk.vtkImplicitPlaneWidget()
        plane_widget.GetNormalProperty().SetColor(parse_color(color))
        plane_widget.GetOutlineProperty().SetColor(parse_color(color))
        plane_widget.GetOutlineProperty().SetColor(parse_color(color))
        plane_widget.GetPlaneProperty().SetOpacity(0.5)
        plane_widget.SetTubing(tubing)
        plane_widget.SetInteractor(self.iren)
        plane_widget.SetCurrentRenderer(self.renderer)
        plane_widget.SetPlaceFactor(factor)
        plane_widget.PlaceWidget(bounds)
        plane_widget.SetOrigin(origin)
        plane_widget.SetOutlineTranslation(outline_translation)
        plane_widget.SetOriginTranslation(origin_translation)
        if assign_to_axis:
            # TODO: how do we now disable/hide the arrow?
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
        plane_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(plane_widget, None) # Trigger immediate update

        _start_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(True)
        _stop_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(False)

        plane_widget.SetDrawPlane(False)
        plane_widget.AddObserver(vtk.vtkCommand.StartInteractionEvent, _start_interact)
        plane_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _stop_interact)

        self.plane_widgets.append(plane_widget)
        return plane_widget


    def clear_plane_widgets(self):
        """ Disables all of the plane widgets """
        if hasattr(self, 'plane_widgets'):
            for widget in self.plane_widgets:
                widget.Off()
            del self.plane_widgets
        return


    def add_mesh_clip_plane(self, mesh, normal='x', invert=False,
                            widget_color=None, value=0.0, assign_to_axis=None,
                            tubing=False, origin_translation=True,
                            outline_translation=False, **kwargs):
        """Add a mesh to the scene with a plane widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.plane_clipped_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and clip

        noraml : str or tuple(flaot)
            The starting normal vector of the plane

        invert : bool
            Flag on whether to flip/invert the clip

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        name = kwargs.get('name', str(hex(id(mesh))))
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        if isinstance(mesh, vtk.vtkPolyData):
            alg = vtk.vtkClipPolyData()
        # elif isinstance(mesh, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = vtk.vtkTableBasedClipDataSet()
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
            alg.Update() # Perfrom the Cut
            plane_clipped_mesh.shallow_copy(alg.GetOutput())

        self.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal=normal,
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation)

        actor = self.add_mesh(plane_clipped_mesh, **kwargs)

        return actor



    def add_mesh_slice(self, mesh, normal='x', generate_triangles=False,
                       widget_color=None, assign_to_axis=None,
                       tubing=False, origin_translation=True,
                       outline_translation=False, **kwargs):
        """Add a mesh to the scene with a plane widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.plane_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and slice

        noraml : str or tuple(flaot)
            The starting normal vector of the plane

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        name = kwargs.get('name', str(hex(id(mesh))))
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = vtk.vtkCutter() # Construct the cutter object
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
            alg.SetCutFunction(plane) # the the cutter to use the plane we made
            alg.Update() # Perfrom the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())

        self.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal=normal,
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation)

        actor = self.add_mesh(plane_sliced_mesh, **kwargs)

        return actor


    def add_mesh_slice_orthogonal(self, mesh, generate_triangles=False,
                                  widget_color=None, tubing=False, **kwargs):
        """Adds three interactive plane slicing widgets for orthogonal slicing
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
                        **kwargs):
        """Add a line widget to the scene. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the PolyData line output from this widget, and performs a
        task with that line.

        Parameters
        ----------
        callback : callable
            The method called everytime the line is updated. This has two
            options: Take a single argument, the ``PolyData`` line (defualt) or
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
        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Line widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')
        if not hasattr(self, "line_widgets"):
            self.line_widgets = []

        if bounds is None:
            bounds = self.bounds

        if color is None:
            color = rcParams['font']['color']

        def _the_callback(widget, event_id):
            pointa = widget.GetPoint1()
            pointb = widget.GetPoint2()
            if hasattr(callback, '__call__'):
                if use_vertices:
                    try_callback(callback, pointa, pointb)
                else:
                    the_line = pyvista.Line(pointa, pointb, resolution=resolution)
                    try_callback(callback, the_line)
            return

        line_widget = vtk.vtkLineWidget()
        line_widget.GetLineProperty().SetColor(parse_color(color))
        line_widget.SetInteractor(self.iren)
        line_widget.SetCurrentRenderer(self.renderer)
        line_widget.SetPlaceFactor(factor)
        line_widget.PlaceWidget(bounds)
        line_widget.SetResolution(resolution)
        line_widget.Modified()
        line_widget.On()
        line_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(line_widget, None)

        self.line_widgets.append(line_widget)
        return line_widget


    def clear_line_widgets(self):
        """ Disables all of the line widgets """
        if hasattr(self, 'line_widgets'):
            for widget in self.line_widgets:
                widget.Off()
            del self.line_widgets
        return


    def add_slider_widget(self, callback, rng, value=None, title=None,
                          pointa=(.4, .9), pointb=(.9, .9),
                          color=None):
        """Add a slider bar widget. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the value of this slider widget, and performs a
        task with that value.

        Parameters
        ----------
        callback : callable
            The method called everytime the slider is updated. This should take
            a single parameter: the float value of the slider

        rng : tuple(float)
            Length two tuple of the minimum and maximum ranges of the slider

        value : float, optional
            The starting value of the slider

        title : str
            The string label of the slider widget

        pointa : tuple(float)
            The relative coordinates of the left point of the slider on the
            display port

        pointb : tuple(float)
            The relative coordinates of the right point of the slider on the
            display port

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.
        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Slider widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')

        if not hasattr(self, "slider_widgets"):
            self.slider_widgets = []

        min, max = rng

        if value is None:
            value = ((max-min) / 2) + min

        if color is None:
            color = rcParams['font']['color']

        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetPickable(False)
        slider_rep.SetMinimumValue(min)
        slider_rep.SetMaximumValue(max)
        slider_rep.SetValue(value)
        slider_rep.SetTitleText(title)
        slider_rep.GetTitleProperty().SetColor(parse_color(color))
        slider_rep.GetSliderProperty().SetColor(parse_color(color))
        slider_rep.GetCapProperty().SetColor(parse_color(color))
        slider_rep.GetLabelProperty().SetColor(parse_color(color))
        slider_rep.GetTubeProperty().SetColor(parse_color(color))
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedViewport()
        slider_rep.GetPoint1Coordinate().SetValue(.4, .9)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedViewport()
        slider_rep.GetPoint2Coordinate().SetValue(.9, .9)
        slider_rep.SetSliderLength(0.05)
        slider_rep.SetSliderWidth(0.05)
        slider_rep.SetEndCapLength(0.01)

        def _the_callback(slider, event):
            value = slider.GetRepresentation().GetValue()
            if hasattr(callback, '__call__'):
                try_callback(callback, value)
            return

        slider_widget = vtk.vtkSliderWidget()
        slider_widget.SetInteractor(self.iren)
        slider_widget.SetCurrentRenderer(self.renderer)
        slider_widget.SetRepresentation(slider_rep)
        slider_widget.On()
        slider_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(slider_widget, None)

        self.slider_widgets.append(slider_widget)
        return slider_widget


    def clear_slider_widgets(self):
        """ Disables all of the slider widgets """
        if hasattr(self, 'slider_widgets'):
            for widget in self.slider_widgets:
                widget.Off()
            del self.slider_widgets
        return


    def add_mesh_threshold(self, mesh, scalars=None, invert=False,
                           widget_color=None, preference='cell',
                           title=None, pointa=(.4, .9), pointb=(.9, .9),
                           continuous=False, **kwargs):
        """Add a mesh to the scene with a slider widget that is used to
        threshold the mesh interactively.

        The threshold mesh is saved to the ``.threshold_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
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
        name = kwargs.get('name', str(hex(id(mesh))))
        if scalars is None:
            field, scalars = mesh.active_scalar_info
        arr, field = get_array(mesh, scalars, preference=preference, info=True)
        if arr is None:
            raise AssertionError('No arrays present to threshold.')
        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = vtk.vtkThreshold()
        alg.SetInputDataObject(mesh)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars) # args: (idx, port, connection, field, name)
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
        """Add a mesh to the scene with a slider widget that is used to
        contour at an isovalue of the *point* data on the mesh interactively.

        The isovalue mesh is saved to the ``.isovalue_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and contour

        scalars : str
            The string name of the scalars on the mesh to threshold and display

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for this widget.')
        name = kwargs.get('name', str(hex(id(mesh))))
        # set the array to contour on
        if mesh.n_arrays < 1:
            raise AssertionError('Input dataset for the contour filter must have scalar data.')
        if scalars is None:
            field, scalars = mesh.active_scalar_info
        else:
            _, field = get_array(mesh, scalars, preference=preference, info=True)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != pyvista.POINT_DATA_FIELD:
            raise AssertionError('Contour filter only works on Point data. Array ({}) is in the Cell data.'.format(scalars))

        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars

        alg = vtk.vtkContourFilter()
        alg.SetInputDataObject(mesh)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        alg.SetInputArrayToProcess(0, 0, 0, field, scalars)
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
                          n_hanldes=5, resolution=25, color="yellow",
                          show_ribbon=False, ribbon_color="pink",
                          ribbon_opacity=0.5, **kwargs):
        """Create and add a spline widget to the scene. Use the bounds
        argument to place this widget. Several "handles" are used to control a
        parametric function for building this spline. Click directly on the
        line to translate the widget.

        Note
        ----
        This widget has trouble displaying certain colors. Use only simple
        colors (white, black, yellow).

        Parameters
        ----------
        callback : callable
            The method called everytime the spline is updated. This passes a
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

        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Spline widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')

        if not hasattr(self, "spline_widgets"):
            self.spline_widgets = []

        if color is None:
            color = rcParams['color']

        if bounds is None:
            bounds = self.bounds

        ribbon = pyvista.PolyData()

        def _the_callback(widget, event_id):
            polyline = pyvista.PolyData()
            widget.GetPolyData(polyline)
            ribbon.shallow_copy(polyline.ribbon(normal=(0,0,1), angle=90.0))
            if hasattr(callback, '__call__'):
                try_callback(callback, polyline)
            return

        spline_widget = vtk.vtkSplineWidget()
        spline_widget.GetLineProperty().SetColor(parse_color(color))
        spline_widget.SetNumberOfHandles(n_hanldes)
        spline_widget.SetInteractor(self.iren)
        spline_widget.SetCurrentRenderer(self.renderer)
        spline_widget.SetPlaceFactor(factor)
        spline_widget.PlaceWidget(bounds)
        spline_widget.SetResolution(resolution)
        spline_widget.Modified()
        spline_widget.On()
        spline_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(spline_widget, None)

        if show_ribbon:
            self.add_mesh(ribbon, color=ribbon_color, opacity=ribbon_opacity)

        self.spline_widgets.append(spline_widget)
        return spline_widget


    def clear_spline_widgets(self):
        """disables all of the spline widgets"""
        if hasattr(self, 'spline_widgets'):
            for widget in self.spline_widgets:
                widget.Off()
            del self.spline_widgets


    def add_mesh_slice_spline(self, mesh, generate_triangles=False,
                              n_hanldes=5, resolution=25,
                              widget_color=None, show_ribbon=False,
                              ribbon_color="pink", ribbon_opacity=0.5,
                              **kwargs):
        """Add a mesh to the scene with a spline widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.spline_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and slice along the spline

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        name = kwargs.get('name', str(hex(id(mesh))))
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))

        self.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = vtk.vtkCutter() # Construct the cutter object
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
            polyplane = vtk.vtkPolyPlane()
            polyplane.SetPolyLine(polyline)
            alg.SetCutFunction(polyplane) # the the cutter to use the poly planes
            alg.Update() # Perfrom the Cut
            spline_sliced_mesh.shallow_copy(alg.GetOutput())

        self.add_spline_widget(callback=callback, bounds=mesh.bounds,
                               factor=1.25, color=widget_color,
                               n_hanldes=n_hanldes, resolution=resolution,
                               show_ribbon=show_ribbon,
                               ribbon_color=ribbon_color,
                               ribbon_opacity=ribbon_opacity)

        actor = self.add_mesh(spline_sliced_mesh, **kwargs)

        return actor


    def add_sphere_widget(self, callback, center=(0, 0, 0), radius=0.5,
                          theta_resolution=30, phi_resolution=30,
                          color=None, style="surface",
                          selected_color="pink", indices=None):
        """Add one or many sphere widgets to a scene. Use a sphere widget
        to control a vertex location.

        Parameters
        ----------
        callback : callable
            The function to call back when the widget is modified. It takes a
            single argument: the center of the sphere as a XYZ coordinate.

        center : tuple(float)
            Length 3 array for the XYZ coordinate of the sphere's center
            when placing it in the scene. If more than one location is passed,
            then that many widgets will be added and the callback will also
            be passed the integer index of that widget.

        radius : float
            The radius of the sphere

        theta_resolution: int , optional
            Set the number of points in the longitude direction (ranging from
            start_theta to end theta).

        phi_resolution : int, optional
            Set the number of points in the latitude direction (ranging from
            start_phi to end_phi).

        color : str
            The color of the sphere's surface

        style : str
            Reprsentation style: surface or wireframe

        selected_color : str
            Color of the widget when selected during interaction
        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Sphere widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')

        if not hasattr(self, "sphere_widgets"):
            self.sphere_widgets = []

        if color is None:
            color = rcParams['color']

        center = np.array(center)
        num = 1
        if center.ndim > 1:
            num = len(center)

        if isinstance(color, (list, tuple, np.ndarray)):
            colors = color
        else:
            colors = [color] * num

        def _the_callback(widget, event_id):
            point = widget.GetCenter()
            index = widget.WIDGET_INDEX
            if hasattr(callback, '__call__'):
                if num > 1:
                    try_callback(callback, point, index)
                else:
                    try_callback(callback, point)
            return

        if indices is None:
            indices = [x for x in range(num)]

        for i in range(num):
            if center.ndim > 1:
                loc = center[i]
            else:
                loc = center
            sphere_widget = vtk.vtkSphereWidget()
            sphere_widget.WIDGET_INDEX = indices[i] # Monkey patch the index
            if style in "wireframe":
                sphere_widget.SetRepresentationToWireframe()
            else:
                sphere_widget.SetRepresentationToSurface()
            sphere_widget.GetSphereProperty().SetColor(parse_color(colors[i]))
            sphere_widget.GetSelectedSphereProperty().SetColor(parse_color(selected_color))
            sphere_widget.SetInteractor(self.iren)
            sphere_widget.SetCurrentRenderer(self.renderer)
            sphere_widget.SetRadius(radius)
            sphere_widget.SetCenter(loc)
            sphere_widget.SetThetaResolution(theta_resolution)
            sphere_widget.SetPhiResolution(phi_resolution)
            sphere_widget.Modified()
            sphere_widget.On()
            sphere_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
            _the_callback(sphere_widget, None)

            self.sphere_widgets.append(sphere_widget)

        if num > 1:
            return self.sphere_widgets

        return sphere_widget


    def clear_sphere_widgets(self):
        """ Disable all of the sphere widgets """
        if hasattr(self, 'sphere_widgets'):
            for widget in self.sphere_widgets:
                widget.Off()
            del self.sphere_widgets
        return


    def close(self):
        """ closes widgets """
        self.clear_box_widgets()
        self.clear_plane_widgets()
        self.clear_line_widgets()
        self.clear_slider_widgets()
        self.clear_sphere_widgets()
        self.clear_spline_widgets()
