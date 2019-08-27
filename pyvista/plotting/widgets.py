import logging
import vtk

import pyvista
from pyvista.utilities import NORMALS, get_array, try_callback

from .theme import *


class WidgetHelper(object):
    """An internal class to manage widgets and other helper methods involving
    widgets"""

    def enable_box_widget(self, callback, bounds=None, factor=1.25,
                          rotation_enabled=True, color=None, use_planes=False,
                          **kwargs):
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

        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Box widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')
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

        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.GetOutlineProperty().SetColor(parse_color(color))
        self.box_widget.SetInteractor(self.iren)
        self.box_widget.SetPlaceFactor(factor)
        self.box_widget.SetRotationEnabled(rotation_enabled)
        self.box_widget.PlaceWidget(bounds)
        self.box_widget.On()
        self.box_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(self.box_widget, None)

        return self.box_widget


    def disable_box_widget(self):
        """ Disables the last active box widget """
        if hasattr(self, 'box_widget'):
            self.box_widget.Off()
        return


    def add_mesh_clip_box(self, mesh, invert=False, rotation_enabled=True,
                          widget_color=None, **kwargs):
        """Add a mesh to the scene with a box widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.box_clipped_mesh`` attribute on
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
        name = kwargs.pop('name', str(hex(id(mesh))))
        kwargs.setdefault('clim', mesh.get_data_range(kwargs.get('scalars', None)))

        actor = self.add_mesh(mesh, name=name, **kwargs)

        def callback(planes):
            bounds = []
            for i in range(planes.GetNumberOfPlanes()):
                plane = planes.GetPlane(i)
                bounds.append(plane.GetNormal())
                bounds.append(plane.GetOrigin())

            self.box_clipped_mesh = mesh.clip_box(bounds=bounds, invert=invert)
            self.add_mesh(self.box_clipped_mesh, name=name, reset_camera=False,
                          **kwargs)

        self.enable_box_widget(callback=callback, bounds=mesh.bounds,
                factor=1.25, rotation_enabled=rotation_enabled,
                use_planes=True, color=widget_color)

        return actor


    def enable_plane_widget(self, callback, normal='x', origin=None,
                            bounds=None, factor=1.25, color=None, **kwargs):
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

        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Plane widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')
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

        self.plane_widget = vtk.vtkImplicitPlaneWidget()
        self.plane_widget.GetNormalProperty().SetColor(parse_color(color))
        self.plane_widget.GetOutlineProperty().SetColor(parse_color(color))
        self.plane_widget.GetOutlineProperty().SetColor(parse_color(color))
        self.plane_widget.GetPlaneProperty().SetOpacity(0.5)
        self.plane_widget.SetInteractor(self.iren)
        self.plane_widget.SetPlaceFactor(factor)
        self.plane_widget.PlaceWidget(bounds)
        self.plane_widget.SetOrigin(origin)
        self.plane_widget.SetNormal(normal)
        self.plane_widget.Modified()
        self.plane_widget.UpdatePlacement()
        self.plane_widget.On()
        self.plane_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(self.plane_widget, None) # Trigger immediate update

        _start_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(True)
        _stop_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(False)

        self.plane_widget.SetDrawPlane(False)
        self.plane_widget.AddObserver(vtk.vtkCommand.StartInteractionEvent, _start_interact)
        self.plane_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _stop_interact)

        return self.plane_widget


    def disable_plane_widget(self):
        """ Disables the last active plane widget """
        if hasattr(self, 'plane_widget'):
            self.plane_widget.Off()
        return


    def add_mesh_clip_plane(self, mesh, normal='x', invert=False,
                            widget_color=None, **kwargs):
        """Add a mesh to the scene with a plane widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.plane_clipped_mesh`` attribute on
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
        name = kwargs.pop('name', str(hex(id(mesh))))
        kwargs.setdefault('clim', mesh.get_data_range(kwargs.get('scalars', None)))

        actor = self.add_mesh(mesh, name=name, **kwargs)

        def callback(normal, origin):
            self.plane_clipped_mesh = mesh.clip(normal=normal, origin=origin,
                                                invert=invert)
            self.add_mesh(self.plane_clipped_mesh, name=name,
                          reset_camera=False, **kwargs)

        self.enable_plane_widget(callback=callback, bounds=mesh.bounds,
                                 factor=1.25, normal=normal, color=widget_color)

        return actor



    def add_mesh_slice(self, mesh, normal='x', contour=False,
                       generate_triangles=False, widget_color=None, **kwargs):
        """Add a mesh to the scene with a plane widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.plane_sliced_mesh`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and clip

        noraml : str or tuple(flaot)
            The starting normal vector of the plane

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        name = kwargs.pop('name', str(hex(id(mesh))))
        kwargs.setdefault('clim', mesh.get_data_range(kwargs.get('scalars', None)))

        actor = self.add_mesh(mesh, name=name, **kwargs)


        def callback(normal, origin):
            self.plane_sliced_mesh = mesh.slice(normal=normal, origin=origin,
                        contour=contour, generate_triangles=generate_triangles)
            self.add_mesh(self.plane_sliced_mesh, name=name, reset_camera=False,
                          **kwargs)

        self.enable_plane_widget(callback=callback, bounds=mesh.bounds,
                                 factor=1.25, normal=normal, color=widget_color)

        return actor



    def enable_line_widget(self, callback, bounds=None, factor=1.25,
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
            raise AssertionError('Box widget not available in notebook plotting')
        if not hasattr(self, 'iren'):
            raise AttributeError('Widgets must be used with an intereactive renderer. No off screen plotting.')
        if bounds is None:
            bounds = self.bounds

        if color is None:
            color = rcParams['font']['color']

        def _the_callback(widget, event_id):
            pointa = self.line_widget.GetPoint1()
            pointb = self.line_widget.GetPoint2()
            if hasattr(callback, '__call__'):
                if use_vertices:
                    try_callback(callback, pointa, pointb)
                else:
                    the_line = pyvista.Line(pointa, pointb, resolution=resolution)
                    try_callback(callback, the_line)
            return

        self.line_widget = vtk.vtkLineWidget()
        self.line_widget.GetLineProperty().SetColor(parse_color(color))
        self.line_widget.SetInteractor(self.iren)
        self.line_widget.SetPlaceFactor(factor)
        self.line_widget.PlaceWidget(bounds)
        self.line_widget.SetResolution(resolution)
        self.line_widget.Modified()
        self.line_widget.On()
        self.line_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(self.line_widget, None)

        return self.line_widget


    def disable_line_widget(self):
        """ Disables the last active line widget """
        if hasattr(self, 'line_widget'):
            self.line_widget.Off()
        return


    def enable_slider_widget(self, callback, rng, value=None, title=None,
                             pointa=(.4 ,.9), pointb=(.9, .9),
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
        slider_rep.GetPoint1Coordinate().SetValue(.4 ,.9)
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

        self.slider_widget = vtk.vtkSliderWidget()
        self.slider_widget.SetInteractor(self.iren)
        self.slider_widget.SetRepresentation(slider_rep)
        self.slider_widget.On()
        self.slider_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(self.slider_widget, None)

        return self.slider_widget


    def disable_slider_widget(self):
        """ Disables the last active slider widget """
        if hasattr(self, 'slider_widget'):
            self.slider_widget.Off()
        return


    def add_mesh_threshold(self, mesh, scalars=None, invert=False,
                           widget_color=None, preference='cell',
                           title=None, pointa=(.4 ,.9), pointb=(.9, .9),
                           **kwargs):
        """Add a mesh to the scene with a slider widget that is used to
        threshold the mesh interactively.

        The threshold mesh is saved to the ``.threshold_mesh`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and clip

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
        name = kwargs.pop('name', str(hex(id(mesh))))
        if scalars is None:
            field, scalars = mesh.active_scalar_info
        arr, field = get_array(mesh, scalars, preference=preference, info=True)
        if arr is None:
            raise AssertionError('No arrays present to threshold.')
        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars

        actor = self.add_mesh(mesh, name=name, **kwargs)

        def callback(value):
            self.threshold_mesh = pyvista.DataSetFilters.threshold(mesh, value,
                        scalars=scalars, preference=preference, invert=invert)
            if self.threshold_mesh.n_points < 1:
                self.remove_actor(name)
            else:
                self.add_mesh(self.threshold_mesh, name=name, scalars=scalars,
                              reset_camera=False, **kwargs)

        self.enable_slider_widget(callback=callback, rng=rng, title=title,
                                  color=widget_color, pointa=pointa,
                                  pointb=pointb)

        return actor


    def close(self):
        """ closes widgets """
        if hasattr(self, 'box_widget'):
            del self.box_widget

        if hasattr(self, 'plane_widget'):
            del self.plane_widget

        if hasattr(self, 'line_widget'):
            del self.line_widget

        if hasattr(self, 'slider_widget'):
            del self.slider_widget
