import vtk

import pyvista
from pyvista.utilities import NORMALS

from .theme import *


class WidgetHelper(object):
    """An internal class to manage widgets and other helper methods involving
    widgets"""

    def enable_box_widget(self, bounds=None, factor=1.0, callback=None,
                          rotation_enabled=True, color=None, use_planes=False,
                          **kwargs):
        """Add a box widget to the scene. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the PolyData box output from this widget, and performs a
        task with that box.
        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Box widget not available in notebook plotting')
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
                    callback(planes)
                else:
                    callback(the_box)
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

        return


    def disable_box_widget(self):
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

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for box widget clipping.')
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
            self.add_mesh(self.box_clipped_mesh, name=name, **kwargs)

        self.enable_box_widget(bounds=mesh.bounds, factor=1.25,
                rotation_enabled=rotation_enabled, callback=callback,
                use_planes=True, color=widget_color)

        return actor


    def enable_plane_widget(self, origin=None, normal='x', factor=1.25,
                            callback=None, bounds=None, color=None, **kwargs):
        """Add a plane widget to the scene. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the vtkPlane implicit function output from this widget, and
        performs a task with that plane.
        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Plane widget not available in notebook plotting')
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
            if hasattr(callback, '__call__'):
                callback(the_plane)
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

        return


    def disable_plane_widget(self):
        self.plane_widget.Off()
        return


    def add_mesh_clip_plane(self, mesh, invert=False, normal='x',
                            widget_color=None, **kwargs):
        """Add a mesh to the scene with a plane widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.plane_clipped_mesh`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : pyvista.Common
            The input dataset to add to the scene and clip

        invert : bool
            Flag on whether to flip/invert the clip

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for plane widget clipping.')
        name = kwargs.pop('name', str(hex(id(mesh))))
        kwargs.setdefault('clim', mesh.get_data_range(kwargs.get('scalars', None)))

        actor = self.add_mesh(mesh, name=name, **kwargs)

        def callback(plane):
            self.plane_clipped_mesh = pyvista.DataSetFilters._clip_with_function(mesh, plane,
                            invert=invert)
            self.add_mesh(self.plane_clipped_mesh, name=name, **kwargs)

        self.enable_plane_widget(bounds=mesh.bounds, factor=1.25, normal=normal,
                                 callback=callback, color=widget_color)

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

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        kwargs : dict
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.
        """
        if isinstance(mesh, pyvista.MultiBlock):
            raise TypeError('MultiBlock datasets are not supported for plane widget clipping.')
        name = kwargs.pop('name', str(hex(id(mesh))))
        kwargs.setdefault('clim', mesh.get_data_range(kwargs.get('scalars', None)))

        actor = self.add_mesh(mesh, name=name, **kwargs)


        def callback(plane):
            normal = plane.GetNormal()
            origin = plane.GetOrigin()
            self.plane_sliced_mesh = pyvista.DataSetFilters.slice(mesh,
                        normal=normal, origin=origin, contour=contour,
                        generate_triangles=generate_triangles)
            self.add_mesh(self.plane_sliced_mesh, name=name, **kwargs)

        self.enable_plane_widget(bounds=mesh.bounds, factor=1.25, normal=normal,
                                 callback=callback, color=widget_color)

        _start_interact = lambda obj, event: self.plane_widget.SetDrawPlane(True)
        _stop_interact = lambda obj, event: self.plane_widget.SetDrawPlane(False)

        self.plane_widget.SetDrawPlane(False)
        self.plane_widget.AddObserver(vtk.vtkCommand.StartInteractionEvent, _start_interact)
        self.plane_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _stop_interact)

        return actor



    def enable_line_widget(self, bounds=None, factor=1.0, callback=None,
                           rotation_enabled=True, resolution=100,
                           color=None, **kwargs):
        """Add a line widget to the scene. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the PolyData line output from this widget, and performs a
        task with that line.
        """
        if hasattr(self, 'notebook') and self.notebook:
            raise AssertionError('Box widget not available in notebook plotting')
        if bounds is None:
            bounds = self.bounds

        if color is None:
            color = rcParams['font']['color']

        # This dataset is continually updated by the widget and is return to
        # the user for use
        the_line = pyvista.PolyData()

        def _the_callback(widget, event_id):
            pointa = self.line_widget.GetPoint1()
            pointb = self.line_widget.GetPoint2()
            the_line.DeepCopy(pyvista.Line(pointa, pointb, resolution=resolution))
            if hasattr(callback, '__call__'):
                callback(the_line)
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

        self.line_widget.GetPolyData(the_line)

        return the_line


    def disable_line_widget(self):
        self.line_widget.Off()
        return


    def enable_slider_widget(self, min, max, value=None, title=None,
                             pointa=(.4 ,.9), pointb=(.9, .9), callback=None,
                             color=None):
        """Add a slider bar widget. This is useless without a callback
        function. You can pass a callable function that takes a single
        argument, the value of this slider widget, and performs a
        task with that value.
        """

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
                callback(value)
            return

        self.slider_widget = vtk.vtkSliderWidget()
        self.slider_widget.SetInteractor(self.iren)
        self.slider_widget.SetRepresentation(slider_rep)
        self.slider_widget.On()
        self.slider_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, _the_callback)
        _the_callback(self.slider_widget, None)

        return


    def disable_slider_widget(self):
        self.slider_widget.Off()
        return


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
