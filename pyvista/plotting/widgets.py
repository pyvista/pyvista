"""Module dedicated to widgets."""

from __future__ import annotations

from itertools import product
import pathlib
from typing import TYPE_CHECKING

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import get_array_association
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.helpers import generate_plane
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.misc import try_callback

from . import _vtk
from .affine_widget import AffineWidget3D
from .colors import Color
from .opts import PickerType
from .utilities.algorithms import add_ids_algorithm
from .utilities.algorithms import algorithm_to_mesh_handler
from .utilities.algorithms import crinkle_algorithm
from .utilities.algorithms import outline_algorithm
from .utilities.algorithms import pointset_to_polydata_algorithm
from .utilities.algorithms import set_algorithm_input

if TYPE_CHECKING:
    from pyvista.core._typing_core import InteractionEventType
    from pyvista.core._typing_core import VectorLike


def _parse_interaction_event(interaction_event: InteractionEventType):
    """Parse the interaction event.

    Parameters
    ----------
    interaction_event : InteractionEventType
        The VTK interaction event to use for triggering the callback. Accepts
        either the strings ``'start'``, ``'end'``, ``'always'`` or a
        :vtk:`vtkCommand.EventIds`.

    Returns
    -------
    :vtk:`vtkCommand.EventIds`
        VTK Event type.

    """
    if not isinstance(interaction_event, (_vtk.vtkCommand.EventIds, str)):
        msg = (  # type: ignore[unreachable]
            'Expected type for `interaction_event` is either a str '
            'or an instance of `vtk.vtkCommand.EventIds`.'
            f' ({type(interaction_event)}) was given.'
        )
        raise TypeError(msg)

    if isinstance(interaction_event, _vtk.vtkCommand.EventIds):
        return interaction_event

    event_map = {
        'start': _vtk.vtkCommand.StartInteractionEvent,
        'end': _vtk.vtkCommand.EndInteractionEvent,
        'always': _vtk.vtkCommand.InteractionEvent,
    }
    if interaction_event not in event_map:
        expected = ', '.join(f'`{e}`' for e in event_map)
        msg = (
            f'Expected value for `interaction_event` is {expected}. {interaction_event} was given.'
        )
        raise ValueError(msg)

    return event_map[interaction_event]


@abstract_class
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
        self.radio_button_widget_dict = {}
        self.radio_button_title_dict = {}
        self.distance_widgets = []
        self.logo_widgets = []
        self.camera3d_widgets = []

    @_deprecate_positional_args(allowed=['callback'])
    def add_box_widget(  # noqa: PLR0917
        self,
        callback,
        bounds=None,
        factor=1.25,
        rotation_enabled: bool = True,  # noqa: FBT001, FBT002
        color=None,
        use_planes: bool = False,  # noqa: FBT001, FBT002
        outline_translation: bool = True,  # noqa: FBT001, FBT002
        pass_widget: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
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
            single argument of the plane collection as a :vtk:`vtkPlanes`
            object.

        bounds : tuple(float)
            Length 6 tuple of the bounding box where the widget is
            placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing.

        rotation_enabled : bool, optional
            If ``False``, the box widget cannot be rotated and is
            strictly orthogonal to the Cartesian axes.

        color : ColorLike, optional
            Either a string, rgb sequence, or hex color string.
            Defaults to :attr:`pyvista.global_theme.font.color
            <pyvista.plotting.themes._Font.color>`.

        use_planes : bool, optional
            Changes the arguments passed to the callback to the planes
            that make up the box.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is
            strictly placed at the given bounds.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

            .. versionchanged:: 0.38.0
               Now accepts either strings or :vtk:`vtkCommand.EventIds`.

        Returns
        -------
        :vtk:`vtkBoxWidget`
            Box widget.

        Examples
        --------
        Shows an interactive box that is used to resize and relocate a sphere.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> plotter = pv.Plotter()
        >>> def simulate(widget):
        ...     bounds = widget.bounds
        ...     new_center = np.array(
        ...         [
        ...             (bounds[0] + bounds[1]) / 2,
        ...             (bounds[2] + bounds[3]) / 2,
        ...             (bounds[4] + bounds[5]) / 2,
        ...         ]
        ...     )
        ...     new_radius = (
        ...         min(
        ...             (bounds[1] - bounds[0]) / 2,
        ...             (bounds[3] - bounds[2]) / 2,
        ...             (bounds[5] - bounds[4]) / 2,
        ...         )
        ...         - 0.3
        ...     )
        ...     sphere = pv.Sphere(radius=new_radius, center=new_center)
        ...     _ = plotter.add_mesh(sphere, name='Sphere')
        >>> _ = plotter.add_box_widget(callback=simulate)
        >>> plotter.show()

        """
        if bounds is None:
            bounds = self.bounds  # type: ignore[attr-defined]

        def _the_callback(box_widget, _event):
            the_box = pyvista.PolyData()
            box_widget.GetPolyData(the_box)
            planes = _vtk.vtkPlanes()
            box_widget.GetPlanes(planes)
            if callable(callback):
                args = [planes] if use_planes else [the_box]  # type: ignore[list-item]
                if pass_widget:
                    args.append(box_widget)
                try_callback(callback, *args)

        box_widget = _vtk.vtkBoxWidget()
        box_widget.GetOutlineProperty().SetColor(
            Color(color, default_color=pyvista.global_theme.font.color).float_rgb,
        )
        box_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        box_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
        box_widget.SetPlaceFactor(factor)
        box_widget.SetRotationEnabled(rotation_enabled)
        box_widget.SetTranslationEnabled(outline_translation)
        box_widget.PlaceWidget(bounds)
        box_widget.On()
        box_widget.AddObserver(
            _parse_interaction_event(interaction_event),
            _the_callback,
        )
        _the_callback(box_widget, None)

        self.box_widgets.append(box_widget)
        return box_widget

    def clear_box_widgets(self):
        """Remove all of the box widgets."""
        for box_widget in self.box_widgets:
            box_widget.Off()
        self.box_widgets.clear()

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_clip_box(  # noqa: PLR0917
        self,
        mesh,
        invert: bool = False,  # noqa: FBT001, FBT002
        rotation_enabled: bool = True,  # noqa: FBT001, FBT002
        widget_color=None,
        outline_translation: bool = True,  # noqa: FBT001, FBT002
        merge_points: bool = True,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        **kwargs,
    ):
        """Clip a mesh using a box widget.

        Add a mesh to the scene with a box widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.box_clipped_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : DataSet | :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and clip or algorithm that
            produces said mesh.

        invert : bool, optional
            Flag on whether to flip/invert the clip.

        rotation_enabled : bool, optional
            If ``False``, the box widget cannot be rotated and is strictly
            orthogonal to the cartesian axes.

        widget_color : ColorLike, optional
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

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

            .. versionchanged:: 0.38.0
               Changed from ``event_type`` to ``interaction_event`` and now
               accepts either strings and :vtk:`vtkCommand.EventIds`.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the mesh.

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
        from pyvista.core.filters import _get_output  # avoids circular import

        mesh, algo = algorithm_to_mesh_handler(
            add_ids_algorithm(mesh, point_ids=False, cell_ids=True),
        )

        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars'))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(outline_algorithm(algo), name=f'{name}-outline', opacity=0.0)  # type: ignore[attr-defined]

        port = 1 if invert else 0

        clipper = _vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            clipper.SetLocator(_vtk.vtkNonMergingPointLocator())
        set_algorithm_input(clipper, algo)
        clipper.GenerateClippedOutputOn()

        if crinkle:
            crinkler = crinkle_algorithm(clipper.GetOutputPort(port), algo)
            box_clipped_mesh = _get_output(crinkler)
        else:
            box_clipped_mesh = _get_output(clipper, oport=port)

        self.box_clipped_meshes.append(box_clipped_mesh)

        def callback(planes):
            bounds = []
            for i in range(planes.GetNumberOfPlanes()):
                plane = planes.GetPlane(i)
                bounds.append(plane.GetNormal())
                bounds.append(plane.GetOrigin())

            clipper.SetBoxClip(*bounds)
            clipper.Update()
            if crinkle:
                clipped = pyvista.wrap(crinkler.GetOutputDataObject(0))
            else:
                clipped = _get_output(clipper, oport=port)
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

        if crinkle:
            return self.add_mesh(crinkler, reset_camera=False, **kwargs)  # type: ignore[attr-defined]
        return self.add_mesh(clipper.GetOutputPort(port), reset_camera=False, **kwargs)  # type: ignore[attr-defined]

    @_deprecate_positional_args(allowed=['callback'])
    def add_plane_widget(  # noqa: PLR0917
        self,
        callback,
        normal='x',
        origin=None,
        bounds=None,
        factor=1.25,
        color=None,
        assign_to_axis=None,
        tubing: bool = False,  # noqa: FBT001, FBT002
        outline_translation: bool = False,  # noqa: FBT001, FBT002
        origin_translation: bool = True,  # noqa: FBT001, FBT002
        implicit: bool = True,  # noqa: FBT001, FBT002
        pass_widget: bool = False,  # noqa: FBT001, FBT002
        test_callback: bool = True,  # noqa: FBT001, FBT002
        normal_rotation: bool = True,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        outline_opacity=None,
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

        color : ColorLike, optional
            Either a string, rgb list, or hex color string.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given
            axis: options are ``(0, 'x')``, ``(1, 'y')``, or ``(2,
            'z')``.

        tubing : bool, optional
            When using an implicit plane widget, this controls whether
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
            When ``True``, a :vtk:`vtkImplicitPlaneWidget` is used and
            when ``False``, a :vtk:`vtkPlaneWidget` is used.

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

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

            .. versionchanged:: 0.38.0
               Now accepts either strings and :vtk:`vtkCommand.EventIds`.

        outline_opacity : bool or float, optional
            Set the visible of outline. Only valid when using
            an implicit plane. Either a bool or float.

            .. versionadded:: 0.44.0

        Returns
        -------
        :vtk:`vtkImplicitPlaneWidget` | :vtk:`vtkPlaneWidget`
            Plane widget.

        Examples
        --------
        Shows an interactive plane moving along the x-axis in the random-hill example,
        which is used to mark the max altitude at a particular distance x.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_random_hills()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> def callback(normal, origin):
        ...     slc = mesh.slice(normal=normal, origin=origin)
        ...     origin = list(origin)
        ...     origin[2] = slc.bounds.z_max
        ...     peak_plane = pv.Plane(
        ...         center=origin,
        ...         direction=[0, 0, 1],
        ...         i_size=20,
        ...         j_size=20,
        ...     )
        ...     _ = pl.add_mesh(peak_plane, name='Peak', color='red', opacity=0.4)
        >>> _ = pl.add_plane_widget(callback, normal_rotation=False)
        >>> pl.show()

        """
        if origin is None:
            origin = self.center  # type: ignore[attr-defined]
        if bounds is None:
            bounds = self.bounds  # type: ignore[attr-defined]

        if isinstance(normal, str):
            normal = NORMALS[normal.lower()]

        color = Color(color, default_color=pyvista.global_theme.font.color)

        if assign_to_axis:
            normal_rotation = False

        def _the_callback(widget, _event):
            the_plane = _vtk.vtkPlane()
            widget.GetPlane(the_plane)
            normal = the_plane.GetNormal()
            origin = the_plane.GetOrigin()
            if callable(callback):
                if pass_widget:
                    try_callback(callback, normal, origin, widget)
                else:
                    try_callback(callback, normal, origin)

        if implicit:
            plane_widget = _vtk.vtkImplicitPlaneWidget()
            plane_widget.GetNormalProperty().SetColor(color.float_rgb)
            plane_widget.GetOutlineProperty().SetColor(color.float_rgb)
            plane_widget.GetOutlineProperty().SetColor(color.float_rgb)
            plane_widget.GetOutlineProperty().SetOpacity(color.opacity)
            plane_widget.SetTubing(tubing)
            plane_widget.SetOutlineTranslation(outline_translation)
            plane_widget.SetOriginTranslation(origin_translation)

            _start_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(True)  # noqa: ARG005
            _stop_interact = lambda plane_widget, event: plane_widget.SetDrawPlane(False)  # noqa: ARG005

            plane_widget.SetDrawPlane(False)
            plane_widget.AddObserver(_vtk.vtkCommand.StartInteractionEvent, _start_interact)
            plane_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _stop_interact)
            plane_widget.SetPlaceFactor(factor)
            plane_widget.PlaceWidget(bounds)
            plane_widget.SetOrigin(origin)

            if not normal_rotation:
                plane_widget.GetNormalProperty().SetOpacity(0)

            if outline_opacity is not None:
                plane_widget.GetOutlineProperty().SetOpacity(float(outline_opacity))

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
            plane_widget = _vtk.vtkPlaneWidget()  # type: ignore[assignment]
            plane_widget.SetHandleSize(0.01)
            # Position of the widget
            plane_widget.SetInputData(source.GetOutput())
            plane_widget.SetRepresentationToOutline()  # type: ignore[attr-defined]
            plane_widget.SetPlaceFactor(factor)
            plane_widget.PlaceWidget(bounds)
            plane_widget.SetCenter(origin)  # type: ignore[attr-defined] # Necessary
            plane_widget.GetPlaneProperty().SetColor(color.float_rgb)  # self.C_LOT[fn])
            plane_widget.GetHandleProperty().SetColor(color.float_rgb)  # type: ignore[attr-defined]

            if not normal_rotation:
                plane_widget.GetHandleProperty().SetOpacity(0)  # type: ignore[attr-defined]

        plane_widget.GetPlaneProperty().SetOpacity(0.5)
        plane_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        plane_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]

        if assign_to_axis:
            # Note that normal_rotation was forced to False
            if assign_to_axis in [0, 'x', 'X']:
                plane_widget.NormalToXAxisOn()
                plane_widget.SetNormal(NORMALS['x'])  # type: ignore[arg-type]
            elif assign_to_axis in [1, 'y', 'Y']:
                plane_widget.NormalToYAxisOn()
                plane_widget.SetNormal(NORMALS['y'])  # type: ignore[arg-type]
            elif assign_to_axis in [2, 'z', 'Z']:
                plane_widget.NormalToZAxisOn()
                plane_widget.SetNormal(NORMALS['z'])  # type: ignore[arg-type]
            else:
                msg = 'assign_to_axis not understood'
                raise RuntimeError(msg)
        else:
            plane_widget.SetNormal(normal)

        plane_widget.Modified()
        plane_widget.UpdatePlacement()
        plane_widget.On()
        plane_widget.AddObserver(
            _parse_interaction_event(interaction_event),
            _the_callback,
        )
        if test_callback:
            _the_callback(plane_widget, None)  # Trigger immediate update

        self.plane_widgets.append(plane_widget)
        return plane_widget

    def clear_plane_widgets(self):
        """Remove all of the plane widgets."""
        for plane_widget in self.plane_widgets:
            plane_widget.Off()
        self.plane_widgets.clear()

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_clip_plane(  # noqa: PLR0917
        self,
        mesh,
        normal='x',
        invert: bool = False,  # noqa: FBT001, FBT002
        widget_color=None,
        value=0.0,
        assign_to_axis=None,
        tubing: bool = False,  # noqa: FBT001, FBT002
        origin_translation: bool = True,  # noqa: FBT001, FBT002
        outline_translation: bool = False,  # noqa: FBT001, FBT002
        implicit: bool = True,  # noqa: FBT001, FBT002
        normal_rotation: bool = True,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        origin=None,
        outline_opacity=None,
        **kwargs,
    ):
        """Clip a mesh using a plane widget.

        Add a mesh to the scene with a plane widget that is used to clip
        the mesh interactively.

        The clipped mesh is saved to the ``.plane_clipped_meshes``
        attribute on the plotter.

        Parameters
        ----------
        mesh : DataSet or :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and clip or algorithm that
            produces said mesh.

        normal : str or tuple(float), optional
            The starting normal vector of the plane.

        invert : bool, optional
            Flag on whether to flip/invert the clip.

        widget_color : ColorLike, optional
            Either a string, RGB list, or hex color string.

        value : float, optional
            Set the clipping value along the normal direction.
            The default value is 0.0.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given
            axis.  Options are ``(0, 'x')``, ``(1, 'y')``, or ``(2,
            'z')``.

        tubing : bool, optional
            When using an implicit plane widget, this controls whether
            or not tubing is shown around the plane's boundaries.

        origin_translation : bool, optional
            If ``False``, the plane widget cannot be translated by its
            origin and is strictly placed at the given origin. Only
            valid when using an implicit plane.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is
            strictly placed at the given bounds.

        implicit : bool, optional
            When ``True``, a :vtk:`vtkImplicitPlaneWidget` is used and
            when ``False``, a :vtk:`vtkPlaneWidget` is used.

        normal_rotation : bool, optional
            Set the opacity of the normal vector arrow to 0 such that
            it is effectively disabled. This prevents the user from
            rotating the normal. This is forced to ``False`` when
            ``assign_to_axis`` is set.

        crinkle : bool, optional
            Crinkle the clip by extracting the entire cells along the clip.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

            .. versionchanged:: 0.38.0
               Now accepts either strings or :vtk:`vtkCommand.EventIds`.

        origin : tuple(float), optional
            The starting coordinate of the center of the plane.

        outline_opacity : bool or float, optional
            Set the visible of outline. Only valid when using
            an implicit plane. Either a bool or float.

            .. versionadded:: 0.44.0

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the mesh.

        Examples
        --------
        Shows an interactive plane used to clip the mesh and store it.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> vol = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh_clip_plane(vol, normal=[0, -1, 0])
        >>> pl.show(cpos=[-2.1, 0.6, 1.5])
        >>> pl.plane_clipped_meshes  # doctest:+SKIP

        For a full example see :ref:`plane_widget_example`.

        """
        from pyvista.core.filters import _get_output  # avoids circular import

        mesh, algo = algorithm_to_mesh_handler(
            add_ids_algorithm(mesh, point_ids=False, cell_ids=True),
        )

        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars'))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))
        if origin is None:
            origin = mesh.center

        self.add_mesh(outline_algorithm(algo), name=f'{name}-outline', opacity=0.0)  # type: ignore[attr-defined]

        if isinstance(mesh, _vtk.vtkPolyData):
            clipper = _vtk.vtkClipPolyData()
        # elif isinstance(mesh, vtk.vtkImageData):
        #     clipper = vtk.vtkClipVolume()
        #     clipper.SetMixed3DCellGeneration(True)
        else:
            clipper = _vtk.vtkTableBasedClipDataSet()  # type: ignore[assignment]
        set_algorithm_input(clipper, algo)
        clipper.SetValue(value)
        clipper.SetInsideOut(invert)  # invert the clip if needed

        if crinkle:
            crinkler = crinkle_algorithm(clipper, algo)
            plane_clipped_mesh = _get_output(crinkler)
        else:
            plane_clipped_mesh = _get_output(clipper)
        self.plane_clipped_meshes.append(plane_clipped_mesh)

        def callback(normal, loc):
            function = generate_plane(normal, loc)
            clipper.SetClipFunction(function)  # the implicit function
            clipper.Update()  # Perform the Cut
            if crinkle:
                clipped = pyvista.wrap(crinkler.GetOutputDataObject(0))
            else:
                clipped = pyvista.wrap(clipper.GetOutput())
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
            outline_opacity=outline_opacity,
        )

        if crinkle:
            return self.add_mesh(crinkler, **kwargs)  # type: ignore[attr-defined]
        return self.add_mesh(clipper, **kwargs)  # type: ignore[attr-defined]

    @_deprecate_positional_args(allowed=['volume'])
    def add_volume_clip_plane(  # noqa: PLR0917
        self,
        volume,
        normal='x',
        invert: bool = False,  # noqa: ARG002, FBT001, FBT002
        widget_color=None,
        value=0.0,  # noqa: ARG002
        assign_to_axis=None,
        tubing: bool = False,  # noqa: FBT001, FBT002
        origin_translation: bool = True,  # noqa: FBT001, FBT002
        outline_translation: bool = False,  # noqa: FBT001, FBT002
        implicit: bool = True,  # noqa: FBT001, FBT002
        normal_rotation: bool = True,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        origin=None,
        outline_opacity=None,
        **kwargs,
    ):
        """Clip a volume using a plane widget.

        Parameters
        ----------
        volume : pyvista.plotting.volume.Volume or pyvista.ImageData or pyvista.RectilinearGrid
            New dataset of type :class:`pyvista.ImageData` or
            :class:`pyvista.RectilinearGrid`, or the return value from
            :class:`pyvista.plotting.volume.Volume` from :func:`pyvista.Plotter.add_volume`.

        normal : str or tuple(float), optional
            The starting normal vector of the plane.

        invert : bool, optional
            Flag on whether to flip/invert the clip.

        widget_color : ColorLike, optional
            Either a string, RGB list, or hex color string.

        value : float, optional
            Set the clipping value along the normal direction.
            The default value is 0.0.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given
            axis.  Options are ``(0, 'x')``, ``(1, 'y')``, or ``(2,
            'z')``.

        tubing : bool, optional
            When using an implicit plane widget, this controls whether
            or not tubing is shown around the plane's boundaries.

        origin_translation : bool, optional
            If ``False``, the plane widget cannot be translated by its
            origin and is strictly placed at the given origin. Only
            valid when using an implicit plane.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is
            strictly placed at the given bounds.

        implicit : bool, optional
            When ``True``, a :vtk:`vtkImplicitPlaneWidget` is used and
            when ``False``, a :vtk:`vtkPlaneWidget` is used.

        normal_rotation : bool, optional
            Set the opacity of the normal vector arrow to 0 such that
            it is effectively disabled. This prevents the user from
            rotating the normal. This is forced to ``False`` when
            ``assign_to_axis`` is set.

        interaction_event : :vtk:`vtkCommand.EventIds`, optional
            The VTK interaction event to use for triggering the callback.

        origin : tuple(float), optional
            The starting coordinate of the center of the plane.

        outline_opacity : bool or float, optional
            Set the visible of outline. Only valid when using
            an implicit plane. Either a bool or float.

            .. versionadded:: 0.44.0

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_volume` to control how the volume is
            displayed. Only applicable if ``volume`` is either a
            :class:`pyvista.ImageData` and :class:`pyvista.RectilinearGrid`.

        Returns
        -------
        :vtk:`vtkPlaneWidget` | :vtk:`vtkImplicitPlaneWidget`
            The VTK plane widget depending on the value of ``implicit``.

        See Also
        --------
        :ref:`clip_volume_widget_example`

        """
        if isinstance(volume, (pyvista.ImageData, pyvista.RectilinearGrid)):
            volume = self.add_volume(volume, **kwargs)  # type: ignore[attr-defined]
        elif not isinstance(volume, pyvista.plotting.volume.Volume):
            msg = (
                'The `volume` parameter type must be either pyvista.ImageData, '
                'pyvista.RectilinearGrid, or a pyvista.plotting.volume.Volume '
                'from `Plotter.add_volume`.'
            )
            raise TypeError(msg)
        else:
            assert_empty_kwargs(**kwargs)

        plane = _vtk.vtkPlane()

        def callback(normal, origin):  # numpydoc ignore=PR01
            """Update the plane used to clip the volume."""
            plane.SetNormal(normal)
            plane.SetOrigin(origin)

        widget = self.add_plane_widget(
            callback=callback,
            bounds=volume.bounds,
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
            outline_opacity=outline_opacity,
        )
        widget.GetPlane(plane)
        volume.mapper.AddClippingPlane(plane)
        self.plane_widgets.append(widget)

        return widget

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_slice(  # noqa: PLR0917
        self,
        mesh,
        normal='x',
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        widget_color=None,
        assign_to_axis=None,
        tubing: bool = False,  # noqa: FBT001, FBT002
        origin_translation: bool = True,  # noqa: FBT001, FBT002
        outline_translation: bool = False,  # noqa: FBT001, FBT002
        implicit: bool = True,  # noqa: FBT001, FBT002
        normal_rotation: bool = True,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        origin=None,
        outline_opacity=None,
        **kwargs,
    ):
        """Slice a mesh using a plane widget.

        Add a mesh to the scene with a plane widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.plane_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : DataSet | :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and slice or algorithm that
            produces said mesh.

        normal : str or tuple(float), optional
            The starting normal vector of the plane.

        generate_triangles : bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        widget_color : ColorLike, optional
            Either a string, RGB sequence, or hex color string.  Defaults
            to ``'white'``.

        assign_to_axis : str or int, optional
            Assign the normal of the plane to be parallel with a given axis:
            options are (0, 'x'), (1, 'y'), or (2, 'z').

        tubing : bool, optional
            When using an implicit plane widget, this controls whether or not
            tubing is shown around the plane's boundaries.

        origin_translation : bool, optional
            If ``False``, the plane widget cannot be translated by its origin
            and is strictly placed at the given origin. Only valid when using
            an implicit plane.

        outline_translation : bool, optional
            If ``False``, the box widget cannot be translated and is strictly
            placed at the given bounds.

        implicit : bool, optional
            When ``True``, a :vtk:`vtkImplicitPlaneWidget` is used and when
            ``False``, a :vtk:`vtkPlaneWidget` is used.

        normal_rotation : bool, optional
            Set the opacity of the normal vector arrow to 0 such that it is
            effectively disabled. This prevents the user from rotating the
            normal. This is forced to ``False`` when ``assign_to_axis`` is set.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

        origin : tuple(float), optional
            The starting coordinate of the center of the plane.

        outline_opacity : bool or float, optional
            Set the visible of outline. Only valid when using
            an implicit plane. Either a bool or float.

            .. versionadded:: 0.44.0

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the mesh.

        Examples
        --------
        Shows an interactive plane used specifically for slicing.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter()
        >>> mesh = examples.load_channels()
        >>> _ = pl.add_mesh(mesh.outline())
        >>> _ = pl.add_mesh_slice(mesh, normal=[1, 0, 0.3])
        >>> pl.show()

        For a full example see :ref:`plane_widget_example`.

        """
        mesh, algo = algorithm_to_mesh_handler(mesh)

        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars'))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))
        if origin is None:
            origin = mesh.center

        self.add_mesh(outline_algorithm(algo or mesh), name=f'{name}-outline', opacity=0.0)  # type: ignore[attr-defined]

        alg = _vtk.vtkCutter()  # Construct the cutter object
        set_algorithm_input(alg, algo or mesh)
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
            interaction_event=_parse_interaction_event(interaction_event),
            outline_opacity=outline_opacity,
        )

        return self.add_mesh(alg, **kwargs)  # type: ignore[attr-defined]

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_slice_orthogonal(  # noqa: PLR0917
        self,
        mesh,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        widget_color=None,
        tubing: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        **kwargs,
    ):
        """Slice a mesh with three interactive planes.

        Adds three interactive plane slicing widgets for orthogonal slicing
        along each cartesian axis.

        Parameters
        ----------
        mesh : DataSet or :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and threshold or algorithm
            that produces said mesh.

        generate_triangles : bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        widget_color : ColorLike, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        tubing : bool, optional
            When using an implicit plane widget, this controls whether or not
            tubing is shown around the plane's boundaries.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        list
            List of :vtk:`vtkActor`.

        Examples
        --------
        Shows an interactive plane sliced along each cartesian axis of the mesh.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> mesh = pv.Wavelet()
        >>> _ = pl.add_mesh(mesh.outline())
        >>> _ = pl.add_mesh_slice_orthogonal(mesh)
        >>> pl.show()

        """
        actors = []
        name = kwargs.pop('name', None)
        for ax in ['x', 'y', 'z']:
            axkwargs = kwargs.copy()
            if name:
                axkwargs['name'] = f'{name}-{ax}'
            a = self.add_mesh_slice(
                mesh,
                assign_to_axis=ax,
                origin_translation=False,
                outline_translation=False,
                generate_triangles=generate_triangles,
                widget_color=widget_color,
                tubing=tubing,
                interaction_event=_parse_interaction_event(interaction_event),
                **axkwargs,
            )
            actors.append(a)

        return actors

    @_deprecate_positional_args(allowed=['callback'])
    def add_line_widget(  # noqa: PLR0917
        self,
        callback,
        bounds=None,
        factor=1.25,
        resolution=100,
        color=None,
        use_vertices: bool = False,  # noqa: FBT001, FBT002
        pass_widget: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
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

        color : ColorLike, optional
            Either a string, rgb sequence, or hex color string.

        use_vertices : bool, optional
            Changes the arguments of the callback method to take the end
            points of the line instead of a PolyData object.

        pass_widget : bool, default: False
            If ``True``, the widget will be passed as the last
            argument of the callback.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

        Returns
        -------
        :vtk:`vtkLineWidget`
            Created line widget.

        Examples
        --------
        Shows an interactive line widget to move the sliced object
        like in `add_mesh_slice` function.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> import numpy as np
        >>> model = examples.load_channels()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(model, opacity=0.4)
        >>> def move_center(pointa, pointb):
        ...     center = (np.array(pointa) + np.array(pointb)) / 2
        ...     normal = np.array(pointa) - np.array(pointb)
        ...     single_slc = model.slice(normal=normal, origin=center)
        ...
        ...     _ = pl.add_mesh(single_slc, name='slc')
        >>> _ = pl.add_line_widget(callback=move_center, use_vertices=True)
        >>> pl.show()

        """
        if bounds is None:
            bounds = self.bounds  # type: ignore[attr-defined]

        color = Color(color, default_color=pyvista.global_theme.font.color)

        def _the_callback(widget, _event):
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
        line_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        line_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
        line_widget.SetPlaceFactor(factor)
        line_widget.PlaceWidget(bounds)
        line_widget.SetResolution(resolution)
        line_widget.Modified()
        line_widget.On()
        line_widget.AddObserver(
            _parse_interaction_event(interaction_event),
            _the_callback,
        )
        _the_callback(line_widget, None)

        self.line_widgets.append(line_widget)
        return line_widget

    def clear_line_widgets(self):
        """Remove all of the line widgets."""
        for line_widget in self.line_widgets:
            line_widget.Off()
        self.line_widgets.clear()

    @_deprecate_positional_args(allowed=['callback', 'data'])
    def add_text_slider_widget(  # noqa: PLR0917
        self,
        callback,
        data,
        value=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        color=None,
        interaction_event: InteractionEventType = 'end',
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

        color : ColorLike, optional
            Either a string, RGB list, or hex color string.  Defaults
            to :attr:`pyvista.global_theme.font.color
            <pyvista.plotting.themes._Font.color>`.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

            .. versionchanged:: 0.38.0
               Changed from ``event_type`` to ``interaction_event`` and now
               accepts either strings or :vtk:`vtkCommand.EventIds`.

        style : str, optional
            The name of the slider style. The list of available styles
            are in ``pyvista.global_theme.slider_styles``. Defaults to
            ``None``.


        Returns
        -------
        :vtk:`vtkSliderWidget`
            The VTK slider widget configured to display text.

        """
        if not isinstance(data, list):
            msg = (
                f'The `data` parameter must be a list but {type(data).__name__} was passed instead'
            )
            raise TypeError(msg)
        n_states = len(data)
        if n_states == 0:
            msg = 'The input list of values is empty'
            raise ValueError(msg)
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

        slider_widget = self.add_slider_widget(
            callback=_the_callback,
            rng=[0, n_states - 1],
            value=value,
            pointa=pointa,
            pointb=pointb,
            color=color,
            interaction_event=interaction_event,
            style=style,
        )
        slider_rep = slider_widget.GetRepresentation()
        slider_rep.ShowSliderLabelOff()

        def title_callback(widget, _event):
            value = widget.GetRepresentation().GetValue()
            idx = int(value / delta)
            # handle limit index
            if idx == n_states:
                idx = n_states - 1
            slider_rep.SetTitleText(data[idx])

        slider_widget.AddObserver(_parse_interaction_event(interaction_event), title_callback)
        title_callback(slider_widget, None)
        return slider_widget

    @_deprecate_positional_args(allowed=['callback', 'rng'])
    def add_slider_widget(  # noqa: PLR0917
        self,
        callback,
        rng,
        value=None,
        title=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        color=None,
        pass_widget: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
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
            Called every time the slider is updated. This should take a single
            parameter: the float value of the slider. If ``pass_widget=True``,
            callable should take two parameters: the float value of the slider
            and the widget itself.

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

        color : ColorLike, optional
            Either a string, RGB list, or hex color string.  Defaults
            to :attr:`pyvista.global_theme.font.color
            <pyvista.plotting.themes._Font.color>`.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

            .. versionchanged:: 0.38.0
               Changed from ``event_type`` to ``interaction_event`` and now accepts
               either strings or :vtk:`vtkCommand.EventIds`.

        style : str, optional
            The name of the slider style. The list of available styles
            are in ``pyvista.global_theme.slider_styles``. Defaults to
            ``None``.

        title_height : float, optional
            Relative height of the title as compared to the length of
            the slider.

        title_opacity : float, optional
            Opacity of title. Defaults to 1.0.

        title_color : ColorLike, optional
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
        :vtk:`vtkSliderWidget`
            Slider widget.

        See Also
        --------
        :ref:`multi_slider_widget_example`

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> def create_mesh(value):
        ...     res = int(value)
        ...     sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
        ...     pl.add_mesh(sphere, name='sphere', show_edges=True)
        >>> slider = pl.add_slider_widget(
        ...     create_mesh,
        ...     [5, 100],
        ...     title='Resolution',
        ...     title_opacity=0.5,
        ...     title_color='red',
        ...     fmt='%0.9f',
        ...     title_height=0.08,
        ... )
        >>> pl.show()

        """
        msg = 'Cannot add a widget to a closed plotter.'
        iren = self._get_iren_not_none(msg)  # type: ignore[attr-defined]

        if value is None:
            value = ((rng[1] - rng[0]) / 2) + rng[0]

        color = Color(color, default_color=pyvista.global_theme.font.color)
        title_color = Color(title_color, default_color=color)

        if fmt is None:
            fmt = pyvista.global_theme.font.fmt

        def normalize(point, viewport):
            return (
                point[0] * (viewport[2] - viewport[0]),
                point[1] * (viewport[3] - viewport[1]),
            )

        pointa = normalize(pointa, self.renderer.GetViewport())  # type: ignore[attr-defined]
        pointb = normalize(pointb, self.renderer.GetViewport())  # type: ignore[attr-defined]

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
                msg = f'Expected type for ``style`` is str but {type(style).__name__} was given.'
                raise TypeError(msg)
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

        def _the_callback(widget, _event):
            value = widget.GetRepresentation().GetValue()
            if callable(callback):
                if pass_widget:
                    try_callback(callback, value, widget)
                else:
                    try_callback(callback, value)

        slider_widget = _vtk.vtkSliderWidget()
        slider_widget.SetInteractor(iren.interactor)
        slider_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
        slider_widget.SetRepresentation(slider_rep)
        slider_widget.GetRepresentation().SetTitleHeight(title_height)  # type: ignore[attr-defined]
        slider_widget.GetRepresentation().GetTitleProperty().SetOpacity(title_opacity)  # type: ignore[attr-defined]
        slider_widget.GetRepresentation().GetTitleProperty().SetColor(title_color.float_rgb)  # type: ignore[attr-defined]
        if fmt is not None:
            slider_widget.GetRepresentation().SetLabelFormat(fmt)  # type: ignore[attr-defined]
        slider_widget.On()
        slider_widget.AddObserver(_parse_interaction_event(interaction_event), _the_callback)
        _the_callback(slider_widget, None)

        self.slider_widgets.append(slider_widget)
        return slider_widget

    def clear_slider_widgets(self):
        """Remove all of the slider widgets."""
        for slider_widget in self.slider_widgets:
            slider_widget.Off()
        self.slider_widgets.clear()

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_threshold(  # noqa: PLR0917
        self,
        mesh,
        scalars=None,
        invert: bool = False,  # noqa: FBT001, FBT002
        widget_color=None,
        preference='cell',
        title=None,
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        continuous: bool = False,  # noqa: FBT001, FBT002
        all_scalars: bool = False,  # noqa: FBT001, FBT002
        method='upper',
        **kwargs,
    ):
        """Apply a threshold on a mesh with a slider.

        Add a mesh to the scene with a slider widget that is used to
        threshold the mesh interactively.

        The threshold mesh is saved to the ``.threshold_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : DataSet or :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and threshold or algorithm
            that produces said mesh.

        scalars : str, optional
            The string name of the scalars on the mesh to threshold and display.

        invert : bool, default: False
            Invert the threshold results. That is, cells that would have been
            in the output with this option off are excluded, while cells that
            would have been excluded from the output are included.

        widget_color : ColorLike, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        preference : str, default: 'cell'
            When ``mesh.n_points == mesh.n_cells`` and setting
            scalars, this parameter sets how the scalars will be
            mapped to the mesh.  Default ``'cell'``, causes the
            scalars to be associated with the mesh cells.  Can be
            either ``'point'`` or ``'cell'``.

        title : str, optional
            The string label of the slider widget.

        pointa : sequence, default: (0.4, 0.9)
            The relative coordinates of the left point of the slider
            on the display port.

        pointb : sequence, default: (0.9, 0.9)
            The relative coordinates of the right point of the slider
            on the display port.

        continuous : bool, default: False
            If this is enabled (default is ``False``), use the continuous
            interval ``[minimum cell scalar, maximum cell scalar]``
            to intersect the threshold bound, rather than the set of
            discrete scalar values from the vertices.

        all_scalars : bool, default: False
            If using scalars from point data, all
            points in a cell must satisfy the threshold when this
            value is ``True``.  When ``False``, any point of the cell
            with a scalar value satisfying the threshold criterion
            will extract the cell. Has no effect when using cell data.

        method : str, default: 'upper'
            Set the threshold method for single-values, defining which
            threshold bounds to use. If the ``value`` is a range, this
            parameter will be ignored, extracting data between the two
            values. For single values, ``'lower'`` will extract data
            lower than the  ``value``. ``'upper'`` will extract data
            larger than the ``value``.

        **kwargs : dict, optional
            All additional keyword arguments are passed to ``add_mesh`` to
            control how the mesh is displayed.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the mesh.

        """
        # avoid circular import
        from pyvista.core.filters.data_set import _set_threshold_limit

        mesh, algo = algorithm_to_mesh_handler(mesh)

        if isinstance(mesh, pyvista.PointSet):
            # vtkThreshold is CELL-wise and PointSets have no cells
            algo = pointset_to_polydata_algorithm(algo or mesh)
            mesh, algo = algorithm_to_mesh_handler(algo)

        if isinstance(mesh, pyvista.MultiBlock):
            msg = 'MultiBlock datasets are not supported for threshold widget.'
            raise TypeError(msg)
        name = kwargs.get('name', mesh.memory_address)
        if scalars is None:
            field, scalars = mesh.active_scalars_info
        arr = get_array(mesh, scalars, preference=preference)
        if arr is None:
            msg = 'No arrays present to threshold.'
            raise ValueError(msg)
        field = get_array_association(mesh, scalars, preference=preference)

        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars
        mesh.set_active_scalars(scalars)

        self.add_mesh(outline_algorithm(algo or mesh), name=f'{name}-outline', opacity=0.0)  # type: ignore[attr-defined]

        alg = _vtk.vtkThreshold()
        set_algorithm_input(alg, algo or mesh)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetUseContinuousCellRange(continuous)
        alg.SetAllScalars(all_scalars)

        threshold_mesh = pyvista.wrap(alg.GetOutput())
        self.threshold_meshes.append(threshold_mesh)

        def callback(value):
            _set_threshold_limit(alg, value=value, method=method, invert=invert)
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

        kwargs.setdefault('reset_camera', False)
        return self.add_mesh(alg, scalars=scalars, **kwargs)  # type: ignore[attr-defined]

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_isovalue(  # noqa: PLR0917
        self,
        mesh,
        scalars=None,
        compute_normals: bool = False,  # noqa: FBT001, FBT002
        compute_gradients: bool = False,  # noqa: FBT001, FBT002
        compute_scalars: bool = True,  # noqa: FBT001, FBT002
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

        .. warning::
            This will not work with :class:`pyvista.PointSet` as
            creating an isovalue is a dimension reducing operation
            on the geometry and point clouds are zero dimensional.
            This will similarly fail for point clouds in
            :class:`pyvista.PolyData`.

        Parameters
        ----------
        mesh : DataSet or :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and contour or algorithm
            that produces said mesh.

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

        widget_color : ColorLike, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the mesh.

        Examples
        --------
        Shows an interactive slider controlling the altitude of the contours.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter()
        >>> mesh = examples.load_random_hills()
        >>> _ = pl.add_mesh(mesh, opacity=0.4)
        >>> _ = pl.add_mesh_isovalue(mesh)
        >>> pl.show()

        """
        mesh, algo = algorithm_to_mesh_handler(mesh)
        if isinstance(mesh, pyvista.PointSet):
            msg = 'PointSets are 0-dimensional and thus cannot produce contours.'
            raise TypeError(msg)
        if isinstance(mesh, pyvista.MultiBlock):
            msg = 'MultiBlock datasets are not supported for this widget.'
            raise TypeError(msg)
        name = kwargs.get('name', mesh.memory_address)
        # set the array to contour on
        if mesh.n_arrays < 1:
            msg = 'Input dataset for the contour filter must have data arrays.'
            raise ValueError(msg)
        if scalars is None:
            field, scalars = mesh.active_scalars_info
        else:
            field = get_array_association(mesh, scalars, preference=preference)
        # NOTE: only point data is allowed? well cells works but seems buggy?
        if field != pyvista.FieldAssociation.POINT:
            msg = (
                f'Contour filter only works on Point data. Array ({scalars}) is in the Cell data.'
            )
            raise TypeError(msg)

        rng = mesh.get_data_range(scalars)
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        if title is None:
            title = scalars
        mesh.set_active_scalars(scalars)

        alg = _vtk.vtkContourFilter()
        set_algorithm_input(alg, algo or mesh)
        alg.SetComputeNormals(compute_normals)
        alg.SetComputeGradients(compute_gradients)
        alg.SetComputeScalars(compute_scalars)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
        alg.SetNumberOfContours(1)  # Only one contour level

        self.add_mesh(outline_algorithm(algo or mesh), name=f'{name}-outline', opacity=0.0)  # type: ignore[attr-defined]

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

        kwargs.setdefault('reset_camera', False)
        return self.add_mesh(alg, scalars=scalars, **kwargs)  # type: ignore[attr-defined]

    @_deprecate_positional_args(allowed=['callback'])
    def add_spline_widget(  # noqa: PLR0917
        self,
        callback,
        bounds=None,
        factor=1.25,
        n_handles=5,
        resolution=25,
        color='yellow',
        show_ribbon: bool = False,  # noqa: FBT001, FBT002
        ribbon_color='pink',
        ribbon_opacity=0.5,
        pass_widget: bool = False,  # noqa: FBT001, FBT002
        closed: bool = False,  # noqa: FBT001, FBT002
        initial_points=None,
        interaction_event: InteractionEventType = 'end',
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

        bounds : sequence[float], optional
            Length 6 sequence of the bounding box where the widget is placed.

        factor : float, optional
            An inflation factor to expand on the bounds when placing.

        n_handles : int, optional
            The number of interactive spheres to control the spline's
            parametric function.

        resolution : int, optional
            The number of points in the spline created between all the handles.

        color : ColorLike, optional
            Either a string, RGB sequence, or hex color string.

        show_ribbon : bool, optional
            If ``True``, the poly plane used for slicing will also be shown.

        ribbon_color : ColorLike, optional
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

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

        Returns
        -------
        :vtk:`vtkSplineWidget`
            The newly created spline widget.

        See Also
        --------
        :ref:`spline_widget_example`

        Notes
        -----
        This widget has trouble displaying certain colors. Use only simple
        colors (white, black, yellow).

        """
        if initial_points is not None and len(initial_points) != n_handles:
            msg = '`initial_points` must be length `n_handles`.'
            raise ValueError(msg)

        color = Color(color, default_color=pyvista.global_theme.color)

        if bounds is None:
            bounds = self.bounds  # type: ignore[attr-defined]

        ribbon = pyvista.PolyData()

        def _the_callback(widget, _event):
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

        spline_widget = _vtk.vtkSplineWidget()
        spline_widget.GetLineProperty().SetColor(color.float_rgb)
        spline_widget.SetNumberOfHandles(n_handles)
        spline_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        spline_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
        spline_widget.SetPlaceFactor(factor)
        spline_widget.PlaceWidget(bounds)
        spline_widget.SetResolution(resolution)
        if initial_points is not None:
            spline_widget.InitializeHandles(pyvista.vtk_points(initial_points))
        else:
            spline_widget.SetClosed(closed)
        spline_widget.Modified()
        spline_widget.On()
        spline_widget.AddObserver(
            _parse_interaction_event(interaction_event),
            _the_callback,
        )
        _the_callback(spline_widget, None)

        if show_ribbon:
            self.add_mesh(ribbon, color=ribbon_color, opacity=ribbon_opacity)  # type: ignore[attr-defined]

        self.spline_widgets.append(spline_widget)
        return spline_widget

    def clear_spline_widgets(self):
        """Remove all of the spline widgets."""
        for spline_widget in self.spline_widgets:
            spline_widget.Off()
        self.spline_widgets.clear()

    @_deprecate_positional_args(allowed=['mesh'])
    def add_mesh_slice_spline(  # noqa: PLR0917
        self,
        mesh,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        n_handles=5,
        resolution=25,
        widget_color=None,
        show_ribbon: bool = False,  # noqa: FBT001, FBT002
        ribbon_color='pink',
        ribbon_opacity=0.5,
        initial_points=None,
        closed: bool = False,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
        **kwargs,
    ):
        """Slice a mesh with a spline widget.

        Add a mesh to the scene with a spline widget that is used to slice
        the mesh interactively.

        The sliced mesh is saved to the ``.spline_sliced_meshes`` attribute on
        the plotter.

        Parameters
        ----------
        mesh : DataSet or :vtk:`vtkAlgorithm`
            The input dataset to add to the scene and slice along the spline
            or algorithm that produces said mesh.

        generate_triangles : bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        n_handles : int, optional
            The number of interactive spheres to control the spline's
            parametric function.

        resolution : int, optional
            The number of points to generate on the spline.

        widget_color : ColorLike, optional
            Color of the widget.  Either a string, RGB sequence, or
            hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        show_ribbon : bool, optional
            If ``True``, the poly plane used for slicing will also be shown.

        ribbon_color : ColorLike, optional
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

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

        **kwargs : dict, optional
            All additional keyword arguments are passed to
            :func:`pyvista.Plotter.add_mesh` to control how the mesh is
            displayed.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the mesh.

        """
        mesh, algo = algorithm_to_mesh_handler(mesh)
        name = kwargs.get('name')
        if name is None:
            name = mesh.memory_address
        rng = mesh.get_data_range(kwargs.get('scalars'))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.add_mesh(outline_algorithm(algo or mesh), name=f'{name}-outline', opacity=0.0)  # type: ignore[attr-defined]

        alg = _vtk.vtkCutter()  # Construct the cutter object
        # Use the grid as the data we desire to cut
        set_algorithm_input(alg, algo or mesh)
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
            interaction_event=_parse_interaction_event(interaction_event),
        )

        return self.add_mesh(alg, **kwargs)  # type: ignore[attr-defined]

    def add_measurement_widget(
        self,
        callback=None,
        color=None,
    ):
        """Interactively measure distance with a distance widget.

        Creates an overlay documenting the selected line and total
        distance between two mouse left-click interactions.

        The measurement overlay stays on the rendering until the
        widget is deleted. Only one measurement can be added by each
        widget instance.

        Parameters
        ----------
        callback : Callable[[tuple[float, float, float], [tuple[float, float, float], int], float]
            The method called every time the widget calculates a
            distance measurement. This callback receives the start
            point and end point as cartesian coordinate tuples
            and the calculated distance between the two points.

        color : ColorLike, optional
            The color of the measurement widget.

        Returns
        -------
        :vtk:`vtkDistanceWidget`
            The newly created distance widget.

        See Also
        --------
        :ref:`distance_measurement_example`

        """
        msg = 'Cannot add a widget to a closed plotter.'
        iren = self._get_iren_not_none(msg)  # type: ignore[attr-defined]

        if color is None:
            color = pyvista.global_theme.font.color.float_rgb
        color = Color(color)

        compute = lambda a, b: np.sqrt(np.sum((np.array(b) - np.array(a)) ** 2))

        handle = _vtk.vtkPointHandleRepresentation3D()
        representation = _vtk.vtkDistanceRepresentation3D()
        representation.SetHandleRepresentation(handle)
        widget = _vtk.vtkDistanceWidget()
        widget.SetInteractor(iren.interactor)
        widget.SetRepresentation(representation)

        handle.GetProperty().SetColor(*color.float_rgb)
        representation.GetLabelProperty().SetColor(*color.float_rgb)
        representation.GetLineProperty().SetColor(*color.float_rgb)

        iren.picker = PickerType.POINT

        def place_point(*_):
            p1 = [0, 0, 0]
            p2 = [0, 0, 0]
            representation.GetPoint1DisplayPosition(p1)  # type: ignore[arg-type]
            representation.GetPoint2DisplayPosition(p2)  # type: ignore[arg-type]
            if iren.picker.Pick(p1, self.renderer):  # type: ignore[attr-defined]
                pos1 = iren.picker.GetPickPosition()
                representation.GetPoint1Representation().SetWorldPosition(pos1)
            if iren.picker.Pick(p2, self.renderer):  # type: ignore[attr-defined]
                pos2 = iren.picker.GetPickPosition()
                representation.GetPoint2Representation().SetWorldPosition(pos2)
            representation.BuildRepresentation()

            a = representation.GetPoint1Representation().GetWorldPosition()
            b = representation.GetPoint2Representation().GetWorldPosition()
            if callable(callback):
                try_callback(callback, a, b, compute(a, b))

        widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, place_point)

        widget.On()
        self.distance_widgets.append(widget)
        return widget

    def clear_measure_widgets(self):
        """Remove all of the measurement widgets."""
        for distance_widget in self.distance_widgets:
            distance_widget.Off()
        self.distance_widgets.clear()

    @_deprecate_positional_args(allowed=['callback'])
    def add_sphere_widget(  # noqa: PLR0917
        self,
        callback,
        center=(0, 0, 0),
        radius=0.5,
        theta_resolution=30,
        phi_resolution=30,
        color=None,
        style='surface',
        selected_color='pink',
        indices=None,
        pass_widget: bool = False,  # noqa: FBT001, FBT002
        test_callback: bool = True,  # noqa: FBT001, FBT002
        interaction_event: InteractionEventType = 'end',
    ):
        """Add one or many sphere widgets to a scene.

        Use a sphere widget to control a vertex location.

        Parameters
        ----------
        callback : callable
            The function to call back when the widget is modified. It takes a
            single argument: the center of the sphere as an XYZ coordinate (a
            3-length sequence), unless ``pass_widget=True``, in which case the
            callback must accept the widget object as the second parameter.  If
            multiple centers are passed in the ``center`` parameter, the
            callback must also accept an index of that widget.

        center : sequence[float], optional
            The cartesian coordinate of the sphere's center when placing it in
            the scene. If more than one location is passed, then that many
            widgets will be added and the callback will also be passed the
            integer index of that widget.

        radius : float, optional
            The radius of the sphere.

        theta_resolution : int, optional
            Set the number of points in the longitude direction.

        phi_resolution : int, optional
            Set the number of points in the latitude direction.

        color : ColorLike, optional
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

        selected_color : ColorLike, optional
            Color of the widget when selected during interaction.

        indices : sequence[int], optional
            Indices to assign the sphere widgets.

        pass_widget : bool, optional
            If ``True``, the widget will be passed as the last
            argument of the callback.

        test_callback : bool, optional
            If ``True``, run the callback function after the widget is
            created.

        interaction_event : InteractionEventType, optional
            The VTK interaction event to use for triggering the
            callback. Accepts either the strings ``'start'``, ``'end'``,
            ``'always'`` or a :vtk:`vtkCommand.EventIds`.

        Returns
        -------
        :vtk:`vtkSphereWidget`
            The sphere widget.

        See Also
        --------
        :ref:`sphere_widget_example`

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

        def _the_callback(widget, _event):
            point = widget.GetCenter()
            index = widget.WIDGET_INDEX
            if callable(callback):
                args = [point, index] if num > 1 else [point]
                if pass_widget:
                    args.append(widget)
                try_callback(callback, *args)

        if indices is None:
            indices = list(range(num))

        for i in range(num):
            loc = center[i] if center.ndim > 1 else center
            sphere_widget = _vtk.vtkSphereWidget()
            sphere_widget.WIDGET_INDEX = indices[i]  # type: ignore[attr-defined] # Monkey patch the index
            if style in 'wireframe':
                sphere_widget.SetRepresentationToWireframe()
            else:
                sphere_widget.SetRepresentationToSurface()
            sphere_widget.GetSphereProperty().SetColor(Color(colors[i]).float_rgb)
            sphere_widget.GetSelectedSphereProperty().SetColor(selected_color.float_rgb)
            sphere_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
            sphere_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
            sphere_widget.SetRadius(radius)
            sphere_widget.SetCenter(loc)
            sphere_widget.SetThetaResolution(theta_resolution)
            sphere_widget.SetPhiResolution(phi_resolution)
            sphere_widget.Modified()
            sphere_widget.On()
            sphere_widget.AddObserver(
                _parse_interaction_event(interaction_event),
                _the_callback,
            )
            self.sphere_widgets.append(sphere_widget)

        if test_callback is True:
            # Test call back in the last
            _the_callback(sphere_widget, None)
        if num > 1:
            return self.sphere_widgets

        return sphere_widget

    def clear_sphere_widgets(self):
        """Remove all of the sphere widgets."""
        for sphere_widget in self.sphere_widgets:
            sphere_widget.Off()
        self.sphere_widgets.clear()

    @_deprecate_positional_args(allowed=['actor'])
    def add_affine_transform_widget(  # noqa: PLR0917
        self,
        actor,
        origin=None,
        start: bool = True,  # noqa: FBT001, FBT002
        scale=0.15,
        line_radius=0.02,
        always_visible: bool = True,  # noqa: FBT001, FBT002
        axes_colors=None,
        axes=None,
        release_callback=None,
        interact_callback=None,
    ):
        """Add a 3D affine transform widget.

        This widget allows interactive transformations including translation and
        rotation using the left mouse button.

        Parameters
        ----------
        actor : pyvista.Actor
            The actor to which the widget is attached to.
        origin : sequence[float], optional
            Origin of the widget. Default is the origin of the main actor.
        start : bool, default: True
            If True, start the widget immediately.
        scale : float, default: 0.15
            Scale factor for the widget relative to the length of the actor.
        line_radius : float, default: 0.02
            Relative radius of the lines composing the widget.
        always_visible : bool, default: True
            Make the widget always visible. Setting this to ``False`` will cause
            the widget geometry to be hidden by other actors in the plotter.
        axes_colors : tuple[ColorLike], optional
            Uses the theme by default. Configure the individual axis colors by
            modifying either the theme with ``pyvista.global_theme.axes.x_color =
            <COLOR>`` or setting this with a ``tuple`` as in ``('r', 'g', 'b')``.
        axes : numpy.ndarray, optional
            ``(3, 3)`` Numpy array defining the X, Y, and Z axes. By default
            this matches the default coordinate system.
        release_callback : callable, optional
            Call this method when releasing the left mouse button. It is passed
            the ``user_matrix`` of the actor.
        interact_callback : callable, optional
            Call this method when moving the mouse with the left mouse button
            pressed down and a valid movement actor selected. It is passed the
            ``user_matrix`` of the actor.

        Returns
        -------
        pyvista.plotting.widgets.AffineWidget3D
            The affine widget.

        Notes
        -----
        After interacting with the actor, the transform will be stored within
        :attr:`pyvista.Prop3D.user_matrix` but will not be applied to the
        dataset. Use this matrix in conjunction with
        :func:`pyvista.DataObjectFilters.transform` to transform the dataset.

        Examples
        --------
        Add the 3d affine widget.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> widget = pl.add_affine_transform_widget(actor)
        >>> pl.show()

        Access the transform from the actor.

        >>> actor.user_matrix
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        """
        return AffineWidget3D(
            self,
            actor,
            origin=origin,
            start=start,
            scale=scale,
            line_radius=line_radius,
            always_visible=always_visible,
            axes_colors=axes_colors,
            axes=axes,
            release_callback=release_callback,
            interact_callback=interact_callback,
        )

    @_deprecate_positional_args(allowed=['callback'])
    def add_checkbox_button_widget(  # noqa: PLR0917
        self,
        callback,
        value: bool = False,  # noqa: FBT001, FBT002
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

        value : bool, default: False
            The default state of the button.

        position : sequence[float], default: (10.0, 10.0)
            The absolute coordinates of the bottom left point of the button.

        size : int, default: 50
            The size of the button in number of pixels.

        border_size : int, default: 5
            The size of the borders of the button in pixels.

        color_on : ColorLike, optional
            The color used when the button is checked. Default is ``'blue'``.

        color_off : ColorLike, optional
            The color used when the button is not checked. Default is ``'grey'``.

        background_color : ColorLike, optional
            The background color of the button. Default is ``'white'``.

        Returns
        -------
        :vtk:`vtkButtonWidget`
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
        msg = 'Cannot add a widget to a closed plotter.'
        self._get_iren_not_none(msg)  # type: ignore[attr-defined]

        def create_button(color1, color2, color3, *, dims=(size, size, 1)):
            color1 = np.array(Color(color1).int_rgb)
            color2 = np.array(Color(color2).int_rgb)
            color3 = np.array(Color(color3).int_rgb)

            n_points = dims[0] * dims[1]
            button = pyvista.ImageData(dimensions=dims)
            arr = np.array([color1] * n_points).reshape(dims[0], dims[1], 3)  # fill with color1
            arr[1 : dims[0] - 1, 1 : dims[1] - 1] = color2  # apply color2
            arr[border_size : dims[0] - border_size, border_size : dims[1] - border_size] = (
                color3  # apply color3
            )
            button.point_data['texture'] = arr.reshape(n_points, 3).astype(np.uint8)
            return button

        button_on = create_button(color_on, background_color, color_on)
        button_off = create_button(color_on, background_color, color_off)

        bounds = [
            position[0],
            position[0] + size,
            position[1],
            position[1] + size,
            0.0,
            0.0,
        ]

        button_rep = _vtk.vtkTexturedButtonRepresentation2D()
        button_rep.SetNumberOfStates(2)
        button_rep.SetState(value)
        button_rep.SetButtonTexture(0, button_off)
        button_rep.SetButtonTexture(1, button_on)
        button_rep.SetPlaceFactor(1)
        button_rep.PlaceWidget(bounds)

        button_widget = _vtk.vtkButtonWidget()
        button_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        button_widget.SetRepresentation(button_rep)
        button_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
        button_widget.On()

        def _the_callback(widget, _event):
            state = widget.GetRepresentation().GetState()
            if callable(callback):
                try_callback(callback, bool(state))

        button_widget.AddObserver(_vtk.vtkCommand.StateChangedEvent, _the_callback)
        self.button_widgets.append(button_widget)
        return button_widget

    @_deprecate_positional_args(allowed=['callback', 'radio_button_group'])
    def add_radio_button_widget(  # noqa: PLR0917
        self,
        callback,
        radio_button_group,
        value: bool = False,  # noqa: FBT001, FBT002
        title=None,
        position=(10.0, 10.0),
        size=50,
        border_size=8,
        color_on='blue',
        color_off='grey',
        background_color=None,
    ):
        """Add a radio button widget to the scene.

        Radio buttons work in groups. Only one button in a group can be on at
        at the same time. Typically you should add two or more buttons belonging
        to a same radio button group. Each button should be passed a callback
        function. This function will be called when a radio button in a group
        is switched on, assuming it was not already on.

        Parameters
        ----------
        callback : callable
            The method called when a radio button's state changes from off to
            on.

        radio_button_group: str
            Name of the group for the radio button.

        value : bool, default: False
            The default state of the button. If multiple buttons in the same
            group are initialized with to True state, only the last initialized
            button will remain on.

        title: str, optional
            String title to be displayed next to the radio button.

        position : sequence[float], default: (10.0, 10.0)
            The absolute coordinates of the bottom left point of the button.

        size : int, default: 50
            The diameter of the button in number of pixels.

        border_size : int, default: 8
            The size of the borders of the button in pixels.

        color_on : ColorLike, default: ``'blue'``
            The color used when the button is checked.

        color_off : ColorLike, default: ``'grey'``
            The color used when the button is not checked.

        background_color : ColorLike, optional
            The background color of the button. If not set, default  will be set
            as ``self.background_color``.

        Returns
        -------
        :vtk:`vtkButtonWidget`
            The VTK button widget configured as a radio button.

        Examples
        --------
        The following example creates a background color switcher.

        >>> import pyvista as pv
        >>> p = pv.Plotter()
        >>> def set_bg(color):
        ...     def wrapped_callback():
        ...         p.background_color = color
        ...
        ...     return wrapped_callback
        >>> _ = p.add_radio_button_widget(
        ...     set_bg('white'),
        ...     'bgcolor',
        ...     position=(10.0, 200.0),
        ...     title='White',
        ...     value=True,
        ... )
        >>> _ = p.add_radio_button_widget(
        ...     set_bg('lightblue'),
        ...     'bgcolor',
        ...     position=(10.0, 140.0),
        ...     title='Light Blue',
        ... )
        >>> _ = p.add_radio_button_widget(
        ...     set_bg('pink'),
        ...     'bgcolor',
        ...     position=(10.0, 80.0),
        ...     title='Pink',
        ... )
        >>> p.show()

        """
        msg = 'Cannot add a widget to a closed plotter.'
        self._get_iren_not_none(msg)  # type: ignore[attr-defined]

        if radio_button_group not in self.radio_button_widget_dict:
            self.radio_button_widget_dict[radio_button_group] = []
        if title is not None:
            if radio_button_group not in self.radio_button_title_dict:
                self.radio_button_title_dict[radio_button_group] = []
            button_title = self.add_text(  # type: ignore[attr-defined]
                title,
                position=(position[0] + size + 10.0, position[1] + 7.5),
                font_size=15,
            )
            self.radio_button_title_dict[radio_button_group].append(button_title)

        color_on = Color(color_on)
        color_off = Color(color_off)
        background_color = Color(background_color, default_color=self.background_color)  # type: ignore[attr-defined]

        def create_radio_button(fg_color, bg_color, size=size, smooth=2):  # noqa: PLR0917
            fg_color = np.array(fg_color.int_rgb)
            bg_color = np.array(bg_color.int_rgb)

            n_points = size**2
            button = pyvista.ImageData(dimensions=(size, size, 1))
            arr = np.array([bg_color] * n_points).reshape(size, size, 3)  # fill background

            centre = size / 2
            rad_outer = centre
            rad_inner = centre - border_size
            # Paint radio button with simple anti-aliasing
            for i, j in product(range(size), range(size)):
                distance = np.sqrt((i - size / 2) ** 2 + (j - size / 2) ** 2)
                if distance < rad_inner:
                    arr[i, j] = fg_color
                elif rad_inner <= distance <= rad_inner + smooth:
                    blend = (distance - rad_inner) / smooth
                    arr[i, j] = (1 - blend) * fg_color + blend * bg_color
                elif rad_outer - 2 * smooth <= distance <= rad_outer:
                    blend = abs(distance - rad_outer + smooth) / smooth
                    arr[i, j] = (1 - blend) * fg_color + blend * bg_color

            button.point_data['texture'] = arr.reshape(n_points, 3).astype(np.uint8)
            return button

        button_on = create_radio_button(color_on, background_color)
        button_off = create_radio_button(color_off, background_color)

        bounds = [
            position[0],
            position[0] + size,
            position[1],
            position[1] + size,
            0.0,
            0.0,
        ]

        button_rep = _vtk.vtkTexturedButtonRepresentation2D()
        button_rep.SetNumberOfStates(2)
        button_rep.SetState(value)
        button_rep.SetButtonTexture(0, button_off)
        button_rep.SetButtonTexture(1, button_on)
        button_rep.SetPlaceFactor(1)
        button_rep.PlaceWidget(bounds)
        button_rep.GetProperty().SetColor((1, 1, 1))

        button_widget = _vtk.vtkButtonWidget()
        button_widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        button_widget.SetRepresentation(button_rep)
        button_widget.SetCurrentRenderer(self.renderer)  # type: ignore[attr-defined]
        button_widget.On()

        def toggle_other_buttons_off(widget):
            other_buttons = [
                w for w in self.radio_button_widget_dict[radio_button_group] if w is not widget
            ]
            for w in other_buttons:
                w.GetRepresentation().SetState(0)

        def _the_callback(widget, _event):
            widget_rep = widget.GetRepresentation()
            state = widget_rep.GetState()
            # Toggle back on, if button was already on, and was clicked off
            if not state:
                widget_rep.SetState(1)
                state = True
            else:
                toggle_other_buttons_off(widget)
            if callable(callback):
                try_callback(callback)

        button_widget.AddObserver(_vtk.vtkCommand.StateChangedEvent, _the_callback)
        self.radio_button_widget_dict[radio_button_group].append(button_widget)
        if value:
            toggle_other_buttons_off(button_widget)
        return button_widget

    def clear_radio_button_widgets(self):
        """Remove all of the radio button widgets."""
        for widgets in self.radio_button_widget_dict.values():
            for widget in widgets:
                widget.Off()
        self.radio_button_widget_dict.clear()
        for titles in self.radio_button_title_dict.values():
            for title in titles:
                title.VisibilityOff()
        self.radio_button_title_dict.clear()

    @_deprecate_positional_args
    def add_camera_orientation_widget(self, animate: bool = True, n_frames=20):  # noqa: FBT001, FBT002
        """Add a camera orientation widget to the active renderer.

        Parameters
        ----------
        animate : bool, default: True
            Enable or disable jump-to-axis-view animation.

        n_frames : int, default: 20
            The number of frames to animate the jump-to-axis-viewpoint feature.

        Returns
        -------
        :vtk:`vtkCameraOrientationWidget`
            Camera orientation widget.

        See Also
        --------
        :meth:`~pyvista.Plotter.add_axes`
            Add arrow-style axes as an orientation widget.

        :meth:`~pyvista.Plotter.add_box_axes`
            Add an axes box as an orientation widget.

        :ref:`axes_objects_example`
            Example showing different axes objects.

        Examples
        --------
        Add a camera orientation widget to the scene.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(mesh, scalars=range(6), show_scalar_bar=False)
        >>> _ = plotter.add_camera_orientation_widget()
        >>> plotter.show()

        """
        widget = _vtk.vtkCameraOrientationWidget()
        widget.SetParentRenderer(self.renderer)  # type: ignore[attr-defined]
        widget.SetAnimate(animate)
        widget.SetAnimatorTotalFrames(n_frames)
        widget.On()
        self.camera_widgets.append(widget)
        return widget

    def clear_camera_widgets(self):
        """Remove all of the camera widgets."""
        for camera_widget in self.camera_widgets:
            camera_widget.Off()
        self.camera_widgets.clear()

    def clear_button_widgets(self):
        """Remove all of the button widgets."""
        for button_widget in self.button_widgets:
            button_widget.Off()
        self.button_widgets.clear()

    @_deprecate_positional_args(allowed=['logo'])
    def add_logo_widget(  # noqa: PLR0917
        self,
        logo: pyvista.ImageData | str | pathlib.Path | None = None,
        position: VectorLike[float] = (0.75, 0.8),
        size: VectorLike[float] = (0.2, 0.2),
        opacity: float = 1.0,
    ):
        """Add a logo widget to the top of the viewport.

        If no logo is passed, the PyVista logo will be used.

        Parameters
        ----------
        logo : pyvista.ImageData or pathlib.Path, optional
            The logo to display. If a pathlike is passed, it is assumed to be a
            file path to an image.

        position : tuple(float), optional
            The position of the logo in the viewport. The first value is the
            horizontal position and the second value is the vertical position.
            Both values must be between 0 and 1.

        size : tuple(float), optional
            The size of the logo in the viewport. The first value is the
            horizontal size and the second value is the vertical size. Both
            values must be between 0 and 1.

        opacity : float, optional
            The opacity of the logo. Must be between 0 and 1.

        Returns
        -------
        :vtk:`vtkLogoWidget`
            The logo widget.

        Examples
        --------
        Add a logo widget to the scene.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_logo_widget()
        >>> _ = pl.add_mesh(pv.Sphere(), show_edges=True)
        >>> pl.show()

        """
        if logo is None:
            logo = pyvista.global_theme.logo_file
        if logo is None:
            # Fallback to PyVista logo
            from pyvista import examples

            logo = examples.logofile

        # Read dataset and narrow the logo type to ImageData
        logo_maybe: pyvista.DataObject | str | pathlib.Path | None
        logo_maybe = pyvista.read(logo) if isinstance(logo, (str, pathlib.Path)) else logo
        if not isinstance(logo_maybe, pyvista.ImageData):
            msg = 'Logo must be a pyvista.ImageData or a file path to an image.'
            raise TypeError(msg)
        else:
            logo = logo_maybe

        representation = _vtk.vtkLogoRepresentation()
        representation.SetImage(logo)
        representation.SetPosition(*position)
        representation.SetPosition2(*size)
        representation.GetImageProperty().SetOpacity(opacity)
        widget = _vtk.vtkLogoWidget()
        widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        widget.SetRepresentation(representation)
        widget.On()
        self.logo_widgets.append(widget)
        return widget

    def clear_logo_widgets(self):
        """Remove all of the logo widgets."""
        for logo_widget in self.logo_widgets:
            logo_widget.Off()
        self.logo_widgets.clear()

    def add_camera3d_widget(self):
        """Add a camera3d widget allow to move the camera.

        .. note::
           This widget requires ``vtk>=9.3.0``.

        Returns
        -------
        :vtk:`vtkCamera3DWidget`
            The camera3d widget.

        Examples
        --------
        Add a camera3d widget to the scene.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> plotter = pv.Plotter(shape=(1, 2))
        >>> _ = plotter.add_mesh(sphere, show_edges=True)
        >>> plotter.subplot(0, 1)
        >>> _ = plotter.add_mesh(sphere, show_edges=True)
        >>> _ = plotter.add_camera3d_widget()
        >>> plotter.show(cpos=plotter.camera_position)

        """
        try:
            from vtkmodules.vtkInteractionWidgets import vtkCamera3DRepresentation
            from vtkmodules.vtkInteractionWidgets import vtkCamera3DWidget
        except ImportError:  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            msg = 'vtkCamera3DWidget requires vtk>=9.3.0'
            raise VTKVersionError(msg)
        representation = vtkCamera3DRepresentation()
        representation.SetCamera(self.renderer.GetActiveCamera())  # type: ignore[attr-defined]
        widget = vtkCamera3DWidget()
        widget.SetInteractor(self.iren.interactor)  # type: ignore[attr-defined]
        widget.SetRepresentation(representation)
        widget.On()
        self.camera3d_widgets.append(widget)
        return widget

    def clear_camera3d_widgets(self):
        """Remove all of the camera3d widgets."""
        for camera3d_widget in self.camera3d_widgets:
            camera3d_widget.Off()
        self.camera3d_widgets.clear()

    def close(self):
        """Close the widgets."""
        self.clear_box_widgets()
        self.clear_plane_widgets()
        self.clear_line_widgets()
        self.clear_slider_widgets()
        self.clear_sphere_widgets()
        self.clear_spline_widgets()
        self.clear_button_widgets()
        self.clear_radio_button_widgets()
        self.clear_camera_widgets()
        self.clear_measure_widgets()
        self.clear_logo_widgets()
        self.clear_camera3d_widgets()
