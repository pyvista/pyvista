"""Module managing picking events.

The plotter exposes the public picking API through the
:class:`PickingComponent` plotter component, registered under
``plotter.picking``. The component owns every picking observer, picked-
result attribute, and per-pick representation actor, and centralizes
teardown in ``__plotter_close__``.

Top-level methods on :class:`pyvista.BasePlotter` such as
``enable_point_picking``, ``disable_picking``, and ``picked_point``
forward to the component for backward compatibility.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import cast
import weakref

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import try_callback

from .composite_mapper import CompositePolyDataMapper
from .errors import PyVistaPickingError
from .mapper import _mapper_get_data_set_input
from .mapper import _prop_get_data_set_input
from .opts import ElementType
from .opts import PickerType

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from numpy.typing import NDArray

    from pyvista.core._typing_core import VectorLike
    from pyvista.core.pointset import PolyData

    from ._typing import ColorLike
    from ._typing import StyleOptions
    from .plotter import BasePlotter
    from .render_window_interactor import InteractorStyleCaptureMixin
    from .text import CornerAnnotation

    # Pickers that resolve a dataset.
    _DataSetPicker: TypeAlias = _vtk.vtkPicker | _vtk.vtkHardwarePicker

PICKED_REPRESENTATION_NAMES = {
    'point': '_picked_point',
    'mesh': '_picked_mesh',
    'through': '_picked_through_selection',
    'visible': '_picked_visible_selection',
    'element': '_picked_element',
    'path': '_picked_path',
    'horizon': '_picked_horizon',
    'frustum': '_rectangle_selection_frustum',
}


def _launch_pick_event(interactor: _vtk.vtkRenderWindowInteractor, _event: str) -> None:
    """Create a Pick event based on coordinate or left-click."""
    click_x, click_y = interactor.GetEventPosition()
    click_z = 0

    picker = interactor.GetPicker()
    style = cast('InteractorStyleCaptureMixin', interactor.GetInteractorStyle())
    parent = style._parent()
    if parent is None:  # pragma: no cover
        return
    picker.Pick(click_x, click_y, click_z, parent.get_poked_renderer())


def _poked_context_callback(plotter: BasePlotter, *args, **kwargs) -> None:
    """Invoke a picking callback from within a poked renderer subplot context."""
    with plotter._get_iren_not_none().poked_subplot():
        try_callback(*args, **kwargs)


class RectangleSelection(_NoNewAttrMixin):
    """Internal data structure for rectangle based selections.

    Parameters
    ----------
    frustum : :vtk:`vtkPlanes`
        Frustum that defines the selection.
    viewport : tuple[float, float, float, float]
        The selected viewport coordinates, given as ``(x0, y0, x1, y1)``.

    """

    def __init__(
        self, frustum: _vtk.vtkPlanes, viewport: tuple[float, float, float, float]
    ) -> None:
        self._frustum = frustum
        self._viewport = viewport

    @property
    def frustum(self) -> _vtk.vtkPlanes:  # numpydoc ignore=RT01
        """Get the selected frustum through the scene."""
        return self._frustum

    @property
    def frustum_mesh(self) -> pv.PolyData:  # numpydoc ignore=RT01
        """Get the frustum as a PyVista mesh."""
        frustum_source = _vtk.vtkFrustumSource()
        frustum_source.ShowLinesOff()
        frustum_source.SetPlanes(self.frustum)
        frustum_source.Update()
        return pv.wrap(frustum_source.GetOutput())

    @property
    def viewport(self) -> tuple[float, float, float, float]:  # numpydoc ignore=RT01
        """Get the selected viewport coordinates.

        Coordinates are given as: ``(x0, y0, x1, y1)``
        """
        return self._viewport


class PointPickingElementHandler(_NoNewAttrMixin):
    """Internal picking handler for element-based picking.

    This handler is only valid for single point picking operations.

    Parameters
    ----------
    mode : ElementType, optional
        The element type to pick.
    callback : callable, optional
        A callback function to be executed on picking events.

    """

    def __init__(
        self,
        mode: ElementType | str | int = ElementType.CELL,
        callback: Callable[..., None] | None = None,
    ) -> None:
        self._picker_: weakref.ref[_DataSetPicker] | None = None
        self.callback = callback
        self.mode = ElementType.from_any(mode)

    @property
    def picker(self) -> _DataSetPicker:  # numpydoc ignore=RT01
        """Get or set the picker instance."""
        picker = None if self._picker_ is None else self._picker_()
        if picker is None:  # pragma: no cover
            msg = 'No picker has been set on this handler.'
            raise PyVistaPickingError(msg)
        return picker

    @picker.setter
    def picker(self, picker: _DataSetPicker) -> None:
        self._picker_ = weakref.ref(picker)

    def get_mesh(self) -> pv.DataSet | None:
        """Get the picked mesh.

        Returns
        -------
        pyvista.DataSet | None
            Picked mesh, or ``None`` when the pick missed every mesh.

        """
        ds = self.picker.GetDataSet()
        if ds is not None:
            return pv.wrap(ds)
        return None

    def get_cell(self, picked_point: VectorLike[float]) -> pv.UnstructuredGrid | None:
        """Get the picked cell of the picked mesh.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.UnstructuredGrid | None
            UnstructuredGrid containing the picked cell, or ``None`` when no
            cell contains the point.

        """
        mesh = self._get_mesh_not_none()
        cell_id = mesh.find_containing_cell(picked_point)  # more accurate
        if cell_id < 0:
            return None  # TODO: this happens but shouldn't  # pragma: no cover
        cell = mesh.extract_cells(cell_id)
        cell.cell_data['vtkOriginalCellIds'] = np.array([cell_id])
        return cell

    def get_face(self, picked_point: VectorLike[float]) -> pv.UnstructuredGrid | None:
        """Get the picked face of the picked cell.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.UnstructuredGrid | None
            UnstructuredGrid containing the picked face, or ``None`` when no
            cell contains the point.

        """
        picked_cell = self.get_cell(picked_point)
        if picked_cell is None:  # pragma: no cover
            return None
        cell = picked_cell.get_cell(0)
        if cell.n_faces > 1:
            face = None
            for face_id, cell_face in enumerate(cell.faces):
                grid = cell_face.cast_to_unstructured_grid()
                if grid.find_containing_cell(picked_point) > -1:
                    face = grid
                    face.field_data['vtkOriginalFaceIds'] = np.array([face_id])
                    break
            if face is None:
                # this shouldn't happen
                msg = 'Trouble aligning point with face.'
                raise RuntimeError(msg)
        else:
            face = cell.cast_to_unstructured_grid()
            face.field_data['vtkOriginalFaceIds'] = np.array([0])

        return face

    def get_edge(self, picked_point: VectorLike[float]) -> pv.UnstructuredGrid | None:
        """Get the picked edge of the picked cell.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.UnstructuredGrid | None
            UnstructuredGrid containing the picked edge, or ``None`` when no
            cell contains the point.

        """
        picked_cell = self.get_cell(picked_point)
        if picked_cell is None:  # pragma: no cover
            return None
        cell = picked_cell.get_cell(0)
        if cell.n_edges > 1:
            ei = (
                cell.cast_to_unstructured_grid()
                .extract_all_edges()
                .find_closest_cell(picked_point)
            )
            edge = cell.edges[ei].cast_to_unstructured_grid()
            edge.field_data['vtkOriginalEdgeIds'] = np.array([ei])
        else:
            edge = cell.cast_to_unstructured_grid()

        return edge

    def get_point(self, picked_point: VectorLike[float]) -> PolyData:
        """Get the picked point of the picked mesh.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.PolyData
            Picked mesh containing the point.

        """
        mesh = self._get_mesh_not_none()
        pid = mesh.find_closest_point(np.asarray(picked_point))
        picked = mesh.extract_points(pid, adjacent_cells=False, include_cells=False)
        return picked.cast_to_poly_points()

    def _get_mesh_not_none(self) -> pv.DataSet:
        """Return the picked mesh, raising when the pick missed every mesh."""
        mesh = self.get_mesh()
        if mesh is None:  # pragma: no cover
            msg = 'The pick did not hit a mesh.'
            raise PyVistaPickingError(msg)
        return mesh

    def __call__(self, picked_point: VectorLike[float], picker: _DataSetPicker) -> None:
        """Perform the pick."""
        self.picker = picker
        mesh = self.get_mesh()
        if mesh is None:
            return  # No selected mesh (point not on surface of mesh)

        picked: pv.DataSet | None
        if self.mode == ElementType.MESH:
            picked = mesh
        elif self.mode == ElementType.CELL:
            picked = self.get_cell(picked_point)
        elif self.mode == ElementType.FACE:
            picked = self.get_face(picked_point)
        elif self.mode == ElementType.EDGE:
            picked = self.get_edge(picked_point)
        else:
            picked = self.get_point(picked_point)
        if picked is None:
            return  # TODO: handle

        if self.callback:
            try_callback(self.callback, picked)


class PickingComponent(_NoNewAttrMixin):
    """Plotter picking component.

    Owns every picking observer, picked-result attribute, and per-pick
    representation actor for a single :class:`pyvista.BasePlotter`.
    Constructed lazily on first access of ``plotter.picking`` and
    registered for close-time teardown via ``__plotter_close__``.

    The plotter exposes the public picking surface (``enable_*_picking``,
    ``disable_picking``, ``picked_point`` and friends) as forwarding
    shims that delegate here.

    Parameters
    ----------
    plotter : pyvista.BasePlotter
        Owning plotter. Stored as a strong reference; the component's
        lifetime is bounded by the plotter's lifetime.

    Attributes
    ----------
    picked_path : pyvista.PolyData | None
        PolyLine accumulated by :meth:`enable_path_picking`.
    picked_geodesic : pyvista.PolyData | None
        Geodesic polyline accumulated by :meth:`enable_geodesic_picking`.
    picked_horizon : pyvista.PolyData | None
        Ribbon surface produced by :meth:`enable_horizon_picking`.

    .. versionadded:: 0.48.0

    """

    def __init__(self, plotter: BasePlotter) -> None:
        """Initialize the picking component."""
        self._plotter = plotter
        # Low-level picking state
        self._picking_left_clicking_observer: int | None = None
        self._picking_right_clicking_observer: int | None = None
        self._picker_in_use = False
        self._picked_point: NDArray[np.float64] | None = None
        # Mesh-aware picking state
        self._picked_actor: _vtk.vtkActor | None = None
        self._picked_mesh: pv.DataSet | None = None
        self._picked_cell: pv.UnstructuredGrid | pv.MultiBlock | None = None
        self._picking_text: CornerAnnotation | None = None
        self._picked_block_index: int | None = None
        # Path / geodesic / horizon state
        self.picked_path: pv.PolyData | None = None
        self.picked_geodesic: pv.PolyData | None = None
        self.picked_horizon: pv.PolyData | None = None
        self._last_picked_idx: int | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _get_picked_mesh_not_none(self) -> pv.DataSet:
        """Return the picked mesh, raising when nothing has been picked."""
        if self._picked_mesh is None:  # pragma: no cover
            msg = 'No mesh has been picked.'
            raise PyVistaPickingError(msg)
        return self._picked_mesh

    def __plotter_close__(self) -> None:
        """Release picking observers when the owning plotter closes."""
        self.disable_picking()

    def __plotter_deep_clean__(self) -> None:
        """Release picking observers on deep clean."""
        self.disable_picking()

    # =========================================================================
    # Picked-result properties
    # =========================================================================

    @property
    def picked_point(self) -> NDArray[np.float64] | None:  # numpydoc ignore=RT01
        """Return the picked point."""
        return self._picked_point

    @property
    def picked_actor(self) -> _vtk.vtkActor | None:  # numpydoc ignore=RT01
        """Return the picked actor."""
        return self._picked_actor

    @property
    def picked_mesh(self) -> pv.DataSet | None:  # numpydoc ignore=RT01
        """Return the picked mesh."""
        return self._picked_mesh

    @property
    def picked_cells(self) -> pv.UnstructuredGrid | pv.MultiBlock | None:
        r"""Return the cell-picked object.

        Returns
        -------
        output : None | pyvista.UnstructuredGrid | pyvista.MultiBlock
            Picked object if available.

        """
        return self._picked_cell

    @property
    def picked_block_index(self) -> int | None:  # numpydoc ignore=RT01
        """Return the picked block index."""
        return self._picked_block_index

    # =========================================================================
    # Pick position helpers
    # =========================================================================

    def get_pick_position(self) -> tuple[int, int, int, int]:
        """Get the pick position or area.

        Returns
        -------
        sequence
            Picked position or area as ``(x0, y0, x1, y1)``.

        """
        renderer = self._plotter._get_iren_not_none().get_poked_renderer()
        return renderer.get_pick_position()

    def pick_click_position(self) -> tuple[float, float, float]:
        """Get corresponding click location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        plotter = self._plotter
        if plotter.click_position is None:
            plotter.store_click_position()
        click_position = plotter.click_position or (0, 0)
        iren = plotter._get_iren_not_none()
        iren.picker.Pick(click_position[0], click_position[1], 0, iren.get_poked_renderer())
        return iren.picker.GetPickPosition()

    def pick_mouse_position(self) -> tuple[float, float, float]:
        """Get corresponding mouse location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        plotter = self._plotter
        if plotter.mouse_position is None:
            plotter.store_mouse_position()
        mouse_position = plotter.mouse_position or (0, 0)
        iren = plotter._get_iren_not_none()
        iren.picker.Pick(mouse_position[0], mouse_position[1], 0, iren.get_poked_renderer())
        return iren.picker.GetPickPosition()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _clear_picking_representations(self) -> None:
        """Clear all picking representations."""
        for name in PICKED_REPRESENTATION_NAMES.values():
            self._plotter.remove_actor(name)

    def _init_click_picking_callback(self, *, left_clicking: bool = False) -> None:
        if left_clicking:
            self._picking_left_clicking_observer = self._plotter._get_iren_not_none().add_observer(
                'LeftButtonPressEvent',
                functools.partial(try_callback, _launch_pick_event),
            )
        else:
            self._picking_right_clicking_observer = (
                self._plotter._get_iren_not_none().add_observer(
                    'RightButtonPressEvent',
                    functools.partial(try_callback, _launch_pick_event),
                )
            )

    def _validate_picker_not_in_use(self) -> None:
        if self._picker_in_use:
            msg = (
                'Picking is already enabled, please disable previous picking '
                'with `disable_picking()`.'
            )
            raise PyVistaPickingError(msg)

    # =========================================================================
    # disable_picking
    # =========================================================================

    def disable_picking(self) -> None:
        """Disable any active picking and remove observers.

        Examples
        --------
        Enable and then disable picking.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 0, 0))
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_mesh_picking()
        >>> pl.disable_picking()

        """
        iren = getattr(self._plotter, 'iren', None)
        if iren is not None:
            iren.remove_observer(self._picking_left_clicking_observer)
            iren.remove_observer(self._picking_right_clicking_observer)
            iren.reset_picker()
        self._picking_left_clicking_observer = None
        self._picking_right_clicking_observer = None
        self._picker_in_use = False

        # Remove picking-text actor from every renderer.
        if self._picking_text is not None and hasattr(self._plotter, 'renderers'):
            for renderer in self._plotter.renderers:
                renderer.remove_actor(self._picking_text, render=False)
        self._picking_text = None

    # =========================================================================
    # Low-level picking entrypoints
    # =========================================================================

    def enable_point_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        tolerance: float = 0.025,
        left_clicking: bool = False,
        picker: PickerType | str | int | None = PickerType.POINT,
        show_message: bool | str = True,
        font_size: int = 18,
        color: ColorLike = 'pink',
        point_size: float = 10,
        show_point: bool = True,
        use_picker: bool = False,
        pickable_window: bool = False,
        clear_on_no_selection: bool = True,
        **kwargs,
    ) -> None:
        """Enable picking at points under the cursor.

        Enable picking a point at the mouse location in the render
        view using the right mouse button. This point is saved to the
        ``.picked_point`` attribute on the plotter. Pass a callback
        that takes that point as an argument. The picked
        point can either be a point on the first intersecting mesh, or
        a point in the 3D window.

        The ``picker`` choice will help determine how the point picking
        is performed.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a pick is made. The
            picked point is input as the first parameter to this
            callable.

        tolerance : float, tolerance: 0.025
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size. Rendering window size is measured across diagonal.
            This is only valid for some choices of ``picker``.

        left_clicking : bool, default: False
            When ``True``, points can be picked by clicking the left mouse
            button. Default is to use the right mouse button.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries.
                * ``'cell'``: Uses :vtk:`vtkCellPicker`.
                * ``'point'``: Uses :vtk:`vtkPointPicker` which will snap to
                  points on the surface of the mesh (default).
                * ``'volume'``: Uses :vtk:`vtkVolumePicker`.

        show_message : bool | str, default: True
            Show the message about how to use the point picking
            tool. If this is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the size of the message.

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

        point_size : int, default: 10
            Size of picked points if ``show_point`` is ``True``.

        show_point : bool, default: True
            Show the picked point after clicking.

        use_picker : bool, default: False
            When ``True``, the callback will also be passed the picker.

        pickable_window : bool, default: False
            When ``True`` and the chosen picker supports it, points in the
            3D window are pickable.

        clear_on_no_selection : bool, default: True
            Clear the selections when no point is selected.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked point is interactively displayed.

        Examples
        --------
        Enable point picking with a custom message.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> _ = pl.add_mesh(pv.Cube(), pickable=False)
        >>> pl.enable_point_picking(show_message='Pick a point')


        """
        self._validate_picker_not_in_use()
        if 'use_mesh' in kwargs:
            warn_external(
                '`use_mesh` is deprecated. See `use_picker` instead.',
                PyVistaDeprecationWarning,
            )
            use_mesh = kwargs.pop('use_mesh')
        else:
            use_mesh = False

        self_ = weakref.ref(self)

        def _end_pick_event(picker: _vtk.vtkAbstractPicker, _event: str) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if (
                not pickable_window
                and hasattr(picker, 'GetDataSet')
                and picker.GetDataSet() is None
            ):
                component._picked_point = None
                if clear_on_no_selection:
                    with plotter._get_iren_not_none().poked_subplot():
                        component._clear_picking_representations()
                return
            with plotter._get_iren_not_none().poked_subplot():
                point = np.array(picker.GetPickPosition())
                point /= plotter.scale  # HACK: handle scale
                component._picked_point = point
                if show_point:
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        point,
                        color=color,
                        point_size=point_size,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['point']),
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
                if callable(callback):
                    if use_picker:
                        _poked_context_callback(plotter, callback, point, picker)
                    elif use_mesh:  # Lower priority
                        # Only a picker that resolves a dataset and a point id
                        # can serve this deprecated mode.
                        point_picker = cast('_vtk.vtkPointPicker', picker)
                        _poked_context_callback(
                            plotter,
                            callback,
                            point_picker.GetDataSet(),
                            point_picker.GetPointId(),
                        )
                    else:
                        _poked_context_callback(plotter, callback, point)

        iren = self._plotter._get_iren_not_none()
        if picker is not None:  # If None, use the already-set picker
            iren.picker = picker
        active_picker = iren.picker
        if hasattr(active_picker, 'SetTolerance'):
            active_picker.SetTolerance(tolerance)
        iren.add_pick_observer(_end_pick_event)
        self._init_click_picking_callback(left_clicking=left_clicking)
        self._picker_in_use = True

        if show_message:
            if show_message is True:
                show_message = 'Left-click' if left_clicking else 'Right-click'
                show_message += ' or press P to pick under the mouse'
            self._picking_text = self._plotter.add_text(
                str(show_message),
                font_size=font_size,
                name='_point_picking_message',
            )

    def enable_rectangle_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show_message: bool | str = True,
        font_size: int = 18,
        start: bool = False,
        show_frustum: bool = False,
        style: StyleOptions = 'wireframe',
        color: ColorLike = 'pink',
        **kwargs,
    ) -> None:
        """Enable rectangle based picking at cells.

        Press ``"r"`` to enable rectangle based selection. Press
        ``"r"`` again to turn it off.

        Picking with the rectangle selection tool provides two values that
        are passed as the ``RectangleSelection`` object in the callback:

        1. ``RectangleSelection.viewport``: the viewport coordinates of the
           selection rectangle.
        2. ``RectangleSelection.frustum``: the full frustum made from
           the selection rectangle into the scene.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made.
            The ``RectangleSelection`` is the only passed argument
            containing the viewport coordinates of the selection and the
            projected frustum.

        show_message : bool | str, default: True
            Show the message about how to use the cell picking tool. If this
            is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the font size of the message.

        start : bool, default: False
            Automatically start the cell selection tool.

        show_frustum : bool, default: False
            Show the frustum in the scene.

        style : str, default: "wireframe"
            Visualization style of the selection frustum. One of the
            following: ``style='surface'``, ``style='wireframe'``, or
            ``style='points'``.

        color : ColorLike, default: "pink"
            The color of the selected frustum when shown.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the selection frustum is interactively displayed.

        Examples
        --------
        Add a mesh and a cube to a plot and enable cell picking.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 0, 0))
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_rectangle_picking()


        """
        self._validate_picker_not_in_use()

        self_ = weakref.ref(self)

        def _end_pick_helper(picker: _vtk.vtkAreaPicker, *_) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            renderer = picker.GetRenderer()
            x0 = int(renderer.GetPickX1())
            x1 = int(renderer.GetPickX2())
            y0 = int(renderer.GetPickY1())
            y1 = int(renderer.GetPickY2())

            selection = RectangleSelection(frustum=picker.GetFrustum(), viewport=(x0, y0, x1, y1))

            if show_frustum:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        selection.frustum_mesh,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['frustum']),
                        style=style,
                        color=color,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )

            if callback is not None:
                _poked_context_callback(plotter, callback, selection)

        self._plotter.enable_rubber_band_style()
        self._plotter._get_iren_not_none().picker = 'rendered'
        self._plotter._get_iren_not_none().add_pick_observer(_end_pick_helper)
        self._picker_in_use = True

        if show_message:
            if show_message is True:
                show_message = 'Press R to toggle selection tool'
            self._picking_text = self._plotter.add_text(
                str(show_message),
                font_size=font_size,
                name='_rectangle_picking_message',
            )

        if start:
            rubber_band_style = cast(
                '_vtk.vtkInteractorStyleRubberBandPick',
                self._plotter._get_iren_not_none().style,
            )
            rubber_band_style.StartSelect()

    # =========================================================================
    # Mesh-aware picking
    # =========================================================================

    def enable_surface_point_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show_message: bool | str = True,
        font_size: int = 18,
        color: ColorLike = 'pink',
        show_point: bool = True,
        point_size: float = 10,
        tolerance: float = 0.025,
        pickable_window: bool = False,
        left_clicking: bool = False,
        picker: PickerType | str | int = PickerType.CELL,
        use_picker: bool = False,
        clear_on_no_selection: bool = True,
        **kwargs,
    ) -> None:
        """Enable picking of a point on the surface of a mesh.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made. The
            picked point is input as the first parameter to this callable.

        show_message : bool | str, default: True
            Show the message about how to use the mesh picking tool. If this
            is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the font size of the message.

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

        show_point : bool, default: True
            Show the selection interactively.

        point_size : int, default: 10
            Size of picked points if ``show_point`` is ``True``.

        tolerance : float, default: 0.025
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size. Rendering window size is measured across diagonal.

            .. warning::
                This is ignored with the ``'hardware'`` ``picker``.

        pickable_window : bool, default: False
            When ``True``, points in the 3D window are pickable.

        left_clicking : bool, default: False
            When ``True``, meshes can be picked by clicking the left
            mouse button.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked mesh.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries.
                * ``'cell'``: Uses :vtk:`vtkCellPicker` (default).
                * ``'point'``: Uses :vtk:`vtkPointPicker` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses :vtk:`vtkVolumePicker`.

        use_picker : bool, default: False
            When ``True``, the callback will also be passed the picker.

        clear_on_no_selection : bool, default: True
            Clear the selections when no point is selected.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        Notes
        -----
        Picked point can be accessed from :attr:`picked_point
        <pyvista.Plotter.picked_point>` attribute.

        Examples
        --------
        Add a cube to a plot and enable cell picking.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_surface_point_picking()


        """
        picker = PickerType.from_any(picker)
        valid_pickers = [
            PickerType.POINT,
            PickerType.CELL,
            PickerType.HARDWARE,
            PickerType.VOLUME,
        ]
        if picker not in valid_pickers:
            msg = f'Invalid picker choice for surface picking. Use one of: {valid_pickers}'
            raise ValueError(msg)

        self_ = weakref.ref(self)

        def _end_pick_event(picked_point: VectorLike[float], picker: _DataSetPicker) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if not pickable_window and picker.GetActor() is None:
                component._picked_point = None
                component._picked_actor = None
                component._picked_mesh = None
                if clear_on_no_selection:
                    with plotter._get_iren_not_none().poked_subplot():
                        component._clear_picking_representations()
                return
            component._picked_actor = picker.GetActor()
            component._picked_mesh = cast('pv.DataSet | None', picker.GetDataSet())

            if show_point:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        picked_point,
                        color=color,
                        point_size=point_size,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['point']),
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
            if callable(callback):
                if use_picker:
                    _poked_context_callback(plotter, callback, picked_point, picker)
                else:
                    _poked_context_callback(plotter, callback, picked_point)

        self.enable_point_picking(
            callback=_end_pick_event,
            picker=picker,
            show_point=False,
            show_message=show_message,
            left_clicking=left_clicking,
            use_picker=True,
            font_size=font_size,
            tolerance=tolerance,
            pickable_window=True,  # let this callback handle pickable window
            clear_on_no_selection=clear_on_no_selection,
        )

    def enable_mesh_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show: bool = True,
        show_message: bool | str = True,
        style: StyleOptions = 'wireframe',
        line_width: float = 5,
        color: ColorLike = 'pink',
        font_size: int = 18,
        left_clicking: bool = False,
        use_actor: bool = False,
        picker: PickerType | str | int = PickerType.CELL,
        **kwargs,
    ) -> None:
        """Enable picking of a mesh.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made. The
            ``mesh`` is input as the first parameter to this callable.

        show : bool, default: True
            Show the selection interactively. Best when combined with
            ``left_clicking``.

        show_message : bool | str, default: True
            Show the message about how to use the mesh picking tool. If this
            is a string, that will be the message shown.

        style : str, default: "wireframe"
            Visualization style of the selection. One of the following:

            * ``'surface'``
            * ``'wireframe'``
            * ``'points'``

        line_width : float, default: 5.0
            Thickness of selected mesh edges.

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

        font_size : int, default: 18
            Sets the font size of the message.

        left_clicking : bool, default: False
            When ``True``, meshes can be picked by clicking the left
            mouse button.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked point.

        use_actor : bool, default: False
            If True, the callback will be passed the picked actor instead of
            the mesh object.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries.
                * ``'cell'``: Uses :vtk:`vtkCellPicker` (default).
                * ``'point'``: Uses :vtk:`vtkPointPicker` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses :vtk:`vtkVolumePicker`.


        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        Returns
        -------
        :vtk:`vtkPropPicker`
            Property picker.

        Examples
        --------
        Add a sphere and a cube to a plot and enable mesh picking. Enable
        ``left_clicking`` to immediately start picking on the left click and
        disable showing the box. You can still press the ``p`` key to select
        meshes.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 0, 0))
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_mesh_picking()


        """
        self_ = weakref.ref(self)

        def end_pick_call_back(*args) -> None:  # noqa: ARG001
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if callback:
                if use_actor:
                    _poked_context_callback(plotter, callback, component._picked_actor)
                else:
                    _poked_context_callback(plotter, callback, component._picked_mesh)

            if show:
                # Select the renderer where the mesh is added.
                active_renderer_index = plotter.renderers._active_index
                loc = plotter._get_iren_not_none().get_event_subplot_loc()
                plotter.subplot(*np.atleast_1d(loc))

                # Use try in case selection is empty or invalid
                try:
                    with plotter._get_iren_not_none().poked_subplot():
                        _kwargs = kwargs.copy()
                        plotter.add_mesh(
                            component._get_picked_mesh_not_none(),
                            name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['mesh']),
                            style=style,
                            color=color,
                            line_width=line_width,
                            pickable=_kwargs.pop('pickable', False),
                            reset_camera=_kwargs.pop('reset_camera', False),
                            **_kwargs,
                        )
                except Exception as e:  # noqa: BLE001  # pragma: no cover
                    warn_external(f'Unable to show mesh when picking:\n\n{e}')

                # Reset to the active renderer.
                loc = plotter.renderers.index_to_loc(active_renderer_index)
                plotter.subplot(*np.atleast_1d(loc))

                # render here prior to running the callback
                plotter.render()

        if show_message and show_message is True:
            show_message = 'Left-click' if left_clicking else 'Right-click'
            show_message += ' or press P to pick single dataset under the mouse pointer'

        self.enable_surface_point_picking(
            callback=end_pick_call_back,
            picker=picker,
            show_point=False,
            show_message=show_message,
            left_clicking=left_clicking,
            use_picker=True,
            font_size=font_size,
            pickable_window=False,
        )

    def enable_rectangle_through_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show: bool = True,
        style: StyleOptions = 'wireframe',
        line_width: float = 5,
        color: ColorLike = 'pink',
        show_message: bool | str = True,
        font_size: int = 18,
        start: bool = False,
        show_frustum: bool = False,
        **kwargs,
    ) -> None:
        """Enable rectangle based cell picking through the scene.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made.
            The picked cells is the only passed argument.

        show : bool, default: True
            Show the selection interactively.

        style : str, default: "wireframe"
            Visualization style of the selection frustum. One of the
            following: ``style='surface'``, ``style='wireframe'``, or
            ``style='points'``.

        line_width : float, default: 5.0
            Thickness of selected mesh edges.

        color : ColorLike, default: "pink"
            The color of the selected frustum when shown.

        show_message : bool | str, default: True
            Show the message about how to use the cell picking tool. If this
            is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the font size of the message.

        start : bool, default: False
            Automatically start the cell selection tool.

        show_frustum : bool, default: False
            Show the frustum in the scene.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the selection frustum is interactively displayed.


        """
        self_ = weakref.ref(self)

        def finalize(picked: pv.UnstructuredGrid | pv.MultiBlock | None) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if picked is None:
                # Indicates invalid pick
                with plotter._get_iren_not_none().poked_subplot():
                    component._clear_picking_representations()
                return

            component._picked_cell = picked

            if show:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        picked,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['through']),
                        style=style,
                        color=color,
                        line_width=line_width,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )

            if callback is not None:
                _poked_context_callback(plotter, callback, component.picked_cells)

        def through_pick_callback(selection: RectangleSelection) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            picked = pv.MultiBlock()
            renderer = plotter._get_iren_not_none().get_poked_renderer()
            for actor in renderer.actors.values():
                dataset = _prop_get_data_set_input(actor)
                if dataset is not None and actor.GetPickable():
                    input_mesh = pv.wrap(dataset)
                    input_mesh.cell_data['original_cell_ids'] = np.arange(input_mesh.n_cells)
                    extract = _vtk.vtkExtractGeometry()
                    extract.SetInputData(input_mesh)
                    extract.SetImplicitFunction(selection.frustum)
                    extract.Update()

                    if (wrapped := pv.wrap(extract.GetOutput())).n_cells > 0:
                        picked.append(wrapped)

            if picked.n_blocks == 0 or picked.combine().n_cells < 1:
                component._picked_cell = None
            elif picked.n_blocks == 1:
                component._picked_cell = cast('pv.UnstructuredGrid', picked[0])
            else:
                component._picked_cell = picked

            finalize(component._picked_cell)

        self.enable_rectangle_picking(
            callback=through_pick_callback,
            show_message=show_message,
            font_size=font_size,
            show_frustum=show_frustum,
            start=start,
            style=style,
            color=color,
        )

    def enable_rectangle_visible_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show: bool = True,
        style: StyleOptions = 'wireframe',
        line_width: float = 5,
        color: ColorLike = 'pink',
        show_message: bool | str = True,
        font_size: int = 18,
        start: bool = False,
        show_frustum: bool = False,
        **kwargs,
    ) -> None:
        """Enable rectangle based cell picking on visible surfaces.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made.
            The picked cells is the only passed argument.

        show : bool, default: True
            Show the selection interactively.

        style : str, default: "wireframe"
            Visualization style of the selection frustum. One of the
            following: ``style='surface'``, ``style='wireframe'``, or
            ``style='points'``.

        line_width : float, default: 5.0
            Thickness of selected mesh edges.

        color : ColorLike, default: "pink"
            The color of the selected frustum when shown.

        show_message : bool | str, default: True
            Show the message about how to use the cell picking tool. If this
            is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the font size of the message.

        start : bool, default: False
            Automatically start the cell selection tool.

        show_frustum : bool, default: False
            Show the frustum in the scene.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the selection frustum is interactively displayed.


        """
        self_ = weakref.ref(self)

        def finalize(picked: pv.UnstructuredGrid | pv.MultiBlock | None) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if picked is None:
                with plotter._get_iren_not_none().poked_subplot():
                    component._clear_picking_representations()
                return

            if show:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        picked,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['visible']),
                        style=style,
                        color=color,
                        line_width=line_width,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )

            if callback is not None:
                _poked_context_callback(plotter, callback, picked)

        def visible_pick_callback(selection: RectangleSelection) -> None:  # noqa: ARG001
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            picked = pv.MultiBlock()
            renderer = plotter._get_iren_not_none().get_poked_renderer()
            x0, y0, x1, y1 = renderer.get_pick_position()
            if x0 >= 0:  # initial pick position is (-1, -1, -1, -1)
                selector = _vtk.vtkOpenGLHardwareSelector()
                selector.SetFieldAssociation(_vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
                selector.SetRenderer(renderer)
                selector.SetArea(x0, y0, x1, y1)
                hardware_selection = selector.Select()

                for node in range(hardware_selection.GetNumberOfNodes()):
                    selection_node = hardware_selection.GetNode(node)
                    if selection_node is None:  # pragma: no cover
                        continue
                    cids = pv.convert_array(selection_node.GetSelectionList())
                    actor = selection_node.GetProperties().Get(_vtk.vtkSelectionNode.PROP())

                    if not actor.GetMapper() or not hasattr(
                        actor.GetProperty(),
                        'GetRepresentation',
                    ):
                        continue

                    if actor.GetProperty().GetRepresentation() != 2:  # pragma: no cover
                        warn_external(
                            'Display representations other than `surface` will result '
                            'in incorrect results.',
                        )
                    smesh = pv.wrap(_mapper_get_data_set_input(actor.GetMapper()))
                    smesh = smesh.copy()
                    smesh.cell_data['original_cell_ids'] = np.arange(smesh.n_cells)
                    tri_smesh = smesh.extract_surface(
                        algorithm=None, pass_pointid=False, pass_cellid=False
                    ).triangulate()
                    # The hardware selector can report ids past the triangulated mesh
                    cids = cids[cids < tri_smesh.n_cells]
                    cids_to_get = tri_smesh.extract_cells(cids)['original_cell_ids']
                    picked.append(smesh.extract_cells(cids_to_get))

                # memory leak issues on vtk==9.0.20210612.dev0
                # See https://gitlab.kitware.com/vtk/vtk/-/issues/18239#note_973826
                hardware_selection.UnRegister(hardware_selection)

            if len(picked) == 0 or picked.combine().n_cells < 1:
                component._picked_cell = None
            elif len(picked) == 1:
                component._picked_cell = cast('pv.UnstructuredGrid', picked[0])
            else:
                component._picked_cell = picked

            finalize(component._picked_cell)

        self.enable_rectangle_picking(
            callback=visible_pick_callback,
            show_message=show_message,
            font_size=font_size,
            start=start,
            show_frustum=show_frustum,
            style=style,
            color=color,
        )

    def enable_cell_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        through: bool = True,
        show: bool = True,
        show_message: bool | str = True,
        style: StyleOptions = 'wireframe',
        line_width: float = 5,
        color: ColorLike = 'pink',
        font_size: int = 18,
        start: bool = False,
        show_frustum: bool = False,
        **kwargs,
    ) -> None:
        """Enable picking of cells with a rectangle selection tool.

        Press ``"r"`` to enable rectangle based selection.  Press
        ``"r"`` again to turn it off. Selection will be saved to
        :attr:`picked_cells <pyvista.Plotter.picked_cells>` as:

        * a :class:`pyvista.MultiBlock` when multiple meshes have been picked,
        * an :class:`pyvista.UnstructuredGrid` if a single mesh have been picked.

        All meshes in the scene are available for picking by default.
        If you would like to only pick a single mesh in the scene,
        use the ``pickable=False`` argument when adding the other
        meshes to the scene.

        Uses last input mesh for input by default.

        .. warning::
           Visible cell picking (``through=False``) will only work if
           the mesh is displayed with a ``'surface'`` representation
           style (the default).

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made.
            The :attr:`picked_cells <pyvista.Plotter.picked_cells>` is given
            as the first parameter to this callable.

        through : bool, default: True
            When ``True`` the picker will select all cells
            through the mesh(es). When ``False``, the picker will select
            only visible cells on the selected surfaces.

        show : bool, default: True
            Show the selection interactively.

        show_message : bool | str, default: True
            Show the message about how to use the cell picking tool. If this
            is a string, that will be the message shown.

        style : str, default: "wireframe"
            Visualization style of the selection.  One of the
            following: ``style='surface'``, ``style='wireframe'``, or
            ``style='points'``.

        line_width : float, default: 5.0
            Thickness of selected mesh edges.

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

        font_size : int, default: 18
            Sets the font size of the message.

        start : bool, default: False
            Automatically start the cell selection tool.

        show_frustum : bool, default: False
            Show the frustum in the scene.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the selection is interactively displayed.

        Examples
        --------
        Add a mesh and a cube to a plot and enable cell picking.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 0, 0))
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_cell_picking()


        """
        if through:
            method = self.enable_rectangle_through_picking
        else:
            method = self.enable_rectangle_visible_picking
        method(
            callback=callback,
            show=show,
            show_message=show_message,
            style=style,
            line_width=line_width,
            color=color,
            font_size=font_size,
            start=start,
            show_frustum=show_frustum,
            **kwargs,
        )

    def enable_element_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        mode: ElementType | str | int = 'cell',
        show: bool = True,
        show_message: bool | str = True,
        font_size: int = 18,
        tolerance: float = 0.025,
        pickable_window: bool = False,
        left_clicking: bool = False,
        picker: PickerType | str | int = PickerType.CELL,
        **kwargs,
    ) -> None:
        """Select individual elements on a mesh.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made. The
            picked element is input as the first parameter to this callable,
            or the mesh when ``mode`` is ``"mesh"``.

        mode : str | ElementType, default: "cell"
            The picking mode. Either ``"mesh"``, ``"cell"``, ``"face"``,
            ``"edge"``, or ``"point"``.

        show : bool, default: True
            Show the selection interactively.

        show_message : bool | str, default: True
            Show the message about how to use the mesh picking tool. If this
            is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the font size of the message.

        tolerance : float, default: 0.025
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size. Rendering window size is measured across diagonal.

            .. warning::
                This is ignored with the ``'hardware'`` ``picker``.

        pickable_window : bool, default: False
            When ``True``, points in the 3D window are pickable.

        left_clicking : bool, default: False
            When ``True``, meshes can be picked by clicking the left
            mouse button.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked mesh.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries.
                * ``'cell'``: Uses :vtk:`vtkCellPicker` (default).
                * ``'point'``: Uses :vtk:`vtkPointPicker` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses :vtk:`vtkVolumePicker`.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        """
        mode = ElementType.from_any(mode)
        self_ = weakref.ref(self)

        def _end_handler(picked: pv.DataSet) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if callback:
                _poked_context_callback(plotter, callback, picked)

            if mode == ElementType.CELL:
                component._picked_cell = cast('pv.UnstructuredGrid', picked)

            if show:
                if mode == ElementType.CELL:
                    kwargs.setdefault('color', 'pink')
                elif mode == ElementType.EDGE:
                    kwargs.setdefault('color', 'magenta')
                else:
                    kwargs.setdefault('color', 'pink')

                if mode in [ElementType.CELL, ElementType.FACE]:
                    picked = picked.extract_all_edges()

                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        picked,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['element']),
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        point_size=_kwargs.pop('point_size', 5),
                        line_width=_kwargs.pop('line_width', 5),
                        **_kwargs,
                    )

        handler = PointPickingElementHandler(mode=mode, callback=_end_handler)

        self.enable_surface_point_picking(
            callback=handler,
            show_message=show_message,
            font_size=font_size,
            show_point=False,
            tolerance=tolerance,
            pickable_window=pickable_window,
            left_clicking=left_clicking,
            picker=picker,
            use_picker=True,
            **kwargs,
        )

    def enable_block_picking(
        self, callback: Callable[..., None] | None = None, side: str = 'left'
    ) -> None:
        """Enable composite block picking.

        Use this picker to return the index of a DataSet when using composite
        dataset like :class:`pyvista.MultiBlock` and pass it to a callback.

        Parameters
        ----------
        callback : callable, optional
            When input, this picker calls this callable after a selection is
            made. The composite index is passed to ``callback`` as the first
            argument and the dataset as the second argument.

        side : str, default: "left"
            The mouse button to track (either ``'left'`` or ``'right'``).
            Also accepts ``'r'`` or ``'l'``.

        Notes
        -----
        The picked block index can be accessed from :attr:`picked_block_index
        <pyvista.Plotter.picked_block_index>` attribute.

        Examples
        --------
        Enable block picking with a multiblock dataset. Left clicking will turn
        blocks blue while right picking will turn the block back to the default
        color.

        >>> import pyvista as pv
        >>> multiblock = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(multiblock)
        >>> def turn_blue(index, dataset):
        ...     mapper.block_attr[index].color = 'blue'
        >>> pl.enable_block_picking(callback=turn_blue, side='left')
        >>> def clear_color(index, dataset):
        ...     mapper.block_attr[index].color = None
        >>> pl.enable_block_picking(callback=clear_color, side='right')
        >>> pl.show()


        """
        self_ = weakref.ref(self)

        sel_index = _vtk.vtkSelectionNode.COMPOSITE_INDEX()
        sel_prop = _vtk.vtkSelectionNode.PROP()

        def get_picked_block(*args, **kwargs) -> None:  # noqa: ARG001  # numpydoc ignore=PR01
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            mouse_position = plotter.mouse_position
            if mouse_position is None:  # pragma: no cover
                return
            x, y = mouse_position
            loc = plotter._get_iren_not_none().get_event_subplot_loc()
            index = plotter.renderers.loc_to_index(loc)
            renderer = plotter.renderers[index]

            selector = _vtk.vtkOpenGLHardwareSelector()
            selector.SetRenderer(renderer)
            selector.SetArea(x, y, x, y)
            selection = selector.Select()

            for ii in range(selection.GetNumberOfNodes()):
                node = selection.GetNode(ii)
                if node is None:  # pragma: no cover
                    continue
                node_prop = node.GetProperties()
                component._picked_block_index = node_prop.Get(sel_index)

                # Safely return the dataset (a non-pyvista mapper may have been added).
                mapper = node_prop.Get(sel_prop).GetMapper()
                if isinstance(mapper, CompositePolyDataMapper):
                    dataset = mapper.block_attr.get_block(component._picked_block_index)
                else:  # pragma: no cover
                    dataset = None

                if callable(callback):
                    _poked_context_callback(
                        plotter, callback, component._picked_block_index, dataset
                    )

        self._plotter.track_click_position(callback=get_picked_block, viewport=True, side=side)

    # =========================================================================
    # Higher-level convenience pickers
    # =========================================================================

    def fly_to_mouse_position(self, *, focus: bool = False) -> None:
        """Focus on last stored mouse position.

        Parameters
        ----------
        focus : bool, default: False
            Set the camera focal point to the picked point instead of flying to it.

        """
        plotter = self._plotter
        if plotter.mouse_position is None:
            plotter.store_mouse_position()
        click_point = self.pick_mouse_position()
        if focus:
            plotter.set_focus(click_point)
        else:
            plotter.fly_to(click_point)

    def enable_fly_to_right_click(self, callback: Callable[..., None] | None = None) -> None:
        """Set the camera to track right click positions.

        A convenience method to track right click positions and fly to
        the picked point in the scene. The callback will be passed the
        point in 3D space.

        Parameters
        ----------
        callback : callable
            Callback to call immediately after right clicking.

        """
        self_ = weakref.ref(self)

        def _the_callback(*_) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            click_point = component.pick_mouse_position()
            plotter.fly_to(click_point)
            if callable(callback):
                _poked_context_callback(plotter, callback, click_point)

        self._plotter.track_click_position(callback=_the_callback, side='right')

    def enable_path_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show_message: bool | str = True,
        font_size: int = 18,
        color: ColorLike = 'pink',
        point_size: float = 10,
        line_width: float = 5,
        show_path: bool = True,
        tolerance: float = 0.025,
        **kwargs,
    ) -> None:
        """Enable picking at paths.

        This is a convenience method for :func:`enable_point_picking
        <pyvista.Plotter.enable_point_picking>` to keep track of the
        picked points and create a line using those points.

        The line is saved to the ``.picked_path`` attribute of this
        plotter

        Parameters
        ----------
        callback : callable, optional
            When given, calls this callable after a pick is made.  The
            entire picked path is passed as the only parameter to this
            callable.

        show_message : bool | str, default: True
            Show the message about how to use the point picking
            tool. If this is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the size of the message.

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

        point_size : int, default: 10
            Size of picked points if ``show_path`` is ``True``.

        line_width : float, default: 5.0
            Thickness of path representation if ``show_path`` is
            ``True``.

        show_path : bool, default: True
            Show the picked path interactively.

        tolerance : float, default: 0.025
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size.  Rendering window size is measured across diagonal.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.


        """
        self_ = weakref.ref(self)
        kwargs.setdefault('pickable', False)

        def make_line_cells(n_points: int) -> NDArray[np.int_]:
            cells = np.arange(0, n_points, dtype=np.int_)
            return np.insert(cells, 0, n_points)

        the_points = []

        def _the_callback(picked_point: VectorLike[float], picker: _vtk.vtkPicker) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if picker.GetDataSet() is None:
                return
            the_points.append(picked_point)
            component.picked_path = pv.PolyData(np.array(the_points))
            component.picked_path.lines = make_line_cells(len(the_points))
            if show_path:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        component.picked_path,
                        color=color,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['path']),
                        line_width=line_width,
                        point_size=point_size,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
            if callable(callback):
                _poked_context_callback(plotter, callback, component.picked_path)

        def _clear_path_event_watcher() -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            del the_points[:]
            with plotter._get_iren_not_none().poked_subplot():
                component._clear_picking_representations()

        self._plotter.add_key_event('c', _clear_path_event_watcher)
        if show_message is True:
            show_message = 'Press P to pick under the mouse\nPress C to clear'

        self.enable_surface_point_picking(
            callback=_the_callback,
            use_picker=True,
            font_size=font_size,
            show_message=show_message,
            show_point=False,
            tolerance=tolerance,
            clear_on_no_selection=False,
        )

    def enable_geodesic_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        show_message: bool | str = True,
        font_size: int = 18,
        color: ColorLike = 'pink',
        point_size: float = 10,
        line_width: float = 5,
        tolerance: float = 0.025,
        show_path: bool = True,
        keep_order: bool = True,
        **kwargs,
    ) -> None:
        """Enable picking at geodesic paths.

        This is a convenience method for ``enable_point_picking`` to
        keep track of the picked points and create a geodesic path
        using those points.

        The geodesic path is saved to the ``.picked_geodesic``
        attribute of this plotter.

        Parameters
        ----------
        callback : callable, optional
            When given, calls this callable after a pick is made.  The
            entire picked, geodesic path is passed as the only
            parameter to this callable.

        show_message : bool | str, default: True
            Show the message about how to use the point picking
            tool. If this is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the size of the message.

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

        point_size : int, default: 10
            Size of picked points if ``show_path`` is ``True``.

        line_width : float, default: 5.0
            Thickness of path representation if ``show_path`` is
            ``True``.

        tolerance : float, default: 0.025
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size.  Rendering window size is measured across diagonal.

        show_path : bool, default: True
            Show the picked path interactively.

        keep_order : bool, default: True
            If ``True``, the created geodesic path is a single ordered
            and cleaned line from the first point to the last.

            .. note::

                In older versions there were apparent discontinuities
                in the resulting path due to the behavior of the
                underlying VTK filter which corresponds to
                ``keep_order=False``.

            .. versionadded:: 0.32.0

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.


        """
        self_ = weakref.ref(self)

        kwargs.setdefault('pickable', False)

        self.picked_geodesic = pv.PolyData()

        def _the_callback(picked_point: VectorLike[float], picker: _vtk.vtkPicker) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if picker.GetDataSet() is None:
                return
            mesh = pv.wrap(picker.GetDataSet())
            idx = mesh.find_closest_point(np.asarray(picked_point))
            point = mesh.points[idx]
            if component._last_picked_idx is None:
                component.picked_geodesic = pv.PolyData(point)
                component.picked_geodesic['vtkOriginalPointIds'] = [idx]
            else:
                surface = mesh.extract_surface(algorithm=None).triangulate()
                locator = _vtk.vtkPointLocator()
                locator.SetDataSet(surface)
                locator.BuildLocator()
                start_idx = locator.FindClosestPoint(mesh.points[component._last_picked_idx])
                end_idx = locator.FindClosestPoint(point)
                if component.picked_geodesic is None:  # pragma: no cover
                    return
                component.picked_geodesic += surface.geodesic(
                    start_idx, end_idx, keep_order=keep_order
                )
                if keep_order:
                    component.picked_geodesic.clean(
                        inplace=True,
                        lines_to_points=False,
                        polys_to_lines=False,
                        strips_to_polys=False,
                    )
            component._last_picked_idx = idx

            if show_path:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        component.picked_geodesic,
                        color=color,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['path']),
                        line_width=line_width,
                        point_size=point_size,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
            if callable(callback):
                _poked_context_callback(plotter, callback, component.picked_geodesic)

        def _clear_g_path_event_watcher() -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            component.picked_geodesic = pv.PolyData()
            with plotter._get_iren_not_none().poked_subplot():
                component._clear_picking_representations()
            component._last_picked_idx = None

        self._plotter.add_key_event('c', _clear_g_path_event_watcher)
        if show_message is True:
            show_message = 'Press P to pick under the mouse\nPress C to clear'

        self.enable_surface_point_picking(
            callback=_the_callback,
            use_picker=True,
            font_size=font_size,
            show_message=show_message,
            tolerance=tolerance,
            show_point=False,
            clear_on_no_selection=False,
        )

    def enable_horizon_picking(
        self,
        /,
        callback: Callable[..., None] | None = None,
        *,
        normal: VectorLike[float] = (0.0, 0.0, 1.0),
        width: float | None = None,
        show_message: bool | str = True,
        font_size: int = 18,
        color: ColorLike = 'pink',
        point_size: float = 10,
        line_width: float = 5,
        show_path: bool = True,
        opacity: float = 0.75,
        show_horizon: bool = True,
        **kwargs,
    ) -> None:
        """Enable horizon picking.

        Helper for the ``enable_path_picking`` method to also show a
        ribbon surface along the picked path. Ribbon is saved under
        ``.picked_horizon``.

        Parameters
        ----------
        callback : callable, optional
            When given, calls this callable after a pick is made.  The
            entire picked path is passed as the only parameter to this
            callable.

        normal : sequence[float], default: (0.0, 0.0, 1.0)
            The normal to the horizon surface's projection plane.

        width : float, optional
            The width of the horizon surface. Default behaviour will
            dynamically change the surface width depending on its
            length.

        show_message : bool | str, default: True
            Show the message about how to use the horizon picking
            tool. If this is a string, that will be the message shown.

        font_size : int, default: 18
            Sets the font size of the message.

        color : ColorLike, default: "pink"
            The color of the horizon surface if shown.

        point_size : int, default: 10
            Size of picked points if ``show_horizon`` is ``True``.

        line_width : float, default: 5.0
            Thickness of path representation if ``show_horizon`` is
            ``True``.

        show_path : bool, default: True
            Show the picked path that the horizon is built from
            interactively.

        opacity : float, default: 0.75
            The opacity of the horizon surface if shown.

        show_horizon : bool, default: True
            Show the picked horizon surface interactively.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.


        """
        self_ = weakref.ref(self)

        def _clear_horizon_event_watcher() -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            component.picked_horizon = pv.PolyData()
            with plotter._get_iren_not_none().poked_subplot():
                component._clear_picking_representations()

        self._plotter.add_key_event('c', _clear_horizon_event_watcher)

        def _the_callback(path: pv.PolyData) -> None:
            component = self_()
            if component is None:
                return
            plotter = component._plotter
            if path.n_points < 2:
                _clear_horizon_event_watcher()
                return
            component.picked_horizon = path.ribbon(normal=normal, width=width)

            if show_horizon:
                with plotter._get_iren_not_none().poked_subplot():
                    _kwargs = kwargs.copy()
                    plotter.add_mesh(
                        component.picked_horizon,
                        name=_kwargs.get('name', PICKED_REPRESENTATION_NAMES['horizon']),
                        color=color,
                        opacity=opacity,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )

            if callable(callback):
                _poked_context_callback(plotter, callback, path)

        self.enable_path_picking(
            callback=_the_callback,
            show_message=show_message,
            font_size=font_size,
            color=color,
            point_size=point_size,
            line_width=line_width,
            show_path=show_path,
            **kwargs,
        )
