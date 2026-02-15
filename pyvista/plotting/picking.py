"""Module managing picking events."""

from __future__ import annotations

from functools import partial
from functools import wraps
import weakref

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.misc import try_callback

from . import _vtk
from .composite_mapper import CompositePolyDataMapper
from .errors import PyVistaPickingError
from .mapper import _mapper_get_data_set_input
from .mapper import _mapper_has_data_set_input
from .opts import ElementType
from .opts import PickerType

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


def _launch_pick_event(interactor, _event):
    """Create a Pick event based on coordinate or left-click."""
    click_x, click_y = interactor.GetEventPosition()
    click_z = 0

    picker = interactor.GetPicker()
    renderer = interactor.GetInteractorStyle()._parent()._plotter.iren.get_poked_renderer()
    picker.Pick(click_x, click_y, click_z, renderer)


def _poked_context_callback(plotter, *args, **kwargs):
    """Use _poked_context_callback in a poked renderer context."""
    with plotter.iren.poked_subplot():
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

    def __init__(self, frustum, viewport):
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

    def __init__(self, mode: ElementType = ElementType.CELL, callback=None):
        self._picker_ = None
        self.callback = callback
        self.mode = ElementType.from_any(mode)

    @property
    def picker(self):  # numpydoc ignore=RT01
        """Get or set the picker instance."""
        return self._picker_()  # type: ignore[misc]

    @picker.setter
    def picker(self, picker):
        self._picker_ = weakref.ref(picker)  # type: ignore[assignment]

    def get_mesh(self):
        """Get the picked mesh.

        Returns
        -------
        pyvista.DataSet
            Picked mesh.

        """
        ds = self.picker.GetDataSet()
        if ds is not None:
            return pv.wrap(ds)
        return None

    def get_cell(self, picked_point):
        """Get the picked cell of the picked mesh.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the picked cell.

        """
        mesh = self.get_mesh()
        # cell_id = self.picker.GetCellId()
        cell_id = mesh.find_containing_cell(picked_point)  # more accurate
        if cell_id < 0:
            return None  # TODO: this happens but shouldn't  # pragma: no cover
        cell = mesh.extract_cells(cell_id)
        cell.cell_data['vtkOriginalCellIds'] = np.array([cell_id])
        return cell

    def get_face(self, picked_point):
        """Get the picked face of the picked cell.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the picked face.

        """
        cell = self.get_cell(picked_point).get_cell(0)
        if cell.n_faces > 1:
            for face in cell.faces:
                contains = face.cast_to_unstructured_grid().find_containing_cell(picked_point)
                if contains > -1:
                    break
            if contains < 0:
                # this shouldn't happen
                msg = 'Trouble aligning point with face.'
                raise RuntimeError(msg)
            face = face.cast_to_unstructured_grid()
            face.field_data['vtkOriginalFaceIds'] = np.array([len(cell.faces) - 1])
        else:
            face = cell.cast_to_unstructured_grid()
            face.field_data['vtkOriginalFaceIds'] = np.array([0])

        return face

    def get_edge(self, picked_point):
        """Get the picked edge of the picked cell.

        Parameters
        ----------
        picked_point : sequence[float]
            Coordinates of the picked point.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the picked edge.

        """
        cell = self.get_cell(picked_point).get_cell(0)
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

    def get_point(self, picked_point):
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
        mesh = self.get_mesh()
        pid = mesh.find_closest_point(picked_point)
        picked = mesh.extract_points(pid, adjacent_cells=False, include_cells=False)
        return picked.cast_to_poly_points()

    def __call__(self, picked_point, picker):
        """Perform the pick."""
        self.picker = picker
        mesh = self.get_mesh()
        if mesh is None:
            return  # No selected mesh (point not on surface of mesh)

        if self.mode == ElementType.MESH:
            picked = mesh
        elif self.mode == ElementType.CELL:
            picked = self.get_cell(picked_point)
            if picked is None:
                return  # TODO: handle
        elif self.mode == ElementType.FACE:
            picked = self.get_face(picked_point)
        elif self.mode == ElementType.EDGE:
            picked = self.get_edge(picked_point)
        elif self.mode == ElementType.POINT:
            picked = self.get_point(picked_point)

        if self.callback:
            try_callback(self.callback, picked)


@abstract_class
class PickingInterface:  # numpydoc ignore=PR01
    """An internal class to hold core picking related features."""

    def __init__(self, *args, **kwargs):
        """Initialize the picking interface."""
        super().__init__(*args, **kwargs)
        self._picking_left_clicking_observer = None
        self._picking_right_clicking_observer = None
        self._picker_in_use = False
        self._picked_point = None

    def _clear_picking_representations(self):
        """Clear all picking representations."""
        for name in PICKED_REPRESENTATION_NAMES.values():
            self.remove_actor(name)  # type: ignore[attr-defined]

    @property
    def picked_point(self):  # numpydoc ignore=RT01
        """Return the picked point.

        This returns the picked point after selecting a point.

        Returns
        -------
        output : numpy.ndarray | None
            Picked point if available.

        """
        return self._picked_point

    def get_pick_position(self):
        """Get the pick position or area.

        Returns
        -------
        sequence
            Picked position or area as ``(x0, y0, x1, y1)``.

        """
        renderer = self.iren.get_poked_renderer()  # type: ignore[attr-defined]
        return renderer.get_pick_position()

    def pick_click_position(self):
        """Get corresponding click location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        if self.click_position is None:  # type: ignore[attr-defined]
            self.store_click_position()  # type: ignore[attr-defined]
        renderer = self.iren.get_poked_renderer()  # type: ignore[attr-defined]
        self.iren.picker.Pick(self.click_position[0], self.click_position[1], 0, renderer)  # type: ignore[attr-defined]
        return self.iren.picker.GetPickPosition()  # type: ignore[attr-defined]

    def pick_mouse_position(self):
        """Get corresponding mouse location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        if self.mouse_position is None:  # type: ignore[attr-defined]
            self.store_mouse_position()  # type: ignore[attr-defined]
        renderer = self.iren.get_poked_renderer()  # type: ignore[attr-defined]
        self.iren.picker.Pick(self.mouse_position[0], self.mouse_position[1], 0, renderer)  # type: ignore[attr-defined]
        return self.iren.picker.GetPickPosition()  # type: ignore[attr-defined]

    def _init_click_picking_callback(self, *, left_clicking=False):
        if left_clicking:
            self._picking_left_clicking_observer = self.iren.add_observer(  # type: ignore[attr-defined]
                'LeftButtonPressEvent',
                partial(try_callback, _launch_pick_event),
            )
        else:
            self._picking_right_clicking_observer = self.iren.add_observer(  # type: ignore[attr-defined]
                'RightButtonPressEvent',
                partial(try_callback, _launch_pick_event),
            )

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
        # remove left and right clicking observer if available
        if getattr(self, 'iren', None):
            self.iren.remove_observer(self._picking_left_clicking_observer)  # type: ignore[attr-defined]
            self.iren.remove_observer(self._picking_right_clicking_observer)  # type: ignore[attr-defined]
            # Reset to default picker
            self.iren.reset_picker()  # type: ignore[attr-defined]
        self._picking_left_clicking_observer = None
        self._picking_right_clicking_observer = None

        self._picker_in_use = False

    def _validate_picker_not_in_use(self):
        if self._picker_in_use:
            msg = (
                'Picking is already enabled, please disable previous picking '
                'with `disable_picking()`.'
            )
            raise PyVistaPickingError(msg)

    @_deprecate_positional_args(allowed=['callback'])
    def enable_point_picking(  # noqa: PLR0917
        self,
        callback=None,
        tolerance=0.025,
        left_clicking=False,  # noqa: FBT002
        picker=PickerType.POINT,
        show_message=True,  # noqa: FBT002
        font_size=18,
        color='pink',
        point_size=10,
        show_point=True,  # noqa: FBT002
        use_picker=False,  # noqa: FBT002
        pickable_window=False,  # noqa: FBT002
        clear_on_no_selection=True,  # noqa: FBT002
        **kwargs,
    ):
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
                  performant for large geometries (default).
                * ``'cell'``: Uses :vtk:`vtkCellPicker`.
                * ``'point'``: Uses :vtk:`vtkPointPicker` which will snap to
                  points on the surface of the mesh.
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

        See :ref:`point_picking_example` for a full example using this method.

        """
        self._validate_picker_not_in_use()
        if 'use_mesh' in kwargs:
            warn_external(
                '`use_mesh` is deprecated. See `use_picker` instead.', PyVistaDeprecationWarning
            )
            use_mesh = kwargs.pop('use_mesh')
        else:
            use_mesh = False

        self_ = weakref.ref(self)

        def _end_pick_event(picker, _event):
            if (
                not pickable_window
                and hasattr(picker, 'GetDataSet')
                and picker.GetDataSet() is None
            ):
                # Clear the selection
                self._picked_point = None
                if clear_on_no_selection:
                    with self_().iren.poked_subplot():  # type: ignore[union-attr]
                        self_()._clear_picking_representations()  # type: ignore[union-attr]
                return
            with self_().iren.poked_subplot():  # type: ignore[union-attr]
                point = np.array(picker.GetPickPosition())
                point /= self_().scale  # type: ignore[union-attr] # HACK: handle scale
                self_()._picked_point = point  # type: ignore[union-attr]
                if show_point:
                    _kwargs = kwargs.copy()
                    self_().add_mesh(  # type: ignore[union-attr]
                        self_().picked_point,  # type: ignore[union-attr]
                        color=color,
                        point_size=point_size,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['point']),
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
                if callable(callback):
                    if use_picker:
                        _poked_context_callback(self_(), callback, self.picked_point, picker)
                    elif use_mesh:  # Lower priority
                        _poked_context_callback(
                            self_(),
                            callback,
                            picker.GetDataSet(),
                            picker.GetPointId(),
                        )
                    else:
                        _poked_context_callback(self_(), callback, self.picked_point)

        if picker is not None:  # If None, that means use already set picker
            self.iren.picker = picker  # type: ignore[attr-defined]
        if hasattr(self.iren.picker, 'SetTolerance'):  # type: ignore[attr-defined]
            self.iren.picker.SetTolerance(tolerance)  # type: ignore[attr-defined]
        self.iren.add_pick_observer(_end_pick_event)  # type: ignore[attr-defined]
        self._init_click_picking_callback(left_clicking=left_clicking)
        self._picker_in_use = True

        # Now add text about cell-selection
        if show_message:
            if show_message is True:
                show_message = 'Left-click' if left_clicking else 'Right-click'
                show_message += ' or press P to pick under the mouse'
            self._picking_text = self.add_text(  # type: ignore[attr-defined]
                str(show_message),
                font_size=font_size,
                name='_point_picking_message',
            )

    @_deprecate_positional_args(allowed=['callback'])
    def enable_rectangle_picking(  # noqa: PLR0917
        self,
        callback=None,
        show_message=True,  # noqa: FBT002
        font_size=18,
        start=False,  # noqa: FBT002
        show_frustum=False,  # noqa: FBT002
        style='wireframe',
        color='pink',
        **kwargs,
    ):
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

        def _end_pick_helper(picker, *_):
            renderer = picker.GetRenderer()  # TODO: double check this is poked renderer
            x0 = int(renderer.GetPickX1())
            x1 = int(renderer.GetPickX2())
            y0 = int(renderer.GetPickY1())
            y1 = int(renderer.GetPickY2())

            selection = RectangleSelection(frustum=picker.GetFrustum(), viewport=(x0, y0, x1, y1))

            if show_frustum:
                with self_().iren.poked_subplot():  # type: ignore[union-attr]
                    _kwargs = kwargs.copy()
                    self_().add_mesh(  # type: ignore[union-attr]
                        selection.frustum_mesh,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['frustum']),
                        style=style,
                        color=color,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )

            if callback is not None:
                _poked_context_callback(self_(), callback, selection)

        self.enable_rubber_band_style()  # type: ignore[attr-defined] # TODO: better handle?
        self.iren.picker = 'rendered'  # type: ignore[attr-defined]
        self.iren.add_pick_observer(_end_pick_helper)  # type: ignore[attr-defined]
        self._picker_in_use = True

        # Now add text about cell-selection
        if show_message:
            if show_message is True:
                show_message = 'Press R to toggle selection tool'
            self._picking_text = self.add_text(  # type: ignore[attr-defined]
                str(show_message),
                font_size=font_size,
                name='_rectangle_picking_message',
            )

        if start:
            self.iren._style_class.StartSelect()  # type: ignore[attr-defined]


@abstract_class
class PickingMethods(PickingInterface):  # numpydoc ignore=PR01
    """Internal class to contain picking utilities."""

    def __init__(self, *args, **kwargs):
        """Initialize the picking methods."""
        super().__init__(*args, **kwargs)
        self._picked_actor = None
        self._picked_mesh = None
        self._picked_cell: None | pv.MultiBlock | pv.UnstructuredGrid = None
        self._picking_text = None
        self._picked_block_index = None

    @property
    def picked_actor(self):  # numpydoc ignore=RT01
        """Return the picked mesh.

        This returns the picked actor after selecting a mesh with
        :func:`enable_surface_point_picking <pyvista.Plotter.enable_surface_point_picking>` or
        :func:`enable_mesh_picking <pyvista.Plotter.enable_mesh_picking>`.

        Returns
        -------
        output : pyvista.Actor | None
            Picked actor if available.

        """
        return self._picked_actor

    @property
    def picked_mesh(self):  # numpydoc ignore=RT01
        """Return the picked mesh.

        This returns the picked mesh after selecting a mesh with
        :func:`enable_surface_point_picking <pyvista.Plotter.enable_surface_point_picking>` or
        :func:`enable_mesh_picking <pyvista.Plotter.enable_mesh_picking>`.

        Returns
        -------
        output : pyvista.DataSet | None
            Picked mesh if available.

        """
        return self._picked_mesh

    @property
    def picked_cell(self) -> None | pv.UnstructuredGrid | pv.MultiBlock:
        r"""Return the cell-picked object.

        This returns the object containing cells that were interactively picked with
        :func:`enable_cell_picking <pyvista.Plotter.enable_cell_picking>`,
        :func:`enable_rectangle_through_picking <pyvista.Plotter.enable_rectangle_through_picking>`
        or
        :func:`enable_rectangle_visible_picking <pyvista.Plotter.enable_rectangle_visible_picking>`.

        Its value depends on the picking result:

        * if no cells have been picked, returns :py:data:`None`
        * if all picked cells belong to a single actor, returns an :class:`UnstructuredGrid`
        * if picked cells belong to multiple actors, returns a :class:`MultiBlock`
          containing ``n`` ``pyvista.UnstructuredGrid``\s, with ``n`` being the number of picked actors.

        Note that a cell data ``original_cell_ids`` is added to help identifying
        cell ids picked from the original dataset.

        .. deprecated:: 0.47
            Use the :attr:`picked_cells <pyvista.Plotter.picked_cells>` attribute instead.

        Returns
        -------
        output : None | pyvista.UnstructuredGrid | pyvista.MultiBlock
            Picked object if available.

        """  # noqa: E501
        # deprecated in 0.47, error in 0.48, remove in 0.49
        warn_external(
            category=PyVistaDeprecationWarning, message='Use the `picked_cells` attribute instead.'
        )
        return self._picked_cell

    @property
    def picked_cells(self) -> None | pv.UnstructuredGrid | pv.MultiBlock:
        r"""Return the cell-picked object.

        This returns the object containing cells that were interactively picked with
        :func:`enable_cell_picking <pyvista.Plotter.enable_cell_picking>`,
        :func:`enable_rectangle_through_picking <pyvista.Plotter.enable_rectangle_through_picking>`
        or
        :func:`enable_rectangle_visible_picking <pyvista.Plotter.enable_rectangle_visible_picking>`.

        Its value depends on the picking result:

        * if no cells have been picked, returns :py:data:`None`
        * if all picked cells belong to a single actor, returns an :class:`UnstructuredGrid`
        * if picked cells belong to multiple actors, returns a :class:`MultiBlock`
          containing ``n`` ``pyvista.UnstructuredGrid``\s, with ``n`` being the number of picked actors.

        Note that a cell data ``original_cell_ids`` is added to help identifying
        cell ids picked from the original dataset.


        Returns
        -------
        output : None | pyvista.UnstructuredGrid | pyvista.MultiBlock
            Picked object if available.

        """  # noqa: E501
        return self._picked_cell

    @property
    def picked_block_index(self):  # numpydoc ignore=RT01
        """Return the picked block index.

        This returns the picked block index after selecting a point with
        :func:`enable_point_picking <pyvista.Plotter.enable_point_picking>`.

        Returns
        -------
        output : int | None
            Picked block if available. If ``-1``, then a non-composite dataset
            was selected.

        """
        return self._picked_block_index

    @wraps(PickingInterface.disable_picking)
    def disable_picking(self) -> None:  # type: ignore[override]
        """Disable picking."""
        super().disable_picking()
        # remove any picking text
        if hasattr(self, 'renderers'):
            for renderer in self.renderers:
                renderer.remove_actor(self._picking_text, render=False)
        self._picking_text = None

    @_deprecate_positional_args(allowed=['callback'])
    def enable_surface_point_picking(  # noqa: PLR0917
        self,
        callback=None,
        show_message=True,  # noqa: FBT002
        font_size=18,
        color='pink',
        show_point=True,  # noqa: FBT002
        point_size=10,
        tolerance=0.025,
        pickable_window=False,  # noqa: FBT002
        left_clicking=False,  # noqa: FBT002
        picker=PickerType.CELL,
        use_picker=False,  # noqa: FBT002
        clear_on_no_selection=True,  # noqa: FBT002
        **kwargs,
    ):
        """Enable picking of a point on the surface of a mesh.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made. The
            ``mesh`` is input as the first parameter to this callable.

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
            mousebutton.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked mesh.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses :vtk:`vtkCellPicker`.
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

        See :ref:`surface_point_picking_example` for a full example using this method.

        """
        # only allow certain pickers to be used for surface picking
        #  the picker class needs to have `GetDataSet()`
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

        def _end_pick_event(picked_point, picker):
            if not pickable_window and picker.GetActor() is None:
                self_()._picked_point = None  # type: ignore[union-attr]
                self_()._picked_actor = None  # type: ignore[union-attr]
                self_()._picked_mesh = None  # type: ignore[union-attr]
                if clear_on_no_selection:
                    with self_().iren.poked_subplot():  # type: ignore[union-attr]
                        self_()._clear_picking_representations()  # type: ignore[union-attr]
                return
            self_()._picked_actor = picker.GetActor()  # type: ignore[union-attr]
            self_()._picked_mesh = picker.GetDataSet()  # type: ignore[union-attr]

            if show_point:
                with self_().iren.poked_subplot():  # type: ignore[union-attr]
                    _kwargs = kwargs.copy()
                    self_().add_mesh(  # type: ignore[union-attr]
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
                    _poked_context_callback(self_(), callback, picked_point, picker)
                else:
                    _poked_context_callback(self_(), callback, picked_point)

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

    @_deprecate_positional_args(allowed=['callback'])
    def enable_mesh_picking(  # noqa: PLR0917
        self,
        callback=None,
        show=True,  # noqa: FBT002
        show_message=True,  # noqa: FBT002
        style='wireframe',
        line_width=5,
        color='pink',
        font_size=18,
        left_clicking=False,  # noqa: FBT002
        use_actor=False,  # noqa: FBT002
        picker=PickerType.CELL,
        **kwargs,
    ):
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
            mousebutton.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked point.

        use_actor : bool, default: False
            If True, the callback will be passed the picked actor instead of
            the mesh object.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses :vtk:`vtkCellPicker`.
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

        See :ref:`mesh_picking_example` for a full example using this method.

        """
        self_ = weakref.ref(self)

        def end_pick_call_back(*args):  # noqa: ARG001
            if callback:
                if use_actor:
                    _poked_context_callback(self_(), callback, self_()._picked_actor)  # type: ignore[union-attr]
                else:
                    _poked_context_callback(self_(), callback, self_()._picked_mesh)  # type: ignore[union-attr]

            if show:
                # Select the renderer where the mesh is added.
                active_renderer_index = self_().renderers._active_index  # type: ignore[union-attr]
                loc = self_().iren.get_event_subplot_loc()  # type: ignore[union-attr]
                self_().subplot(*loc)  # type: ignore[union-attr]

                # Use try in case selection is empty or invalid
                try:
                    with self_().iren.poked_subplot():  # type: ignore[union-attr]
                        _kwargs = kwargs.copy()
                        self_().add_mesh(  # type: ignore[union-attr]
                            self_()._picked_mesh,  # type: ignore[union-attr]
                            name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['mesh']),
                            style=style,
                            color=color,
                            line_width=line_width,
                            pickable=_kwargs.pop('pickable', False),
                            reset_camera=_kwargs.pop('reset_camera', False),
                            **_kwargs,
                        )
                except Exception as e:  # noqa: BLE001  # pragma: no cover
                    warn_external('Unable to show mesh when picking:\n\n%s', str(e))  # type: ignore[arg-type]

                # Reset to the active renderer.
                loc = self_().renderers.index_to_loc(active_renderer_index)  # type: ignore[union-attr]
                self_().subplot(*loc)  # type: ignore[union-attr]

                # render here prior to running the callback
                self_().render()  # type: ignore[union-attr]

        # add on-screen message about point-selection
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

    @_deprecate_positional_args(allowed=['callback'])
    def enable_rectangle_through_picking(  # noqa: PLR0917
        self,
        callback=None,
        show=True,  # noqa: FBT002
        style='wireframe',
        line_width=5,
        color='pink',
        show_message=True,  # noqa: FBT002
        font_size=18,
        start=False,  # noqa: FBT002
        show_frustum=False,  # noqa: FBT002
        **kwargs,
    ):
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

        def finalize(picked):
            if picked is None:
                # Indicates invalid pick
                with self_().iren.poked_subplot():  # type: ignore[union-attr]
                    self_()._clear_picking_representations()  # type: ignore[union-attr]
                return

            self._picked_cell = picked

            if show:
                # Use try in case selection is empty
                with self_().iren.poked_subplot():  # type: ignore[union-attr]
                    _kwargs = kwargs.copy()
                    self_().add_mesh(  # type: ignore[union-attr]
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
                _poked_context_callback(self_(), callback, self_().picked_cells)  # type: ignore[union-attr]

        def through_pick_callback(selection: RectangleSelection):
            picked = pv.MultiBlock()
            renderer = self_().iren.get_poked_renderer()  # type: ignore[union-attr]
            for actor in renderer.actors.values():
                if (
                    (mapper := actor.GetMapper())
                    and _mapper_has_data_set_input(mapper)
                    and actor.GetPickable()
                ):
                    input_mesh = pv.wrap(_mapper_get_data_set_input(actor.GetMapper()))
                    old_name, new_name = 'orig_extract_id', 'original_cell_ids'

                    #  deprecated in 0.47, rename in v0.49
                    warn_external(
                        category=PyVistaDeprecationWarning,
                        message=(
                            f'The `{old_name}` cell data has been deprecated and will be renamed'
                            f' to `{new_name} in a future version of PyVista.'
                        ),
                    )
                    input_mesh.cell_data[old_name] = (ids := np.arange(input_mesh.n_cells))
                    input_mesh.cell_data[new_name] = ids
                    extract = _vtk.vtkExtractGeometry()
                    extract.SetInputData(input_mesh)
                    extract.SetImplicitFunction(selection.frustum)
                    extract.Update()

                    if (wrapped := pv.wrap(extract.GetOutput())).n_cells > 0:
                        picked.append(wrapped)

            if picked.n_blocks == 0 or picked.combine().n_cells < 1:
                self_()._picked_cell = None  # type: ignore[union-attr]
            elif picked.n_blocks == 1:
                self_()._picked_cell = picked[0]  # type: ignore[union-attr]
            else:
                self_()._picked_cell = picked  # type: ignore[union-attr]

            finalize(self_()._picked_cell)  # type: ignore[union-attr]

        self.enable_rectangle_picking(
            callback=through_pick_callback,
            show_message=show_message,
            font_size=font_size,
            show_frustum=show_frustum,
            start=start,
            style=style,
            color=color,
        )

    @_deprecate_positional_args(allowed=['callback'])
    def enable_rectangle_visible_picking(  # noqa: PLR0917
        self,
        callback=None,
        show=True,  # noqa: FBT002
        style='wireframe',
        line_width=5,
        color='pink',
        show_message=True,  # noqa: FBT002
        font_size=18,
        start=False,  # noqa: FBT002
        show_frustum=False,  # noqa: FBT002
        **kwargs,
    ):
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

        def finalize(picked):
            if picked is None:
                # Indicates invalid pick
                with self_().iren.poked_subplot():  # type: ignore[union-attr]
                    self_()._clear_picking_representations()  # type: ignore[union-attr]
                return

            if show:
                # Use try in case selection is empty
                with self_().iren.poked_subplot():  # type: ignore[union-attr]
                    _kwargs = kwargs.copy()
                    self_().add_mesh(  # type: ignore[union-attr]
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
                _poked_context_callback(self_(), callback, picked)

        def visible_pick_callback(selection):
            picked = pv.MultiBlock()
            renderer = self_().iren.get_poked_renderer()  # type: ignore[union-attr]
            x0, y0, x1, y1 = renderer.get_pick_position()
            # x0, y0, x1, y1 = selection.viewport
            if x0 >= 0:  # initial pick position is (-1, -1, -1, -1)
                selector = _vtk.vtkOpenGLHardwareSelector()
                selector.SetFieldAssociation(_vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
                selector.SetRenderer(renderer)
                selector.SetArea(x0, y0, x1, y1)
                selection = selector.Select()

                for node in range(selection.GetNumberOfNodes()):
                    selection_node = selection.GetNode(node)
                    if selection_node is None:  # pragma: no cover
                        # No selection
                        continue
                    cids = pv.convert_array(selection_node.GetSelectionList())
                    actor = selection_node.GetProperties().Get(_vtk.vtkSelectionNode.PROP())

                    # TODO: this is too hacky - find better way to avoid non-dataset actors
                    if not actor.GetMapper() or not hasattr(
                        actor.GetProperty(),
                        'GetRepresentation',
                    ):
                        continue

                    # if not a surface
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
                    cids_to_get = tri_smesh.extract_cells(cids)['original_cell_ids']
                    picked.append(smesh.extract_cells(cids_to_get))

                # memory leak issues on vtk==9.0.20210612.dev0
                # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18239#note_973826
                selection.UnRegister(selection)

            if len(picked) == 0 or picked.combine().n_cells < 1:
                self_()._picked_cell = None  # type: ignore[union-attr]
            elif len(picked) == 1:
                self_()._picked_cell = picked[0]  # type: ignore[union-attr]
            else:
                self_()._picked_cell = picked  # type: ignore[union-attr]

            finalize(self_()._picked_cell)  # type: ignore[union-attr]

        self.enable_rectangle_picking(
            callback=visible_pick_callback,
            show_message=show_message,
            font_size=font_size,
            start=start,
            show_frustum=show_frustum,
            style=style,
            color=color,
        )

    @_deprecate_positional_args(allowed=['callback'])
    def enable_cell_picking(  # noqa: PLR0917
        self,
        callback=None,
        through=True,  # noqa: FBT002
        show=True,  # noqa: FBT002
        show_message=True,  # noqa: FBT002
        style='wireframe',
        line_width=5,
        color='pink',
        font_size=18,
        start=False,  # noqa: FBT002
        show_frustum=False,  # noqa: FBT002
        **kwargs,
    ):
        """Enable picking of cells with a rectangle selection tool.

        Press ``"r"`` to enable rectangle based selection.  Press
        ``"r"`` again to turn it off. Selection will be saved to
        :attr:`picked_cells <pyvista.Plotter.picked_cells>` as:

        * a :class:`MultiBlock` when multiple meshes have been picked,
        * an :class:`UnstructuredGrid` if a single mesh have been picked.

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
            only visible cells on the selected surface(s).

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

    @_deprecate_positional_args(allowed=['callback'])
    def enable_element_picking(  # noqa: PLR0917
        self,
        callback=None,
        mode='cell',
        show=True,  # noqa: FBT002
        show_message=True,  # noqa: FBT002
        font_size=18,
        tolerance=0.025,
        pickable_window=False,  # noqa: FBT002
        left_clicking=False,  # noqa: FBT002
        picker=PickerType.CELL,
        **kwargs,
    ):
        """Select individual elements on a mesh.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made. The
            ``mesh`` is input as the first parameter to this callable.

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
            mousebutton.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked mesh.

        picker : str | PickerType, optional
            Choice of VTK picker class type:

                * ``'hardware'``: Uses :vtk:`vtkHardwarePicker` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses :vtk:`vtkCellPicker`.
                * ``'point'``: Uses :vtk:`vtkPointPicker` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses :vtk:`vtkVolumePicker`.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        See Also
        --------
        :ref:`element_picking_example`

        """
        mode = ElementType.from_any(mode)
        self_ = weakref.ref(self)

        def _end_handler(picked):
            if callback:
                _poked_context_callback(self_(), callback, picked)

            if mode == ElementType.CELL:
                self._picked_cell = picked

            if show:
                if mode == ElementType.CELL:
                    kwargs.setdefault('color', 'pink')
                elif mode == ElementType.EDGE:
                    kwargs.setdefault('color', 'magenta')
                else:
                    kwargs.setdefault('color', 'pink')

                if mode in [ElementType.CELL, ElementType.FACE]:
                    picked = picked.extract_all_edges()

                with self.iren.poked_subplot():  # type: ignore[attr-defined]
                    _kwargs = kwargs.copy()
                    self.add_mesh(  # type: ignore[attr-defined]
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

    def enable_block_picking(self, callback=None, side='left'):
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
        # use a weak reference to enable garbage collection
        self_ = weakref.ref(self)

        sel_index = _vtk.vtkSelectionNode.COMPOSITE_INDEX()
        sel_prop = _vtk.vtkSelectionNode.PROP()

        def get_picked_block(*args, **kwargs):  # numpydoc ignore=PR01  # noqa: ARG001
            """Get the picked block and pass it to the user callback."""
            x, y = self.mouse_position  # type: ignore[attr-defined]
            loc = self_().iren.get_event_subplot_loc()  # type: ignore[union-attr]
            index = self_().renderers.loc_to_index(loc)  # type: ignore[union-attr]
            renderer = self_().renderers[index]  # type: ignore[union-attr]

            selector = _vtk.vtkOpenGLHardwareSelector()
            selector.SetRenderer(renderer)
            selector.SetArea(x, y, x, y)
            selection = selector.Select()

            for ii in range(selection.GetNumberOfNodes()):
                node = selection.GetNode(ii)
                if node is None:  # pragma: no cover
                    continue
                node_prop = node.GetProperties()
                self._picked_block_index = node_prop.Get(sel_index)

                # Safely return the dataset as it's possible a non pyvista
                # mapper was added
                mapper = node_prop.Get(sel_prop).GetMapper()
                if isinstance(mapper, CompositePolyDataMapper):
                    dataset = mapper.block_attr.get_block(self._picked_block_index)
                else:  # pragma: no cover
                    dataset = None

                if callable(callback):
                    _poked_context_callback(self_(), callback, self._picked_block_index, dataset)

        self.track_click_position(callback=get_picked_block, viewport=True, side=side)  # type: ignore[attr-defined]


@abstract_class
class PickingHelper(PickingMethods):
    """Internal container class to contain picking helper methods."""

    def __init__(self, *args, **kwargs):
        """Initialize the picking methods."""
        super().__init__(*args, **kwargs)
        self.picked_path = None
        self.picked_geodesic = None
        self.picked_horizon = None
        self._last_picked_idx: int | None = None

    @_deprecate_positional_args
    def fly_to_mouse_position(self, focus=False):  # noqa: FBT002
        """Focus on last stored mouse position."""
        if self.mouse_position is None:  # type: ignore[attr-defined]
            self.store_mouse_position()  # type: ignore[attr-defined]
        click_point = self.pick_mouse_position()
        if focus:
            self.set_focus(click_point)  # type: ignore[attr-defined]
        else:
            self.fly_to(click_point)  # type: ignore[attr-defined]

    def enable_fly_to_right_click(self, callback=None):
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

        def _the_callback(*_):
            click_point = self.pick_mouse_position()
            self.fly_to(click_point)  # type: ignore[attr-defined]
            if callable(callback):
                _poked_context_callback(self_(), callback, click_point)

        self.track_click_position(callback=_the_callback, side='right')  # type: ignore[attr-defined]

    @_deprecate_positional_args(allowed=['callback'])
    def enable_path_picking(  # noqa: PLR0917
        self,
        callback=None,
        show_message=True,  # noqa: FBT002
        font_size=18,
        color='pink',
        point_size=10,
        line_width=5,
        show_path=True,  # noqa: FBT002
        tolerance=0.025,
        **kwargs,
    ):
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

        def make_line_cells(n_points):
            cells = np.arange(0, n_points, dtype=np.int_)
            return np.insert(cells, 0, n_points)

        the_points = []

        def _the_callback(picked_point, picker):
            if picker.GetDataSet() is None:
                return
            the_points.append(picked_point)
            self.picked_path = pv.PolyData(np.array(the_points))
            self.picked_path.lines = make_line_cells(len(the_points))
            if show_path:
                with self.iren.poked_subplot():  # type: ignore[attr-defined]
                    _kwargs = kwargs.copy()
                    self.add_mesh(  # type: ignore[attr-defined]
                        self.picked_path,
                        color=color,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['path']),
                        line_width=line_width,
                        point_size=point_size,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
            if callable(callback):
                _poked_context_callback(self_(), callback, self.picked_path)

        def _clear_path_event_watcher():
            del the_points[:]
            with self.iren.poked_subplot():  # type: ignore[attr-defined]
                self._clear_picking_representations()

        self.add_key_event('c', _clear_path_event_watcher)  # type: ignore[attr-defined]
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

    @_deprecate_positional_args(allowed=['callback'])
    def enable_geodesic_picking(  # noqa: PLR0917
        self,
        callback=None,
        show_message=True,  # noqa: FBT002
        font_size=18,
        color='pink',
        point_size=10,
        line_width=5,
        tolerance=0.025,
        show_path=True,  # noqa: FBT002
        keep_order=True,  # noqa: FBT002
        **kwargs,
    ):
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

        def _the_callback(picked_point, picker):
            if picker.GetDataSet() is None:
                return
            mesh = pv.wrap(picker.GetDataSet())
            idx = mesh.find_closest_point(picked_point)
            point = mesh.points[idx]
            if self._last_picked_idx is None:
                self.picked_geodesic = pv.PolyData(point)
                self.picked_geodesic['vtkOriginalPointIds'] = [idx]
            else:
                surface = mesh.extract_surface(algorithm=None).triangulate()
                locator = _vtk.vtkPointLocator()
                locator.SetDataSet(surface)
                locator.BuildLocator()
                start_idx = locator.FindClosestPoint(mesh.points[self._last_picked_idx])
                end_idx = locator.FindClosestPoint(point)
                self.picked_geodesic += surface.geodesic(start_idx, end_idx, keep_order=keep_order)
                if keep_order:
                    # it makes sense to remove adjacent duplicate points
                    self.picked_geodesic.clean(
                        inplace=True,
                        lines_to_points=False,
                        polys_to_lines=False,
                        strips_to_polys=False,
                    )
            self._last_picked_idx = idx

            if show_path:
                with self.iren.poked_subplot():  # type: ignore[attr-defined]
                    _kwargs = kwargs.copy()
                    self.add_mesh(  # type: ignore[attr-defined]
                        self.picked_geodesic,
                        color=color,
                        name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['path']),
                        line_width=line_width,
                        point_size=point_size,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )
            if callable(callback):
                _poked_context_callback(self_(), callback, self.picked_geodesic)

        def _clear_g_path_event_watcher():
            self.picked_geodesic = pv.PolyData()
            with self.iren.poked_subplot():  # type: ignore[attr-defined]
                self._clear_picking_representations()
            self._last_picked_idx = None

        self.add_key_event('c', _clear_g_path_event_watcher)  # type: ignore[attr-defined]
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

    @_deprecate_positional_args(allowed=['callback'])
    def enable_horizon_picking(  # noqa: PLR0917
        self,
        callback=None,
        normal=(0.0, 0.0, 1.0),
        width=None,
        show_message=True,  # noqa: FBT002
        font_size=18,
        color='pink',
        point_size=10,
        line_width=5,
        show_path=True,  # noqa: FBT002
        opacity=0.75,
        show_horizon=True,  # noqa: FBT002
        **kwargs,
    ):
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

        def _clear_horizon_event_watcher():
            self.picked_horizon = pv.PolyData()
            with self.iren.poked_subplot():  # type: ignore[attr-defined]
                self._clear_picking_representations()

        self.add_key_event('c', _clear_horizon_event_watcher)  # type: ignore[attr-defined]

        def _the_callback(path):
            if path.n_points < 2:
                _clear_horizon_event_watcher()
                return
            self.picked_horizon = path.ribbon(normal=normal, width=width)

            if show_horizon:
                with self.iren.poked_subplot():  # type: ignore[attr-defined]
                    _kwargs = kwargs.copy()
                    self.add_mesh(  # type: ignore[attr-defined]
                        self.picked_horizon,
                        name=_kwargs.get('name', PICKED_REPRESENTATION_NAMES['horizon']),
                        color=color,
                        opacity=opacity,
                        pickable=_kwargs.pop('pickable', False),
                        reset_camera=_kwargs.pop('reset_camera', False),
                        **_kwargs,
                    )

            if callable(callback):
                _poked_context_callback(self_(), callback, path)

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
