"""Module managing picking events."""
from functools import partial, wraps
from typing import Tuple
import warnings
import weakref

import numpy as np

import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import try_callback

from . import _vtk
from .composite_mapper import CompositePolyDataMapper
from .errors import PyVistaPickingError
from .opts import ElementType, PickerType

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


def _launch_pick_event(interactor, event):
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


class RectangleSelection:
    """Internal data structure for rectangle based selections."""

    def __init__(self, frustum, viewport):
        self._frustum = frustum
        self._viewport = viewport

    @property
    def frustum(self) -> _vtk.vtkPlanes:
        """Get the selected frustum through the scene."""
        return self._frustum

    @property
    def frustum_mesh(self) -> 'pyvista.PolyData':
        """Get the frustum as a PyVista mesh."""
        frustum_source = _vtk.vtkFrustumSource()
        frustum_source.ShowLinesOff()
        frustum_source.SetPlanes(self.frustum)
        frustum_source.Update()
        return pyvista.wrap(frustum_source.GetOutput())

    @property
    def viewport(self) -> Tuple[float, float, float, float]:
        """Get the selected viewport coordinates.

        Coordinates are given as: x0, y0, x1, y1
        """
        return self._viewport


class PointPickingElementHandler:
    """Internal picking handler for element-based picking.

    This handler is only valid for single point picking operations.
    """

    def __init__(self, mode: ElementType = ElementType.CELL, callback=None):
        self._picker_ = None
        self.callback = callback
        self.mode = ElementType.from_any(mode)

    @property
    def picker(self):
        """Get or set the picker instance."""
        return self._picker_()

    @picker.setter
    def picker(self, picker):
        self._picker_ = weakref.ref(picker)

    def get_mesh(self):
        """Get the picked mesh."""
        ds = self.picker.GetDataSet()
        if ds is not None:
            return pyvista.wrap(ds)

    def get_cell(self, picked_point):
        """Get the picked cell of the picked mesh."""
        mesh = self.get_mesh()
        # cell_id = self.picker.GetCellId()
        cell_id = mesh.find_containing_cell(picked_point)  # more accurate
        if cell_id < 0:
            return  # TODO: this happens but shouldn't
        cell = mesh.extract_cells(cell_id)
        cell.cell_data['vtkOriginalCellIds'] = np.array([cell_id])
        return cell

    def get_face(self, picked_point):
        """Get the picked face of the picked cell."""
        cell = self.get_cell(picked_point).get_cell(0)
        if cell.n_faces > 1:
            for i, face in enumerate(cell.faces):
                contains = face.cast_to_unstructured_grid().find_containing_cell(picked_point)
                if contains > -1:
                    break
            if contains < 0:
                # this shouldn't happen
                raise RuntimeError('Trouble aligning point with face.')
            face = face.cast_to_unstructured_grid()
            face.field_data['vtkOriginalFaceIds'] = np.array([i])
        else:
            face = cell.cast_to_unstructured_grid()
            face.field_data['vtkOriginalFaceIds'] = np.array([0])
        return face

    def get_edge(self, picked_point):
        """Get the picked edge of the picked cell."""
        cell = self.get_cell(picked_point).get_cell(0)
        if cell.n_edges > 1:
            ei = (
                cell.cast_to_unstructured_grid().extract_all_edges().find_closest_cell(picked_point)
            )
            edge = cell.edges[ei].cast_to_unstructured_grid()
            edge.field_data['vtkOriginalEdgeIds'] = np.array([ei])
        else:
            edge = cell.cast_to_unstructured_grid()
        return edge

    def get_point(self, picked_point):
        """Get the picked point of the picked mesh."""
        mesh = self.get_mesh()
        pid = mesh.find_closest_point(picked_point)
        picked = pyvista.PolyData(mesh.points[pid])
        picked.point_data['vtkOriginalPointIds'] = np.array([pid])
        return picked

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


class PickingInterface:
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
            self.remove_actor(name)

    @property
    def picked_point(self):
        """Return the picked point.

        This returns the picked point after selecting a point.

        Returns
        -------
        numpy.ndarray or None
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
        renderer = self.iren.get_poked_renderer()
        return renderer.get_pick_position()

    def pick_click_position(self):
        """Get corresponding click location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        if self.click_position is None:
            self.store_click_position()
        renderer = self.iren.get_poked_renderer()
        self.iren.picker.Pick(self.click_position[0], self.click_position[1], 0, renderer)
        return self.iren.picker.GetPickPosition()

    def pick_mouse_position(self):
        """Get corresponding mouse location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        if self.mouse_position is None:
            self.store_mouse_position()
        renderer = self.iren.get_poked_renderer()
        self.iren.picker.Pick(self.mouse_position[0], self.mouse_position[1], 0, renderer)
        return self.iren.picker.GetPickPosition()

    def _init_click_picking_callback(self, left_clicking=False):
        if left_clicking:
            self._picking_left_clicking_observer = self.iren.add_observer(
                "LeftButtonPressEvent",
                partial(try_callback, _launch_pick_event),
            )
        else:
            self._picking_right_clicking_observer = self.iren.add_observer(
                "RightButtonPressEvent",
                partial(try_callback, _launch_pick_event),
            )

    def disable_picking(self):
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
            self.iren.remove_observer(self._picking_left_clicking_observer)
            self.iren.remove_observer(self._picking_right_clicking_observer)
            # Reset to default picker
            self.iren.reset_picker()
        self._picking_left_clicking_observer = None
        self._picking_right_clicking_observer = None

        self._picker_in_use = False

    def _validate_picker_not_in_use(self):
        if self._picker_in_use:
            raise PyVistaPickingError(
                'Picking is already enabled, please disable previous picking with `disable_picking()`.'
            )

    def enable_point_picking(
        self,
        callback=None,
        tolerance=0.025,
        left_clicking=False,
        picker=PickerType.POINT,
        show_message=True,
        font_size=18,
        color='pink',
        point_size=10,
        show_point=True,
        use_picker=False,
        pickable_window=False,
        clear_on_no_selection=True,
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

                * ``'hardware'``: Uses ``vtkHardwarePicker`` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses ``vtkCellPicker``.
                * ``'point'``: Uses ``vtkPointPicker`` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses ``vtkVolumePicker``.

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
            warnings.warn(
                '`use_mesh` is deprecated. See `use_picker` instead.', PyVistaDeprecationWarning
            )
            use_mesh = kwargs.pop('use_mesh')
        else:
            use_mesh = False

        self_ = weakref.ref(self)

        def _end_pick_event(picker, event):
            if (
                not pickable_window
                and hasattr(picker, 'GetDataSet')
                and picker.GetDataSet() is None
            ):
                # Clear the selection
                self._picked_point = None
                if clear_on_no_selection:
                    with self_().iren.poked_subplot():
                        self_()._clear_picking_representations()
                return
            with self_().iren.poked_subplot():
                point = np.array(picker.GetPickPosition())
                point /= self_().scale  # HACK: handle scale
                self_()._picked_point = point
                if show_point:
                    _kwargs = kwargs.copy()
                    self_().add_mesh(
                        self_().picked_point,
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
                            self_(), callback, picker.GetDataSet(), picker.GetPointId()
                        )
                    else:
                        _poked_context_callback(self_(), callback, self.picked_point)

        if picker is not None:  # If None, that means use already set picker
            self.iren.picker = picker
        if hasattr(self.iren.picker, 'SetTolerance'):
            self.iren.picker.SetTolerance(tolerance)
        self.iren.add_pick_obeserver(_end_pick_event)
        self._init_click_picking_callback(left_clicking=left_clicking)
        self._picker_in_use = True

        # Now add text about cell-selection
        if show_message:
            if show_message is True:
                if left_clicking:
                    show_message = "Left-click"
                else:
                    show_message = "Right-click"
                show_message += ' or press P to pick under the mouse'
            self._picking_text = self.add_text(
                str(show_message), font_size=font_size, name='_point_picking_message'
            )

    def enable_rectangle_picking(
        self,
        callback=None,
        show_message=True,
        font_size=18,
        start=False,
        show_frustum=False,
        style='wireframe',
        color='pink',
        **kwargs,
    ):
        """Enable rectangle based picking at cells.

        Press ``"r"`` to enable retangle based selection. Press
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

        start : bool, default: True
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

        def _end_pick_helper(picker, *args):
            renderer = picker.GetRenderer()  # TODO: double check this is poked renderer
            x0 = int(renderer.GetPickX1())
            x1 = int(renderer.GetPickX2())
            y0 = int(renderer.GetPickY1())
            y1 = int(renderer.GetPickY2())

            selection = RectangleSelection(frustum=picker.GetFrustum(), viewport=(x0, y0, x1, y1))

            if show_frustum:
                with self_().iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self_().add_mesh(
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

        self.enable_rubber_band_style()  # TODO: better handle?
        self.iren.picker = 'rendered'
        self.iren.add_pick_obeserver(_end_pick_helper)
        self._picker_in_use = True

        # Now add text about cell-selection
        if show_message:
            if show_message is True:
                show_message = "Press R to toggle selection tool"
            self._picking_text = self.add_text(
                str(show_message), font_size=font_size, name='_rectangle_picking_message'
            )

        if start:
            self.iren._style_class.StartSelect()


class PickingMethods(PickingInterface):
    """Internal class to contain picking utilities."""

    def __init__(self, *args, **kwargs):
        """Initialize the picking methods."""
        super().__init__(*args, **kwargs)
        self._picked_actor = None
        self._picked_mesh = None
        self._picked_cell = None
        self._picking_text = None
        self._picked_block_index = None

    @property
    def picked_actor(self):
        """Return the picked mesh.

        This returns the picked actor after selecting a mesh with
        :func:`enable_surface_point_picking <pyvista.Plotter.enable_surface_point_picking>` or
        :func:`enable_mesh_picking <pyvista.Plotter.enable_mesh_picking>`.

        Returns
        -------
        pyvista.Actor or None
            Picked actor if available.

        """
        return self._picked_actor

    @property
    def picked_mesh(self):
        """Return the picked mesh.

        This returns the picked mesh after selecting a mesh with
        :func:`enable_surface_point_picking <pyvista.Plotter.enable_surface_point_picking>` or
        :func:`enable_mesh_picking <pyvista.Plotter.enable_mesh_picking>`.

        Returns
        -------
        pyvista.DataSet or None
            Picked mesh if available.

        """
        return self._picked_mesh

    @property
    def picked_cell(self):
        """Return the picked cell.

        This returns the picked cell after selecting a cell.

        Returns
        -------
        pyvista.Cell or None
            Picked cell if available.

        """
        return self._picked_cell

    @property
    def picked_cells(self):
        """Return the picked cells.

        This returns the picked cells after selecting cells.

        Returns
        -------
        pyvista.Cell or None
            Picked cell if available.

        """
        return self._picked_cell

    @property
    def picked_block_index(self):
        """Return the picked block index.

        This returns the picked block index after selecting a point with
        :func:`enable_point_picking <pyvista.Plotter.enable_point_picking>`.

        Returns
        -------
        int or None
            Picked block if available. If ``-1``, then a non-composite dataset
            was selected.

        """
        return self._picked_block_index

    @wraps(PickingInterface.disable_picking)
    def disable_picking(self):
        """Disable picking."""
        super().disable_picking()
        # remove any picking text
        if hasattr(self, 'renderers'):
            for renderer in self.renderers:
                renderer.remove_actor(self._picking_text, render=False)
        self._picking_text = None

    def enable_surface_point_picking(
        self,
        callback=None,
        show_message=True,
        font_size=18,
        color='pink',
        show_point=True,
        point_size=10,
        tolerance=0.025,
        pickable_window=False,
        left_clicking=False,
        picker=PickerType.CELL,
        use_picker=False,
        clear_on_no_selection=True,
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

                * ``'hardware'``: Uses ``vtkHardwarePicker`` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses ``vtkCellPicker``.
                * ``'point'``: Uses ``vtkPointPicker`` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses ``vtkVolumePicker``.

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
        valid_pickers = [PickerType.POINT, PickerType.CELL, PickerType.HARDWARE, PickerType.VOLUME]
        if picker not in valid_pickers:
            raise ValueError(
                f'Invalid picker choice for surface picking. Use one of: {valid_pickers}'
            )

        self_ = weakref.ref(self)

        def _end_pick_event(picked_point, picker):
            if not pickable_window and picker.GetActor() is None:
                self_()._picked_point = None
                self_()._picked_actor = None
                self_()._picked_mesh = None
                if clear_on_no_selection:
                    with self_().iren.poked_subplot():
                        self_()._clear_picking_representations()
                return
            self_()._picked_actor = picker.GetActor()
            self_()._picked_mesh = picker.GetDataSet()

            if show_point:
                with self_().iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self_().add_mesh(
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

    def enable_surface_picking(self, *args, **kwargs):
        """Surface picking.

        .. deprecated:: 0.40.0
            This method has been renamed to ``enable_surface_point_picking``.

        Parameters
        ----------
        *args : tuple
            Positional arguments.

        **kwargs : dict
            Keyword arguments.

        """
        # Deprecated on v0.40.0, estimated removal on v0.42.0
        warnings.warn(
            "This method has been renamed to `enable_surface_point_picking`.",
            PyVistaDeprecationWarning,
        )
        self.enable_surface_point_picking(*args, **kwargs)

    def enable_mesh_picking(
        self,
        callback=None,
        show=True,
        show_message=True,
        style='wireframe',
        line_width=5,
        color='pink',
        font_size=18,
        left_clicking=False,
        use_actor=False,
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

                * ``'hardware'``: Uses ``vtkHardwarePicker`` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses ``vtkCellPicker``.
                * ``'point'``: Uses ``vtkPointPicker`` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses ``vtkVolumePicker``.


        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        Returns
        -------
        vtk.vtkPropPicker
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

        def end_pick_call_back(_, picker):
            if callback:
                if use_actor:
                    _poked_context_callback(self_(), callback, self_()._picked_actor)
                else:
                    _poked_context_callback(self_(), callback, self_()._picked_mesh)

            if show:
                # Select the renderer where the mesh is added.
                active_renderer_index = self_().renderers._active_index
                loc = self_().iren.get_event_subplot_loc()
                self_().subplot(*loc)

                # Use try in case selection is empty or invalid
                try:
                    with self_().iren.poked_subplot():
                        _kwargs = kwargs.copy()
                        self_().add_mesh(
                            self_()._picked_mesh,
                            name=_kwargs.pop('name', PICKED_REPRESENTATION_NAMES['mesh']),
                            style=style,
                            color=color,
                            line_width=line_width,
                            pickable=_kwargs.pop('pickable', False),
                            reset_camera=_kwargs.pop('reset_camera', False),
                            **_kwargs,
                        )
                except Exception as e:  # pragma: no cover
                    warnings.warn("Unable to show mesh when picking:\n\n%s", str(e))

                # Reset to the active renderer.
                loc = self_().renderers.index_to_loc(active_renderer_index)
                self_().subplot(*loc)

                # render here prior to running the callback
                self_().render()

        # add on-screen message about point-selection
        if show_message:
            if show_message is True:
                if left_clicking:
                    show_message = "Left-click"
                else:
                    show_message = "Right-click"
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
        callback=None,
        show=True,
        style='wireframe',
        line_width=5,
        color='pink',
        show_message=True,
        font_size=18,
        start=False,
        show_frustum=False,
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

        start : bool, default: True
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
                # Inidcates invalid pick
                with self_().iren.poked_subplot():
                    self_()._clear_picking_representations()
                return

            self._picked_cell = picked

            if show:
                # Use try in case selection is empty
                with self_().iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self_().add_mesh(
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
                _poked_context_callback(self_(), callback, self_().picked_cells)

        def through_pick_callback(selection):
            picked = pyvista.MultiBlock()
            renderer = self_().iren.get_poked_renderer()
            for actor in renderer.actors.values():
                if actor.GetMapper() and actor.GetPickable():
                    input_mesh = pyvista.wrap(actor.GetMapper().GetInputAsDataSet())
                    input_mesh.cell_data['orig_extract_id'] = np.arange(input_mesh.n_cells)
                    extract = _vtk.vtkExtractGeometry()
                    extract.SetInputData(input_mesh)
                    extract.SetImplicitFunction(selection.frustum)
                    extract.Update()
                    picked.append(pyvista.wrap(extract.GetOutput()))

            if picked.n_blocks == 0:
                self_()._picked_cell = None
            elif picked.combine().n_cells < 1:
                self_()._picked_cell = None
            elif picked.n_blocks == 1:
                self_()._picked_cell = picked[0]
            else:
                self_()._picked_cell = picked

            finalize(self_()._picked_cell)

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
        callback=None,
        show=True,
        style='wireframe',
        line_width=5,
        color='pink',
        show_message=True,
        font_size=18,
        start=False,
        show_frustum=False,
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

        start : bool, default: True
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
                # Inidcates invalid pick
                with self_().iren.poked_subplot():
                    self_()._clear_picking_representations()
                return

            if show:
                # Use try in case selection is empty
                with self_().iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self_().add_mesh(
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
            picked = pyvista.MultiBlock()
            renderer = self_().iren.get_poked_renderer()
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
                    cids = pyvista.convert_array(selection_node.GetSelectionList())
                    actor = selection_node.GetProperties().Get(_vtk.vtkSelectionNode.PROP())

                    # TODO: this is too hacky - find better way to avoid non-dataset actors
                    if not actor.GetMapper() or not hasattr(
                        actor.GetProperty(), 'GetRepresentation'
                    ):
                        continue

                    # if not a surface
                    if actor.GetProperty().GetRepresentation() != 2:  # pragma: no cover
                        warnings.warn(
                            "Display representations other than `surface` will result in incorrect results."
                        )
                    smesh = pyvista.wrap(actor.GetMapper().GetInputAsDataSet())
                    smesh = smesh.copy()
                    smesh["original_cell_ids"] = np.arange(smesh.n_cells)
                    tri_smesh = smesh.extract_surface().triangulate()
                    cids_to_get = tri_smesh.extract_cells(cids)["original_cell_ids"]
                    picked.append(smesh.extract_cells(cids_to_get))

                # memory leak issues on vtk==9.0.20210612.dev0
                # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18239#note_973826
                selection.UnRegister(selection)

            if len(picked) == 0:
                self_()._picked_cell = None
            elif picked.combine().n_cells < 1:
                self_()._picked_cell = None
            elif len(picked) == 1:
                self_()._picked_cell = picked[0]
            else:
                self_()._picked_cell = picked

            finalize(self_()._picked_cell)

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
        callback=None,
        through=True,
        show=True,
        show_message=True,
        style='wireframe',
        line_width=5,
        color='pink',
        font_size=18,
        start=False,
        show_frustum=False,
        **kwargs,
    ):
        """Enable picking of cells with a rectangle selection tool.

        Press ``"r"`` to enable retangle based selection.  Press
        ``"r"`` again to turn it off. Selection will be saved to
        ``self.picked_cells``.

        All meshes in the scene are available for picking by default.
        If you would like to only pick a single mesh in the scene,
        use the ``pickable=False`` argument when adding the other
        meshes to the scene.

        When multiple meshes are being picked, the picked cells
        in ``self.picked_cells`` will be a :class:`MultiBlock`
        dataset for each mesh's selection.

        Uses last input mesh for input by default.

        .. warning::
           Visible cell picking (``through=False``) will only work if
           the mesh is displayed with a ``'surface'`` representation
           style (the default).

        Parameters
        ----------
        callback : callable, optional
            When input, calls this callable after a selection is made.
            The picked_cells are input as the first parameter to this
            callable.

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

        start : bool, default: True
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
        callback=None,
        mode='cell',
        show=True,
        show_message=True,
        font_size=18,
        color='pink',
        tolerance=0.025,
        pickable_window=False,
        left_clicking=False,
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

        color : ColorLike, default: "pink"
            The color of the selected mesh when shown.

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

                * ``'hardware'``: Uses ``vtkHardwarePicker`` which is more
                  performant for large geometries (default).
                * ``'cell'``: Uses ``vtkCellPicker``.
                * ``'point'``: Uses ``vtkPointPicker`` which will snap to
                  points on the surface of the mesh.
                * ``'volume'``: Uses ``vtkVolumePicker``.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

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

                with self.iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self.add_mesh(
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
            color=color,
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
        >>> multiblock = pv.MultiBlock(
        ...     [pv.Cube(), pv.Sphere(center=(0, 0, 1))]
        ... )
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(multiblock)
        >>> def turn_blue(index, dataset):
        ...     mapper.block_attr[index].color = 'blue'
        ...
        >>> pl.enable_block_picking(callback=turn_blue, side='left')
        >>> def clear_color(index, dataset):
        ...     mapper.block_attr[index].color = None
        ...
        >>> pl.enable_block_picking(callback=clear_color, side='right')
        >>> pl.show()

        """
        # use a weak reference to enable garbage collection
        renderer_ = weakref.ref(self.renderer)
        self_ = weakref.ref(self)

        sel_index = _vtk.vtkSelectionNode.COMPOSITE_INDEX()
        sel_prop = _vtk.vtkSelectionNode.PROP()

        def get_picked_block(*args, **kwargs):
            """Get the picked block and pass it to the user callback."""
            x, y = self.mouse_position
            selector = _vtk.vtkOpenGLHardwareSelector()
            selector.SetRenderer(renderer_())
            selector.SetArea(x, y, x, y)  # single pixel
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

        self.track_click_position(callback=get_picked_block, viewport=True, side=side)


class PickingHelper(PickingMethods):
    """Internal container class to contain picking helper methods."""

    def __init__(self, *args, **kwargs):
        """Initialize the picking methods."""
        super().__init__(*args, **kwargs)
        self.picked_path = None
        self.picked_geodesic = None
        self.picked_horizon = None

    def fly_to_mouse_position(self, focus=False):
        """Focus on last stored mouse position."""
        if self.mouse_position is None:
            self.store_mouse_position()
        click_point = self.pick_mouse_position()
        if focus:
            self.set_focus(click_point)
        else:
            self.fly_to(click_point)

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

        def _the_callback(*args):
            click_point = self.pick_mouse_position()
            self.fly_to(click_point)
            if callable(callback):
                _poked_context_callback(self_(), callback, click_point)

        self.track_click_position(callback=_the_callback, side="right")

    def enable_path_picking(
        self,
        callback=None,
        show_message=True,
        font_size=18,
        color='pink',
        point_size=10,
        line_width=5,
        show_path=True,
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
            cells = np.insert(cells, 0, n_points)
            return cells

        the_points = []

        def _the_callback(picked_point, picker):
            if picker.GetDataSet() is None:
                return
            the_points.append(picked_point)
            self.picked_path = pyvista.PolyData(np.array(the_points))
            self.picked_path.lines = make_line_cells(len(the_points))
            if show_path:
                with self.iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self.add_mesh(
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
            with self.iren.poked_subplot():
                self._clear_picking_representations()

        self.add_key_event('c', _clear_path_event_watcher)
        if show_message is True:
            show_message = "Press P to pick under the mouse\nPress C to clear"

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
        callback=None,
        show_message=True,
        font_size=18,
        color='pink',
        point_size=10,
        line_width=5,
        tolerance=0.025,
        show_path=True,
        keep_order=True,
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

        self.picked_geodesic = pyvista.PolyData()
        self._last_picked_idx = None

        def _the_callback(picked_point, picker):
            if picker.GetDataSet() is None:
                return
            mesh = pyvista.wrap(picker.GetDataSet())
            idx = mesh.find_closest_point(picked_point)
            point = mesh.points[idx]
            if self._last_picked_idx is None:
                self.picked_geodesic = pyvista.PolyData(point)
                self.picked_geodesic['vtkOriginalPointIds'] = [idx]
            else:
                surface = mesh.extract_surface().triangulate()
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
                with self.iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self.add_mesh(
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
            self.picked_geodesic = pyvista.PolyData()
            with self.iren.poked_subplot():
                self._clear_picking_representations()
            self._last_picked_idx = None

        self.add_key_event('c', _clear_g_path_event_watcher)
        if show_message is True:
            show_message = "Press P to pick under the mouse\nPress C to clear"

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
        callback=None,
        normal=(0.0, 0.0, 1.0),
        width=None,
        show_message=True,
        font_size=18,
        color='pink',
        point_size=10,
        line_width=5,
        show_path=True,
        opacity=0.75,
        show_horizon=True,
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
            self.picked_horizon = pyvista.PolyData()
            with self.iren.poked_subplot():
                self._clear_picking_representations()

        self.add_key_event('c', _clear_horizon_event_watcher)

        def _the_callback(path):
            if path.n_points < 2:
                _clear_horizon_event_watcher()
                return
            self.picked_horizon = path.ribbon(normal=normal, width=width)

            if show_horizon:
                with self.iren.poked_subplot():
                    _kwargs = kwargs.copy()
                    self.add_mesh(
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
