"""Module managing picking events."""

from functools import partial
import logging
import weakref

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import try_callback

from .composite_mapper import CompositePolyDataMapper


def _launch_pick_event(interactor, event):
    """Create a Pick event based on coordinate or left-click."""
    click_x, click_y = interactor.GetEventPosition()
    click_z = 0

    picker = interactor.GetPicker()
    renderer = interactor.GetInteractorStyle()._parent()._plotter.renderer
    picker.Pick(click_x, click_y, click_z, renderer)


class PickingHelper:
    """An internal class to hold picking related features."""

    def __init__(self, *args, **kwargs):
        """Initialize the picking helper."""
        super().__init__(*args, **kwargs)
        self.picked_cells = None
        self._picked_point = None
        self._picked_mesh = None
        self.picked_path = None
        self.picked_geodesic = None
        self.picked_horizon = None
        self._picking_left_clicking_observer = None
        self._picking_text = None
        self._picker = None
        self._picked_block_index = None

    def get_pick_position(self):
        """Get the pick position or area.

        Returns
        -------
        sequence
            Picked position or area as ``(x0, y0, x1, y1)``.

        """
        return self.renderer.get_pick_position()

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
        **kwargs,
    ):
        """Enable picking of a mesh.

        Parameters
        ----------
        callback : function, optional
            When input, calls this function after a selection is made. The
            ``mesh`` is input as the first parameter to this function.

        show : bool, optional
            Show the selection interactively. Best when combined with
            ``left_clicking``.

        show_message : bool or str, optional
            Show the message about how to use the mesh picking tool. If this
            is a string, that will be the message shown.

        style : str, optional
            Visualization style of the selection.  Defaults to
            ``'wireframe'``. One of the following:

            * ``'surface'``
            * ``'wireframe'``
            * ``'points'``

        line_width : float, optional
            Thickness of selected mesh edges. Default 5.

        color : color_like, optional
            The color of the selected mesh when shown.

        font_size : int, optional
            Sets the font size of the message.

        left_clicking : bool, optional
            When ``True``, meshes can be picked by clicking the left
            mousebutton.  Default to ``False``.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked point.

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
        >>> _ = pl.enable_mesh_picking(left_clicking=True)

        See :ref:`mesh_picking_example` for a full example using this method.

        """

        def end_pick_call_back(picked, event):
            is_valid_selection = False
            self_ = weakref.ref(self)
            actor = picked.GetActor()
            if actor:
                mesh = actor.GetMapper().GetInputAsDataSet()
                is_valid_selection = True

            if is_valid_selection:
                self_()._picked_mesh = mesh

            if callback and is_valid_selection:
                try_callback(callback, mesh)

            if show and is_valid_selection:

                # Select the renderer where the mesh is added.
                active_renderer_index = self_().renderers._active_index
                for index in range(len(self.renderers)):
                    renderer = self.renderers[index]
                    for actor in renderer._actors.values():
                        mapper = actor.GetMapper()
                        if (
                            isinstance(mapper, _vtk.vtkDataSetMapper)
                            and mapper.GetInputAsDataSet() == mesh
                        ):
                            loc = self_().renderers.index_to_loc(index)
                            self_().subplot(*loc)
                            break

                # Use try in case selection is empty or invalid
                try:
                    self_().add_mesh(
                        mesh,
                        name='_mesh_picking_selection',
                        style=style,
                        color=color,
                        line_width=line_width,
                        pickable=False,
                        reset_camera=False,
                        **kwargs,
                    )
                except Exception as e:  # pragma: no cover
                    logging.warning("Unable to show mesh when picking:\n\n%s", str(e))

                # Reset to the active renderer.
                loc = self_().renderers.index_to_loc(active_renderer_index)
                self_().subplot(*loc)

                # render here prior to running the callback
                self_().render()
            elif not is_valid_selection:
                self.remove_actor('_mesh_picking_selection')
                self_()._picked_mesh = None

        # add on-screen message about point-selection
        if show_message:
            if show_message is True:
                show_message = "\nPress P to pick a single dataset under the mouse pointer."
                if left_clicking:
                    show_message += "\nor click to select a dataset under the mouse pointer."

            self._picking_text = self.add_text(
                str(show_message), font_size=font_size, name='_mesh_picking_message'
            )

        if left_clicking:
            self._picking_left_clicking_observer = self.iren.interactor.AddObserver(
                "LeftButtonPressEvent",
                partial(try_callback, _launch_pick_event),
            )

        self._picker = _vtk.vtkPropPicker()
        self._picker.AddObserver(_vtk.vtkCommand.EndPickEvent, end_pick_call_back)
        self.enable_trackball_style()
        self.iren.set_picker(self._picker)
        return self._picker

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
        **kwargs,
    ):
        """Enable picking at cells.

        Press ``"r"`` to enable retangle based selection.  Press
        ``"r"`` again to turn it off. Selection will be saved to
        ``self.picked_cells``. Also press ``"p"`` to pick a single
        cell under the mouse location.

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

        .. warning::
            Cell picking can only be enabled for a single renderer
            or subplot at a time.

        Parameters
        ----------
        callback : function, optional
            When input, calls this function after a selection is made.
            The picked_cells are input as the first parameter to this
            function.

        through : bool, optional
            When ``True`` (default) the picker will select all cells
            through the mesh(es). When ``False``, the picker will select
            only visible cells on the selected surface(s).

        show : bool, optional
            Show the selection interactively.

        show_message : bool or str, optional
            Show the message about how to use the cell picking tool. If this
            is a string, that will be the message shown.

        style : str, optional
            Visualization style of the selection.  One of the
            following: ``style='surface'``, ``style='wireframe'``, or
            ``style='points'``.  Defaults to ``'wireframe'``.

        line_width : float, optional
            Thickness of selected mesh edges. Default 5.

        color : color_like, optional
            The color of the selected mesh when shown.

        font_size : int, optional
            Sets the font size of the message.

        start : bool, optional
            Automatically start the cell selection tool.

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
        >>> _ = pl.enable_cell_picking(left_clicking=True)

        """
        self_ = weakref.ref(self)

        # make sure to consistently use renderer
        renderer_ = weakref.ref(self.renderer)

        def end_pick_helper(picker, event_id):
            # Merge the selection into a single mesh
            picked = self_().picked_cells
            if isinstance(picked, pyvista.MultiBlock):
                if picked.n_blocks > 0:
                    picked = picked.combine()
                else:
                    picked = pyvista.UnstructuredGrid()
            # Check if valid
            is_valid_selection = picked.n_cells > 0

            if show and is_valid_selection:
                # Select the renderer where the mesh is added.
                active_renderer_index = self_().renderers._active_index
                for index in range(len(self.renderers)):
                    renderer = self.renderers[index]
                    for actor in renderer._actors.values():
                        mapper = actor.GetMapper()
                        if isinstance(mapper, _vtk.vtkDataSetMapper):
                            loc = self_().renderers.index_to_loc(index)
                            self_().subplot(*loc)
                            break

                # Use try in case selection is empty
                self_().add_mesh(
                    picked,
                    name='_cell_picking_selection',
                    style=style,
                    color=color,
                    line_width=line_width,
                    pickable=False,
                    reset_camera=False,
                    **kwargs,
                )

                # Reset to the active renderer.
                loc = self_().renderers.index_to_loc(active_renderer_index)
                self_().subplot(*loc)

                # render here prior to running the callback
                self_().render()
            elif not is_valid_selection:
                self.remove_actor('_cell_picking_selection')
                self_().picked_cells = None

            if callback is not None:
                try_callback(callback, self_().picked_cells)

            # TODO: Deactivate selection tool
            return

        def through_pick_call_back(picker, event_id):
            picked = pyvista.MultiBlock()
            for actor in renderer_().actors.values():
                if actor.GetMapper() and actor.GetPickable():
                    input_mesh = pyvista.wrap(actor.GetMapper().GetInputAsDataSet())
                    input_mesh.cell_data['orig_extract_id'] = np.arange(input_mesh.n_cells)
                    extract = _vtk.vtkExtractGeometry()
                    extract.SetInputData(input_mesh)
                    extract.SetImplicitFunction(picker.GetFrustum())
                    extract.Update()
                    picked.append(pyvista.wrap(extract.GetOutput()))

            if len(picked) == 1:
                self_().picked_cells = picked[0]
            else:
                self_().picked_cells = picked
            return end_pick_helper(picker, event_id)

        def visible_pick_call_back(picker, event_id):
            picked = pyvista.MultiBlock()
            x0, y0, x1, y1 = renderer_().get_pick_position()
            if x0 >= 0:  # initial pick position is (-1, -1, -1, -1)
                selector = _vtk.vtkOpenGLHardwareSelector()
                selector.SetFieldAssociation(_vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
                selector.SetRenderer(renderer_())
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
                        logging.warning(
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

            if len(picked) == 1:
                self_().picked_cells = picked[0]
            else:
                self_().picked_cells = picked
            return end_pick_helper(picker, event_id)

        self._picker = _vtk.vtkRenderedAreaPicker()
        if through:
            self._picker.AddObserver(_vtk.vtkCommand.EndPickEvent, through_pick_call_back)
        else:
            # NOTE: there can be issues with non-triangulated meshes
            # Reference:
            #     https://github.com/pyvista/pyvista/issues/277
            #     https://github.com/pyvista/pyvista/pull/281
            #     https://discourse.vtk.org/t/visible-cell-selection-hardwareselector-py-example-is-not-working-reliably/1262
            self._picker.AddObserver(_vtk.vtkCommand.EndPickEvent, visible_pick_call_back)

        self.enable_rubber_band_style()
        self.iren.set_picker(self._picker)

        # Now add text about cell-selection
        if show_message:
            if show_message is True:
                show_message = "Press R to toggle selection tool"
                if not through:
                    show_message += "\nPress P to pick a single cell under the mouse"
            self._picking_text = self.add_text(
                str(show_message), font_size=font_size, name='_cell_picking_message'
            )

        if start:
            self.iren._style_class.StartSelect()

    @property
    def picked_mesh(self):
        """Return the picked mesh.

        This returns the picked mesh after selecting a mesh with
        :func:`<enable_mesh_picking> pyvista.Plotter.enable_mesh_picking` or
        :func:`<enable_point_picking> pyvista.Plotter.enable_point_picking`.

        Returns
        -------
        pyvista.DataSet or None
            Picked mesh if available.

        """
        return self._picked_mesh

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

    @property
    def picked_block_index(self):
        """Return the picked block index.

        This returns the picked block index after selecting a point with
        :func:`<enable_point_picking> pyvista.Plotter.enable_point_picking`.

        Returns
        -------
        int or None
            Picked block if available. If ``-1``, then a non-composite dataset
            was selected.

        """
        return self._picked_block_index

    def enable_surface_picking(
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
        **kwargs,
    ):
        """Enable picking of a point on the surface of a mesh.

        Parameters
        ----------
        callback : function, optional
            When input, calls this function after a selection is made. The
            ``mesh`` is input as the first parameter to this function.

        show_message : bool or str, optional
            Show the message about how to use the mesh picking tool. If this
            is a string, that will be the message shown.

        font_size : int, optional
            Sets the font size of the message.

        color : color_like, optional
            The color of the selected mesh when shown.

        show_point : bool, optional
            Show the selection interactively. Default ``True``.

        point_size : int, optional
            Size of picked points if ``show_point`` is
            ``True``. Default 10.

        tolerance : float, optional
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size. Rendering window size is measured across diagonal.

        pickable_window : bool, optional
            When ``True``, points in the 3D window are pickable. Default to ``False``.

        left_clicking : bool, optional
            When ``True``, meshes can be picked by clicking the left
            mousebutton.  Default to ``False``.

            .. note::
               If enabled, left-clicking will **not** display the bounding box
               around the picked mesh.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        Returns
        -------
        vtk.vtkPropPicker
            Property picker.

        Notes
        -----
        Picked point can be accessed from :attr:`picked_point
        <PickingHelper.picked_point>` attribute.

        Examples
        --------
        Add a cube to a plot and enable cell picking. Enable ``left_clicking``
        to immediately start picking on the left click and disable showing the
        box. You can still press the ``p`` key to select meshes.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_surface_picking(left_clicking=True)

        See :ref:`surface_picking_example` for a full example using this method.

        """

        def _end_pick_event(picker, event):

            picked_point_id = picker.GetPointId()
            if not pickable_window and picked_point_id < 0:
                self._picked_point = None
                return

            self._picked_point = np.array(picker.GetPickPosition())
            self._picked_mesh = picker.GetDataSet()
            if show_point:
                self.add_mesh(
                    self.picked_point,
                    color=color,
                    point_size=point_size,
                    name='_picked_point',
                    pickable=False,
                    reset_camera=False,
                    **kwargs,
                )
            if callable(callback):
                try_callback(callback, self.picked_point)

        picker = _vtk.vtkCellPicker()
        picker.SetTolerance(tolerance)
        self._picker = picker
        picker.AddObserver(_vtk.vtkCommand.EndPickEvent, _end_pick_event)

        self.enable_trackball_style()
        self.iren.set_picker(picker)

        if left_clicking:
            self._picking_left_clicking_observer = self.iren.interactor.AddObserver(
                "LeftButtonPressEvent",
                partial(try_callback, _launch_pick_event),
            )

        # add on-screen message about point-selection
        if show_message:
            if show_message is True:
                show_message = "\nPress P to pick a point on the surface under the mouse pointer."
                if left_clicking:
                    show_message += "\nor click to select a point under the mouse pointer."

            self._picking_text = self.add_text(
                str(show_message), font_size=font_size, name='_surf_picking_message'
            )

        return picker

    def enable_point_picking(
        self,
        callback=None,
        show_message=True,
        font_size=18,
        color='pink',
        point_size=10,
        use_mesh=False,
        show_point=True,
        tolerance=0.025,
        pickable_window=False,
        left_clicking=False,
        **kwargs,
    ):
        """Enable picking at points.

        Enable picking a point at the mouse location in the render
        view using the ``P`` key. This point is saved to the
        ``.picked_point`` attribute on the plotter. Pass a callback
        function that takes that point as an argument. The picked
        point can either be a point on the first intersecting mesh, or
        a point in the 3D window.

        Parameters
        ----------
        callback : callable, optional
            When input, calls this function after a pick is made.  The
            picked point is input as the first parameter to this
            function.  If ``use_mesh`` is ``True``, the callback
            function will be passed a pointer to the picked mesh and
            the point ID of the selected mesh.

        show_message : bool or str, optional
            Show the message about how to use the point picking
            tool. If this is a string, that will be the message shown.

        font_size : int, optional
            Sets the size of the message.

        color : color_like, optional
            The color of the selected mesh when shown.

        point_size : int, optional
            Size of picked points if ``show_point`` is
            ``True``. Default 10.

        use_mesh : bool, optional
            If ``True``, the callback function will be passed a
            pointer to the picked mesh and the point ID of the
            selected mesh.

        show_point : bool, optional
            Show the picked point after clicking.

        tolerance : float, optional
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size. Rendering window size is measured across diagonal.

        pickable_window : bool, optional
            When ``True``, points in the 3D window are pickable. Default to ``False``.

        left_clicking : bool, optional
            When ``True``, points can be picked by clicking the left mouse button.
            Default to ``False``. Note, if enabled, left-clicking will **not**
            display the bounding box around the picked mesh.

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
        >>> pl.enable_point_picking(show_message="Press P to pick")

        See :ref:`point_picking_example` for a full example using this method.

        """

        def _end_pick_event(picker, event):

            picked_point_id = picker.GetPointId()
            if (not pickable_window) and (picked_point_id < 0):
                return None

            self._picked_point = np.array(picker.GetPickPosition())
            self._picked_mesh = picker.GetDataSet()
            self._picked_block_index = picker.GetFlatBlockIndex()

            self.picked_point_id = picked_point_id
            if show_point:
                self.add_mesh(
                    self.picked_point,
                    color=color,
                    point_size=point_size,
                    name='_picked_point',
                    pickable=False,
                    reset_camera=False,
                    **kwargs,
                )
            if callable(callback):
                if use_mesh:
                    try_callback(callback, self.picked_mesh, self.picked_point_id)
                else:
                    try_callback(callback, self.picked_point)

        point_picker = _vtk.vtkPointPicker()
        point_picker.SetTolerance(tolerance)
        self._picker = point_picker
        point_picker.AddObserver(_vtk.vtkCommand.EndPickEvent, _end_pick_event)

        self.enable_trackball_style()
        self.iren.set_picker(point_picker)

        if left_clicking:
            self._picking_left_clicking_observer = self.iren.add_observer(
                "LeftButtonPressEvent",
                partial(try_callback, _launch_pick_event),
            )

        # Now add text about cell-selection
        if show_message:
            if show_message is True and left_clicking:
                show_message = "Left-click or press P to pick under the mouse"
            elif show_message is True:
                show_message = "Press P to pick under the mouse"
            self._picking_text = self.add_text(
                str(show_message), font_size=font_size, name='_point_picking_message'
            )

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
        <PickingHelper.enable_point_picking>` to keep track of the
        picked points and create a line using those points.

        The line is saved to the ``.picked_path`` attribute of this
        plotter

        Parameters
        ----------
        callback : callable, optional
            When given, calls this function after a pick is made.  The
            entire picked path is passed as the only parameter to this
            function.

        show_message : bool or str, optional
            Show the message about how to use the point picking
            tool. If this is a string, that will be the message shown.

        font_size : int, optional
            Sets the size of the message.

        color : color_like, optional
            The color of the selected mesh when shown.

        point_size : int, optional
            Size of picked points if ``show_path`` is
            ``True``. Default 10.

        line_width : float, optional
            Thickness of path representation if ``show_path`` is
            ``True``.  Default 5.

        show_path : bool, optional
            Show the picked path interactively.

        tolerance : float, optional
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size.  Rendering window size is measured across diagonal.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        """
        kwargs.setdefault('pickable', False)

        def make_line_cells(n_points):
            cells = np.arange(0, n_points, dtype=np.int_)
            cells = np.insert(cells, 0, n_points)
            return cells

        the_points = []
        the_ids = []

        def _the_callback(mesh, idx):
            if mesh is None:
                return
            the_ids.append(idx)
            the_points.append(mesh.points[idx])
            self.picked_path = pyvista.PolyData(np.array(the_points))
            self.picked_path.lines = make_line_cells(len(the_points))
            if show_path:
                self.add_mesh(
                    self.picked_path,
                    color=color,
                    name='_picked_path',
                    line_width=line_width,
                    point_size=point_size,
                    reset_camera=False,
                    **kwargs,
                )
            if callable(callback):
                try_callback(callback, self.picked_path)

        def _clear_path_event_watcher():
            del the_points[:]
            del the_ids[:]
            self.remove_actor('_picked_path')

        self.add_key_event('c', _clear_path_event_watcher)
        if show_message is True:
            show_message = "Press P to pick under the mouse\nPress C to clear"

        self.enable_point_picking(
            callback=_the_callback,
            use_mesh=True,
            font_size=font_size,
            show_message=show_message,
            show_point=False,
            tolerance=tolerance,
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
            When given, calls this function after a pick is made.  The
            entire picked, geodesic path is passed as the only
            parameter to this function.

        show_message : bool or str, optional
            Show the message about how to use the point picking
            tool. If this is a string, that will be the message shown.

        font_size : int, optional
            Sets the size of the message.

        color : color_like, optional
            The color of the selected mesh when shown.

        point_size : int, optional
            Size of picked points if ``show_path`` is
            ``True``. Default 10.

        line_width : float, optional
            Thickness of path representation if ``show_path`` is
            ``True``.  Default 5.

        tolerance : float, optional
            Specify tolerance for performing pick operation. Tolerance
            is specified as fraction of rendering window
            size.  Rendering window size is measured across diagonal.

        show_path : bool, optional
            Show the picked path interactively.

        keep_order : bool, optional
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
        kwargs.setdefault('pickable', False)

        self.picked_geodesic = pyvista.PolyData()
        self._last_picked_idx = None

        def _the_callback(mesh, idx):
            if mesh is None:
                return
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
                self.add_mesh(
                    self.picked_geodesic,
                    color=color,
                    name='_picked_path',
                    line_width=line_width,
                    point_size=point_size,
                    reset_camera=False,
                    **kwargs,
                )
            if callable(callback):
                try_callback(callback, self.picked_geodesic)

        def _clear_g_path_event_watcher():
            self.picked_geodesic = pyvista.PolyData()
            self.remove_actor('_picked_path')
            self._last_picked_idx = None

        self.add_key_event('c', _clear_g_path_event_watcher)
        if show_message is True:
            show_message = "Press P to pick under the mouse\nPress C to clear"

        self.enable_point_picking(
            callback=_the_callback,
            use_mesh=True,
            font_size=font_size,
            show_message=show_message,
            tolerance=tolerance,
            show_point=False,
        )

    def enable_horizon_picking(
        self,
        callback=None,
        normal=(0, 0, 1),
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
            When given, calls this function after a pick is made.  The
            entire picked path is passed as the only parameter to this
            function.

        normal : tuple(float), optional
            The normal to the horizon surface's projection plane.

        width : float, optional
            The width of the horizon surface. Default behaviour will
            dynamically change the surface width depending on its
            length.

        show_message : bool or str, optional
            Show the message about how to use the horizon picking
            tool. If this is a string, that will be the message shown.

        font_size : int, optional
            Sets the font size of the message.

        color : color_like, optional
            The color of the horizon surface if shown.

        point_size : int, optional
            Size of picked points if ``show_horizon`` is
            ``True``. Default 10.

        line_width : float, optional
            Thickness of path representation if ``show_horizon`` is
            ``True``.  Default 5.

        show_path : bool, optional
            Show the picked path that the horizon is built from
            interactively.

        opacity : float, optional
            The opacity of the horizon surface if shown.

        show_horizon : bool, optional
            Show the picked horizon surface interactively.

        **kwargs : dict, optional
            All remaining keyword arguments are used to control how
            the picked path is interactively displayed.

        """
        name = '_horizon'
        self.add_key_event('c', lambda: self.remove_actor(name))

        def _the_callback(path):
            if path.n_points < 2:
                self.remove_actor(name)
                return
            self.picked_horizon = path.ribbon(normal=normal, width=width)

            if show_horizon:
                self.add_mesh(
                    self.picked_horizon,
                    name=name,
                    color=color,
                    opacity=opacity,
                    pickable=False,
                    reset_camera=False,
                )

            if callable(callback):
                try_callback(callback, path)

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

    def enable_block_picking(
        self,
        callback=None,
        side='left',
    ):
        """Enable composite block picking.

        Use this picker to return the index of a DataSet when using composite
        dataset like :class:`pyvista.MultiBlock` and pass it to a callback.

        Parameters
        ----------
        callback : callable, optional
            When input, this picker calls this function after a selection is
            made. The composite index is passed to ``callback`` as the first
            argument and the dataset as the second argument.

        side : str, optional
            The mouse button to track (either ``'left'`` or ``'right'``).
            Default is ``'right'``. Also accepts ``'r'`` or ``'l'``.

        Notes
        -----
        The picked block index can be accessed from :attr:`picked_block_index
        <PickingHelper.picked_block_index>` attribute.

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
        renderer_ = weakref.ref(self.renderer)

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
                    try_callback(callback, self._picked_block_index, dataset)

        self.track_click_position(callback=get_picked_block, viewport=True, side=side)

    def pick_click_position(self):
        """Get corresponding click location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        if self.click_position is None:
            self.store_click_position()
        picker = _vtk.vtkWorldPointPicker()
        picker.Pick(self.click_position[0], self.click_position[1], 0, self.renderer)
        return picker.GetPickPosition()

    def pick_mouse_position(self):
        """Get corresponding mouse location in the 3D plot.

        Returns
        -------
        tuple
            Three item tuple with the 3D picked position.

        """
        if self.mouse_position is None:
            self.store_mouse_position()
        picker = _vtk.vtkWorldPointPicker()
        picker.Pick(self.mouse_position[0], self.mouse_position[1], 0, self.renderer)
        return picker.GetPickPosition()

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
            Callback function to call immediately after right
            clicking.

        """

        def _the_callback(*args):
            click_point = self.pick_mouse_position()
            self.fly_to(click_point)
            if callable(callback):
                try_callback(callback, click_point)

        self.track_click_position(callback=_the_callback, side="right")

    def disable_picking(self):
        """Disable any active picking.

        Examples
        --------
        Enable and then disable picking.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 0, 0))
        >>> cube = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
        >>> _ = pl.add_mesh(cube)
        >>> _ = pl.enable_mesh_picking(left_clicking=True)
        >>> pl.disable_picking()

        """
        if getattr(self, '_picker', None):
            self._picker.RemoveObservers(_vtk.vtkCommand.EndPickEvent)

        # remove left clicking observer if available
        if getattr(self, 'iren', None):
            self.iren.remove_observer(self._picking_left_clicking_observer)
        self._picking_left_clicking_observer = None

        # remove any picking text
        if hasattr(self, 'renderers'):
            self.remove_actor(self._picking_text, render=False)
        self._picking_text = None
