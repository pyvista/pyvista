"""Organize Renderers for ``pyvista.Plotter``."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from weakref import proxy

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .background_renderer import BackgroundRenderer
from .renderer import Renderer


class Renderers(_NoNewAttrMixin):
    """Organize Renderers for ``pyvista.Plotter``.

    Parameters
    ----------
    plotter : str
        The PyVista plotter.

    shape : tuple[int], optional
        The initial shape of the PyVista plotter, (rows, columns).

    splitting_position : float, optional
        The position to place the splitting line between plots.

    row_weights : sequence, optional
        The weights of the rows when the plot window is resized.

    col_weights : sequence, optional
        The weights of the columns when the plot window is resized.

    groups : list, optional
        A list of sequences that defines the grouping of the sub-datasets.

    border : bool, optional
        Whether or not a border should be added around each subplot.

    border_color : str, optional
        The color of the border around each subplot.

    border_width : float, optional
        The width of the border around each subplot.

    """

    @_deprecate_positional_args(allowed=['plotter'])
    def __init__(  # noqa: PLR0917
        self,
        plotter,
        shape=(1, 1),
        splitting_position=None,
        row_weights=None,
        col_weights=None,
        groups=None,
        border=None,
        border_color='k',
        border_width=2.0,
    ):
        """Initialize renderers."""
        self._active_index = 0  # index of the active renderer
        self._plotter = proxy(plotter)
        self._renderers = []
        self._shadow_renderer = None

        # by default add border for multiple plots
        if border is None:
            border = shape != (1, 1)

        self.groups = np.empty((0, 4), dtype=int)

        if isinstance(shape, str):
            if '|' in shape:
                n = int(shape.split('|')[0])
                m = int(shape.split('|')[1])
                rangen = reversed(range(n))
                rangem = reversed(range(m))
            else:
                m = int(shape.split('/')[0])
                n = int(shape.split('/')[1])
                rangen = range(n)  # type: ignore[assignment]
                rangem = range(m)  # type: ignore[assignment]

            if splitting_position is None:
                splitting_position = pv.global_theme.multi_rendering_splitting_position

            if splitting_position is None:
                xsplit = m / (n + m) if n >= m else 1 - n / (n + m)
            else:
                xsplit = splitting_position

            for i in rangen:
                arenderer = Renderer(
                    self._plotter,
                    border=border,
                    border_color=border_color,
                    border_width=border_width,
                )
                if '|' in shape:
                    arenderer.viewport = (0, i / n, xsplit, (i + 1) / n)
                else:
                    arenderer.viewport = (i / n, 0, (i + 1) / n, xsplit)
                self._renderers.append(arenderer)
            for i in rangem:
                arenderer = Renderer(
                    self._plotter,
                    border=border,
                    border_color=border_color,
                    border_width=border_width,
                )
                if '|' in shape:
                    arenderer.viewport = (xsplit, i / m, 1, (i + 1) / m)
                else:
                    arenderer.viewport = (i / m, xsplit, (i + 1) / m, 1)
                self._renderers.append(arenderer)

            self._shape = (n + m,)
            self._render_idxs = np.arange(n + m)

        else:
            if not isinstance(shape, (np.ndarray, Sequence)):
                msg = '"shape" should be a list, tuple or string descriptor'
                raise TypeError(msg)
            if len(shape) != 2:
                msg = '"shape" must have length 2.'
                raise ValueError(msg)
            shape = np.asarray(shape)
            if not np.issubdtype(shape.dtype, np.integer) or (shape <= 0).any():
                msg = '"shape" must contain only positive integers.'
                raise ValueError(msg)
            # always assign shape as a tuple of native ints
            self._shape = tuple(size.item() for size in shape)
            self._render_idxs = np.empty(self._shape, dtype=int)
            # Check if row and col weights correspond to given shape,
            # or initialize them to defaults (equally weighted).

            # and convert to normalized offsets
            if row_weights is None:
                row_weights = np.ones(shape[0])
            if col_weights is None:
                col_weights = np.ones(shape[1])

            # also make flattening and abs explicit
            row_weights = np.abs(np.asanyarray(row_weights).ravel())
            col_weights = np.abs(np.asanyarray(col_weights).ravel())
            if row_weights.size != shape[0]:
                msg = (
                    f'"row_weights" must have {shape[0]} items '
                    f'for {shape[0]} rows of subplots, not '
                    f'{row_weights.size}.'
                )
                raise ValueError(msg)
            if col_weights.size != shape[1]:
                msg = (
                    f'"col_weights" must have {shape[1]} items '
                    f'for {shape[1]} columns of subplots, not '
                    f'{col_weights.size}.'
                )
                raise ValueError(msg)
            row_off = np.cumsum(row_weights) / np.sum(row_weights)
            row_off = 1 - np.concatenate(([0], row_off))
            col_off = np.cumsum(col_weights) / np.sum(col_weights)
            col_off = np.concatenate(([0], col_off))

            # Check and convert groups to internal format (Nx4 matrix
            # where every row contains the row and col index of the
            # top left cell)

            if groups is not None:
                if not isinstance(groups, Sequence):
                    msg = f'"groups" should be a list or tuple, not {type(groups).__name__}.'
                    raise TypeError(msg)
                for group in groups:
                    if not isinstance(group, Sequence):
                        msg = (
                            'Each group entry should be a list or '
                            f'tuple, not {type(group).__name__}.'
                        )
                        raise TypeError(msg)
                    if len(group) != 2:
                        msg = 'Each group entry must have length 2.'
                        raise ValueError(msg)

                    rows = group[0]
                    if isinstance(rows, slice):
                        rows = np.arange(self.shape[0], dtype=int)[rows]
                    cols = group[1]
                    if isinstance(cols, slice):
                        cols = np.arange(self.shape[1], dtype=int)[cols]  # type: ignore[misc]
                    # Get the normalized group, i.e. extract top left corner
                    # and bottom right corner from the given rows and cols
                    norm_group = [np.min(rows), np.min(cols), np.max(rows), np.max(cols)]
                    # Check for overlap with already defined groups:
                    for i, j in product(
                        range(norm_group[0], norm_group[2] + 1),
                        range(norm_group[1], norm_group[3] + 1),
                    ):
                        if self.loc_to_group((i, j)) is not None:
                            msg = f'Groups cannot overlap. Overlap found at position {(i, j)}.'
                            raise ValueError(msg)
                    self.groups = np.concatenate(
                        (self.groups, np.array([norm_group], dtype=int)),
                        axis=0,
                    )
            # Create subplot renderers
            for row, col in product(range(shape[0]), range(shape[1])):
                group = self.loc_to_group((row, col))
                nb_rows = None
                nb_cols = None
                if group is not None:
                    if row == self.groups[group, 0] and col == self.groups[group, 1]:
                        # Only add renderer for first location of the group
                        nb_rows = 1 + self.groups[group, 2] - self.groups[group, 0]
                        nb_cols = 1 + self.groups[group, 3] - self.groups[group, 1]
                else:
                    nb_rows = 1
                    nb_cols = 1
                if nb_rows is not None:
                    renderer = Renderer(
                        self._plotter,
                        border=border,
                        border_color=border_color,
                        border_width=border_width,
                    )
                    x0 = col_off[col]
                    y0 = row_off[row + nb_rows]
                    x1 = col_off[col + nb_cols]  # type: ignore[operator]
                    y1 = row_off[row]
                    renderer.viewport = (x0, y0, x1, y1)
                    self._render_idxs[row, col] = len(self)
                    self._renderers.append(renderer)
                else:
                    self._render_idxs[row, col] = self._render_idxs[
                        self.groups[group, 0],
                        self.groups[group, 1],
                    ]

        # each render will also have an associated background renderer
        self._background_renderers: list[None | BackgroundRenderer] = [
            None for _ in range(len(self))
        ]

        # create a shadow renderer that lives on top of all others
        self._shadow_renderer = Renderer(
            self._plotter, border=border, border_color=border_color, border_width=border_width
        )
        self._shadow_renderer.viewport = (0, 0, 1, 1)
        self._shadow_renderer.SetDraw(False)

    def loc_to_group(self, loc):
        """Return index of the render window given a location index.

        Parameters
        ----------
        loc : int | sequence[int]
            Index of the renderer to add the actor to.  For example, ``loc=2``
            or ``loc=(1, 1)``.

        Returns
        -------
        int
            Index of the render window.

        """
        group_idxs = np.arange(self.groups.shape[0])
        index = (
            (loc[0] >= self.groups[:, 0])
            & (loc[0] <= self.groups[:, 2])
            & (loc[1] >= self.groups[:, 1])
            & (loc[1] <= self.groups[:, 3])
        )
        group = group_idxs[index]
        return None if group.size == 0 else group[0]

    def loc_to_index(self, loc):
        """Return index of the render window given a location index.

        Parameters
        ----------
        loc : int | sequence[int]
            Index of the renderer to add the actor to. For example, ``loc=2``
            or ``loc=(1, 1)``.

        Returns
        -------
        int
            Index of the render window.

        """
        if isinstance(loc, (int, np.integer)):
            return loc
        elif isinstance(loc, (np.ndarray, Sequence)):
            if len(loc) != 2:
                msg = '"loc" must contain two items'
                raise ValueError(msg)
            index_row = loc[0]
            index_column = loc[1]
            if index_row < 0 or index_row >= self.shape[0]:
                msg = f'Row index is out of range ({self.shape[0]})'
                raise IndexError(msg)
            if index_column < 0 or index_column >= self.shape[1]:  # type: ignore[misc]
                msg = f'Column index is out of range ({self.shape[1]})'  # type: ignore[misc]
                raise IndexError(msg)
            return self._render_idxs[index_row, index_column]
        else:
            msg = '"loc" must be an integer or a sequence.'
            raise TypeError(msg)

    def __getitem__(self, index):
        """Return a renderer based on an index."""
        return self._renderers[index]

    def __len__(self):
        """Return number of renderers."""
        return len(self._renderers)

    def __iter__(self):
        """Return a iterable of renderers."""
        yield from self._renderers

    @property
    def active_index(self):  # numpydoc ignore=RT01
        """Return the active index.

        Returns
        -------
        int
            Active index.

        """
        return self._active_index

    def index_to_loc(self, index):
        """Convert a 1D index location to the 2D location on the plotting grid.

        Parameters
        ----------
        index : int
            A scalar integer that refers to the 1D location index.

        Returns
        -------
        numpy.ndarray or numpy.int64
            2D location on the plotting grid.

        """
        if not isinstance(index, (int, np.integer)):
            msg = '"index" must be a scalar integer.'
            raise TypeError(msg)
        if len(self.shape) == 1:
            return np.intp(index)
        args = np.argwhere(self._render_idxs == index)
        if len(args) < 1:
            msg = f'Index ({index}) is out of range.'
            raise IndexError(msg)
        return args[0]

    @property
    def active_renderer(self):  # numpydoc ignore=RT01
        """Return the active renderer.

        Returns
        -------
        Renderer
            Active renderer.

        """
        return self._renderers[self._active_index]

    @property
    def shape(self) -> tuple[int] | tuple[int, int]:
        """Return the shape of the renderers.

        Returns
        -------
        tuple[int] | tuple[int, int]
            Shape of the renderers.

        """
        return self._shape

    def set_active_renderer(self, index_row, index_column=None):
        """Set the index of the active renderer.

        Parameters
        ----------
        index_row : int
            Index of the subplot to activate along the rows.

        index_column : int, optional
            Index of the subplot to activate along the columns.

        """
        if len(self.shape) == 1:
            self._active_index = index_row
            return

        if index_row < 0 or index_row >= self.shape[0]:
            msg = f'Row index is out of range ({self.shape[0]})'
            raise IndexError(msg)
        if index_column < 0 or index_column >= self.shape[1]:
            msg = f'Column index is out of range ({self.shape[1]})'
            raise IndexError(msg)
        self._active_index = self.loc_to_index((index_row, index_column))

    @_deprecate_positional_args(allowed=['interactive'])
    def set_chart_interaction(self, interactive, toggle: bool = False):  # noqa: FBT001, FBT002
        """Set or toggle interaction with charts for the active renderer.

        Interaction with other charts in other renderers is disabled.
        Interaction with other charts in the active renderer is only disabled
        when ``toggle`` is ``False``.

        Parameters
        ----------
        interactive : bool | Chart | int | sequence[Chart] | sequence[int]
            Following parameter values are accepted:

            * A boolean to enable (``True``) or disable (``False``) interaction
              with all charts in the active renderer.
            * The chart or its index to enable interaction with. Interaction
              with multiple charts can be enabled by passing a list of charts
              or indices.

        toggle : bool, default: False
            Instead of enabling interaction with the provided chart(s), interaction
            with the provided chart(s) is toggled. Only applicable when ``interactive``
            is not a boolean.

        Returns
        -------
        list[Chart]
            The list of all interactive charts for the active renderer.

        """
        interactive_scene, interactive_charts = None, []
        if self.active_renderer.has_charts:
            interactive_scene = self.active_renderer._charts._scene
            interactive_charts = self.active_renderer.set_chart_interaction(
                interactive, toggle=toggle
            )
        # Disable chart interaction for other renderers
        for renderer in self:
            if renderer is not self.active_renderer:
                renderer.set_chart_interaction(False)
        # Setup the context interactor style based on the resulting amount of interactive charts.
        self._plotter.iren._set_context_style(interactive_scene if interactive_charts else None)
        return interactive_charts

    def on_plotter_render(self):
        """Notify all renderers of explicit plotter render call."""
        for renderer in self:
            renderer.on_plotter_render()

    def deep_clean(self):
        """Clean all renderers."""
        # Do not remove the renderers on the clean
        for renderer in self:
            renderer.deep_clean()
        if self._shadow_renderer is not None:
            self._shadow_renderer.deep_clean()
        if hasattr(self, '_background_renderers'):
            for renderer in self._background_renderers:
                if renderer is not None:
                    renderer.deep_clean()

    def add_background_renderer(self, image_path, scale, as_global):
        """Add a background image to the renderers.

        Parameters
        ----------
        image_path : str
            Path to an image file.

        scale : float
            Scale the image larger or smaller relative to the size of
            the window.  For example, a scale size of 2 will make the
            largest dimension of the image twice as large as the
            largest dimension of the render window.  Defaults to 1.

        as_global : bool
            When multiple render windows are present, setting
            ``as_global=False`` will cause the background to only
            appear in one window.

        Returns
        -------
        pyvista.BackgroundRenderer
            Newly created background renderer.

        """
        # verify no render exists
        if as_global:
            for renderer in self:
                renderer.layer = 2
            view_port = None
        else:
            self.active_renderer.layer = 2
            view_port = self.active_renderer.GetViewport()

        renderer = BackgroundRenderer(self._plotter, image_path, scale=scale, view_port=view_port)
        renderer.layer = 1
        self._background_renderers[self.active_index] = renderer
        return renderer

    @property
    def has_active_background_renderer(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` when Renderer has an active background renderer.

        Returns
        -------
        bool
            Whether or not the active renderer has a background renderer.

        """
        return self._background_renderers[self.active_index] is not None

    def clear_background_renderers(self):
        """Clear all background renderers."""
        for renderer in self._background_renderers:
            if renderer is not None:
                renderer.clear()

    def clear_actors(self):
        """Clear actors from all renderers."""
        for renderer in self:
            renderer.clear_actors()

    def clear(self):
        """Clear all renders."""
        for renderer in self:
            renderer.clear()
        self._shadow_renderer.clear()  # type: ignore[union-attr]
        self.clear_background_renderers()

    def close(self):
        """Close all renderers."""
        for renderer in self:
            renderer.close()

        self._shadow_renderer.close()  # type: ignore[union-attr]

        for renderer in self._background_renderers:
            if renderer is not None:
                renderer.close()

    def remove_all_lights(self):
        """Remove all lights from all renderers."""
        for renderer in self:
            renderer.remove_all_lights()

    @property
    def shadow_renderer(self):  # numpydoc ignore=RT01
        """Shadow renderer.

        Returns
        -------
        pyvista.plotting.renderer.Renderer
            Shadow renderer.

        """
        return self._shadow_renderer

    @_deprecate_positional_args(allowed=['color'])
    def set_background(  # noqa: PLR0917
        self,
        color,
        top=None,
        right=None,
        side=None,
        corner=None,
        all_renderers: bool = True,  # noqa: FBT001, FBT002
    ):
        """Set the background color.

        Parameters
        ----------
        color : ColorLike, optional
            Either a string, rgb list, or hex color string.  Defaults
            to current theme parameters.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        top : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the bottom and the color given in ``top``
            will be the color at the top of the renderer.

        right : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the left and the color given in ``right``
            will be the color at the right of the renderer.

        side : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the center and the color given in ``side``
            will be the color at the side of the renderer.

        corner : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the center and the color given in ``corner``
            will be the color at the corner of the renderer.

        all_renderers : bool, default: True
            If ``True``, applies to all renderers in subplots. If ``False``,
            then only applies to the active renderer.

        Examples
        --------
        Set the background color to black.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.set_background('black')
        >>> pl.background_color
        Color(name='black', hex='#000000ff', opacity=255)
        >>> pl.close()

        Set the background color at the bottom to black and white at
        the top.  Display a cone as well.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Cone())
        >>> pl.set_background('black', top='white')
        >>> pl.show()

        """
        if all_renderers:
            for renderer in self:
                renderer.set_background(color, top=top, right=right, side=side, corner=corner)
            self._shadow_renderer.set_background(color)  # type: ignore[union-attr]
        else:
            self.active_renderer.set_background(
                color,
                top=top,
                right=right,
                side=side,
                corner=corner,
            )

    @_deprecate_positional_args(allowed=['color_cycler'])
    def set_color_cycler(self, color_cycler, all_renderers: bool = True):  # noqa: FBT001, FBT002
        """Set or reset the color cycler.

        This color cycler is iterated over by each sequential :class:`~pyvista.Plotter.add_mesh`
        call to set the default color of the dataset being plotted.

        When setting, the value must be either a list of color-like objects,
        or a cycler of color-like objects. If the value passed is a single
        string, it must be one of:

            * ``'default'`` - Use the default color cycler (matches matplotlib's default)
            * ``'matplotlib`` - Dynamically get matplotlib's current theme's color cycler.
            * ``'all'`` - Cycle through all available colors in
              ``pyvista.plotting.colors.hexcolors``

        Setting to ``None`` will disable the use of the color cycler on this
        renderer.

        .. note::
            If a mesh has scalar data, set ``color=True`` in the call to :meth:`add_mesh`
            to color the mesh with the next color in the cycler. Otherwise the mesh's
            scalars are used to color the mesh by default.

        Parameters
        ----------
        color_cycler : str | cycler.Cycler | sequence[ColorLike]
            The colors to cycle through.

        all_renderers : bool, default: True
            If ``True``, applies to all renderers in subplots. If ``False``,
            then only applies to the active renderer.

        See Also
        --------
        :ref:`color_cycler_example`

        Examples
        --------
        Set the default color cycler to iterate through red, green, and blue.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.set_color_cycler(['red', 'green', 'blue'])
        >>> _ = pl.add_mesh(pv.Cone(center=(0, 0, 0)))  # red
        >>> _ = pl.add_mesh(pv.Cube(center=(1, 0, 0)))  # green
        >>> _ = pl.add_mesh(pv.Sphere(center=(1, 1, 0)))  # blue
        >>> _ = pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))  # red again
        >>> pl.show()

        """
        if all_renderers:
            for renderer in self:
                renderer.set_color_cycler(color_cycler)
        else:
            self.active_renderer.set_color_cycler(color_cycler)

    def remove_background_image(self):
        """Remove the background image at the current renderer.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> pl.add_background_image(examples.mapfile, as_global=False)
        >>> pl.subplot(0, 1)
        >>> actor = pl.add_mesh(pv.Cube())
        >>> pl.add_background_image(examples.mapfile, as_global=False)
        >>> pl.remove_background_image()
        >>> pl.show()

        """
        renderer = self._background_renderers[self.active_index]
        if renderer is None:
            msg = 'No background image to remove at this subplot'
            raise RuntimeError(msg)
        renderer.deep_clean()
        self._background_renderers[self.active_index] = None

    def __del__(self):
        """Destructor."""
        self._shadow_renderer = None
