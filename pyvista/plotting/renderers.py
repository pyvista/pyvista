"""Organize Renderers for ``pyvista.Plotter``."""
import collections

import numpy as np

import pyvista
from .background_renderer import BackgroundRenderer
from .renderer import Renderer


class Renderers():
    """Organize Renderers for ``pyvista.Plotter``."""

    def __init__(self, plotter, shape=(1, 1), splitting_position=None,
                 row_weights=None, col_weights=None, groups=None,
                 border=None, border_color='k', border_width=2.0):
        """Initialize renderers."""
        self._active_index = 0  # index of the active renderer
        self._plotter = plotter
        self._renderers = []

        # by default add border for multiple plots
        if border is None:
            if shape != (1, 1):
                border = True
            else:
                border = False

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
                rangen = range(n)
                rangem = range(m)

            if splitting_position is None:
                splitting_position = pyvista.global_theme.multi_rendering_splitting_position

            if splitting_position is None:
                if n >= m:
                    xsplit = m/(n+m)
                else:
                    xsplit = 1-n/(n+m)
            else:
                xsplit = splitting_position

            for i in rangen:
                arenderer = Renderer(self._plotter, border, border_color, border_width)
                if '|' in shape:
                    arenderer.SetViewport(0, i/n, xsplit, (i+1)/n)
                else:
                    arenderer.SetViewport(i/n, 0, (i+1)/n, xsplit)
                self._renderers.append(arenderer)
            for i in rangem:
                arenderer = Renderer(self._plotter, border, border_color, border_width)
                if '|' in shape:
                    arenderer.SetViewport(xsplit, i/m, 1, (i+1)/m)
                else:
                    arenderer.SetViewport(i/m, xsplit, (i+1)/m, 1)
                self._renderers.append(arenderer)

            self._shape = (n+m,)
            self._render_idxs = np.arange(n+m)

        else:
            if not isinstance(shape, (np.ndarray, collections.abc.Sequence)):
                raise TypeError('"shape" should be a list, tuple or string descriptor')
            if len(shape) != 2:
                raise ValueError('"shape" must have length 2.')
            shape = np.asarray(shape)
            if not np.issubdtype(shape.dtype, np.integer) or (shape <= 0).any():
                raise ValueError('"shape" must contain only positive integers.')
            # always assign shape as a tuple
            self._shape = tuple(shape)
            self._render_idxs = np.empty(self._shape, dtype=int)
            # Check if row and col weights correspond to given shape,
            # or initialize them to defaults (equally weighted)

            # and convert to normalized offsets
            if row_weights is None:
                row_weights = np.ones(shape[0])
            if col_weights is None:
                col_weights = np.ones(shape[1])
            assert(np.array(row_weights).size == shape[0])
            assert(np.array(col_weights).size == shape[1])
            row_off = np.cumsum(np.abs(row_weights))/np.sum(np.abs(row_weights))
            row_off = 1-np.concatenate(([0], row_off))
            col_off = np.cumsum(np.abs(col_weights))/np.sum(np.abs(col_weights))
            col_off = np.concatenate(([0], col_off))

            # Check and convert groups to internal format (Nx4 matrix
            # where every row contains the row and col index of the
            # top left cell

            if groups is not None:
                assert isinstance(groups, collections.abc.Sequence), '"groups" should be a list or tuple'
                for group in groups:
                    assert isinstance(group, collections.abc.Sequence) and len(group)==2, 'each group entry should be a list or tuple of 2 elements'
                    rows = group[0]
                    if isinstance(rows,slice):
                        rows = np.arange(self.shape[0],dtype=int)[rows]
                    cols = group[1]
                    if isinstance(cols,slice):
                        cols = np.arange(self.shape[1],dtype=int)[cols]
                    # Get the normalized group, i.e. extract top left corner and bottom right corner from the given rows and cols
                    norm_group = [np.min(rows),np.min(cols),np.max(rows),np.max(cols)]
                    # Check for overlap with already defined groups:
                    for i in range(norm_group[0],norm_group[2]+1):
                        for j in range(norm_group[1],norm_group[3]+1):
                            assert self.loc_to_group((i,j)) is None, 'groups cannot overlap'
                    self.groups = np.concatenate((self.groups,np.array([norm_group],dtype=int)),axis=0)
            # Create subplot renderers
            for row in range(shape[0]):
                for col in range(shape[1]):
                    group = self.loc_to_group((row,col))
                    nb_rows = None
                    nb_cols = None
                    if group is not None:
                        if row==self.groups[group,0] and col==self.groups[group,1]:
                            # Only add renderer for first location of the group
                            nb_rows = 1+self.groups[group,2]-self.groups[group,0]
                            nb_cols = 1+self.groups[group,3]-self.groups[group,1]
                    else:
                        nb_rows = 1
                        nb_cols = 1
                    if nb_rows is not None:
                        renderer = Renderer(self._plotter, border, border_color, border_width)
                        x0 = col_off[col]
                        y0 = row_off[row+nb_rows]
                        x1 = col_off[col+nb_cols]
                        y1 = row_off[row]
                        renderer.SetViewport(x0, y0, x1, y1)
                        self._render_idxs[row,col] = len(self)
                        self._renderers.append(renderer)
                    else:
                        self._render_idxs[row,col] = self._render_idxs[self.groups[group,0],self.groups[group,1]]

        # each render will also have an associated background renderer
        self._background_renderers = [None for _ in range(len(self))]

        # create a shadow renderer that lives on top of all others
        self._shadow_renderer = Renderer(self._plotter, border, border_color,
                                         border_width)
        self._shadow_renderer.SetViewport(0, 0, 1, 1)
        self._shadow_renderer.SetDraw(False)

    def loc_to_group(self, loc):
        """Return group id of the given location index or ``None`` if this location is not part of any group."""
        group_idxs = np.arange(self.groups.shape[0])
        index = (loc[0] >= self.groups[:, 0]) & \
                (loc[0] <= self.groups[:, 2]) & \
                (loc[1] >= self.groups[:, 1]) & \
                (loc[1] <= self.groups[:, 3])
        group = group_idxs[index]
        return None if group.size == 0 else group[0]

    def loc_to_index(self, loc):
        """Return index of the render window given a location index.

        Parameters
        ----------
        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example,
            ``loc=2`` or ``loc=(1, 1)``.

        Returns
        -------
        idx : int
            Index of the render window.

        """
        if isinstance(loc, (int, np.integer)):
            return loc
        elif isinstance(loc, (np.ndarray, collections.abc.Sequence)):
            if not len(loc) == 2:
                raise ValueError('"loc" must contain two items')
            index_row = loc[0]
            index_column = loc[1]
            if index_row < 0 or index_row >= self.shape[0]:
                raise IndexError(f'Row index is out of range ({self.shape[0]})')
            if index_column < 0 or index_column >= self.shape[1]:
                raise IndexError(f'Column index is out of range ({self.shape[1]})')
            return self._render_idxs[index_row, index_column]
        else:
            raise TypeError('"loc" must be an integer or a sequence.')

    def __getitem__(self, index):
        """Return a renderer based on an index."""
        return self._renderers[index]

    def __len__(self):
        """Return number of renderers."""
        return len(self._renderers)

    def __iter__(self):
        """Return a iterable of renderers."""
        for renderer in self._renderers:
            yield renderer

    @property
    def active_index(self):
        """Return the active index."""
        return self._active_index

    def index_to_loc(self, index):
        """Convert a 1D index location to the 2D location on the plotting grid."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError('"index" must be a scalar integer.')
        if len(self.shape) == 1:
            return index
        args = np.argwhere(self._render_idxs == index)
        if len(args) < 1:
            raise IndexError(f'Index ({index}) is out of range.')
        return args[0]

    @property
    def active_renderer(self):
        """Return the active renderer."""
        return self._renderers[self._active_index]

    @property
    def shape(self):
        """Return the shape of the renderers."""
        return self._shape

    def set_active_renderer(self, index_row, index_column=None):
        """Set the index of the active renderer.

        Parameters
        ----------
        index_row : int
            Index of the subplot to activate along the rows.

        index_column : int
            Index of the subplot to activate along the columns.

        """
        if len(self.shape) == 1:
            self._active_index = index_row
            return

        if index_row < 0 or index_row >= self.shape[0]:
            raise IndexError(f'Row index is out of range ({self.shape[0]})')
        if index_column < 0 or index_column >= self.shape[1]:
            raise IndexError(f'Column index is out of range ({self.shape[1]})')
        self._active_index = self.loc_to_index((index_row, index_column))

    def deep_clean(self):
        """Clean all renderers."""
        # Do not remove the renderers on the clean
        for renderer in self:
            renderer.deep_clean()
        if hasattr(self, '_shadow_renderer'):
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

        scale : float, optional
            Scale the image larger or smaller relative to the size of
            the window.  For example, a scale size of 2 will make the
            largest dimension of the image twice as large as the
            largest dimension of the render window.  Defaults to 1.

        as_global : bool, optional
            When multiple render windows are present, setting
            ``as_global=False`` will cause the background to only
            appear in one window.

        Returns
        pyvista.BackgroundRenderer
            Newly created background renderer.

        """
        # verify no render exists
        if as_global:
            for renderer in self:
                renderer.SetLayer(2)
            view_port = None
        else:
            self.active_renderer.SetLayer(2)
            view_port = self.active_renderer.GetViewport()

        renderer = BackgroundRenderer(self._plotter, image_path, scale, view_port)
        renderer.SetLayer(1)
        self._background_renderers[self.active_index] = renderer
        return renderer

    @property
    def has_active_background_renderer(self):
        """Return ``True`` when Renderer has an active background renderer."""
        return self._background_renderers[self.active_index] is not None

    def clear_background_renderers(self):
        """Clear all background renderers."""
        for renderer in self._background_renderers:
            if renderer is not None:
                renderer.clear()

    def clear(self):
        """Clear all renders."""
        for renderer in self:
            renderer.clear()
        self._shadow_renderer.clear()
        self.clear_background_renderers()

    def close(self):
        """Close all renderers."""
        for renderer in self:
            renderer.close()

        self._shadow_renderer.close()

        for renderer in self._background_renderers:
            if renderer is not None:
                renderer.close()

    def remove_all_lights(self):
        """Remove all lights from all renderers."""
        for renderer in self:
            renderer.remove_all_lights()

    @property
    def shadow_renderer(self):
        """Shadow renderer."""
        return self._shadow_renderer

    def set_background(self, color, top=None, all_renderers=True):
        """Set the background color.

        Parameters
        ----------
        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        top : string or 3 item list, optional, defaults to None
            If given, this will enable a gradient background where the
            ``color`` argument is at the bottom and the color given in ``top``
            will be the color at the top of the renderer.

        all_renderers : bool
            If True, applies to all renderers in subplots. If False, then
            only applies to the active renderer.

        Examples
        --------
        Set the background color to black.

        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> plotter.set_background('black')
        >>> plotter.background_color
        (0.0, 0.0, 0.0)

        Set the background color to white.

        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> plotter.set_background('white')
        >>> plotter.background_color
        (1.0, 1.0, 1.0)

        """
        if all_renderers:
            for renderer in self:
                renderer.set_background(color, top=top)
            self._shadow_renderer.set_background(color)
        else:
            self.active_renderer.set_background(color, top=top)

    def remove_background_image(self):
        """Remove the background image at the current renderer."""
        renderer = self._background_renderers[self.active_index]
        if renderer is None:
            raise RuntimeError('No background image to remove at this subplot')
        renderer.deep_clean()
        self._background_renderers[self.active_index] = None

    def __del__(self):
        """Destructor."""
        if hasattr(self, '_shadow_renderer'):
            del self._shadow_renderer
