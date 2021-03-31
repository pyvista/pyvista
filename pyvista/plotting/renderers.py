"""Renderers"""
import collections

import numpy as np

from .theme import rcParams
from .renderer import Renderer

class Renderers():
    """Renderers."""

    def __init__(self, plotter, shape=(1, 1), splitting_position=None,
                 row_weights=None, col_weights=None, groups=None,
                 border=None, border_color='k', border_width=2.0):
        """Initialize renderers"""
        self._active_index = 0  # index of the active renderer
        self._plotter = plotter
        self._renderers = []

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
                splitting_position = rcParams['multi_rendering_splitting_position']

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

            # together with the row and col index of the bottom right cell)
            if groups is not None:
                if not isinstance(groups, collections.abc.Sequence):
                    raise TypeError('"groups" should be a list or tuple')
                for group in groups:
                    if not (isinstance(group, collections.abc.Sequence) and len(group) == 2):
                        raise ValueError('Each group entry should be a list or tuple of 2 elements')
                    rows = group[0]
                    if isinstance(rows, slice):
                        rows = np.arange(self._shape[0], dtype=int)[rows]
                    cols = group[1]
                    if isinstance(cols, slice):
                        cols = np.arange(self._shape[1], dtype=int)[cols]
                    # Get the normalized group, i.e. extract top left
                    # corner and bottom right corner from the given
                    # rows and cols
                    norm_group = [np.min(rows), np.min(cols), np.max(rows), np.max(cols)]

                    # Check for overlap with already defined groups:
                    for i in range(norm_group[0], norm_group[2]+1):
                        for j in range(norm_group[1], norm_group[3]+1):
                            if self.loc_to_group((i, j)) is not None:
                                raise ValueError('Groups cannot overlap')
                    self.groups = np.concatenate((self.groups, np.array([norm_group], dtype=int)), axis=0)

            # Create subplot renderers
            for row in range(shape[0]):
                for col in range(shape[1]):
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
                        renderer = Renderer(self._plotter, border, border_color,
                                            border_width)
                        x0 = col_off[col]
                        y0 = row_off[row+nb_rows]
                        x1 = col_off[col+nb_cols]
                        y1 = row_off[row]
                        renderer.SetViewport(x0, y0, x1, y1)
                        self._render_idxs[row, col] = len(self._renderers)
                        self._renderers.append(renderer)
                    else:
                        self._render_idxs[row, col] = self._render_idxs[self.groups[group, 0], self.groups[group, 1]]

    # def loc_to_group(self, loc):
    #     """Return group id of the given location index. Or None if this location is not part of any group."""
    #     group_idxs = np.arange(self.groups.shape[0])
    #     I = (loc[0]>=self.groups[:,0]) & (loc[0]<=self.groups[:,2]) & (loc[1]>=self.groups[:,1]) & (loc[1]<=self.groups[:,3])
    #     group = group_idxs[I]
    #     return None if group.size==0 else group[0]

    def loc_to_group(self, loc):
        """Return group id of the given location index, or ``None`` if this location is not part of any group."""
        group_idxs = np.arange(self.groups.shape[0])

        mask_a = (loc[0] >= self.groups[:, 0])
        mask_b = (loc[0] <= self.groups[:, 2])
        mask_c = (loc[1] >= self.groups[:, 1])
        mask_d = (loc[1] <= self.groups[:, 3])
        index = mask_a + mask_b + mask_c + mask_d

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
        if loc is None:
            return self._active_index
        elif isinstance(loc, (int, np.integer)):
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
        return self._renderers[index]

    def __len__(self):
        return len(self._renderers)

    def __iter__(self):
        for renderer in self._renderers:
            yield renderer

    @property
    def active_renderer_index(self):
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
        """Clean all renderers"""
        for renderer in self:
            renderer.deep_clean()
