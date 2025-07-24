"""Module containing composite data mapper."""

from __future__ import annotations

from itertools import cycle
import sys
from typing import TYPE_CHECKING
import weakref

import numpy as np

import pyvista
from pyvista import vtk_version_info
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.arrays import convert_array
from pyvista.core.utilities.arrays import convert_string_array
from pyvista.core.utilities.misc import _check_range
from pyvista.core.utilities.misc import _NoNewAttrMixin

from . import _vtk
from .colors import Color
from .colors import get_cycler
from .mapper import _BaseMapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cycler

    from ._typing import ColorLike


class BlockAttributes(_NoNewAttrMixin):
    """Block attributes used to set the attributes of a block.

    Parameters
    ----------
    block : pyvista.DataObject
        PyVista data object.

    attr : pyvista.plotting.composite_mapper.CompositeAttributes
        Parent attributes.

    Notes
    -----
    This class employs VTK's flat indexing and allows for accessing both
    the blocks of a composite dataset as well as the entire composite
    dataset. If there is only one composite dataset, ``A``, which contains
    datasets ``[b, c]``, the indexing would be ``[A, b, c]``.

    If there are two composite datasets ``[B, C]`` in one composite
    dataset, ``A``, each of which containing three additional datasets
    ``[d, e, f]``, and ``[g, h, i]``, respectively, then the head node,
    ``A``, would be the zero index, followed by the first child, ``B``,
    followed by all the children of ``B``, ``[d, e, f]``. In data
    structures, this flat indexing would be known as "Depth-first search"
    and the entire indexing would be::

       [A, B, d, e, f, C, g, h, i]

    Note how the composite datasets themselves are capitalized and are
    accessible in the flat indexing, and not just the datasets.

    Examples
    --------
    Add a sphere and a cube as a multiblock dataset to a plotter and then
    change the visibility and color of the blocks. Note how the index of the
    cube is ``1`` as the index of the entire multiblock is ``0``.

    >>> import pyvista as pv
    >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
    >>> pl = pv.Plotter()
    >>> actor, mapper = pl.add_composite(dataset)
    >>> mapper.block_attr[1].color = 'b'
    >>> mapper.block_attr[1].opacity = 0.1
    >>> mapper.block_attr[1]
    Composite Block Addr=... Attributes
    Visible:   None
    Opacity:   0.1
    Color:     Color(name='blue', hex='#0000ffff', opacity=255)
    Pickable   None

    >>> pl.show()

    """

    def __init__(self, block, attr):
        """Initialize the block attributes class."""
        self._block = block
        self.__attr = weakref.ref(attr)

    @property
    def _attr(self):
        """Return the CompositeAttributes."""
        return self.__attr()

    @property
    def _has_color(self):
        """Return if a block has its color set."""
        return self._attr.HasBlockColor(self._block)

    @property
    def _has_visibility(self):
        """Return if a block has its visibility set."""
        return self._attr.HasBlockVisibility(self._block)

    @property
    def _has_opacity(self):
        """Return if a block has its opacity set."""
        return self._attr.HasBlockOpacity(self._block)

    @property
    def _has_pickable(self):
        """Return if a block has its pickability set."""
        return self._attr.HasBlockPickability(self._block)

    @property
    def color(self):  # numpydoc ignore=RT01
        """Get or set the color of a block.

        Examples
        --------
        Set the colors of a composite dataset to red and blue.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].color = 'r'
        >>> mapper.block_attr[2].color = 'b'
        >>> pl.show()

        """
        if not self._has_color:
            return None
        return Color(tuple(self._attr.GetBlockColor(self._block)))

    @color.setter
    def color(self, new_color):
        if new_color is None:
            self._attr.RemoveBlockColor(self._block)
            self._attr.Modified()
            return
        self._attr.SetBlockColor(self._block, Color(new_color).float_rgb)

    @property
    def visible(self) -> bool | None:  # numpydoc ignore=RT01
        """Get or set the visibility of a block.

        Examples
        --------
        Hide the first block of a composite dataset.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].visible = False
        >>> pl.show()

        """
        if not self._has_visibility:
            return None
        return self._attr.GetBlockVisibility(self._block)

    @visible.setter
    def visible(self, new_visible: bool | None):
        if new_visible is None:
            self._attr.RemoveBlockVisibility(self._block)
            self._attr.Modified()
            return
        self._attr.SetBlockVisibility(self._block, new_visible)

    @property
    def opacity(self) -> float | None:  # numpydoc ignore=RT01
        """Get or set the opacity of a block.

        If opacity has not been set this will be ``None``.

        Warnings
        --------
        VTK 9.0.3 has a bug where changing the opacity to less than 1.0 also
        changes the edge visibility on the block that is partially transparent.

        Examples
        --------
        Change the opacity of the second block of the dataset.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[2].opacity = 0.5
        >>> pl.show()

        """
        if not self._has_opacity:
            return None
        return self._attr.GetBlockOpacity(self._block)

    @opacity.setter
    def opacity(self, new_opacity: float | None):
        if new_opacity is None:
            self._attr.RemoveBlockOpacity(self._block)
            self._attr.Modified()
            return

        _check_range(new_opacity, (0, 1), 'opacity')
        self._attr.SetBlockOpacity(self._block, new_opacity)

    @property
    def pickable(self) -> bool | None:  # numpydoc ignore=RT01
        """Get or set the pickability of a block.

        Examples
        --------
        Make the cube of a multiblock dataset pickable and the sphere unpickable.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].pickable = True
        >>> mapper.block_attr[2].pickable = False
        >>> pl.close()

        See :ref:`composite_picking_example` for a full example using block
        picking.

        """
        if not self._has_pickable:
            return None
        return self._attr.GetBlockPickability(self._block)

    @pickable.setter
    def pickable(self, new_pickable: bool | None):
        if new_pickable is None:
            self._attr.RemoveBlockPickability(self._block)
            self._attr.Modified()
            return
        self._attr.SetBlockPickability(self._block, new_pickable)

    def __repr__(self):
        """Representation of block properties."""
        return '\n'.join(
            [
                f'Composite Block {self._block.memory_address} Attributes',
                f'Visible:   {self.visible}',
                f'Opacity:   {self.opacity}',
                f'Color:     {self.color}',
                f'Pickable   {self.pickable}',
            ],
        )


class CompositeAttributes(
    _NoNewAttrMixin,
    _vtk.DisableVtkSnakeCase,
    _vtk.vtkCompositeDataDisplayAttributes,
):
    """Block attributes.

    Parameters
    ----------
    mapper : pyvista.plotting.composite_mapper.CompositePolyDataMapper
        Parent mapper.

    dataset : pyvista.MultiBlock
        Multiblock dataset.

    Notes
    -----
    This class employs VTK's flat indexing and allows for accessing both
    the blocks of a composite dataset as well as the entire composite
    dataset. If there is only one composite dataset, ``A``, which contains
    datasets ``[b, c]``, the indexing would be ``[A, b, c]``.

    If there are two composite datasets ``[B, C]`` in one composite
    dataset, ``A``, each of which containing three additional datasets
    ``[d, e, f]``, and ``[g, h, i]``, respectively, then the head node,
    ``A``, would be the zero index, followed by the first child, ``B``,
    followed by all the children of ``B``, ``[d, e, f]``. In data
    structures, this flat indexing would be known as "Depth-first search"
    and the entire indexing would be::

       [A, B, d, e, f, C, g, h, i]

    Note how the composite datasets themselves are capitalized and are
    accessible in the flat indexing, and not just the datasets.

    Examples
    --------
    Add a sphere and a cube as a multiblock dataset to a plotter and then
    change the visibility and color of the blocks. Note how the index of the
    cube is ``1`` as the index of the entire multiblock is ``0``.

    >>> import pyvista as pv
    >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
    >>> pl = pv.Plotter()
    >>> actor, mapper = pl.add_composite(dataset)
    >>> mapper.block_attr[1].color = 'b'
    >>> mapper.block_attr[1].opacity = 0.1
    >>> mapper.block_attr[1]
    Composite Block Addr=... Attributes
    Visible:   None
    Opacity:   0.1
    Color:     Color(name='blue', hex='#0000ffff', opacity=255)
    Pickable   None

    >>> pl.show()

    """

    def __init__(self, mapper, dataset):
        """Initialize CompositeAttributes."""
        super().__init__()
        mapper.SetCompositeDataDisplayAttributes(self)
        self._dataset = dataset

    def reset_visibilities(self):
        """Reset the visibility of all blocks.

        Examples
        --------
        Hide the first block of a composite dataset and then show all by
        resetting visibilities.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].visible = False
        >>> mapper.block_attr.reset_visibilities()
        >>> pl.show()

        """
        self.RemoveBlockVisibilities()

    def reset_pickabilities(self):
        """Reset the pickability of all blocks.

        Examples
        --------
        Make the cube of a multiblock dataset pickable and the sphere
        unpickable, then reset it.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].pickable = True
        >>> mapper.block_attr[2].pickable = False
        >>> mapper.block_attr.reset_pickabilities()
        >>> [
        ...     mapper.block_attr[1].pickable,
        ...     mapper.block_attr[2].pickable,
        ... ]
        [None, None]
        >>> pl.close()

        """
        self.RemoveBlockPickabilities()

    def reset_colors(self):
        """Reset the color of all blocks.

        Examples
        --------
        Set individual block colors and then reset them.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset, color='w')
        >>> mapper.block_attr[1].color = 'r'
        >>> mapper.block_attr[2].color = 'b'
        >>> mapper.block_attr.reset_colors()
        >>> pl.show()

        """
        self.RemoveBlockColors()

    def reset_opacities(self):
        """Reset the opacities of all blocks.

        Examples
        --------
        Change the opacity of the second block of the dataset then reset all
        opacities.

        Note how the zero index is the entire multiblock, so we have to add 1
        to our indexing to access the right block.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[2].opacity = 0.5
        >>> mapper.block_attr.reset_opacities()
        >>> pl.show()

        """
        self.RemoveBlockOpacities()

    def get_block(self, index):
        """Return a block by its flat index.

        Parameters
        ----------
        index : int
            Flat index of the block to retrieve.

        Returns
        -------
        pyvista.DataObject
            PyVista data object.

        Notes
        -----
        This method employs VTK's flat indexing and allows for accessing both
        the blocks of a composite dataset as well as the entire composite
        dataset. If there is only one composite dataset, ``A``, which contains
        datasets ``[b, c]``, the indexing would be ``[A, b, c]``.

        If there are two composite datasets ``[B, C]`` in one composite
        dataset, ``A``, each of which containing three additional datasets
        ``[d, e, f]``, and ``[g, h, i]``, respectively, then the head node,
        ``A``, would be the zero index, followed by the first child, ``B``,
        followed by all the children of ``B``, ``[d, e, f]``. In data
        structures, this flat indexing would be known as "Depth-first search"
        and the entire indexing would be::

           [A, B, d, e, f, C, g, h, i]

        Note how the composite datasets themselves are capitalized and are
        accessible in the flat indexing, and not just the datasets.

        Examples
        --------
        Add a composite dataset to a plotter and access its block attributes.
        Note how the zero index is the entire multiblock and you can use ``1``
        and ``2`` to access the individual sub-blocks.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr.get_block(0)
        MultiBlock (...)
          N Blocks:   2
          X Bounds:   -5.000e-01, 5.000e-01
          Y Bounds:   -5.000e-01, 5.000e-01
          Z Bounds:   -5.000e-01, 1.500e+00

        Note this is the same as using ``__getitem__``

        >>> mapper.block_attr[0]
        Composite Block Addr=... Attributes
        Visible:   None
        Opacity:   None
        Color:     None
        Pickable   None

        """
        try:
            if vtk_version_info <= (9, 0, 3):  # pragma: no cover
                vtk_ref = _vtk.reference(0)  # needed for <=9.0.3
                block = self.DataObjectFromIndex(index, self._dataset, vtk_ref)  # type: ignore[arg-type]
            else:
                block = self.DataObjectFromIndex(index, self._dataset)
        except OverflowError:
            msg = f'Invalid block key: {index}'
            raise KeyError(msg) from None
        if block is None and index > len(self) - 1:
            msg = f'index {index} is out of bounds. There are only {len(self)} blocks.'
            raise KeyError(msg) from None
        return block

    def __getitem__(self, index):
        """Return a block attribute by its flat index."""
        return BlockAttributes(self.get_block(index), self)

    def __len__(self):
        """Return the number of blocks in this dataset."""
        from pyvista import MultiBlock  # avoid circular  # noqa: PLC0415

        # start with 1 as there is always a composite dataset and this is the
        # root of the tree
        cc = 1
        for dataset in self._dataset:
            if isinstance(dataset, MultiBlock):
                cc += len(dataset) + 1  # include the block itself
            else:
                cc += 1
        return cc

    def __iter__(self):
        """Return an iterator of all the block attributes."""
        for ii in range(len(self)):
            yield self[ii]


class CompositePolyDataMapper(
    _BaseMapper,
    (
        _vtk.vtkCompositePolyDataMapper  # type: ignore[misc]
        if vtk_version_info >= (9, 3)
        else _vtk.vtkCompositePolyDataMapper2
    ),
):
    """Composite PolyData mapper.

    Parameters
    ----------
    dataset : pyvista.MultiBlock, optional
        Multiblock dataset.

    theme : pyvista.plotting.themes.Theme, optional
        Plot-specific theme.

    color_missing_with_nan : bool, optional
        Color any missing values with the ``nan_color``. This is useful
        when not all blocks of the composite dataset have the specified
        ``scalars``.

    interpolate_before_map : bool, optional
        Enabling makes for a smoother scalars display.  Default is
        ``True``.  When ``False``, OpenGL will interpolate the
        mapped colors which can result is showing colors that are
        not present in the color map.

    """

    @_deprecate_positional_args(allowed=['dataset'])
    def __init__(  # noqa: PLR0917
        self,
        dataset=None,
        theme=None,
        color_missing_with_nan=None,
        interpolate_before_map=None,
    ):
        """Initialize this composite mapper."""
        super().__init__(theme=theme)
        # this must be added to set the color, opacity, and visibility of
        # individual blocks
        self._attr = CompositeAttributes(self, dataset)
        self.dataset = dataset

        if color_missing_with_nan is not None:
            self.color_missing_with_nan = color_missing_with_nan
        if interpolate_before_map is not None:
            self.interpolate_before_map = interpolate_before_map

        self._orig_scalars_name: str | None = None

    @property
    def dataset(self) -> pyvista.MultiBlock:  # numpydoc ignore=RT01
        """Return the composite dataset assigned to this mapper.

        Examples
        --------
        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.dataset
        MultiBlock (...)
          N Blocks:   2
          X Bounds:   -5.000e-01, 5.000e-01
          Y Bounds:   -5.000e-01, 5.000e-01
          Z Bounds:   -5.000e-01, 1.500e+00

        """
        return self._dataset

    @dataset.setter
    def dataset(self, obj: pyvista.MultiBlock):
        self.SetInputDataObject(obj)
        self._dataset = obj
        self._attr._dataset = obj

    @property
    def block_attr(self) -> CompositeAttributes:  # numpydoc ignore=RT01
        """Return the block attributes.

        Notes
        -----
        ``block_attr`` employs VTK's flat indexing and allows for accessing
        both the blocks of a composite dataset as well as the entire composite
        dataset. If there is only one composite dataset, ``A``, which contains
        datasets ``[b, c]``, the indexing would be ``[A, b, c]``.

        If there are two composite datasets ``[B, C]`` in one composite
        dataset, ``A``, each of which containing three additional datasets
        ``[d, e, f]``, and ``[g, h, i]``, respectively, then the head node,
        ``A``, would be the zero index, followed by the first child, ``B``,
        followed by all the children of ``B``, ``[d, e, f]``. In data
        structures, this flat indexing would be known as "Depth-first search"
        and the entire indexing would be::

           [A, B, d, e, f, C, g, h, i]

        Examples
        --------
        Add a sphere and a cube as a multiblock dataset to a plotter and then
        change the visibility and color of the blocks.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.block_attr[1].color = 'b'
        >>> mapper.block_attr[1].opacity = 0.1
        >>> mapper.block_attr[1]
        Composite Block Addr=... Attributes
        Visible:   None
        Opacity:   0.1
        Color:     Color(name='blue', hex='#0000ffff', opacity=255)
        Pickable   None

        """
        return self._attr

    @property
    def color_missing_with_nan(self) -> bool:  # numpydoc ignore=RT01
        """Color missing arrays with the NaN color.

        Examples
        --------
        Enable coloring missing values with NaN.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> dataset[0].point_data['data'] = dataset[0].points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(
        ...     dataset, scalars='data', show_scalar_bar=False
        ... )
        >>> pv.global_theme.nan_color = 'r'
        >>> mapper.color_missing_with_nan = True
        >>> pl.show()

        """
        return self.GetColorMissingArraysWithNanColor()

    @color_missing_with_nan.setter
    def color_missing_with_nan(self, value: bool):
        self.SetColorMissingArraysWithNanColor(value)

    def set_unique_colors(
        self,
        color_cycler: bool | str | cycler.Cycler[str, ColorLike] | Sequence[ColorLike] = True,  # noqa: FBT001, FBT002
    ):
        """Set each block of the dataset to a unique color.

        This uses ``matplotlib``'s color cycler by default.

        When a custom color cycler, or a sequence of
        color-like objects, is passed it sets the blocks
        to the corresponding colors.

        Parameters
        ----------
        color_cycler : bool | str | cycler.Cycler | sequence[ColorLike]
            The sequence of colors to cycle through,
            if ``True``, uses matplotlib cycler.

        Examples
        --------
        Set each block of the composite dataset to a unique color.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.set_unique_colors()
        >>> mapper.block_attr[1].color
        Color(name='tab:orange', hex='#ff7f0eff', opacity=255)
        >>> mapper.block_attr[2].color
        Color(name='tab:green', hex='#2ca02cff', opacity=255)

        >>> pl.show()

        """
        self.scalar_visibility = False

        if isinstance(color_cycler, bool):
            colors = cycle(get_cycler('matplotlib'))
        else:
            colors = cycle(get_cycler(color_cycler))

        for attr in self.block_attr:
            attr.color = next(colors)['color']

    @_deprecate_positional_args(allowed=['scalars_name'])
    def set_scalars(  # noqa: PLR0917
        self,
        scalars_name,
        preference,
        component,
        annotations,
        rgb,
        scalar_bar_args,
        n_colors,
        nan_color,
        above_color,
        below_color,
        clim,
        cmap,
        flip_scalars,
        log_scale,
    ):
        """Set the scalars of the mapper.

        Parameters
        ----------
        scalars_name : str
            Name of the scalars in the dataset. Must already exist in at least
            of the blocks.

        preference : str
            For each block, when ``block.n_points == block.n_cells`` and
            setting scalars, this parameter sets how the scalars will be mapped
            to the mesh.  Default ``'point'``, causes the scalars will be
            associated with the mesh points.  Can be either ``'point'`` or
            ``'cell'``.

        component : int
            Set component of vector valued scalars to plot.  Must be
            nonnegative, if supplied. If ``None``, the magnitude of
            the vector is plotted.

        annotations : dict
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the string annotations.

        rgb : bool
            If the ``scalars_name`` corresponds to a 2 dimensional array, plot
            those values as RGB(A) colors.

        scalar_bar_args : dict
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.Plotter.add_scalar_bar`.

        n_colors : int
            Number of colors to use when displaying scalars.

        nan_color : ColorLike
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        above_color : ColorLike
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'above'``.

        below_color : ColorLike
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'below'``.

        clim : Sequence
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        cmap : str | list | pyvista.LookupTable
            Name of the Matplotlib colormap to use when mapping the
            ``scalars``.  See available Matplotlib colormaps.  Only applicable
            for when displaying ``scalars``.
            ``colormap`` is also an accepted alias for this. If
            ``colorcet`` or ``cmocean`` are installed, their colormaps can be
            specified by name.

            You can also specify a list of colors to override an existing
            colormap with a custom one.  For example, to create a three color
            colormap you might specify ``['green', 'red', 'blue']``.

            This parameter also accepts a :class:`pyvista.LookupTable`. If this
            is set, all parameters controlling the color map like ``n_colors``
            will be ignored.
            are installed, their colormaps can be specified by name.

        flip_scalars : bool
            Flip direction of cmap. Most colormaps allow ``*_r``
            suffix to do this as well.

        log_scale : bool
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float.

        Returns
        -------
        dict
            Dictionary of scalar bar arguments.

        """
        self._orig_scalars_name = scalars_name

        field, scalars_name, dtype = self._dataset._activate_plotting_scalars(
            scalars_name=scalars_name,
            preference=preference,
            component=component,
            rgb=rgb,
        )

        self.scalar_visibility = True
        if rgb:
            self.color_mode = 'direct'
            return scalar_bar_args
        else:
            self.scalar_map_mode = field.name.lower()

        scalar_bar_args.setdefault('title', scalars_name)

        if clim is None:
            clim = self._dataset.get_data_range(scalars_name, allow_missing=True)
        self.scalar_range = clim

        if log_scale and clim[0] <= 0:
            clim = [sys.float_info.min, clim[1]]

        if isinstance(cmap, pyvista.LookupTable):
            self.lookup_table = cmap
        else:
            if dtype == np.bool_:
                cats = np.array([b'False', b'True'], dtype='|S5')
                values = np.array([0, 1])
                n_colors = 2
                scalar_bar_args.setdefault('n_labels', 0)
                self.lookup_table.SetAnnotations(convert_array(values), convert_string_array(cats))
                clim = [-0.5, 1.5]

            self.lookup_table.log_scale = log_scale

            if isinstance(annotations, dict):
                self.lookup_table.annotations = annotations

            # self.lookup_table.SetNumberOfTableValues(n_colors)
            if nan_color:
                self.lookup_table.nan_color = nan_color
            if above_color:
                self.lookup_table.above_range_color = above_color
                scalar_bar_args.setdefault('above_label', 'above')
            if below_color:
                self.lookup_table.below_range_color = below_color
                scalar_bar_args.setdefault('below_label', 'below')

            if cmap is None:
                cmap = pyvista.global_theme.cmap if self._theme is None else self._theme.cmap

            if cmap is not None:
                self.lookup_table.apply_cmap(cmap, n_colors, flip=flip_scalars)
            elif flip_scalars:
                self.lookup_table.SetHueRange(0.0, 0.66667)
            else:
                self.lookup_table.SetHueRange(0.66667, 0.0)

        return scalar_bar_args
