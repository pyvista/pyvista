"""Module containing composite data mapper."""
from itertools import cycle
import logging
import sys
from typing import Optional
import weakref

import numpy as np

from pyvista import _vtk

from ._plotting import _has_matplotlib
from .colors import Color, get_cmap_safe


class _BlockAttributes:
    """Internal class to set the attributes of a block."""

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

    def _remove_block_color(self):
        """Remove block color."""
        self._attr.RemoveBlockColors()

    @property
    def color(self):
        """Get or set the color of a block."""
        if not self._has_color:
            return None
        return tuple(self._attr.GetBlockColor(self._block))

    @color.setter
    def color(self, new_color):
        """Get or set the color of a block."""
        if new_color is None:
            self._attr.RemoveBlockColor(self._block)
            self._attr.Modified()
            return
        self._attr.SetBlockColor(self._block, Color(new_color).float_rgb)

    @property
    def visible(self) -> Optional[bool]:
        """Get or set the visibility of a block."""
        if not self._has_visibility:
            return None
        return self._attr.GetBlockVisibility(self._block)

    @visible.setter
    def visible(self, new_visible: bool):
        """Get or set the visibility of a block."""
        if new_visible is None:
            self._attr.RemoveBlockVisibility(self._block)
            self._attr.Modified()
            return
        self._attr.SetBlockVisibility(self._block, new_visible)

    @property
    def opacity(self) -> Optional[float]:
        """Get or set the visibility of a block.

        If opacity has not been set this will be ``None``.

        """
        if not self._has_opacity:
            return None
        return self._attr.GetBlockOpacity(self._block)

    @opacity.setter
    def opacity(self, new_opacity: float):
        """Get or set the visibility of a block."""
        if new_opacity is None:
            self._attr.RemoveBlockOpacity(self._block)
            self._attr.Modified()
            return

        self._attr.SetBlockOpacity(self._block, new_opacity)

    @property
    def pickable(self) -> Optional[bool]:
        """Get or set the visibility of a block."""
        if not self._has_pickable:
            return None
        return self._attr.GetBlockPickability(self._block)

    @pickable.setter
    def pickable(self, new_pickable: bool):
        """Get or set the visibility of a block."""
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
            ]
        )


class CompositeAttributes(_vtk.vtkCompositeDataDisplayAttributes):
    """Wrap vtkCompositeDataDisplayAttributes."""

    def __init__(self, mapper, dataset):
        """Initialize CompositeAttributes."""
        super().__init__()
        mapper.SetCompositeDataDisplayAttributes(self)
        self._dataset = dataset

    def reset_visibilities(self):
        """Reset the visibility of all blocks."""
        self.RemoveBlockVisibilities()

    def reset_pickability(self):
        """Reset the pickability of all blocks."""
        self.RemoveBlockPickabilities()

    def reset_colors(self):
        """Reset the color of all blocks."""
        self.RemoveBlockColors()

    def reset_opacities(self):
        """Reset the opacities of all blocks."""
        self.RemoveBlockOpacities()

    def get_block(self, index):
        """Return a block by its flat index."""
        try:
            if _vtk.VTK9:
                block = self.DataObjectFromIndex(index, self._dataset)
            else:
                vtk_ref = _vtk.reference(0)  # needed for <9.0
                block = self.DataObjectFromIndex(index, self._dataset, vtk_ref)
        except OverflowError:
            raise KeyError(f'Invalid block key: {index}') from None
        if block is None:
            if index > len(self):
                raise KeyError(
                    f'index {index} is out of bounds. There are only {len(self)} blocks.'
                ) from None
        return block

    def __getitem__(self, index):
        """Return a block attribute by its flat index."""
        return _BlockAttributes(self.get_block(index), self)

    def __len__(self):
        """Return the number of blocks."""
        return len(self._dataset)

    def __iter__(self):
        """Return an iterator of all the block attributes."""
        for ii in range(len(self)):
            yield self[ii]


class CompositePolyDataMapper(_vtk.vtkCompositePolyDataMapper2):
    """Wrap vtkCompositePolyDataMapper2."""

    def __init__(self, dataset, color_missing_with_nan=None, interpolate_before_map=None):
        """Initialize this composite mapper."""
        super().__init__()
        self.SetInputDataObject(dataset)

        # this must be added to set the color, opacity, and visibility of
        # individual blocks
        self._attr = CompositeAttributes(self, dataset)
        self._dataset = dataset
        self._added_scalars = None
        self._orig_scalars_name = None

        if color_missing_with_nan is not None:
            self.color_missing_with_nan = color_missing_with_nan
        if interpolate_before_map is not None:
            self.interpolate_before_map = interpolate_before_map

    @property
    def interpolate_before_map(self):
        """Return or set the interpolation of scalars before mapping."""
        return self.GetInterpolateScalarsBeforeMapping()

    @interpolate_before_map.setter
    def interpolate_before_map(self, value: bool):
        self.SetInterpolateScalarsBeforeMapping(value)

    @property
    def block_attr(self):
        """Return the block attributes."""
        return self._attr

    @property
    def color_missing_with_nan(self) -> bool:
        """Color missing arrays with the NaN color."""
        return self.GetColorMissingArraysWithNanColor()

    @color_missing_with_nan.setter
    def color_missing_with_nan(self, value: bool):
        self.SetColorMissingArraysWithNanColor(value)

    @property
    def lookup_table(self):
        """Return or set the lookup table."""
        return self.GetLookupTable()

    @lookup_table.setter
    def lookup_table(self, table):
        return self.SetLookupTable(table)

    @property
    def scalar_range(self):
        """Return or set the scalar range."""
        return self.GetScalarRange()

    @scalar_range.setter
    def scalar_range(self, clim):
        self.SetScalarRange(*clim)
        if self.lookup_table is not None:
            self.lookup_table.SetRange(*clim)
        self._scalar_range = clim

    @property
    def scalar_visibility(self) -> bool:
        """Return or set the scalar visibility."""
        return self.GetScalarVisibility()

    @scalar_visibility.setter
    def scalar_visibility(self, value: bool):
        return self.SetScalarVisibility(value)

    def set_unique_colors(self):
        """Compute unique colors for each block of the dataset."""
        self.scalar_visibility = False
        if _has_matplotlib():
            import matplotlib

            colors = cycle(matplotlib.rcParams['axes.prop_cycle'])
            for attr in self.block_attr:
                attr.color = next(colors)['color']

        else:
            logging.warning('Please install matplotlib for color cycles.')

    @property
    def scalar_map_mode(self) -> str:
        """Return or set the scalar map mode."""
        return self.GetScalarModeAsString()

    @scalar_map_mode.setter
    def scalar_map_mode(self, scalar_mode: str):
        """Return or set the scalar map mode."""
        if scalar_mode == 'default':
            self.SetScalarModeToDefault()
        elif scalar_mode == 'point':
            self.SetScalarModeToUsePointData()
        elif scalar_mode == 'cell':
            self.SetScalarModeToUseCellData()
        elif scalar_mode == 'point_field':
            self.SetScalarModeToUsePointFieldData()
        elif scalar_mode == 'cell_field':
            self.SetScalarModeToUseCellFieldData()
        elif scalar_mode == 'field':
            self.SetScalarModeToUseFieldData()
        else:
            raise ValueError(
                f'Invalid `scalar_map_mode` "{scalar_mode}". Should be either '
                '"default", "point", "cell", "point_field", "cell_field" or "field".'
            )

    def set_scalars(
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
        categories,
        theme,
        log_scale,
    ):
        """Set the scalars of the mapper."""
        self._orig_scalars_name = scalars_name

        field, scalars_name = self._dataset._activate_plotting_scalars(
            scalars_name, preference, component, rgb
        )

        self.scalar_visibility = True
        if rgb:
            self.SetColorModeToDirectScalars()
            return scalar_bar_args
        else:
            self.scalar_map_mode = field.name.lower()

        # track if any scalars have been added
        if self._orig_scalars_name != scalars_name:
            self._added_scalars = (scalars_name, field)

        scalar_bar_args.setdefault('title', scalars_name)

        if isinstance(annotations, dict):
            for val, anno in annotations.items():
                self.lookup_table.SetAnnotation(float(val), str(anno))

        self.lookup_table.SetNumberOfTableValues(n_colors)
        self.nan_color = nan_color.float_rgba
        if above_color:
            self.above_range_color = above_color.float_rgba
            scalar_bar_args.setdefault('above_label', 'Above')
        if below_color:
            self.below_range_color = below_color.float_rgba
            scalar_bar_args.setdefault('below_label', 'Above')

        if clim is None:
            clim = self._dataset.get_data_range(scalars_name, allow_missing=True)
        self.scalar_range = clim

        if log_scale:
            if clim[0] <= 0:
                clim = [sys.float_info.min, clim[1]]
            self.lookup_table.SetScaleToLog10()

        if cmap is None:  # Set default map if matplotlib is available
            if _has_matplotlib():
                cmap = theme.cmap

        if cmap is not None:
            cmap = get_cmap_safe(cmap)
            ctable = cmap(np.linspace(0, 1, n_colors)) * 255
            ctable = ctable.astype(np.uint8)
            if flip_scalars:
                ctable = np.ascontiguousarray(ctable[::-1])

            self.lookup_table.SetTable(_vtk.numpy_to_vtk(ctable))
        else:  # no cmap specified
            if flip_scalars:
                self.lookup_table.SetHueRange(0.0, 0.66667)
            else:
                self.lookup_table.SetHueRange(0.66667, 0.0)

        return scalar_bar_args

    @property
    def nan_color(self):
        """Return or set the NaN color."""
        return self.lookup_table.GetNanColor()

    @nan_color.setter
    def nan_color(self, color):
        self.lookup_table.SetNanColor(color)

    @property
    def above_range_color(self):
        """Return or set the above range color."""
        return self.lookup_table.GetAboveRangeColor()

    @above_range_color.setter
    def above_range_color(self, color):
        self.lookup_table.SetUseAboveRangeColor(True)
        self.lookup_table.SetAboveRangeColor(color)

    @property
    def below_range_color(self):
        """Return or set the below range color."""
        return self.lookup_table.GetBelowRangeColor()

    @below_range_color.setter
    def below_range_color(self, color):
        self.lookup_table.SetUseBelowRangeColor(True)
        self.lookup_table.SetBelowRangeColor(color)
