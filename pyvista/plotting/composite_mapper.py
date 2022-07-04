"""Module containing composite data mapper."""
from typing import Optional
import weakref

from pyvista import _vtk

from .colors import Color


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

    def __getitem__(self, index):
        """Return a block by its flat index."""
        try:
            block = self.DataObjectFromIndex(index, self._dataset)
        except OverflowError:
            raise KeyError(f'Invalid block key: {index}') from None
        if block is None:
            raise KeyError(
                f'index {index} is out of bounds. There are only {len(self)} blocks.'
            ) from None
        return _BlockAttributes(block, self)

    def __len__(self):
        """Return the number of blocks."""
        return len(self._dataset)


class CompositePolyDataMapper(_vtk.vtkCompositePolyDataMapper2):
    """Wrap vtkCompositePolyDataMapper2."""

    def __init__(self, dataset):
        """Initialize this composite mapper."""
        super().__init__()
        self.SetInputDataObject(dataset)

        # this must be added to set the color, opacity, and visibility of
        # individual blocks
        self._attr = CompositeAttributes(self, dataset)

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
        """Return the lookup table."""
        return self.GetLookupTable()
