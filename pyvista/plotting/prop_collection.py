"""Wrapper for vtlPropCollection."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import MutableSequence

import numpy as np

from pyvista.plotting import _vtk


class PropCollection(MutableSequence[_vtk.vtkProp]):  # noqa: D101
    def __init__(self, prop_collection: _vtk.vtkPropCollection):
        """Initialize with the object to be wrapped."""
        super().__init__()
        self._prop_collection = prop_collection

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            # lookup from index number
            if item < 0 or item >= len(self):
                raise IndexError('Index out of range.')
            return self._prop_collection.GetItemAsObject(int(item))
        elif isinstance(item, str):
            # lookup from actor name
            names = self.keys()
            try:
                index = names.index(item)
                return self._prop_collection.GetItemAsObject(index)
            except ValueError:
                raise KeyError(f"No item found with name '{item}'")
        raise TypeError(f'Item must be an index or a string, got {type(item).__name__}.')

    def __len__(self):
        return self._prop_collection.GetNumberOfItems()

    def __delitem__(self, key):
        self._prop_collection.RemoveItem(key)

    def __setitem__(self, key, value):
        self._prop_collection.ReplaceItem(key, value)

    def insert(self, index, value):  # noqa: D102
        self._prop_collection.InsertItem(index, value)

    def append(self, value: _vtk.vtkProp):  # noqa: D102
        self._prop_collection.AddItem(value)

    def keys(self) -> list[str]:  # noqa: D102
        return [
            prop.name if hasattr(prop, 'name') else prop.GetAddressAsString('') for prop in self
        ]

    def items(self) -> Iterable[tuple[str, _vtk.vtkProp]]:  # noqa: D102
        return zip(self.keys(), self)

    def __del__(self):
        del self._prop_collection
