"""Wrapper for vtkPropCollection."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import MutableSequence

import numpy as np

from pyvista import _validation
from pyvista.plotting import _vtk


class _PropCollection(MutableSequence[_vtk.vtkProp]):
    """Sequence wrapper for a vtkPropCollection with a dict-like interface.

    .. versionadded:: 0.45

    """

    def __init__(self, prop_collection: _vtk.vtkPropCollection):
        super().__init__()
        self._prop_collection = prop_collection

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # lookup from index number
            if key < 0 or key >= len(self):
                raise IndexError('Index out of range.')
            return self._prop_collection.GetItemAsObject(int(key))
        elif isinstance(key, str):
            # lookup from actor name
            names = self.keys()
            try:
                index = names.index(key)
                return self._prop_collection.GetItemAsObject(index)
            except ValueError:
                raise KeyError(f"No item found with name '{key}'")
        raise TypeError(f'Key must be an index or a string, got {type(key).__name__}.')

    def __len__(self):
        return self._prop_collection.GetNumberOfItems()

    def __delitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # remove by index
            while key < 0:
                key = len(self) + key
            if key >= len(self):
                raise IndexError('Index out of range.')
            self._prop_collection.RemoveItem(int(key))
        elif isinstance(key, str):
            # remove by name
            index = self.keys().index(key)
            del self[index]
        else:
            raise TypeError(f'Key must be an index or a string, got {type(key).__name__}.')

    def __setitem__(self, key, value):
        _validation.check_instance(value, _vtk.vtkProp)
        if isinstance(key, (int, np.integer)):
            # set by index
            while key < 0:
                key = len(self) + key
            if key >= len(self):
                raise IndexError('Index out of range.')
            self._prop_collection.ReplaceItem(key, value)
        elif isinstance(key, str):
            if hasattr(value, 'name') and value.name != key:
                raise ValueError(
                    f"Name of the new actor '{value.name}' must match the key name '{key}'."
                )
            # set by name
            index = self.keys().index(key)
            self[index] = value
        else:
            raise TypeError(f'Key must be an index or a string, got {type(key).__name__}.')

    def insert(self, index, value):
        _validation.check_instance(value, _vtk.vtkProp)
        while index < 0:
            index = len(self) + index + 1
        index = min(index, len(self))
        self._prop_collection.InsertItem(index - 1, value)

    def append(self, value: _vtk.vtkProp):
        _validation.check_instance(value, _vtk.vtkProp)
        self._prop_collection.AddItem(value)

    def keys(self) -> list[str]:
        return [
            prop.name
            if hasattr(prop, 'name')
            else f'{type(prop).__name__}({prop.GetAddressAsString("")})'
            for prop in self
        ]

    def items(self) -> Iterable[tuple[str, _vtk.vtkProp]]:
        yield from zip(self.keys(), self)

    def __del__(self):
        del self._prop_collection
