"""Wrapper for :vtk:`vtkPropCollection`."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import MutableSequence
from typing import TYPE_CHECKING

import numpy as np

from pyvista import _validation
from pyvista.plotting import _vtk

if TYPE_CHECKING:
    from typing import Any


class _PropCollection(MutableSequence[_vtk.vtkProp]):
    """Sequence wrapper for a :vtk:`vtkPropCollection` with a dict-like interface.

    .. versionadded:: 0.45

    """

    def __init__(self, prop_collection: _vtk.vtkPropCollection):
        super().__init__()
        self._prop_collection = prop_collection

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # lookup from index number
            key = self._validate_index(key)
            return self._prop_collection.GetItemAsObject(int(key))
        elif isinstance(key, str):
            # lookup from actor name
            names = self.keys()
            try:
                index = names.index(key)
                return self._prop_collection.GetItemAsObject(index)
            except ValueError:
                msg = f"No item found with name '{key}'."
                raise KeyError(msg)
        msg = f'Key must be an index or a string, got {type(key).__name__}.'
        raise TypeError(msg)

    def __len__(self):
        return self._prop_collection.GetNumberOfItems()

    def __delitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # remove by index
            key = self._validate_index(key)
            self._prop_collection.RemoveItem(key)
        elif isinstance(key, str):
            # remove by name
            names = self.keys()
            try:
                index = names.index(key)
            except ValueError:
                msg = f"No item found with name '{key}'."
                raise KeyError(msg)
            del self[index]
        else:
            msg = f'Key must be an index or a string, got {type(key).__name__}.'
            raise TypeError(msg)

    def __setitem__(self, key, value):
        _validation.check_instance(value, _vtk.vtkProp)
        if isinstance(key, (int, np.integer)):
            # set by index
            key = self._validate_index(key)
            self._prop_collection.ReplaceItem(key, value)
        elif isinstance(key, str):
            if hasattr(value, 'name') and value.name != key:
                msg = f"Name of the new actor '{value.name}' must match the key name '{key}'."
                raise ValueError(msg)
            # set by name
            index = self.keys().index(key)
            self[index] = value
        else:
            msg = f'Key must be an index or a string, got {type(key).__name__}.'
            raise TypeError(msg)

    def insert(self, index, value) -> None:
        _validation.check_instance(value, _vtk.vtkProp)
        if len(self) == 0:
            self.append(value)
            return
        if index < 0:
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
        yield from zip(self.keys(), self, strict=False)

    def __del__(self):
        self._prop_collection = None  # type: ignore[assignment]

    def _validate_index(self, index: int | np.integer[Any]) -> int:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            msg = 'Index out of range.'
            raise IndexError(msg)
        return int(index)
