"""This module contains the Property class."""

from __future__ import annotations

from typing import Any
from typing import Final
from typing import Sequence

from pyvista.plotting import _vtk
from pyvista.plotting._property import Property


class MultiProperty:
    """Represent multiple Property objects as a single Property."""

    _MULTIPLE_VALUES = 'MULTIPLE VALUES'

    # List of @property decorated methods from pv.Property
    _ATTRIBUTES: Final = sorted(
        [name for name, value in vars(Property).items() if isinstance(value, property)]
    )

    def __init__(self, props: Sequence[Property]):
        super().__init__()
        self._props = props

    def __getattr__(self, attr: str):
        if attr in self._ATTRIBUTES:
            # Get attribute from all Property objects
            # Return value if it's the same for all objects, raise error otherwise
            props = self._props
            first_prop_value = getattr(props[0], attr)
            for prop in props:
                prop_value = getattr(prop, attr)
                if prop_value != first_prop_value:
                    raise ValueError(
                        f'Multiple Property values detected for attribute "{attr}". Value can only be returned if all Property objects have the same value.\nGot: `{prop_value}` and `{first_prop_value}`.'
                    )
            return first_prop_value
        # Get attribute from self
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr: str, value: Any):
        if attr in self._ATTRIBUTES:
            # Set attribute for all Property objects
            for prop in self._props:
                setattr(prop, attr, value)
        else:
            # Set attribute for self
            object.__setattr__(self, attr, value)

    def __repr__(self):
        """Representation of this multi-property.

        Show the property values common to all Property objects.
        If values differ between the objects, its value is shown
        as 'MULTIPLE VALUES'.
        """
        from pyvista.core.errors import VTKVersionError

        props = [
            f'{type(self).__name__} ({hex(id(self))})',
        ]

        for attr in self._ATTRIBUTES:
            name = ' '.join(attr.split('_')).capitalize() + ':'
            try:
                value = getattr(self, attr)
                if isinstance(value, str):
                    value = f'"{value}"'
            except VTKVersionError:
                continue
            except ValueError:
                value = MultiProperty._MULTIPLE_VALUES
            props.append(f'  {name:28s} {value}')

        return '\n'.join(props)


class Assembly(_vtk.vtkPropAssembly):
    def add_parts(self, parts: Sequence[_vtk.vtkProp]):
        [self.AddPart(part) for part in parts]

    @property
    def parts(self):
        collection = self.GetParts()
        return [collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems())]

    @property
    def prop(self) -> MultiProperty:
        # Collect all Property objects from all parts
        property_list: list[Property] = []
        for part in self.parts:
            # Parts may be any vtkProp, check if part has a Property object
            if hasattr(part, 'prop'):
                prop = part.prop
            elif hasattr(part, 'GetProperty'):
                prop = part.GetProperty()
            else:
                continue
            # Prop may be a vtkTextProperty, make sure we only collect Property objects
            if isinstance(prop, Property):
                property_list.append(prop)

        return MultiProperty(property_list)
