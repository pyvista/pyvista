"""Typing cases for :meth:`pyvista.core.utilities.state_manager._StateManager.__call__`."""

from __future__ import annotations

from type_assert import assert_types

from pyvista.core.utilities.state_manager import _AllowNewAttributes
from pyvista.core.utilities.state_manager import _AllowNewAttributesOptions
from pyvista.core.utilities.state_manager import _VerbosityOptions
from pyvista.core.utilities.state_manager import _vtkSnakeCase
from pyvista.core.utilities.state_manager import _VtkSnakeCaseOptions
from pyvista.core.utilities.state_manager import _VTKVerbosity
from pyvista.core.utilities.state_manager import allow_new_attributes
from pyvista.core.utilities.state_manager import vtk_snake_case
from pyvista.core.utilities.state_manager import vtk_verbosity

# fmt: off

# Passing a state returns a fresh manager to use as a context manager
assert_types(vtk_verbosity('off'),                _VTKVerbosity)
assert_types(vtk_verbosity('error'),              _VTKVerbosity)
assert_types(vtk_verbosity('max'),                _VTKVerbosity)
assert_types(vtk_snake_case('allow'),             _vtkSnakeCase)
assert_types(vtk_snake_case('error'),             _vtkSnakeCase)
assert_types(allow_new_attributes(True),          _AllowNewAttributes)
assert_types(allow_new_attributes(False),         _AllowNewAttributes)
assert_types(allow_new_attributes('private'),     _AllowNewAttributes)

# Passing nothing reads the current state back
assert_types(vtk_verbosity(None),                 _VerbosityOptions)
assert_types(vtk_snake_case(None),                _VtkSnakeCaseOptions)
assert_types(allow_new_attributes(None),          _AllowNewAttributesOptions)
assert_types(vtk_verbosity(),                     _VerbosityOptions)
assert_types(vtk_snake_case(),                    _VtkSnakeCaseOptions)
assert_types(allow_new_attributes(),              _AllowNewAttributesOptions)
