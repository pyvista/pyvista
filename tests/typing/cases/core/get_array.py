"""Typing cases for :func:`pyvista.core.utilities.arrays.get_array`."""

from __future__ import annotations

from type_assert import assert_types

from pyvista import pyvista_ndarray
from pyvista.core.utilities.arrays import get_array
from tests.typing.meshes import image
from tests.typing.meshes import with_arrays


def a_flag() -> bool:
    """Return a flag typed only as a bool."""
    return True


# Raising on a missing array means an array always comes back
assert_types(get_array(with_arrays(image()), 's', err=True), pyvista_ndarray)

# Otherwise a missing array is ``None``
assert_types(get_array(with_arrays(image()), 's'), pyvista_ndarray | None)
assert_types(get_array(with_arrays(image()), 's', err=False), pyvista_ndarray | None)
assert_types(get_array(with_arrays(image()), 's', err=a_flag()), pyvista_ndarray | None)
assert_types(get_array(image(), 'missing'), pyvista_ndarray | None)
