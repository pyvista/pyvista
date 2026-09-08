"""Typing cases for :func:`pyvista.register_reader`."""

from __future__ import annotations

from typing import Any

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet


class Provider:
    """Read a path and return a dataset."""

    def __call__(self, path: str, /, **kwargs: Any) -> DataSet:  # pragma: no cover
        """Read ``path`` as a mesh."""
        return pv.read(path, cls=pv.PolyData, **kwargs)


SKIP_RUNTIME = dict.fromkeys(
    [
        "pv.register_reader('.type_assert')(Provider())",
        "pv.register_reader('.type_assert')(pv.PLYReader)",
        "pv.register_reader('.type_assert', Provider())",
        "pv.register_reader('.type_assert', pv.PLYReader, override=True)",
    ],
    'registering a reader mutates the process-wide registry',
)


# The decorator form hands the provider back unchanged
assert_types(pv.register_reader('.type_assert')(Provider()), Provider)  # pragma: no cover
assert_types(pv.register_reader('.type_assert')(pv.PLYReader), type[pv.PLYReader])  # pragma: no cover

# Passing the provider outright registers it and returns nothing
assert_types(pv.register_reader('.type_assert', Provider()), None)  # pragma: no cover
assert_types(pv.register_reader('.type_assert', pv.PLYReader, override=True), None)  # pragma: no cover
