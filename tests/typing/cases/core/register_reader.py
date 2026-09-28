"""Typing cases for :func:`pyvista.register_reader`."""

from __future__ import annotations

from typing import Any

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista import DataSet


class Provider:
    """Read a path and return a dataset."""

    def __call__(self, path: str, /, **kwargs: Any) -> DataSet:  # pragma: no cover
        """Read ``path`` as a mesh."""
        return pv.read(path, cls=pv.PolyData, **kwargs)


# The decorator form hands the provider back unchanged
with skip_runtime(reason='registering a reader mutates the process-wide registry'):
    assert_types(pv.register_reader('.type_assert')(Provider()), Provider)
    assert_types(pv.register_reader('.type_assert')(pv.PLYReader), type[pv.PLYReader])

    # Passing the provider outright registers it and returns nothing
    assert_types(pv.register_reader('.type_assert', Provider()), None)
    assert_types(pv.register_reader('.type_assert', pv.PLYReader, override=True), None)
