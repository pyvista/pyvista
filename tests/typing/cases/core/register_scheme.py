"""Typing cases for :func:`pyvista.register_scheme`."""

from __future__ import annotations

from typing import Any

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock


class Handler:
    """Read a URI and return a dataset."""

    def __call__(self, _uri: str, /, **_kwargs: Any) -> DataSet | MultiBlock:  # pragma: no cover
        """Read a URI as a mesh."""
        return pv.Sphere()


# The decorator form hands the handler back unchanged
with skip_runtime(reason='registering a scheme mutates the process-wide registry'):
    assert_types(pv.register_scheme('typeassert')(Handler()), Handler)

    # Passing the handler outright registers it and returns nothing
    assert_types(pv.register_scheme('typeassert', Handler(), override=True), None)

assert_types(pv.registered_schemes(), tuple[pv.SchemeRegistration, ...])
