"""Typing cases for :func:`pyvista.register_writer`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista import DataObject
from pyvista.core.utilities.writer_registry import WriterHandler


def a_handler(dataset: DataObject, path: str, /, **kwargs: Any) -> None:
    """Write nothing, whatever it is given."""


# Without a handler the call only builds the decorator, so nothing is registered yet
assert_types(pv.register_writer('.type_assert'), Callable[[WriterHandler], WriterHandler])
assert_types(pv.register_writer('.type_assert', override=True), Callable[[WriterHandler], WriterHandler])
assert_types(pv.register_writer('.type_assert', None), Callable[[WriterHandler], WriterHandler])

with skip_runtime(reason='registering a writer mutates the process-wide registry'):
    assert_types(pv.register_writer('.type_assert', a_handler), None)
    assert_types(pv.register_writer('.type_assert', a_handler, override=True), None)
