"""Typing cases for :meth:`pyvista.MultiBlock.recursive_iterator`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock


def multi() -> MultiBlock:
    """Return a nested `MultiBlock` with `None` blocks, so the runtime checks see both."""
    return pv.MultiBlock([pv.PolyData(), None, pv.MultiBlock([pv.PolyData(), None])])


# Each case builds its own `MultiBlock`, and `list` forces the iterator so that the
# runtime check can inspect what it yields.
# fmt: off

# Names
assert_types(list(multi().recursive_iterator('names')), list[str])

# Ids
assert_types(list(multi().recursive_iterator('ids')),                    list[tuple[int, ...]])
assert_types(list(multi().recursive_iterator('ids', nested_ids=True)),   list[tuple[int, ...]])
assert_types(list(multi().recursive_iterator('ids', nested_ids=None)),   list[tuple[int, ...]])
assert_types(list(multi().recursive_iterator('ids', nested_ids=False)),  list[int])

# Items
assert_types(list(multi().recursive_iterator('items')),                                     list[tuple[str, DataSet | None]])
assert_types(list(multi().recursive_iterator('items', node_type='child')),                  list[tuple[str, DataSet | None]])
assert_types(list(multi().recursive_iterator('items', skip_none=True)),                     list[tuple[str, DataSet]])
assert_types(list(multi().recursive_iterator('items', node_type='child', skip_none=True)),  list[tuple[str, DataSet]])
assert_types(list(multi().recursive_iterator('items', node_type='parent')),                 list[tuple[str, MultiBlock]])
assert_types(list(multi().recursive_iterator('items', node_type='parent', skip_none=False)), list[tuple[str, MultiBlock]])

# Blocks
assert_types(list(multi().recursive_iterator()),                                     list[DataSet | None])
assert_types(list(multi().recursive_iterator('blocks')),                             list[DataSet | None])
assert_types(list(multi().recursive_iterator(skip_none=True)),                       list[DataSet])
assert_types(list(multi().recursive_iterator('blocks', skip_none=True)),             list[DataSet])
assert_types(list(multi().recursive_iterator(node_type='child')),                    list[DataSet | None])
assert_types(list(multi().recursive_iterator('blocks', node_type='child')),          list[DataSet | None])
assert_types(list(multi().recursive_iterator(skip_none=True, node_type='child')),    list[DataSet])
assert_types(list(multi().recursive_iterator('blocks', skip_none=True, node_type='child')), list[DataSet])
assert_types(list(multi().recursive_iterator(node_type='parent')),                   list[MultiBlock])
assert_types(list(multi().recursive_iterator('blocks', node_type='parent')),         list[MultiBlock])

# All
assert_types(list(multi().recursive_iterator('all')),                                       list[tuple[tuple[int, ...], str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', node_type='child')),                    list[tuple[tuple[int, ...], str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', skip_none=True)),                       list[tuple[tuple[int, ...], str, DataSet]])
assert_types(list(multi().recursive_iterator('all', nested_ids=True)),                      list[tuple[tuple[int, ...], str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', nested_ids=None)),                      list[tuple[tuple[int, ...], str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', nested_ids=False)),                     list[tuple[int, str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', nested_ids=True, skip_none=True)),      list[tuple[tuple[int, ...], str, DataSet]])
assert_types(list(multi().recursive_iterator('all', nested_ids=True, skip_none=False)),     list[tuple[tuple[int, ...], str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', nested_ids=False, skip_none=True)),     list[tuple[int, str, DataSet]])
assert_types(list(multi().recursive_iterator('all', nested_ids=False, skip_none=False)),    list[tuple[int, str, DataSet | None]])

# All, parent nodes
assert_types(list(multi().recursive_iterator('all', node_type='parent')),                                     list[tuple[tuple[int, ...], str, MultiBlock]])
assert_types(list(multi().recursive_iterator('all', node_type='parent', nested_ids=True)),                    list[tuple[tuple[int, ...], str, MultiBlock]])
assert_types(list(multi().recursive_iterator('all', node_type='parent', nested_ids=None)),                    list[tuple[tuple[int, ...], str, MultiBlock]])
assert_types(list(multi().recursive_iterator('all', node_type='parent', nested_ids=False)),                   list[tuple[int, str, MultiBlock]])
assert_types(list(multi().recursive_iterator('all', node_type='parent', nested_ids=True, skip_none=False)),   list[tuple[tuple[int, ...], str, MultiBlock]])
assert_types(list(multi().recursive_iterator('all', node_type='parent', nested_ids=False, skip_none=False)),  list[tuple[int, str, MultiBlock]])

# Options that do not change the return type
assert_types(list(multi().recursive_iterator('blocks', skip_empty=True)),                   list[DataSet | None])
assert_types(list(multi().recursive_iterator('names', prepend_names=True)),                 list[str])
assert_types(list(multi().recursive_iterator('names', prepend_names=True, separator='->')), list[str])
assert_types(list(multi().recursive_iterator('items', prepend_names=True)),                 list[tuple[str, DataSet | None]])
assert_types(list(multi().recursive_iterator('all', prepend_names=True)),                   list[tuple[tuple[int, ...], str, DataSet | None]])

# Positional order
assert_types(list(multi().recursive_iterator('names', 'nested_last')),                      list[str])
assert_types(list(multi().recursive_iterator('blocks', 'nested_first', skip_none=True)),    list[DataSet])
