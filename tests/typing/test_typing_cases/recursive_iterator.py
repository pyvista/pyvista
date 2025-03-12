from __future__ import annotations

from typing_extensions import reveal_type

import pyvista as pv

# Make sure MultiBlock has nested contents with None blocks for the runtime type checks
# For the tests, we also convert the iterator to a `list` so that the type of the
# contents can be inspected dynamically
multi = pv.MultiBlock([pv.PolyData(), None, pv.MultiBlock([pv.PolyData(), None])])

# fmt: off

# Test names
reveal_type(list(multi.recursive_iterator('names')))   # EXPECTED_TYPE: "list[str]"

# Test ids
reveal_type(list(multi.recursive_iterator('ids')))                      # EXPECTED_TYPE: "list[tuple[int, ...]]"
reveal_type(list(multi.recursive_iterator('ids', nested_ids=True)))     # EXPECTED_TYPE: "list[tuple[int, ...]]"
reveal_type(list(multi.recursive_iterator('ids', nested_ids=None)))     # EXPECTED_TYPE: "list[tuple[int, ...]]"
reveal_type(list(multi.recursive_iterator('ids', nested_ids=False)))    # EXPECTED_TYPE: "list[int]"

# Test items
reveal_type(list(multi.recursive_iterator('items')))                                       # EXPECTED_TYPE: "list[tuple[str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('items', node_type='child')))                    # EXPECTED_TYPE: "list[tuple[str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('items', skip_none=True)))                       # EXPECTED_TYPE: "list[tuple[str, DataSet]]"
reveal_type(list(multi.recursive_iterator('items', node_type='child', skip_none=True)))    # EXPECTED_TYPE: "list[tuple[str, DataSet]]"
reveal_type(list(multi.recursive_iterator('items', node_type='parent')))                   # EXPECTED_TYPE: "list[tuple[str, MultiBlock]]"
reveal_type(list(multi.recursive_iterator('items', node_type='parent', skip_none=False)))  # EXPECTED_TYPE: "list[tuple[str, MultiBlock]]"

# Test blocks
reveal_type(list(multi.recursive_iterator()))                          # EXPECTED_TYPE: "list[Union[DataSet, None]]"
reveal_type(list(multi.recursive_iterator('blocks')))                  # EXPECTED_TYPE: "list[Union[DataSet, None]]"
reveal_type(list(multi.recursive_iterator(skip_none=True)))            # EXPECTED_TYPE: "list[DataSet]"
reveal_type(list(multi.recursive_iterator('blocks', skip_none=True)))  # EXPECTED_TYPE: "list[DataSet]"

reveal_type(list(multi.recursive_iterator(node_type='child')))                            # EXPECTED_TYPE: "list[Union[DataSet, None]]"
reveal_type(list(multi.recursive_iterator('blocks', node_type='child')))                  # EXPECTED_TYPE: "list[Union[DataSet, None]]"
reveal_type(list(multi.recursive_iterator(skip_none=True, node_type='child')))            # EXPECTED_TYPE: "list[DataSet]"
reveal_type(list(multi.recursive_iterator('blocks', skip_none=True, node_type='child')))  # EXPECTED_TYPE: "list[DataSet]"

reveal_type(list(multi.recursive_iterator(node_type='parent')))             # EXPECTED_TYPE: "list[MultiBlock]"
reveal_type(list(multi.recursive_iterator('blocks', node_type='parent')))   # EXPECTED_TYPE: "list[MultiBlock]"

# Test all
reveal_type(list(multi.recursive_iterator('all')))                                     # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('all', node_type='child')))                  # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('all', skip_none=True)))                     # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, DataSet]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=True)))                    # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=None)))                    # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=False)))                   # EXPECTED_TYPE: "list[tuple[int, str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=True, skip_none=True)))    # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, DataSet]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=True, skip_none=False)))   # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=False, skip_none=True)))   # EXPECTED_TYPE: "list[tuple[int, str, DataSet]]"
reveal_type(list(multi.recursive_iterator('all', nested_ids=False, skip_none=False)))  # EXPECTED_TYPE: "list[tuple[int, str, Union[DataSet, None]]]"

reveal_type(list(multi.recursive_iterator('all', node_type='parent')))                     # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(list(multi.recursive_iterator('all', node_type='parent', nested_ids=True)))    # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(list(multi.recursive_iterator('all', node_type='parent', nested_ids=None)))    # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(list(multi.recursive_iterator('all', node_type='parent', nested_ids=False)))                   # EXPECTED_TYPE: "list[tuple[int, str, MultiBlock]]"
reveal_type(list(multi.recursive_iterator('all', node_type='parent', nested_ids=True, skip_none=False)))   # EXPECTED_TYPE: "list[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(list(multi.recursive_iterator('all', node_type='parent', nested_ids=False, skip_none=False)))  # EXPECTED_TYPE: "list[tuple[int, str, MultiBlock]]"
