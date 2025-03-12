from __future__ import annotations

from typing_extensions import reveal_type

from pyvista import MultiBlock

# fmt: off

# Test names
reveal_type(MultiBlock().recursive_iterator('names'))   # EXPECTED_TYPE: "Iterator[str]"

# Test ids
reveal_type(MultiBlock().recursive_iterator('ids'))                     # EXPECTED_TYPE: "Iterator[tuple[int, ...]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=True))    # EXPECTED_TYPE: "Iterator[tuple[int, ...]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=None))    # EXPECTED_TYPE: "Iterator[tuple[int, ...]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=False))   # EXPECTED_TYPE: "Iterator[int]"

# Test items
reveal_type(MultiBlock().recursive_iterator('items'))                                       # EXPECTED_TYPE: "Iterator[tuple[str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('items', node_type='child'))                    # EXPECTED_TYPE: "Iterator[tuple[str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('items', skip_none=True))                       # EXPECTED_TYPE: "Iterator[tuple[str, DataSet]]"
reveal_type(MultiBlock().recursive_iterator('items', node_type='child',skip_none=True))     # EXPECTED_TYPE: "Iterator[tuple[str, DataSet]]"
reveal_type(MultiBlock().recursive_iterator('items', node_type='parent'))                   # EXPECTED_TYPE: "Iterator[tuple[str, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('items', node_type='parent', skip_none=False))  # EXPECTED_TYPE: "Iterator[tuple[str, MultiBlock]]"

# Test blocks
reveal_type(MultiBlock().recursive_iterator())                          # EXPECTED_TYPE: "Iterator[Union[DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator('blocks'))                  # EXPECTED_TYPE: "Iterator[Union[DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(skip_none=True))            # EXPECTED_TYPE: "Iterator[DataSet]"
reveal_type(MultiBlock().recursive_iterator('blocks', skip_none=True))  # EXPECTED_TYPE: "Iterator[DataSet]"

reveal_type(MultiBlock().recursive_iterator(node_type='child'))                            # EXPECTED_TYPE: "Iterator[Union[DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator('blocks', node_type='child'))                  # EXPECTED_TYPE: "Iterator[Union[DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(skip_none=True, node_type='child'))            # EXPECTED_TYPE: "Iterator[DataSet]"
reveal_type(MultiBlock().recursive_iterator('blocks', skip_none=True, node_type='child'))  # EXPECTED_TYPE: "Iterator[DataSet]"

reveal_type(MultiBlock().recursive_iterator(node_type='parent'))             # EXPECTED_TYPE: "Iterator[MultiBlock]"
reveal_type(MultiBlock().recursive_iterator('blocks', node_type='parent'))   # EXPECTED_TYPE: "Iterator[MultiBlock]"

# Test all
reveal_type(MultiBlock().recursive_iterator('all'))                                     # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', node_type='child'))                  # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', skip_none=True))                     # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, DataSet]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=True))                    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=None))                    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=False))                   # EXPECTED_TYPE: "Iterator[tuple[int, str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=True, skip_none=True))    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, DataSet]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=True, skip_none=False))   # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=False, skip_none=True))   # EXPECTED_TYPE: "Iterator[tuple[int, str, DataSet]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=False, skip_none=False))  # EXPECTED_TYPE: "Iterator[tuple[int, str, Union[DataSet, None]]]"

reveal_type(MultiBlock().recursive_iterator('all', node_type='parent'))                     # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('all', node_type='parent', nested_ids=True))    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('all', node_type='parent', nested_ids=None))    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('all', node_type='parent', nested_ids=False))                   # EXPECTED_TYPE: "Iterator[tuple[int, str, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('all', node_type='parent', nested_ids=True, skip_none=False))   # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('all', node_type='parent', nested_ids=False, skip_none=False))  # EXPECTED_TYPE: "Iterator[tuple[int, str, MultiBlock]]"
