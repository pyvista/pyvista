from __future__ import annotations

from typing_extensions import reveal_type

from pyvista import MultiBlock

# fmt: off

# Test blocks
reveal_type(MultiBlock().recursive_iterator())                          # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator('blocks'))                  # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(order='nested_first'))      # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(skip_none=True))            # EXPECTED_TYPE: "Iterator[Union[DataSet, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('blocks', skip_none=True))  # EXPECTED_TYPE: "Iterator[Union[DataSet, MultiBlock]]"


reveal_type(MultiBlock().recursive_iterator('names'))                   # EXPECTED_TYPE: "Iterator[str]"

reveal_type(MultiBlock().recursive_iterator('items'))                   # EXPECTED_TYPE: "Iterator[tuple[str, Union[MultiBlock, DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('items', skip_none=True))   # EXPECTED_TYPE: "Iterator[tuple[str, Union[DataSet, MultiBlock]]]"

# Test ids
reveal_type(MultiBlock().recursive_iterator('ids'))                     # EXPECTED_TYPE: "Iterator[tuple[int, ...]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=True))    # EXPECTED_TYPE: "Iterator[tuple[int, ...]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=None))    # EXPECTED_TYPE: "Iterator[tuple[int, ...]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=False))   # EXPECTED_TYPE: "Iterator[int]"

# Test all
reveal_type(MultiBlock().recursive_iterator('all'))                                     # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[MultiBlock, DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', skip_none=True))                     # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, MultiBlock]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=True))                    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[MultiBlock, DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=None))                    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[MultiBlock, DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=False))                   # EXPECTED_TYPE: "Iterator[tuple[int, str, Union[MultiBlock, DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=True, skip_none=True))    # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[DataSet, MultiBlock]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=True, skip_none=False))   # EXPECTED_TYPE: "Iterator[tuple[tuple[int, ...], str, Union[MultiBlock, DataSet, None]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=False, skip_none=True))   # EXPECTED_TYPE: "Iterator[tuple[int, str, Union[DataSet, MultiBlock]]]"
reveal_type(MultiBlock().recursive_iterator('all', nested_ids=False, skip_none=False))  # EXPECTED_TYPE: "Iterator[tuple[int, str, Union[MultiBlock, DataSet, None]]]"
