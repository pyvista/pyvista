from __future__ import annotations

from typing_extensions import reveal_type

from pyvista import MultiBlock

# fmt: off

reveal_type(MultiBlock().recursive_iterator())                          # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator('blocks'))                  # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(order='nested_first'))      # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(skip_none=True))            # EXPECTED_TYPE: "Iterator[Union[DataSet, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('blocks', skip_none=True))  # EXPECTED_TYPE: "Iterator[Union[DataSet, MultiBlock]]"

reveal_type(MultiBlock().recursive_iterator('names'))                   # EXPECTED_TYPE: "Iterator[str]"
reveal_type(MultiBlock().recursive_iterator('ids'))                     # EXPECTED_TYPE: "Iterator[tuple[int]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=True))    # EXPECTED_TYPE: "Iterator[tuple[int]]"
reveal_type(MultiBlock().recursive_iterator('ids', nested_ids=False))   # EXPECTED_TYPE: "Iterator[int]"
