from __future__ import annotations

from typing_extensions import reveal_type

from pyvista import MultiBlock

# fmt: off

reveal_type(MultiBlock().recursive_iterator())                          # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator('blocks'))                  # EXPECTED_TYPE: "Iterator[Union[MultiBlock, DataSet, None]]"
reveal_type(MultiBlock().recursive_iterator(skip_none=True))            # EXPECTED_TYPE: "Iterator[Union[DataSet, MultiBlock]]"
reveal_type(MultiBlock().recursive_iterator('blocks', skip_none=True))  # EXPECTED_TYPE: "Iterator[Union[DataSet, MultiBlock]]"
