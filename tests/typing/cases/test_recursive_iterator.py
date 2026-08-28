"""Static and runtime typing cases for :meth:`pyvista.MultiBlock.recursive_iterator`."""

from __future__ import annotations

import pytest
from typing_extensions import assert_type

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock
from tests.typing.type_assertions import assert_runtime_type


@pytest.fixture
def multi() -> MultiBlock:
    """Nested `MultiBlock` with `None` blocks, so the runtime checks see both."""
    return pv.MultiBlock([pv.PolyData(), None, pv.MultiBlock([pv.PolyData(), None])])


def test_names(multi: MultiBlock) -> None:
    """`names` yields block names."""
    result = list(multi.recursive_iterator('names'))
    assert_type(result, list[str])
    assert_runtime_type(result, list[str])


def test_ids(multi: MultiBlock) -> None:
    """`ids` yields nested ids by default."""
    result = list(multi.recursive_iterator('ids'))
    assert_type(result, list[tuple[int, ...]])
    assert_runtime_type(result, list[tuple[int, ...]])


def test_ids_nested(multi: MultiBlock) -> None:
    """`nested_ids=True` yields nested ids."""
    result = list(multi.recursive_iterator('ids', nested_ids=True))
    assert_type(result, list[tuple[int, ...]])
    assert_runtime_type(result, list[tuple[int, ...]])


def test_ids_nested_none(multi: MultiBlock) -> None:
    """`nested_ids=None` yields nested ids."""
    result = list(multi.recursive_iterator('ids', nested_ids=None))
    assert_type(result, list[tuple[int, ...]])
    assert_runtime_type(result, list[tuple[int, ...]])


def test_ids_flat(multi: MultiBlock) -> None:
    """`nested_ids=False` flattens the ids to plain integers."""
    result = list(multi.recursive_iterator('ids', nested_ids=False))
    assert_type(result, list[int])
    assert_runtime_type(result, list[int])


def test_items(multi: MultiBlock) -> None:
    """`items` yields name/block pairs, and blocks may be `None`."""
    result = list(multi.recursive_iterator('items'))
    assert_type(result, list[tuple[str, DataSet | None]])
    assert_runtime_type(result, list[tuple[str, DataSet | None]])


def test_items_child(multi: MultiBlock) -> None:
    """`node_type='child'` is the default for `items`."""
    result = list(multi.recursive_iterator('items', node_type='child'))
    assert_type(result, list[tuple[str, DataSet | None]])
    assert_runtime_type(result, list[tuple[str, DataSet | None]])


def test_items_skip_none(multi: MultiBlock) -> None:
    """`skip_none=True` drops `None` from the block type."""
    result = list(multi.recursive_iterator('items', skip_none=True))
    assert_type(result, list[tuple[str, DataSet]])
    assert_runtime_type(result, list[tuple[str, DataSet]])


def test_items_child_skip_none(multi: MultiBlock) -> None:
    """`node_type='child'` with `skip_none=True` drops `None`."""
    result = list(multi.recursive_iterator('items', node_type='child', skip_none=True))
    assert_type(result, list[tuple[str, DataSet]])
    assert_runtime_type(result, list[tuple[str, DataSet]])


def test_items_parent(multi: MultiBlock) -> None:
    """`node_type='parent'` yields `MultiBlock` nodes, which are never `None`."""
    result = list(multi.recursive_iterator('items', node_type='parent'))
    assert_type(result, list[tuple[str, MultiBlock]])
    assert_runtime_type(result, list[tuple[str, MultiBlock]])


def test_items_parent_keep_none(multi: MultiBlock) -> None:
    """`skip_none=False` does not add `None` for parent nodes."""
    result = list(multi.recursive_iterator('items', node_type='parent', skip_none=False))
    assert_type(result, list[tuple[str, MultiBlock]])
    assert_runtime_type(result, list[tuple[str, MultiBlock]])


def test_blocks_default(multi: MultiBlock) -> None:
    """`blocks` is the default mode."""
    result = list(multi.recursive_iterator())
    assert_type(result, list[DataSet | None])
    assert_runtime_type(result, list[DataSet | None])


def test_blocks(multi: MultiBlock) -> None:
    """`blocks` yields the blocks themselves, which may be `None`."""
    result = list(multi.recursive_iterator('blocks'))
    assert_type(result, list[DataSet | None])
    assert_runtime_type(result, list[DataSet | None])


def test_blocks_default_skip_none(multi: MultiBlock) -> None:
    """`skip_none=True` drops `None` from the default mode."""
    result = list(multi.recursive_iterator(skip_none=True))
    assert_type(result, list[DataSet])
    assert_runtime_type(result, list[DataSet])


def test_blocks_skip_none(multi: MultiBlock) -> None:
    """`skip_none=True` drops `None` from the block type."""
    result = list(multi.recursive_iterator('blocks', skip_none=True))
    assert_type(result, list[DataSet])
    assert_runtime_type(result, list[DataSet])


def test_blocks_default_child(multi: MultiBlock) -> None:
    """`node_type='child'` is the default for the default mode."""
    result = list(multi.recursive_iterator(node_type='child'))
    assert_type(result, list[DataSet | None])
    assert_runtime_type(result, list[DataSet | None])


def test_blocks_child(multi: MultiBlock) -> None:
    """`node_type='child'` is the default for `blocks`."""
    result = list(multi.recursive_iterator('blocks', node_type='child'))
    assert_type(result, list[DataSet | None])
    assert_runtime_type(result, list[DataSet | None])


def test_blocks_default_child_skip_none(multi: MultiBlock) -> None:
    """`node_type='child'` with `skip_none=True` drops `None`."""
    result = list(multi.recursive_iterator(skip_none=True, node_type='child'))
    assert_type(result, list[DataSet])
    assert_runtime_type(result, list[DataSet])


def test_blocks_child_skip_none(multi: MultiBlock) -> None:
    """`blocks` with `node_type='child'` and `skip_none=True` drops `None`."""
    result = list(multi.recursive_iterator('blocks', skip_none=True, node_type='child'))
    assert_type(result, list[DataSet])
    assert_runtime_type(result, list[DataSet])


def test_blocks_default_parent(multi: MultiBlock) -> None:
    """`node_type='parent'` yields `MultiBlock` nodes."""
    result = list(multi.recursive_iterator(node_type='parent'))
    assert_type(result, list[MultiBlock])
    assert_runtime_type(result, list[MultiBlock])


def test_blocks_parent(multi: MultiBlock) -> None:
    """`blocks` with `node_type='parent'` yields `MultiBlock` nodes."""
    result = list(multi.recursive_iterator('blocks', node_type='parent'))
    assert_type(result, list[MultiBlock])
    assert_runtime_type(result, list[MultiBlock])


def test_all(multi: MultiBlock) -> None:
    """`all` yields id/name/block triplets."""
    result = list(multi.recursive_iterator('all'))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])


def test_all_child(multi: MultiBlock) -> None:
    """`node_type='child'` is the default for `all`."""
    result = list(multi.recursive_iterator('all', node_type='child'))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])


def test_all_skip_none(multi: MultiBlock) -> None:
    """`skip_none=True` drops `None` from the triplet's block."""
    result = list(multi.recursive_iterator('all', skip_none=True))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet]])


def test_all_nested_ids(multi: MultiBlock) -> None:
    """`nested_ids=True` yields nested ids."""
    result = list(multi.recursive_iterator('all', nested_ids=True))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])


def test_all_nested_ids_none(multi: MultiBlock) -> None:
    """`nested_ids=None` yields nested ids."""
    result = list(multi.recursive_iterator('all', nested_ids=None))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])


def test_all_flat_ids(multi: MultiBlock) -> None:
    """`nested_ids=False` flattens the triplet's id."""
    result = list(multi.recursive_iterator('all', nested_ids=False))
    assert_type(result, list[tuple[int, str, DataSet | None]])
    assert_runtime_type(result, list[tuple[int, str, DataSet | None]])


def test_all_nested_ids_skip_none(multi: MultiBlock) -> None:
    """`nested_ids=True` and `skip_none=True` combine."""
    result = list(multi.recursive_iterator('all', nested_ids=True, skip_none=True))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet]])


def test_all_nested_ids_keep_none(multi: MultiBlock) -> None:
    """`nested_ids=True` and `skip_none=False` combine."""
    result = list(multi.recursive_iterator('all', nested_ids=True, skip_none=False))
    assert_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, DataSet | None]])


def test_all_flat_ids_skip_none(multi: MultiBlock) -> None:
    """`nested_ids=False` and `skip_none=True` combine."""
    result = list(multi.recursive_iterator('all', nested_ids=False, skip_none=True))
    assert_type(result, list[tuple[int, str, DataSet]])
    assert_runtime_type(result, list[tuple[int, str, DataSet]])


def test_all_flat_ids_keep_none(multi: MultiBlock) -> None:
    """`nested_ids=False` and `skip_none=False` combine."""
    result = list(multi.recursive_iterator('all', nested_ids=False, skip_none=False))
    assert_type(result, list[tuple[int, str, DataSet | None]])
    assert_runtime_type(result, list[tuple[int, str, DataSet | None]])


def test_all_parent(multi: MultiBlock) -> None:
    """`node_type='parent'` yields `MultiBlock` nodes in the triplet."""
    result = list(multi.recursive_iterator('all', node_type='parent'))
    assert_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])


def test_all_parent_nested_ids(multi: MultiBlock) -> None:
    """`node_type='parent'` with `nested_ids=True`."""
    result = list(multi.recursive_iterator('all', node_type='parent', nested_ids=True))
    assert_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])


def test_all_parent_nested_ids_none(multi: MultiBlock) -> None:
    """`node_type='parent'` with `nested_ids=None`."""
    result = list(multi.recursive_iterator('all', node_type='parent', nested_ids=None))
    assert_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])


def test_all_parent_flat_ids(multi: MultiBlock) -> None:
    """`node_type='parent'` with `nested_ids=False`."""
    result = list(multi.recursive_iterator('all', node_type='parent', nested_ids=False))
    assert_type(result, list[tuple[int, str, MultiBlock]])
    assert_runtime_type(result, list[tuple[int, str, MultiBlock]])


def test_all_parent_nested_ids_keep_none(multi: MultiBlock) -> None:
    """`node_type='parent'` with `nested_ids=True` and `skip_none=False`."""
    result = list(
        multi.recursive_iterator('all', node_type='parent', nested_ids=True, skip_none=False)
    )
    assert_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])
    assert_runtime_type(result, list[tuple[tuple[int, ...], str, MultiBlock]])


def test_all_parent_flat_ids_keep_none(multi: MultiBlock) -> None:
    """`node_type='parent'` with `nested_ids=False` and `skip_none=False`."""
    result = list(
        multi.recursive_iterator('all', node_type='parent', nested_ids=False, skip_none=False)
    )
    assert_type(result, list[tuple[int, str, MultiBlock]])
    assert_runtime_type(result, list[tuple[int, str, MultiBlock]])
