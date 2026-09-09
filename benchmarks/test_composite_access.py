"""Benchmarks for building and reading a composite dataset."""

from __future__ import annotations

import operator

import pyvista as pv


def test_multiblock_construct(blocks, benchmark):
    """Build a composite from a list of meshes."""
    assert benchmark(pv.MultiBlock, blocks).n_blocks == len(blocks)


def test_multiblock_n_blocks(multiblock, benchmark):
    """Read the block count of a composite."""
    assert benchmark(operator.attrgetter('n_blocks'), multiblock)
