import pathlib

import pytest

import pyvista

# Use cleaned data to avoid things like `np`, `os`, etc
# This prevents testing against things that are not intended
# to be in the public namespace
namespace_data = pathlib.Path(__file__).parent / 'namespace-top.txt'
with open(namespace_data) as f:
    namespace = f.read().splitlines()
    # ignore commented data
    namespace = [n.split(', ')[0] for n in namespace if not n.startswith('#')]


@pytest.mark.parametrize("name", namespace)
def test_public_namespace(name):
    assert hasattr(pyvista, name)
