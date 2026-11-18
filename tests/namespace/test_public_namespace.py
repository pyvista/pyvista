from __future__ import annotations

from pathlib import Path
import sys

import pyvista as pv

# Cleaned data, so imported modules like `np` and `os` are not treated as public API.
namespace_data = Path(__file__).parent / 'namespace-top.txt'
with namespace_data.open() as f:
    namespace = f.read().splitlines()
    # ignore commented data
    namespace = [n.split(', ')[0] for n in namespace if not n.startswith('#')]
if sys.version_info >= (3, 15):  # the import system no longer sets module `__cached__`
    namespace.remove('__cached__')


def test_public_namespace():
    """Every recorded public name is still reachable from the top-level namespace."""
    missing = [name for name in namespace if not hasattr(pv, name)]
    assert not missing, f'Missing from the `pyvista` namespace: {missing}'
