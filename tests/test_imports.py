"""Tests to verify that __all__ definitions are correct in all __init__.py files."""

import importlib
from pathlib import Path

import pytest


def pyvista_submodules():
    """Helper that collects all submodules with __init__.py files.

    Returns
    -------
    list
        List of module names (e.g. ``'pyvista.utilities'``).
    """
    # assume that tests are run from project root
    root = Path('pyvista')
    init_paths = root.rglob('__init__.py')
    submodule_names = [path.parent.as_posix().replace('/', '.') for path in init_paths]
    return submodule_names


@pytest.mark.parametrize('module_name', pyvista_submodules())
def test_star_imports(module_name):
    module = importlib.import_module(module_name)
    dunder_all = getattr(module, '__all__', None)
    if dunder_all is None:
        # nothing to go wrong
        pytest.skip()
    # star imports will fail if __all__ contains any names
    # that aren't actually accessible in the namespace
    missing_names = {attr for attr in dunder_all if not hasattr(module, attr)}
    assert not missing_names
