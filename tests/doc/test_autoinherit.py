"""Tests for pyvista.ext._autoinherit, including the invariants it exists to hold.

Every public member reachable on a documented class must be documented exactly once,
and no member VTK implements may be documented at all. ``test_api_coverage.py`` checks
the same thing one level up, for the classes themselves.
"""

from __future__ import annotations

import pytest
from sphinx.ext.autodoc.importer import get_class_members
from sphinx.util.inspect import safe_getattr

import pyvista as pv
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.grid import Grid
from pyvista.core.pointset import _PointSet
from pyvista.ext import _autoinherit as autoinherit
from tests.conftest import PYVISTA_ROOT_DIR

# Kept in step with ``autosummary_context['skipmethods']`` in doc/source/conf.py.
SKIP = ('__init__', 'override', 'check_attribute')


@pytest.fixture(scope='module', autouse=True)
def _srcdir():
    """Point the registry at the real doc sources, the way ``setup()`` does."""
    autoinherit._srcdir = PYVISTA_ROOT_DIR / 'doc' / 'source'
    autoinherit._documented = None
    yield
    autoinherit._documented = None


@pytest.fixture(scope='module')
def documented():
    """Return ``{class: dotted name}`` for every class with an autosummary page."""
    return autoinherit._documented_classes()


def _members(cls):
    """Return the member names class.rst passes to the helpers, in its own order."""
    names = get_class_members(cls, None, safe_getattr)
    return [name for name in names if name not in SKIP]


def test_scan_requires_toctree():
    # Only a ``:toctree:`` block generates pages, so only it may register a class.
    with_toctree = """
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   PolyData
"""
    without = with_toctree.replace('   :toctree: _autosummary\n', '')
    assert autoinherit._scan(with_toctree) == {'pyvista.PolyData'}
    assert autoinherit._scan(without) == set()


def test_scan_qualifies_names_with_the_current_module():
    text = """
.. currentmodule:: pyvista.core.dataset

.. autosummary::
   :toctree: _autosummary

   ~DataSet
   pyvista.core.dataset.ActiveArrayInfoTuple
"""
    assert autoinherit._scan(text) == {
        'pyvista.core.dataset.DataSet',
        'pyvista.core.dataset.ActiveArrayInfoTuple',
    }


def test_provider_finds_the_class_that_defines_the_member():
    assert autoinherit._provider(pv.PolyData, 'remove_cells') is _PointSet
    assert autoinherit._provider(pv.ImageData, 'dimensions') is Grid
    assert autoinherit._provider(pv.PolyData, 'faces') is pv.PolyData


def test_provider_falls_back_to_a_bare_annotation():
    # A dataclass field binds nothing, so ``__dict__`` alone would miss it.
    assert 'time' not in pv.PVDDataSet.__dict__
    assert autoinherit._provider(pv.PVDDataSet, 'time') is pv.PVDDataSet


def test_provider_prefers_a_real_binding_over_an_annotation():
    class Base:
        value = 1

    class Child(Base):
        value: int

    assert autoinherit._provider(Child, 'value') is Base


def test_home_is_the_most_basal_documented_class():
    # DataSetFilters is documented, so it -- not each dataset -- owns the filter's page.
    assert autoinherit._home(pv.PolyData, 'contour') is DataSetFilters
    assert autoinherit._home(pv.ImageData, 'contour') is DataSetFilters


def test_home_falls_through_to_the_class_when_every_provider_is_undocumented(documented):
    assert _PointSet not in documented
    assert Grid not in documented
    assert autoinherit._home(pv.PolyData, 'remove_cells') is pv.PolyData
    assert autoinherit._home(pv.ImageData, 'dimensions') is pv.ImageData


def test_home_is_none_for_a_member_vtk_implements():
    # pv.ImageData().min_spatial_dimension raises PyVistaAttributeError.
    assert autoinherit._home(pv.ImageData, 'min_spatial_dimension') is None
    assert autoinherit._home(pv.ImageData, 'GetDimensions') is None


def test_own_members_excludes_what_a_documented_base_already_documents():
    own = autoinherit.own_members('pyvista', 'ImageData', _members(pv.ImageData))
    assert 'dimensions' in own  # Grid is undocumented, so ImageData is the home
    assert 'dimensionality' not in own  # documented as DataSet.dimensionality
    assert 'contour' not in own
    assert 'min_spatial_dimension' not in own


def test_inherited_member_groups_are_ordered_by_the_mro():
    groups = autoinherit.inherited_member_groups('pyvista', 'PolyData', _members(pv.PolyData))
    names = [f'{module}.{name}' for module, name, _ in groups]
    assert names.index('pyvista.DataSet') < names.index('pyvista.DataObject')
    contour = next(items for _, name, items in groups if name == 'DataSetFilters')
    assert 'contour' in contour


def test_a_member_is_either_owned_or_inherited_never_both():
    for cls, name in [(pv.PolyData, 'PolyData'), (pv.Plotter, 'Plotter')]:
        members = _members(cls)
        own = set(autoinherit.own_members('pyvista', name, members))
        inherited = {
            item
            for _, _, items in autoinherit.inherited_member_groups('pyvista', name, members)
            for item in items
        }
        assert not own & inherited


def test_every_reachable_member_has_exactly_one_page(documented):
    """No documented class may reach a PyVista member that no page documents."""
    pages: set[str] = set()
    for cls, docname in documented.items():
        module, _, objname = docname.rpartition('.')
        for item in autoinherit.own_members(module, objname, _members(cls)):
            pages.add(f'{docname}.{item}')

    missing: set[str] = set()
    for cls, docname in documented.items():
        module, _, objname = docname.rpartition('.')
        members = _members(cls)
        homed = set(autoinherit.own_members(module, objname, members))
        for base_module, base_name, items in autoinherit.inherited_member_groups(
            module, objname, members
        ):
            for item in items:
                if f'{base_module}.{base_name}.{item}' not in pages:
                    missing.add(f'{docname}.{item}')
                homed.add(item)
        for item in autoinherit._candidates(members):
            # Asks _provider, not _home: a member is exempt only because VTK or the
            # standard library implements it, never because _home declined to place it.
            provider = autoinherit._provider(cls, item)
            if provider is None or not provider.__module__.startswith('pyvista'):
                continue
            if item not in homed:
                missing.add(f'{docname}.{item}')

    assert not missing, (
        f'{len(missing)} member(s) reachable on a documented class have no page:\n  '
        + '\n  '.join(sorted(missing))
    )


def test_no_vtk_member_is_documented(documented):
    """A VTK member documented as PyVista API raises PyVistaAttributeError on access."""
    leaked: set[str] = set()
    for cls, docname in documented.items():
        module, _, objname = docname.rpartition('.')
        for item in autoinherit.own_members(module, objname, _members(cls)):
            provider = autoinherit._provider(cls, item)
            if provider is not None and not provider.__module__.startswith('pyvista'):
                leaked.add(f'{docname}.{item}')
    assert not leaked, 'VTK members documented as PyVista API:\n  ' + '\n  '.join(sorted(leaked))


def test_class_template_calls_only_helpers_that_exist():
    """`autosummary` overwrites its own names in the template namespace, so guard them."""
    template = (
        PYVISTA_ROOT_DIR / 'doc' / 'source' / '_templates' / 'autosummary' / 'class.rst'
    ).read_text(encoding='utf-8')
    for helper in ('own_members', 'inherited_member_groups'):
        assert f'{helper}(module, objname,' in template
        assert callable(getattr(autoinherit, helper))
    # ``ns['inherited_members']`` is a set of names by the time the template renders.
    assert 'inherited_members(' not in template


def test_helpers_reject_a_non_class():
    with pytest.raises(TypeError, match='is not a class'):
        autoinherit.own_members('pyvista', 'wrap', [])


def test_setup_records_the_source_directory():
    class _App:
        srcdir = PYVISTA_ROOT_DIR / 'doc' / 'source'

    metadata = autoinherit.setup(_App())
    assert metadata['parallel_read_safe']
    assert autoinherit._srcdir == _App.srcdir
