"""Tests for pyvista.ext._autoinherit, including the invariants it exists to hold.

Every public member reachable on a documented class must be documented exactly once,
and no member VTK implements may be documented at all. ``test_api_coverage.py`` checks
the same thing one level up, for the classes themselves.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum

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


def test_scan_skips_options_other_than_toctree():
    text = """
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary
   :template: enum
   :nosignatures:

   CellType
"""
    assert autoinherit._scan(text) == {'pyvista.CellType'}


def test_resolve_returns_none_for_a_missing_attribute():
    assert autoinherit._resolve('pyvista.NotAnAttribute') is None


def test_resolve_returns_none_when_nothing_imports():
    assert autoinherit._resolve('not_a_module_pyvista_would_ship') is None


def test_documented_classes_needs_the_extension_to_be_loaded(monkeypatch):
    monkeypatch.setattr(autoinherit, '_srcdir', None)
    monkeypatch.setattr(autoinherit, '_documented', None)
    with pytest.raises(RuntimeError, match='is not loaded'):
        autoinherit._documented_classes()


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


def test_home_falls_through_to_the_class_when_every_provider_is_undocumented(monkeypatch):
    class Undocumented:
        shared = 1

    class Left(Undocumented):
        pass

    class Right(Undocumented):
        pass

    Undocumented.__module__ = 'pyvista.fake'  # or _home reads it as VTK's and declines

    monkeypatch.setattr(autoinherit, '_documented', {Left: 'pkg.Left', Right: 'pkg.Right'})
    # Nothing basal is documented, so each subclass has to hold its own page -- which
    # is the duplication that documenting the base collapses.
    assert autoinherit._home(Left, 'shared') is Left
    assert autoinherit._home(Right, 'shared') is Right

    monkeypatch.setattr(
        autoinherit,
        '_documented',
        {Undocumented: 'pkg.Undocumented', Left: 'pkg.Left', Right: 'pkg.Right'},
    )
    assert autoinherit._home(Left, 'shared') is Undocumented
    assert autoinherit._home(Right, 'shared') is Undocumented


def test_the_bases_that_collapse_duplicates_are_documented(documented):
    """Guard the doc entries that ``test_no_implementation_is_documented_twice`` needs."""
    assert _PointSet in documented
    assert Grid in documented


def test_home_is_none_for_a_member_vtk_implements():
    # pv.ImageData().min_spatial_dimension raises PyVistaAttributeError.
    assert autoinherit._home(pv.ImageData, 'min_spatial_dimension') is None
    assert autoinherit._home(pv.ImageData, 'GetDimensions') is None


def test_own_members_excludes_what_a_documented_base_already_documents():
    own = autoinherit.own_members('pyvista', 'ImageData', _members(pv.ImageData))
    assert 'offset' in own  # ImageData defines it
    assert 'dimensions' not in own  # documented as Grid.dimensions
    assert 'dimensionality' not in own  # documented as DataSet.dimensionality
    assert 'contour' not in own  # documented as DataSetFilters.contour
    assert 'min_spatial_dimension' not in own  # implemented by VTK


def test_inherited_member_rows_are_sorted_by_member_name():
    rows = autoinherit.inherited_member_rows('pyvista', 'PolyData', _members(pv.PolyData))
    members = [label.rsplit('.', 1)[1] for label, _, _ in rows]
    assert members == sorted(members)
    assert 'DataObject.copy' in {label for label, _, _ in rows}
    # Filters are split out, so they are not among the inherited rows.
    assert 'DataSetFilters.contour' not in {label for label, _, _ in rows}


def test_inherited_member_rows_label_the_class_without_its_module():
    rows = autoinherit.inherited_member_rows('pyvista', 'Volume', _members(pv.Volume))
    by_label = {label: (target, summary) for label, target, summary in rows}
    # The label keeps the class but drops its module; the link target is complete.
    assert (
        by_label['_BoundsSizeMixin.bounds_size'][0]
        == 'pyvista.core.utilities.misc._BoundsSizeMixin.bounds_size'
    )
    target, summary = by_label['Prop3D.rotate_x']
    assert target == 'pyvista.Prop3D.rotate_x'
    assert summary == 'Rotate the entity about the x-axis.'


def test_summary_reads_the_descriptor_rather_than_evaluating_it():
    # extensions is a _classproperty, so getattr would evaluate it and lose the docstring.
    assert autoinherit._summary(pv.BaseReader, 'extensions').startswith('Return the file')


def test_filters_are_split_out_of_the_inherited_rows():
    """Filters are 142 of the 226 members PolyData inherits, so they get their own table."""
    members = _members(pv.PolyData)
    inherited = autoinherit.inherited_member_rows('pyvista', 'PolyData', members)
    filters = autoinherit.filter_member_rows('pyvista', 'PolyData', members)
    assert 'DataSetFilters.contour' in {label for label, _, _ in filters}
    assert 'DataSet.bounds' in {label for label, _, _ in inherited}
    assert not {label for label, _, _ in inherited} & {label for label, _, _ in filters}


def test_a_class_that_mixes_in_no_filters_has_no_filter_rows():
    assert autoinherit.filter_member_rows('pyvista', 'Camera', _members(pv.Camera)) == []


def test_is_filter_recognises_only_the_filter_classes():
    from pyvista.core.filters.data_set import DataSetFilters

    assert autoinherit._is_filter(DataSetFilters)
    assert not autoinherit._is_filter(pv.DataSet)


def test_a_member_is_either_owned_or_inherited_never_both():
    for cls, name in [(pv.PolyData, 'PolyData'), (pv.Plotter, 'Plotter')]:
        members = _members(cls)
        own = set(autoinherit.own_members('pyvista', name, members))
        inherited = {
            label.rsplit('.', 1)[1]
            for label, _, _ in autoinherit.inherited_member_rows('pyvista', name, members)
            + autoinherit.filter_member_rows('pyvista', name, members)
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
        elsewhere = autoinherit.inherited_member_rows(
            module, objname, members
        ) + autoinherit.filter_member_rows(module, objname, members)
        for _, target, _ in elsewhere:
            if target not in pages:
                missing.add(f'{docname}.{target.rsplit(".", 1)[1]}')
            homed.add(target.rsplit('.', 1)[1])
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


def test_no_implementation_is_documented_twice(documented):
    """One implementation, one page: the reason undocumented bases are documented.

    A member whose every provider is undocumented falls back to the class itself, so
    two public classes sharing an undocumented base each get a page for it. Listing
    that base in ``doc/source/api/`` is what collapses them back to one.
    """
    homes: dict[tuple[type, str], set[str]] = defaultdict(set)
    for cls, docname in documented.items():
        if issubclass(cls, Enum):
            continue  # rendered by enum.rst, which does not route members
        module, _, objname = docname.rpartition('.')
        for item in autoinherit.own_members(module, objname, _members(cls)):
            homes[(autoinherit._provider(cls, item), item)].add(docname)

    duplicated = {
        f'{provider.__qualname__}.{item}': sorted(pages)
        for (provider, item), pages in homes.items()
        if len(pages) > 1
    }
    assert not duplicated, (
        f'{len(duplicated)} implementation(s) have more than one page. Document the '
        f'class that defines each one under doc/source/api/ so every class that '
        f'inherits it links there instead:\n  '
        + '\n  '.join(f'{name}: {pages}' for name, pages in sorted(duplicated.items()))
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
    for helper in ('own_members', 'inherited_member_rows'):
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
