"""Tests for the build manifest schema, the backend registry and the shipped VTK manifest."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from pyvista._build import ManifestError
from pyvista._build import load_manifest
from pyvista._build.manifest import OS_NAMES
from pyvista._build.manifest import Option
from pyvista._build.manifest import cmake_entries
from pyvista._build.manifest import cmake_unset
from pyvista._build.manifest import option_environment
from pyvista._build.manifest import parse_manifest
from pyvista._build.manifest import resolve_groups
from pyvista._build.manifest import resolve_options
from pyvista._build.registry import find_manifest
from pyvista._build.registry import known_backends
from pyvista._build.registry import shipped_manifest
from pyvista.core._vtk_utilities import _MAX_SUPPORTED_VTK_VERSION
from pyvista.core._vtk_utilities import VersionInfo

if TYPE_CHECKING:
    from pathlib import Path


MINIMAL_MANIFEST = {
    'manifest_version': 1,
    'backend': {'name': 'demo', 'default_ref': 'v1.0'},
    'groups': {
        'a': {'help': 'A', 'default': True, 'cmake': ['A=ON'], 'cmake_off': ['A=OFF']},
        'b': {'help': 'B', 'default': False, 'cmake': ['B=ON'], 'conflicts': ['c']},
        'c': {
            'help': 'C',
            'default': False,
            'cmake': ['C=ON'],
            'conflicts': ['b'],
            'os': ['linux'],
        },
        'core': {'help': 'Core', 'required': True, 'cmake': ['CORE=ON']},
    },
    'options': {
        'lto': {'help': 'LTO', 'default': False, 'cmake': 'LTO={value}'},
        'smp': {
            'help': 'SMP',
            'default': 'STDThread',
            'cmake': 'SMP={value}',
            'choices': ['STDThread', 'TBB'],
        },
        'jobs': {'help': 'Jobs', 'default': 2, 'cmake': 'JOBS={value}'},
    },
    'hooks': {'post_build': ['{python} prune.py {build_dir}']},
}


@pytest.fixture
def demo():
    return parse_manifest(MINIMAL_MANIFEST)


def test_shipped_vtk_manifest_loads():
    path = shipped_manifest('vtk')
    assert path is not None
    manifest = load_manifest(path, default_ref='v1')
    assert manifest.name == 'vtk'
    assert 'rendering' in manifest.groups
    assert manifest.groups['headless'].os == ('linux',)
    assert find_manifest('vtk', None).path == path
    with pytest.raises(ManifestError, match='default_ref'):
        load_manifest(path)


def test_vtk_default_ref_is_max_supported():
    maximum = VersionInfo._format(_MAX_SUPPORTED_VTK_VERSION)
    assert find_manifest('vtk', None).default_ref == f'v{maximum}'


def test_load_manifest_default_ref(tmp_path: Path):
    path = tmp_path / 'pyvista-build.toml'
    path.write_text('manifest_version = 1\n[backend]\nname = "x"\ndefault_ref = "r"\n')
    assert load_manifest(path, default_ref='other').default_ref == 'r'
    path.write_text('manifest_version = 1\nbackend = 1\n')
    with pytest.raises(ManifestError, match=re.escape(f'{path}: manifest:')):
        load_manifest(path, default_ref='other')


def test_import_name_defaults_to_the_backend_name():
    assert parse_manifest(MINIMAL_MANIFEST).import_name == 'demo'
    backend = dict(MINIMAL_MANIFEST['backend'], import_name='demo_kit.core')
    assert parse_manifest(dict(MINIMAL_MANIFEST, backend=backend)).import_name == 'demo_kit.core'
    for bad in ('demo-kit', '', 'a..b'):
        backend = dict(MINIMAL_MANIFEST['backend'], import_name=bad)
        with pytest.raises(ManifestError, match="'import_name' must be a Python module name"):
            parse_manifest(dict(MINIMAL_MANIFEST, backend=backend))


@pytest.mark.parametrize('os_name', OS_NAMES)
def test_vtk_manifest_default_entries(os_name):
    manifest = find_manifest('vtk', None)
    groups = resolve_groups(manifest, os_name)
    entries = cmake_entries(
        manifest, os_name, groups=groups, options=resolve_options(manifest, None)
    )
    ci_flags = {
        'VTK_WHEEL_BUILD=ON',
        'VTK_WRAP_PYTHON=ON',
        'VTK_GROUP_ENABLE_Rendering=WANT',
        'VTK_BUILD_TESTING=OFF',
        'VTK_BUILD_DOCUMENTATION=OFF',
        'VTK_BUILD_EXAMPLES=OFF',
        'VTK_USE_PCH=OFF',
        'VTK_DISPATCH_SOA_ARRAYS=ON',
        'VTK_WRAP_SERIALIZATION=ON',
        'VTK_BUILD_PYI_FILES=ON',
        'VTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=WANT',
        'VTK_MODULE_ENABLE_VTK_RenderingMatplotlib=WANT',
        'VTK_MODULE_ENABLE_VTK_IOImport=WANT',
        'VTK_MODULE_ENABLE_VTK_IOParallelExodus=WANT',
        'VTK_MODULE_ENABLE_VTK_IOXdmf2=WANT',
        'VTK_MODULE_ENABLE_VTK_WebCore=WANT',
        'VTK_MODULE_ENABLE_VTK_WebGLExporter=WANT',
        'VTK_MODULE_ENABLE_VTK_WebPython=WANT',
        'VTK_SMP_IMPLEMENTATION_TYPE=Sequential',
    }
    if os_name == 'macos':
        ci_flags.add('CMAKE_OSX_DEPLOYMENT_TARGET=11.0')
    assert set(entries) == ci_flags
    assert not any('EGL' in e or 'OSMESA' in e for e in entries)


def test_resolve_groups_defaults(demo):
    assert resolve_groups(demo, 'linux') == ['a', 'core']
    assert resolve_groups(demo, 'linux', minimal=True) == ['core']
    assert resolve_groups(demo, 'linux', with_=['c']) == ['a', 'c', 'core']
    assert resolve_groups(demo, 'linux', with_=['b'], without=['a']) == ['b', 'core']


def test_resolve_groups_errors(demo):
    with pytest.raises(ManifestError, match='conflicts'):
        resolve_groups(demo, 'linux', with_=['b', 'c'])
    with pytest.raises(ManifestError, match='required'):
        resolve_groups(demo, 'linux', without=['core'])
    with pytest.raises(ManifestError, match='Unknown group'):
        resolve_groups(demo, 'linux', with_=['nope'])
    with pytest.raises(ManifestError, match='not available on macos'):
        resolve_groups(demo, 'macos', with_=['c'])


def test_cmake_entries_use_off_list(demo):
    assert cmake_entries(demo, 'linux', groups=['core'], options={}) == ['A=OFF', 'CORE=ON']
    assert cmake_entries(demo, 'linux', groups=['a', 'core'], options={}) == ['A=ON', 'CORE=ON']


def test_resolve_options(demo):
    values = resolve_options(demo, ['lto=on', 'smp=TBB', 'jobs=8'])
    assert values == {'lto': True, 'smp': 'TBB', 'jobs': 8}
    entries = cmake_entries(demo, 'linux', groups=['core'], options=values)
    assert entries[-3:] == ['LTO=ON', 'SMP=TBB', 'JOBS=8']
    with pytest.raises(ManifestError, match='boolean'):
        resolve_options(demo, ['lto=maybe'])
    with pytest.raises(ManifestError, match='one of'):
        resolve_options(demo, ['smp=OpenMP'])
    with pytest.raises(ManifestError, match='integer'):
        resolve_options(demo, ['jobs=many'])
    with pytest.raises(ManifestError, match='Unknown option'):
        resolve_options(demo, ['nope=1'])
    with pytest.raises(ManifestError, match=re.escape("'lto' needs a value: write lto=VALUE")):
        resolve_options(demo, ['lto'])


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        ({'manifest_version': 2}, 'manifest_version'),
        ({'grups': {}}, 'unknown key'),
        ({'backend': {'name': 'x'}}, 'default_ref'),
        ({'backend': {'name': 'x', 'default_ref': 'r', 'url': 5}}, "'url' must be a string"),
        ({'backend': {'name': 'x', 'default_ref': 'r', 'colour': 1}}, 'unknown key'),
        (
            {'backend': {'name': 'x', 'default_ref': 'r', 'version_command': 'v {home}'}},
            'unknown placeholder',
        ),
        ({'groups': ['g']}, 'table'),
        ({'groups': {'g': 'x'}}, 'table'),
        ({'groups': {'g': {'default': True}}}, 'help'),
        ({'groups': {'g': {'help': 'x', 'defualt': True}}}, 'unknown key'),
        ({'groups': {'g': {'help': 'x', 'default': 'false'}}}, "'default' must be a boolean"),
        ({'groups': {'g': {'help': 'x', 'os': ['beos']}}}, 'unknown os'),
        ({'groups': {'g': {'help': 'x', 'conflicts': ['zz']}}}, 'unknown group'),
        (
            {'groups': {'g': {'help': 'x', 'conflicts': ['h']}, 'h': {'help': 'y'}}},
            "list 'g' in",
        ),
        ({'options': {'o': {'help': 'x', 'default': True, 'cmake': 'X=ON'}}}, 'placeholder'),
        (
            {'options': {'o': {'help': 'x', 'default': True, 'cmake': 5}}},
            "'cmake' must be a string",
        ),
        (
            {'options': {'o': {'help': 'x', 'default': True, 'cmake': 'X=${HOME}{value}'}}},
            'unknown placeholder',
        ),
        ({'options': {'o': {'help': 'x', 'default': [], 'cmake': 'X={value}'}}}, 'default'),
        (
            {'options': {'o': {'help': 'x', 'cmake': 'X={value}'}}},
            "missing required key 'default'",
        ),
        (
            {
                'options': {
                    'o': {'help': 'x', 'default': 'C', 'cmake': 'X={value}', 'choices': ['A']}
                }
            },
            'not one of the choices',
        ),
        ({'hooks': {'pre_build': []}}, 'unknown hook'),
        ({'hooks': {'post_build': ['{interpreter} x.py']}}, 'unknown placeholder'),
        ({'hooks': {'post_build': ['{python x.py']}}, 'literal brace'),
        ({'options': {'o': 5}}, r'\[options.o\] must be a table'),
        (
            {
                'options': {
                    'o': {'help': 'x', 'default': 1, 'cmake': 'X={value}', 'choices': ['A']}
                }
            },
            'needs a string',
        ),
        ({'cmake': {'init_cache': {'beos': 'x'}}}, 'init_cache'),
        ({'cmake': {'init_cache': {'linux': 5}}}, 'init_cache'),
        ({'cmake': {'linx': ['A=1']}}, 'unknown key'),
        ({'cmake': {'common': ['-DA=1']}}, re.escape("write '-DA=1' without the -D prefix")),
        ({'cmake': {'macos': ['A']}}, re.escape("'A' must be KEY=VALUE or KEY:TYPE=VALUE")),
        ({'groups': {'g': {'help': 'x', 'cmake': ['A:BOOL=ON', '-DB=1']}}}, 'without the -D'),
        ({'groups': {'g': {'help': 'x', 'cmake_off': ['=1']}}}, 'KEY=VALUE'),
        (
            {'options': {'o': {'help': 'x', 'default': 1, 'cmake': '-DX={value}'}}},
            'without the -D',
        ),
        ({'manifest_version': '1'}, "'manifest_version' must be"),
        ({'manifest_version': True}, "'manifest_version' must be"),
    ],
)
def test_parse_manifest_errors(patch, match):
    data = dict(MINIMAL_MANIFEST, **patch)
    with pytest.raises(ManifestError, match=match):
        parse_manifest(data)


def test_manifest_version_is_required():
    data = {k: v for k, v in MINIMAL_MANIFEST.items() if k != 'manifest_version'}
    with pytest.raises(ManifestError, match="missing required key 'manifest_version'"):
        parse_manifest(data)


def test_load_manifest_bad_toml(tmp_path: Path):
    path = tmp_path / 'pyvista-build.toml'
    path.write_text('manifest_version = [')
    with pytest.raises(ManifestError, match=re.escape(str(path))):
        load_manifest(path)


def test_find_manifest_prefers_tree(tmp_path: Path):
    (tmp_path / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n[backend]\nname = "vtk"\ndefault_ref = "r"\n'
    )
    assert find_manifest('vtk', tmp_path).default_ref == 'r'
    with pytest.raises(ManifestError, match='ships no manifest'):
        find_manifest('nope', tmp_path.parent / 'missing')


def test_option_needs_cmake_or_env():
    base = {'help': 'x', 'default': True}
    with pytest.raises(ManifestError, match='exactly one'):
        parse_manifest(dict(MINIMAL_MANIFEST, options={'o': base}))
    with pytest.raises(ManifestError, match='exactly one'):
        parse_manifest(
            dict(MINIMAL_MANIFEST, options={'o': dict(base, cmake='X={value}', env='X')})
        )


def test_known_backends():
    assert known_backends() == ['cvista', 'vtk']


def test_option_coerce_and_schema_errors():
    flag = Option('f', help='', default=True, cmake='F={value}')
    assert flag.coerce('on') is True
    assert flag.coerce('off') is False
    with pytest.raises(ManifestError, match='expects a boolean'):
        flag.coerce('maybe')
    count = Option('n', help='', default=2, cmake='N={value}')
    with pytest.raises(ManifestError, match='expects an integer'):
        count.coerce('two')
    with pytest.raises(ManifestError, match="'name' must be a string"):
        parse_manifest(dict(MINIMAL_MANIFEST, backend={'name': 1, 'default_ref': 'v'}))
    with pytest.raises(ManifestError, match='must be a list of strings'):
        parse_manifest(dict(MINIMAL_MANIFEST, groups={'g': {'help': 'x', 'cmake': 'A=ON'}}))
    with pytest.raises(ManifestError, match=r'\[tools\]'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmake': 3}))
    with pytest.raises(ManifestError, match="'cmake' must be a table, got an array"):
        parse_manifest(dict(MINIMAL_MANIFEST, cmake=[]))


def test_tools_table_is_validated():
    """Unknown tools and malformed version specifiers are manifest errors."""
    with pytest.raises(ManifestError, match=r'\[tools\]'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmak': '>=3'}))
    with pytest.raises(ManifestError, match='not a version specifier'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmake': '3.22'}))
    with pytest.raises(ManifestError, match="'cmake' must be a string, got an array"):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmake': ['>=3.22']}))


def test_vtk_groups_needing_rendering_want_their_modules():
    """Modules that depend on rendering are WANT, so ``--without rendering`` still configures."""
    manifest = find_manifest('vtk', None)
    for name in ('io', 'web', 'matplotlib', 'parallel'):
        modules = [e for e in manifest.groups[name].cmake if e.startswith('VTK_MODULE_ENABLE_')]
        assert modules
        assert all(e.endswith('=WANT') for e in modules)
    linux = cmake_entries(manifest, 'linux', groups=['headless'], options={})
    assert {'VTK_OPENGL_HAS_EGL=ON', 'VTK_USE_X=OFF'} <= set(linux)
    with pytest.raises(ManifestError, match='not available on macos'):
        resolve_groups(manifest, 'macos', with_=['headless'])


def test_env_options_and_unset_keys():
    """Environment-backed options export variables, and disabled groups unset their keys."""
    manifest = parse_manifest(
        dict(
            MINIMAL_MANIFEST,
            groups={
                'a': {'help': 'A', 'default': True, 'cmake': ['A=ON']},
                'b': {'help': 'B', 'default': False, 'cmake': ['B=ON', 'SHARED:BOOL=ON', 'A=OFF']},
                'c': {'help': 'C', 'default': False, 'cmake': ['C=ON'], 'cmake_off': ['C=OFF']},
            },
            options={
                'threads': {'help': 'T', 'default': True, 'env': 'X_THREADS'},
                'level': {'help': 'L', 'default': 4, 'env': 'X_LEVEL'},
                'lto': {'help': 'LTO', 'default': False, 'cmake': 'LTO={value}'},
            },
        )
    )
    values = resolve_options(manifest, None)
    assert option_environment(manifest, values) == {'X_THREADS': '1', 'X_LEVEL': '4'}
    assert manifest.options['threads'].render(value=True) is None
    assert manifest.options['lto'].render_env(value=True) is None
    entries = cmake_entries(manifest, 'linux', groups=['a'], options=values)
    assert entries == ['A=ON', 'C=OFF', 'LTO=OFF']
    assert cmake_unset(manifest, 'linux', groups=['a']) == ['B', 'SHARED']
    assert cmake_unset(manifest, 'linux', groups=['a', 'b']) == []
