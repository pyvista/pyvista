"""Tests for pyvista.ext._autoenum: unit tests, plus a real build of tinypages_autoenum."""

from __future__ import annotations

from enum import Enum
from enum import Flag
from enum import IntEnum
from enum import IntFlag
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from pyvista.ext import _autoenum as autoenum

TINYPAGES_AUTOENUM_DIR = Path(__file__).parent / 'tinypages_autoenum'


def test_is_enum():
    class Color(Enum):
        RED = 1

    assert autoenum._is_enum(Color)
    assert not autoenum._is_enum(int)
    assert not autoenum._is_enum(Color.RED)  # an instance, not the class itself


def test_resolve_single_part():
    assert autoenum._resolve('enum', 'IntFlag') is IntFlag


def test_resolve_dotted_path():
    import os

    assert autoenum._resolve('os', 'path.sep') == os.path.sep


def test_metaclass_properties_finds_only_metaclass_properties():
    class Meta(type):
        @property
        def computed(cls):
            return 'value'

    class Plain(metaclass=Meta):
        @property
        def instance_only(self):
            return 'nope'

    props = autoenum._metaclass_properties(Plain)
    assert set(props) == {'computed'}
    assert props['computed'] is Meta.__dict__['computed']


def test_metaclass_properties_empty_for_plain_enum():
    class Color(Enum):
        RED = 1

    assert autoenum._metaclass_properties(Color) == {}


def test_metaclass_properties_skips_enummeta_itself():
    class Color(Enum):
        RED = 1

    assert autoenum._metaclass_properties(type(Color)) == {}


def test_metaclass_property_names_sorted():
    assert autoenum.metaclass_property_names('enum', 'IntFlag') == []


def test_metaclass_property_names_on_celltype():
    assert 'dimension_map' in autoenum.metaclass_property_names('pyvista', 'CellType')


def test_metaclass_properties_first_definition_wins_in_mro():
    class BaseMeta(type):
        @property
        def shared(cls):
            return 'base'

    class SubMeta(BaseMeta):
        @property
        def shared(self):
            return 'sub'

    class Widget(metaclass=SubMeta): ...

    props = autoenum._metaclass_properties(Widget)
    assert props['shared'] is SubMeta.__dict__['shared']


def test_instance_properties_finds_only_public_properties():
    class Widget:
        @property
        def visible(self):
            return True

        @property
        def _hidden(self):
            return False

        constant = 1

    props = autoenum._instance_properties(Widget)
    assert set(props) == {'visible'}
    assert props['visible'] is Widget.__dict__['visible']


def test_instance_property_names_on_celltype():
    names = autoenum.instance_property_names('pyvista', 'CellType')
    assert names == [
        'dimension',
        'is_composite',
        'is_linear',
        'n_edges',
        'n_faces',
        'n_points',
        'vtk_class',
    ]
    assert 'dimension_map' not in names  # that one's a metaclass property, not an instance one


def test_metaclass_property_descriptions_uses_first_docstring_line(monkeypatch):
    class Meta(type):
        @property
        def computed(cls):
            """Return a value.

            More detail that shouldn't appear in the description.
            """
            return 'value'

    class Widget(metaclass=Meta): ...

    monkeypatch.setattr(autoenum, '_resolve', lambda *_args: Widget)
    descriptions = dict(autoenum.metaclass_property_descriptions('unused', 'unused'))
    assert descriptions == {'computed': 'Return a value.'}


def test_metaclass_property_descriptions_on_celltype():
    descriptions = dict(autoenum.metaclass_property_descriptions('pyvista', 'CellType'))
    assert descriptions['dimension_map'].startswith('Return a mapping with sets')


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([0, 1, 2, 4, 8], True),
        ([0], True),
        ([1, 2, 3], False),  # 3 is not a power of two
        ([5, 9, 22], False),  # arbitrary VTK-style type codes, like CellType
    ],
)
def test_is_bitmask_like(values, expected):
    Bits = IntEnum('Bits', {f'V{i}': v for i, v in enumerate(values)})
    assert autoenum._is_bitmask_like(Bits) is expected


def test_is_bitmask_like_true_for_flag_regardless_of_values():
    class NotReallyBits(Flag):
        A = 3  # not a power of two, but Flag membership alone is enough to opt in

    assert autoenum._is_bitmask_like(NotReallyBits)


def test_is_bitmask_like_false_for_non_int_enum():
    class Shape(str, Enum):
        CIRCLE = 'circle'

    assert not autoenum._is_bitmask_like(Shape)


@pytest.mark.parametrize(
    ('value', 'as_hex', 'expected'),
    [
        (0, False, '0'),
        (42, False, '42'),
        (0, True, '0x0'),
        (255, True, '0xff'),
    ],
)
def test_format_value(value, as_hex, expected):
    assert autoenum._format_value(value, as_hex=as_hex) == expected


def test_format_value_str():
    assert autoenum._format_value('circle', as_hex=False) == "'circle'"


def test_enum_documenter_can_document_member():
    class Color(Enum):
        RED = 1

    assert autoenum.EnumDocumenter.can_document_member(Color, 'Color', False, None)
    assert not autoenum.EnumDocumenter.can_document_member(object(), 'obj', False, None)


def test_metaclass_property_documenter_can_document_member_always_declines():
    # Never resolved via generic dispatch -- only via an explicit ``.. autometaclassproperty::``
    # (see setup()) -- so this must decline regardless of what it's asked about.
    assert not autoenum.MetaclassPropertyDocumenter.can_document_member(
        object(), 'anything', True, object()
    )


# --- Integration: build tinypages_autoenum for real, with sphinx-build -------------------


def _build_tinypages_autoenum(
    tmp_path: Path, extra_pages: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, Path]:
    """Build a throwaway copy of tinypages_autoenum; return (process, html build dir).

    ``extra_pages`` adds ``{filename: content}`` .rst files, each linked from a toctree.
    """
    source_dir = tmp_path / 'tinypages_autoenum'
    shutil.rmtree(source_dir, ignore_errors=True)
    shutil.copytree(
        TINYPAGES_AUTOENUM_DIR, source_dir, ignore=shutil.ignore_patterns('__pycache__')
    )

    if extra_pages:
        for filename, content in extra_pages.items():
            (source_dir / filename).write_text(content, encoding='utf-8')
        stems = '\n   '.join(Path(filename).stem for filename in extra_pages)
        with (source_dir / 'index.rst').open('a', encoding='utf-8') as f:
            f.write(f'\n.. toctree::\n\n   {stems}\n')

    build_dir = tmp_path / 'build'
    proc = subprocess.run(
        [sys.executable, '-msphinx', '-b', 'html', str(source_dir), str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, build_dir / '_autosummary'


def _text(html_path: Path) -> str:
    """Strip tags from a built HTML page, collapsing whitespace."""
    return re.sub(r'<[^>]+>', ' ', html_path.read_text(encoding='utf-8'))


def test_tinypages_autoenum_build(tmp_path):
    proc, autosummary_dir = _build_tinypages_autoenum(tmp_path)
    assert proc.returncode == 0, proc.stderr

    # #1: every property -- instance and metaclass alike -- is listed, not just the
    # metaclass one.
    celltype_text = _text(autosummary_dir / 'pyvista.CellType.html')
    for name in (
        'dimension',
        'is_composite',
        'is_linear',
        'n_edges',
        'n_faces',
        'n_points',
        'vtk_class',
        'dimension_map',
    ):
        assert f'CellType.{name}' in celltype_text

    # #2: dimension_map gets its own page, showing its docstring -- not a repr dump.
    dimension_map_html = (autosummary_dir / 'pyvista.CellType.dimension_map.html').read_text(
        encoding='utf-8'
    )
    assert 'mappingproxy(' not in dimension_map_html
    assert 'topological dimension' in dimension_map_html  # from its docstring

    # #3: enum members render under their own rubric heading, flat (no blockquote wrapper).
    celltype_html = (autosummary_dir / 'pyvista.CellType.html').read_text(encoding='utf-8')
    assert '<p class="rubric">Enum Members</p>' in celltype_html
    assert 'blockquote' not in celltype_html

    # Metaclass properties get their own "Class Attributes" table, with a real description
    # (not blank, the way autosummary's own description extraction would leave it) and only
    # one visible link to its page (the toctree-only, :template:-forced block stays hidden,
    # not just duplicated).
    assert '>Class Attributes<' in celltype_html
    assert 'topological dimension' in celltype_text
    assert celltype_html.count('pyvista.CellType.dimension_map.html#') == 1

    # CellStatus: hex-formatted values.
    cellstatus_text = _text(autosummary_dir / 'pyvista.CellStatus.html')
    assert '0x1' in cellstatus_text

    # A str-valued Enum (not int-based) does not crash and reprs its value.
    shape_text = _text(autosummary_dir / 'pyvista.plotting.opts.PointSpriteShape.html')
    assert "'circle'" in shape_text


def test_metaclassproperty_documenter_warns_on_non_metaclass_property(tmp_path):
    """A misuse of the directive (naming something that isn't a metaclass property) warns
    and produces no content, rather than crashing.
    """
    misuse_rst = (
        'Misuse\n======\n\n.. currentmodule:: pyvista\n\n'
        '.. autometaclassproperty:: CellType.EMPTY_CELL\n'
    )
    proc, _ = _build_tinypages_autoenum(tmp_path, extra_pages={'misuse.rst': misuse_rst})
    assert proc.returncode == 0, proc.stderr
    assert 'is not a metaclass property of' in proc.stderr


def test_metaclassproperty_documenter_warns_on_unimportable_name(tmp_path):
    """A name that fails to import at all leaves self.parent unset, not just non-metaclass."""
    misuse_rst = (
        'Misuse\n======\n\n.. autometaclassproperty:: totally.bogus.NonExistentThing12345\n'
    )
    proc, _ = _build_tinypages_autoenum(tmp_path, extra_pages={'misuse.rst': misuse_rst})
    assert proc.returncode == 0, proc.stderr
    assert 'failed to import' in proc.stderr


def test_enum_documenter_filter_members_on_non_enum(tmp_path):
    """.. autoenum:: on a plain class falls back to the stock ClassDocumenter behavior."""
    misuse_rst = 'Misuse\n======\n\n.. autoenum:: pyvista.core.config.Config\n'
    proc, autosummary_dir = _build_tinypages_autoenum(
        tmp_path, extra_pages={'misuse.rst': misuse_rst}
    )
    assert proc.returncode == 0, proc.stderr
    text = _text(autosummary_dir.parent / 'misuse.html')
    assert 'validate_on_wrap' in text  # from Config's own docstring, not crashed/empty
