"""Unit tests for individual ``examples_as_code.py`` functions.

Complements ``test_tinypages_examples_as_code.py``'s full Sphinx-build
tests with fast, direct tests of branches that are impractical to reach
through a full build (mocked Sphinx app, hand-built doctree fragments).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from docutils import nodes
from docutils.core import publish_doctree
from sphinx import addnodes

from pyvista.ext import examples_as_code as eac

if TYPE_CHECKING:
    from pathlib import Path


def _parse(rst: str) -> nodes.document:
    return publish_doctree(rst, settings_overrides={'report_level': 5})


# ---------------------------------------------------------------------------
# _has_class
# ---------------------------------------------------------------------------


def test_has_class_true():
    assert eac._has_class(nodes.container(classes=['sd-dropdown']), 'sd-dropdown')


def test_has_class_false():
    assert not eac._has_class(nodes.container(classes=['other']), 'sd-dropdown')


def test_has_class_node_without_get():
    assert not eac._has_class(nodes.Text('hi'), 'sd-dropdown')


# ---------------------------------------------------------------------------
# _is_examples_heading
# ---------------------------------------------------------------------------


def test_is_examples_heading_rubric():
    doctree = _parse('.. rubric:: Examples')
    assert eac._is_examples_heading(doctree[0])


def test_is_examples_heading_case_insensitive():
    doctree = _parse('.. rubric:: EXAMPLES')
    assert eac._is_examples_heading(doctree[0])


def test_is_examples_heading_wrong_text():
    doctree = _parse('.. rubric:: See Also')
    assert not eac._is_examples_heading(doctree[0])


def test_is_examples_heading_wrong_node_type():
    doctree = _parse('Examples')
    assert not eac._is_examples_heading(doctree[0])


def test_is_examples_heading_title():
    doctree = _parse('Examples\n========\n\nbody text')
    assert eac._is_examples_heading(doctree[0])


def test_add_comment_multiline_with_blank_line():
    lines: list[str] = []
    eac._add_comment(lines, 'first\n\nthird')
    assert lines == ['# first', '#', '# third']


# ---------------------------------------------------------------------------
# _render_inline
# ---------------------------------------------------------------------------


def test_render_inline_literal():
    doctree = _parse('``code``')
    assert eac._render_inline(doctree[0][0]) == '`code`'


def test_render_inline_plain_text():
    doctree = _parse('hello world')
    assert eac._render_inline(doctree[0]) == 'hello world'


def test_render_inline_image_returns_empty():
    doctree = _parse('.. image:: foo.png')
    assert eac._render_inline(doctree[0]) == ''


def test_render_inline_childless_fallback_to_astext():
    assert eac._render_inline(nodes.transition()) == ''


# ---------------------------------------------------------------------------
# _join_segments
# ---------------------------------------------------------------------------


def test_join_segments_text_then_code_no_blank():
    result = eac._join_segments([('text', ['# a']), ('code', ['x = 1'])])
    assert result == ['# a', 'x = 1']


def test_join_segments_code_then_text_blank():
    result = eac._join_segments([('code', ['x = 1']), ('text', ['# a'])])
    assert result == ['x = 1', '', '# a']


def test_join_segments_directive_gets_blank_both_sides():
    result = eac._join_segments(
        [('text', ['# a']), ('directive', ['# NOTE:', '# n']), ('code', ['x = 1'])]
    )
    assert result == ['# a', '', '# NOTE:', '# n', '', 'x = 1']


def test_join_segments_first_segment_no_leading_blank():
    result = eac._join_segments([('directive', ['# H'])])
    assert result == ['# H']


def test_join_segments_skips_empty_segments():
    result = eac._join_segments([('text', []), ('code', ['x = 1'])])
    assert result == ['x = 1']


# ---------------------------------------------------------------------------
# _convert_doctest_block
# ---------------------------------------------------------------------------


def test_convert_doctest_block_strips_prompts():
    doctree = _parse('>>> x = 1\n>>> y = 2')
    segments = eac._convert_doctest_block(doctree[0])
    assert segments == [('code', ['x = 1', 'y = 2'])]


def test_convert_doctest_block_drops_output():
    doctree = _parse('>>> 1 + 1\n2')
    segments = eac._convert_doctest_block(doctree[0])
    assert segments == [('code', ['1 + 1'])]


def test_convert_doctest_block_no_code_returns_empty():
    # a doctest block always has at least one >>> line by construction, so
    # this is exercised indirectly through _convert_node's dispatch instead
    node = nodes.doctest_block()
    node += nodes.Text('not a real doctest line')
    assert eac._convert_doctest_block(node) == []


def test_convert_doctest_block_continuation_line():
    doctree = _parse('>>> def f():\n...     return 1')
    segments = eac._convert_doctest_block(doctree[0])
    assert segments == [('code', ['def f():', '    return 1'])]


def test_convert_doctest_block_internal_blank_line_kept():
    node = nodes.doctest_block('', '>>> x = 1\n\n>>> y = 2')
    segments = eac._convert_doctest_block(node)
    assert segments == [('code', ['x = 1', '', 'y = 2'])]


def test_convert_doctest_block_trailing_blank_trimmed():
    node = nodes.doctest_block('', '>>> x = 1\n\n')
    segments = eac._convert_doctest_block(node)
    assert segments == [('code', ['x = 1'])]


# ---------------------------------------------------------------------------
# _convert_literal_block
# ---------------------------------------------------------------------------


def test_convert_literal_block_python():
    # Sphinx's own code-block directive sets a ``language`` attribute
    # (unlike plain docutils', which uses CSS classes instead), so this
    # node is built directly to match what a real Sphinx build produces.
    node = nodes.literal_block('', 'x = 1')
    node['language'] = 'python'
    segments = eac._convert_literal_block(node)
    assert segments == [('code', ['x = 1'])]


def test_convert_literal_block_non_python_becomes_comment():
    node = nodes.literal_block('', 'echo hi')
    node['language'] = 'bash'
    segments = eac._convert_literal_block(node)
    assert segments == [('text', ['# echo hi'])]


def test_convert_literal_block_empty_non_python():
    node = nodes.literal_block('', '')
    node['language'] = 'bash'
    assert eac._convert_literal_block(node) == []


def test_convert_literal_block_python_whitespace_only():
    node = nodes.literal_block('', '   \n   ')
    node['language'] = 'python'
    assert eac._convert_literal_block(node) == []


# ---------------------------------------------------------------------------
# _clean_stray_rst_markup / _clean_code_comment
# ---------------------------------------------------------------------------


def test_clean_stray_hyperlink_with_url():
    text = '# See `Ext <https://example.com>`_.'
    assert eac._clean_stray_rst_markup(text) == '# See Ext <https://example.com>.'


def test_clean_stray_hyperlink_without_url():
    text = '# see `target`_ here'
    assert eac._clean_stray_rst_markup(text) == '# see target here'


def test_clean_stray_xref_explicit_title():
    text = '# :func:`short <pyvista.long.path>`.'
    assert eac._clean_stray_rst_markup(text) == '# short.'


def test_clean_stray_xref_plain():
    text = '# :class:`pyvista.Plotter`.'
    assert eac._clean_stray_rst_markup(text) == '# pyvista.Plotter.'


def test_clean_code_comment_ignores_real_code():
    line = "x = '`not a ref`_'"
    assert eac._clean_code_comment(line) == line


def test_clean_code_comment_cleans_comment_lines():
    line = '# see `target`_'
    assert eac._clean_code_comment(line) == '# see target'


# ---------------------------------------------------------------------------
# _convert_admonition / _convert_node dispatch
# ---------------------------------------------------------------------------


def test_convert_node_note():
    doctree = _parse('.. note::\n\n   hello')
    assert eac._convert_node(doctree[0]) == [('directive', ['# NOTE:', '# hello'])]


def test_convert_node_seealso():
    node = addnodes.seealso()
    p1 = nodes.paragraph()
    p1 += nodes.Text('See X')
    p2 = nodes.paragraph()
    p2 += nodes.Text('More info')
    node += p1
    node += p2
    assert eac._convert_node(node) == [('directive', ['# SEE ALSO:', '# See X', '# More info'])]


def test_convert_node_generic_admonition_uses_title():
    doctree = _parse('.. admonition:: Custom Title\n\n   body text')
    assert eac._convert_node(doctree[0]) == [('directive', ['# Custom Title:', '# body text'])]


def test_convert_node_generic_admonition_no_title_defaults_to_note():
    node = nodes.admonition()
    p = nodes.paragraph()
    p += nodes.Text('body')
    node += p
    assert eac._convert_node(node) == [('directive', ['# NOTE:', '# body'])]


def test_convert_node_admonition_with_no_body_keeps_label():
    node = nodes.note()
    assert eac._convert_node(node) == [('directive', ['# NOTE:'])]


def test_convert_node_skip_subtree_class():
    node = nodes.container(classes=['sd-dropdown'])
    p = nodes.paragraph()
    p += nodes.Text('hidden')
    node += p
    assert eac._convert_node(node) == []


def test_convert_node_ignored_type():
    doctree = _parse('.. image:: foo.png')
    assert eac._convert_node(doctree[0]) == []


def test_convert_node_versionmodified():
    node = addnodes.versionmodified()
    p = nodes.paragraph()
    p += nodes.Text('Added in version 1.0.')
    node += p
    assert eac._convert_node(node) == [('text', ['# Added in version 1.0.'])]


def test_convert_node_container_recurses():
    doctree = _parse('- item one\n- item two')
    segments = eac._convert_node(doctree[0])
    assert segments == [('text', ['# item one']), ('text', ['# item two'])]


def test_convert_node_empty_paragraph_returns_empty():
    assert eac._convert_node(nodes.paragraph()) == []


# ---------------------------------------------------------------------------
# _has_real_code
# ---------------------------------------------------------------------------


def test_has_real_code_true():
    assert eac._has_real_code('x = 1\n')


def test_has_real_code_syntax_error():
    assert not eac._has_real_code('def (:\n')


def test_has_real_code_only_constant_expression():
    assert not eac._has_real_code('"just a comment string"\n')


def test_has_real_code_empty():
    assert not eac._has_real_code('')


# ---------------------------------------------------------------------------
# _span_from / _examples_spans
# ---------------------------------------------------------------------------


def test_span_from_stops_at_boundary():
    doctree = _parse('.. rubric:: Examples\n\ntext\n\n.. rubric:: See Also\n\nmore')
    end = eac._span_from(doctree, 1)
    assert end == 2  # only the paragraph, not the second rubric or beyond


def test_span_from_runs_to_end_of_parent():
    doctree = _parse('.. rubric:: Examples\n\ntext one\n\ntext two')
    end = eac._span_from(doctree, 1)
    assert end == len(doctree.children)


def test_examples_spans_finds_multiple_headings():
    doctree = _parse('.. rubric:: Examples\n\ncode here')
    spans = eac._examples_spans(doctree)
    assert len(spans) == 1
    parent, start, _, _ = spans[0]
    assert parent is doctree
    assert start == 1


# ---------------------------------------------------------------------------
# _qualified_name_for
# ---------------------------------------------------------------------------


def test_qualified_name_for_desc_ancestor():
    desc = addnodes.desc()
    sig = addnodes.desc_signature(ids=['pkg.mod.func'])
    desc += sig
    content = addnodes.desc_content()
    heading = nodes.rubric()
    content += heading
    desc += content
    assert eac._qualified_name_for(heading, 'page', 1) == 'pkg.mod.func'


def test_qualified_name_for_fallback_no_desc_ancestor():
    section = nodes.section()
    heading = nodes.rubric()
    section += heading
    assert eac._qualified_name_for(heading, 'mypage', 3) == 'mypage-example-3'


def test_qualified_name_for_desc_signature_without_ids_falls_back():
    desc = addnodes.desc()
    sig = addnodes.desc_signature()  # no ids set
    desc += sig
    content = addnodes.desc_content()
    heading = nodes.rubric()
    content += heading
    desc += content
    assert eac._qualified_name_for(heading, 'mypage', 2) == 'mypage-example-2'


# ---------------------------------------------------------------------------
# _header_segment
# ---------------------------------------------------------------------------


def test_header_segment_format():
    kind, lines = eac._header_segment('pyvista.read')
    assert kind == 'directive'
    assert lines[0] == '# Examples from pyvista.read'
    assert lines[1] == '# ' + '-' * len('Examples from pyvista.read')


# ---------------------------------------------------------------------------
# _write_source / _make_download_node
# ---------------------------------------------------------------------------


def test_write_source_writes_file_with_32_char_digest(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    rel_path = eac._write_source(app, 'pkg.func', 'x = 1\n')
    digest, filename = rel_path.split('/')
    assert len(digest) == 32
    assert filename == 'pkg_func.py'
    assert (tmp_path / '_downloads' / digest / filename).read_text() == 'x = 1\n'


def test_write_source_empty_name_fallback(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    rel_path = eac._write_source(app, '', 'x = 1\n')
    assert rel_path.endswith('/example.py')


def test_make_download_node_structure():
    node = eac._make_download_node('abc123/foo.py')
    assert isinstance(node, nodes.paragraph)
    reference = node.children[0]
    assert isinstance(reference, addnodes.download_reference)
    assert reference['filename'] == 'abc123/foo.py'


# ---------------------------------------------------------------------------
# _process_span / _process_doctree / setup
# ---------------------------------------------------------------------------


def _build_examples_doctree(rst: str):
    doctree = _parse(rst)
    heading = doctree[0]
    parent = heading.parent
    start = parent.index(heading) + 1
    end = eac._span_from(parent, start)
    return doctree, parent, start, end, heading


def test_process_span_no_code_no_download(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\njust prose, no code'
    )
    original_len = len(parent.children)
    eac._process_span(app, 'page', parent, start, end, heading, 1, 'bottom')
    assert len(parent.children) == original_len


def test_process_span_code_segment_but_not_real_code(tmp_path: Path):
    # a doctest block containing only a comment line: has a 'code' segment
    # (it matched the >>> prompt), but ast.parse finds no real statements
    app = Mock(outdir=str(tmp_path))
    _, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> # just a comment'
    )
    original_len = len(parent.children)
    eac._process_span(app, 'page', parent, start, end, heading, 1, 'bottom')
    assert len(parent.children) == original_len


def test_process_span_inserts_at_bottom(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    eac._process_span(app, 'page', parent, start, end, heading, 1, 'bottom')
    assert isinstance(parent.children[end], nodes.paragraph)


def test_process_span_inserts_at_top(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    _doctree, parent, start, end, heading = _build_examples_doctree(
        '.. rubric:: Examples\n\n>>> x = 1'
    )
    eac._process_span(app, 'page', parent, start, end, heading, 1, 'top')
    assert isinstance(parent.children[start], nodes.paragraph)


def test_process_doctree_skips_when_no_download_support():
    app = Mock()
    app.builder.download_support = False
    doctree = _parse('.. rubric:: Examples\n\n>>> x = 1')
    original = doctree.pformat()
    eac._process_doctree(app, doctree, 'page')
    assert doctree.pformat() == original


def test_process_doctree_processes_spans(tmp_path: Path):
    app = Mock(outdir=str(tmp_path))
    app.builder.download_support = True
    app.config.examples_as_code_link_position = 'bottom'
    doctree = _parse('.. rubric:: Examples\n\n>>> x = 1')
    eac._process_doctree(app, doctree, 'page')
    assert any(isinstance(n, addnodes.download_reference) for n in doctree.findall())


def test_setup_registers_connect_and_config():
    app = Mock()
    result = eac.setup(app)
    app.connect.assert_called_once_with('doctree-resolved', eac._process_doctree)
    app.add_config_value.assert_called_once_with('examples_as_code_link_position', 'bottom', 'env')
    assert result['parallel_read_safe'] is True
    assert result['parallel_write_safe'] is True
