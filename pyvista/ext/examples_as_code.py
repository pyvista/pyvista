"""Generate downloadable Python source files from docstring "Examples" sections.

This Sphinx extension looks, on every page, for numpydoc-style "Examples"
headings -- rendered as ``.. rubric:: Examples`` for docstrings, or as a
regular section title on hand-written pages that happen to reuse that
heading -- and turns the content of each one into a small, self-contained,
runnable Python script, with a download link inserted immediately below it.

Everything outside of an Examples section is left completely alone: pages
or docstrings without one produce no file and no link. Enabling this
extension (adding it to ``conf.py``'s ``extensions``) is itself the on/off
switch; there is no separate configuration value.

This extension is intentionally independent of ``plot_directive.py``: it
doesn't import anything from it, and works the same whether or not that
extension is even installed. The one thing they informally share is a CSS
class, ``pyvista-plot-source`` (see ``plot_directive.py``'s ``TEMPLATE``),
which marks generated plot-directive code as a distinct node -- but this
extension doesn't need to treat it specially, since a plain
``.. container::`` node is already handled like any other generic
container (its doctest block / code block content is picked up the same
way either way). It only matters in that a numpydoc "Examples" section that
imports pyvista typically gets auto-wrapped in a ``.. pyvista-plot::`` call
(see the ``_str_examples`` monkeypatch in this project's ``conf.py``), and
this extension needs to handle whatever that directive leaves behind in the
resolved doctree.

Conversion rules applied to the nodes within an Examples section:

- doctest blocks (``>>> ...`` / ``... ...``) keep their input lines, with
  the prompts stripped, as real Python source; doctest *output* lines
  (expected results, with no prompt) are dropped entirely -- we only care
  about the input code, not what running it once produced
- ``.. code-block:: python`` (or ``py``) blocks -- as used by
  ``plot_directive.py`` for non-doctest-format examples -- are kept as-is
- ``.. note::``/``.. warning::`` (and the rest of the docutils admonition
  family) become a ``# LABEL:`` comment followed by their content as
  comments
- cross-references and inline code (``:class:``, ``:meth:``, ``:func:``,
  ``:attr:``, double-backtick literals, ...) keep just their display text,
  wrapped in backticks, e.g. :class:`pyvista.Plotter` -> `` `pyvista.Plotter` ``
- plain prose-style references (``:ref:``, ``:doc:``) become their display
  text with no backticks
- everything else text-bearing (prose, captions, other non-python code) is
  turned into a plain ``#`` comment
- figures/images, raw HTML, and sphinx-design dropdowns/tab-sets (as used
  by ``plot_directive.py`` for its vtksz interactive-scene tabs) are
  dropped entirely -- not even as a comment

Generated files start with a title header (the documented object's fully
qualified name, e.g. ``# pyvista.read examples`` followed by a matching
``-----`` underline), and follow a few whitespace conventions so the result
reads like normal, human-written Python rather than a flat dump: prose
immediately preceding a code block stays directly above it with no blank
line, but a code block is always followed by a blank line before whatever
comes next (comment or more code), and a converted directive (the title
header itself, or a ``# NOTE:``-style block) always gets a blank line both
before and after it. The file always ends with a trailing blank line.

If the resulting script contains at least one real executable statement, a
download link for it is inserted at the bottom of the Examples section.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx import addnodes

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# Node types that mark a heading (the start of some other section) and
# therefore bound the end of whatever Examples section precedes them.
# ``addnodes.desc`` and ``addnodes.index`` are included because numpydoc
# renders a class's nested members (e.g. ``:members:``) as flat siblings
# directly inside the *same* ``desc_content`` as the class's own Examples
# section -- without stopping there, a class's Examples section would
# swallow all of its methods too.
_BOUNDARY_TYPES = (
    nodes.rubric,
    nodes.title,
    nodes.section,
    addnodes.desc,
    addnodes.index,
)

# Node types with no textual content worth keeping under any circumstance.
_IGNORED_TYPES = (nodes.image, nodes.figure, nodes.comment, nodes.raw)

# sphinx-design containers identified by class name (checked by CSS class
# rather than node type, so this doesn't require importing sphinx_design and
# keeps working across its versions). Their entire subtree is dropped: a
# dropdown's hidden content isn't part of the visible example, and a
# tab-set's only content in our use case is figures (already ignored) plus
# tab-label cruft like "Static Scene" / "Interactive Scene".
_SKIP_SUBTREE_CLASSES = ('sd-dropdown', 'sd-tab-set')

_CONTAINER_TYPES = (
    nodes.bullet_list,
    nodes.enumerated_list,
    nodes.definition_list,
    nodes.definition_list_item,
    nodes.list_item,
    nodes.definition,
    nodes.term,
    nodes.classifier,
    nodes.block_quote,
    nodes.container,
    nodes.compound,
    addnodes.versionmodified,
)

# The standard docutils admonition family (``.. note::``, ``.. warning::``,
# etc, all no-title fixed-label admonitions -- as opposed to the generic
# ``.. admonition:: Custom Title`` directive, handled separately below).
_ADMONITION_LABELS = {
    nodes.attention: 'ATTENTION',
    nodes.caution: 'CAUTION',
    nodes.danger: 'DANGER',
    nodes.error: 'ERROR',
    nodes.hint: 'HINT',
    nodes.important: 'IMPORTANT',
    nodes.note: 'NOTE',
    nodes.tip: 'TIP',
    nodes.warning: 'WARNING',
}

_PYTHON_LANGUAGES = ('python', 'py', 'python3')

# A chunk of generated lines tagged with how it should be spaced relative to
# its neighbors when segments are joined (see ``_join_segments``):
#   'code'      real Python source
#   'text'      a plain comment (prose, a paragraph, ...)
#   'directive' a comment block that must be visually set off with a blank
#               line both before and after it (the title header; a
#               ``# NOTE:``-style admonition block)
Segment = tuple[str, list[str]]


def _has_class(node: nodes.Node, css_class: str) -> bool:
    getter = getattr(node, 'get', None)
    return bool(getter) and css_class in getter('classes', [])


def _is_examples_heading(node: nodes.Node) -> bool:
    """Check whether ``node`` is a heading (rubric or title) named "Examples"."""
    return (
        isinstance(node, (nodes.rubric, nodes.title))
        and node.astext().strip().lower() == 'examples'
    )


def _add_comment(lines: list[str], text: str) -> None:
    """Append ``text`` to ``lines`` as one or more Python comment lines."""
    for line in text.splitlines():
        line_ = line.rstrip()
        lines.append(f'# {line_}' if line_ else '#')


def _render_inline(node: nodes.Node) -> str:
    """Render a node's inline content to a plain string.

    Code-like spans -- ``nodes.literal``, which is how docutils/Sphinx render
    both plain double-backtick literals and resolved ``:class:``/``:meth:``/
    ``:func:``/``:attr:`` cross-references -- are wrapped in backticks using
    just their display text (the only part of an explicit-title reference
    like ``:class:`Display Name <target>``` that's actually visible).
    Prose-style references (``:ref:``, ``:doc:``, ...), which docutils
    renders as a plain ``inline`` rather than a ``literal``, keep their
    display text with no backticks. Any other inline formatting (emphasis,
    strong, ...) is flattened to plain text.
    """
    if isinstance(node, nodes.literal):
        return f'`{node.astext()}`'
    if isinstance(node, (nodes.image, nodes.figure, nodes.raw, nodes.comment)):
        return ''
    if isinstance(node, nodes.Text):
        return str(node)
    if hasattr(node, 'children') and node.children:
        return ''.join(_render_inline(child) for child in node.children)
    return node.astext()


def _join_segments(segments: list[Segment]) -> list[str]:
    """Flatten segments into lines, applying inter-segment spacing rules.

    - a blank line always follows a ``code`` segment, whatever comes next
    - a ``directive`` segment always gets a blank line both before and
      after it
    - otherwise (e.g. prose directly above a code block, or two prose
      segments back to back), no blank line is forced
    """
    lines: list[str] = []
    prev_kind: str | None = None
    for kind, seg_lines in segments:
        if not seg_lines:
            continue
        need_blank = prev_kind is not None and (
            prev_kind == 'code' or kind == 'directive' or prev_kind == 'directive'
        )
        if need_blank:
            lines.append('')
        lines.extend(seg_lines)
        prev_kind = kind
    return lines


def _convert_doctest_block(node: nodes.doctest_block) -> list[Segment]:
    """Convert a doctest block, stripping ``>>> ``/``... `` prompts.

    Non-prompted, non-blank lines are expected doctest *output* -- we only
    care about the input code, so those are dropped entirely rather than
    kept as comments.
    """
    lines: list[str] = []
    has_code = False
    for line in node.astext().splitlines():
        if line.startswith('>>> ') or line == '>>>':
            lines.append(line[4:])
            has_code = True
        elif line.startswith('... ') or line == '...':
            lines.append(line[4:])
        elif not line.strip():
            lines.append('')
        # else: doctest output line - dropped
    if not has_code:
        return []
    while lines and not lines[-1].strip():
        lines.pop()
    return [('code', lines)]


def _convert_literal_block(node: nodes.literal_block) -> list[Segment]:
    """Convert a ``.. code-block::``. Python blocks stay code, others become comments."""
    language = node.get('language', '')
    if language in _PYTHON_LANGUAGES:
        lines = node.astext().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        return [('code', lines)] if lines else []
    text = node.astext().strip()
    if not text:
        return []
    comment_lines: list[str] = []
    _add_comment(comment_lines, text)
    return [('text', comment_lines)]


def _convert_admonition(
    node: nodes.Element, label: str, *, skip_first_title: bool = False
) -> list[Segment]:
    """Convert an admonition-like container to a ``# LABEL:`` directive segment."""
    inner: list[Segment] = [('text', [f'# {label}:'])]
    for child in node.children:
        if skip_first_title and isinstance(child, nodes.title):
            continue
        inner.extend(_convert_node(child))
    inner_lines = _join_segments(inner)
    return [('directive', inner_lines)] if inner_lines else []


def _convert_node(node: nodes.Node) -> list[Segment]:  # noqa: PLR0911
    """Convert ``node`` into zero or more segments."""
    if any(_has_class(node, css_class) for css_class in _SKIP_SUBTREE_CLASSES):
        return []
    if isinstance(node, _IGNORED_TYPES):
        return []
    if isinstance(node, nodes.doctest_block):
        return _convert_doctest_block(node)
    if isinstance(node, nodes.literal_block):
        return _convert_literal_block(node)
    if type(node) in _ADMONITION_LABELS:
        return _convert_admonition(node, _ADMONITION_LABELS[type(node)])
    if isinstance(node, nodes.admonition):
        # generic ``.. admonition:: Custom Title`` - use its own title as the label
        title_node = node.next_node(nodes.title)
        label = title_node.astext().strip() if title_node is not None else 'NOTE'
        return _convert_admonition(node, label, skip_first_title=True)
    if isinstance(node, _CONTAINER_TYPES):
        segments: list[Segment] = []
        for child in node.children:
            segments.extend(_convert_node(child))
        return segments

    # Plain text-bearing nodes (paragraphs, etc.) - render inline content,
    # backticking code-like cross-references/literals along the way.
    text = _render_inline(node).strip()
    if not text:
        return []
    comment_lines: list[str] = []
    _add_comment(comment_lines, text)
    return [('text', comment_lines)]


def _has_real_code(source: str) -> bool:
    """Check whether ``source`` contains at least one executable statement."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    return any(
        not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant)
        for stmt in tree.body
    )


def _span_from(parent: nodes.Element, start: int) -> int:
    """Return the end index (exclusive) of a content span starting at ``start``.

    The span extends until (but excludes) the next boundary-type node, or to
    the end of ``parent``'s children.
    """
    end = start
    for i in range(start, len(parent.children)):
        if isinstance(parent.children[i], _BOUNDARY_TYPES):
            break
        end = i + 1
    return end


def _examples_spans(doctree: nodes.document) -> list[tuple[nodes.Element, int, int, nodes.Node]]:
    """Find every "Examples" heading's content span.

    Returns a list of ``(parent, start, end, heading)`` tuples.
    """
    spans = []
    for heading in doctree.findall(_is_examples_heading):
        parent = heading.parent
        start = parent.index(heading) + 1
        end = _span_from(parent, start)
        spans.append((parent, start, end, heading))
    return spans


def _qualified_name_for(node: nodes.Node, docname: str, counter: int) -> str:
    """Best-effort identifier used to name the generated file and its title header."""
    ancestor: nodes.Node | None = node.parent
    while ancestor is not None:
        if isinstance(ancestor, addnodes.desc):
            signature = ancestor.next_node(addnodes.desc_signature)
            if signature is not None and signature.get('ids'):
                return signature['ids'][0]
        ancestor = ancestor.parent
    base = Path(docname).name or docname
    return f'{base}-example-{counter}'


def _header_segment(qualified_name: str) -> Segment:
    """Build the title-header segment, e.g. ``# pyvista.read examples`` + underline."""
    title = f'Examples from {qualified_name}'
    underline = '-' * len(title)
    return ('directive', [f'# {title}', f'# {underline}'])


def _write_source(app: Sphinx, name: str, source: str) -> str:
    """Write generated Python source directly into the builder's downloads dir.

    Returns the path of the written file, relative to the downloads
    directory (i.e. the value to use as a ``download_reference``'s
    ``filename`` attribute).

    Note: this writes straight to ``<outdir>/_downloads/...`` instead of
    registering through ``env.dlfiles``, because the HTML builder's
    ``copy_download_files`` task (which copies everything registered in
    ``env.dlfiles``) runs during ``copy_assets()`` -- *before* any
    ``doctree-resolved`` handler (this one included) gets a chance to run.
    Registering through ``env.dlfiles`` here would silently be too late.
    """
    # 32 hex characters, matching the digest length Sphinx's own native
    # download-file handling uses for its ``_downloads/<digest>/...`` layout.
    digest = hashlib.sha256(source.encode()).hexdigest()[:32]
    safe_name = name.replace('.', '_') if name else 'example'
    rel_path = f'{digest}/{safe_name}.py'

    out_path = Path(app.outdir) / '_downloads' / digest / f'{safe_name}.py'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source, encoding='utf-8')
    return rel_path


def _make_download_node(rel_path: str) -> nodes.paragraph:
    """Build a working download link node for a file already in the downloads dir."""
    reference = addnodes.download_reference('', reftarget=rel_path)
    reference['filename'] = rel_path
    reference += nodes.Text('Download examples as a Python script')

    paragraph = nodes.paragraph()
    paragraph += reference
    return paragraph


def _process_span(  # noqa: PLR0917
    app: Sphinx,
    docname: str,
    parent: nodes.Element,
    start: int,
    end: int,
    heading: nodes.Node,
    counter: int,
) -> None:
    """Convert one Examples span and insert a download link if it has real code."""
    segments: list[Segment] = []
    for node in parent.children[start:end]:
        segments.extend(_convert_node(node))

    if not any(kind == 'code' for kind, _lines in segments):
        return

    name = _qualified_name_for(heading, docname, counter)
    lines = _join_segments([_header_segment(name), *segments])
    source = '\n'.join(lines).rstrip() + '\n\n'

    if not _has_real_code(source):
        return

    rel_path = _write_source(app, name, source)
    download_node = _make_download_node(rel_path)

    parent.insert(end, download_node)


def _process_doctree(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Add a download link below every "Examples" section found on this page."""
    if not getattr(app.builder, 'download_support', False):
        # Only HTML-family builders (html, dirhtml, singlehtml, ...) know how
        # to serve a ``_downloads/`` directory. Skip everything else (latex,
        # text, man, epub, ...) rather than writing files nobody can reach.
        return

    # Process spans, per shared parent, from last to first: inserting a
    # download-link node shifts every later sibling index by one, so a
    # page with more than one Examples heading under the same parent (an
    # unusual but possible structure) stays correct if later spans are
    # inserted before earlier ones are processed.
    spans = _examples_spans(doctree)
    numbered_spans = [(*span, i + 1) for i, span in enumerate(spans)]
    for parent, start, end, heading, counter in sorted(
        numbered_spans, key=lambda s: (id(s[0]), -s[1])
    ):
        _process_span(app, docname, parent, start, end, heading, counter)


def setup(app: Sphinx) -> dict:  # numpydoc ignore=RT01
    """Register the extension."""
    app.connect('doctree-resolved', _process_doctree)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
