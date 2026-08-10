"""Open Graph descriptions built from a page's leading prose.

``sphinxext-opengraph`` derives ``og:description`` by walking every leaf node of a
page until it has enough characters. That works for hand-written prose pages, but
produces poor results for the two page shapes that make up most of PyVista's
documentation:

* API pages. Sphinx wraps autodoc output in a ``desc`` node, which subclasses
  :class:`docutils.nodes.Admonition`, and ``sphinxext-opengraph`` skips all
  admonitions. Every docstring is therefore invisible to it, and the description
  falls back to whatever happens to sit outside the ``desc`` node -- usually the
  rendered ``Examples`` section.
* Sphinx-Gallery pages, whose download links, timing footer and "Gallery generated
  by Sphinx-Gallery" signature are all leaf text and end up in the description --
  as do the "Download Python source code" links added by ``sphinx-examples-as-code``.

This module instead collects whole paragraphs of real prose, in document order,
skipping the structural furniture (signatures, parameter tables, code blocks,
admonitions, download links, captions, ...). For a docstring that means the
summary line and the paragraphs that follow it; for a Sphinx-Gallery example it
means the example's own introduction.

The result is written to the page's ``og:description`` field before
``sphinxext-opengraph`` renders its tags, so its own parser never runs.

"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING
from typing import Any

from docutils import nodes
from sphinx import addnodes

from pyvista.ext import _opengraph

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx

CONFIG_VALUE = 'pyvista_opengraph_description'

# Whole subtrees that never contain page prose.
_SKIP_NODES: tuple[type[nodes.Node], ...] = (
    nodes.Invisible,  # comments, targets, ``currentmodule``, index entries, ...
    nodes.Sequential,  # enumerated/bullet lists read badly as a summary
    nodes.decoration,
    nodes.docinfo,
    nodes.doctest_block,
    nodes.field_list,  # numpydoc parameter/return tables
    nodes.figure,
    nodes.image,
    nodes.literal_block,
    nodes.math_block,
    nodes.problematic,
    nodes.raw,
    nodes.rubric,
    nodes.sidebar,
    nodes.system_message,
    nodes.table,
    nodes.title,
    nodes.topic,
    addnodes.compact_paragraph,  # toctree entries; a ``paragraph`` subclass
    addnodes.desc_signature,
    addnodes.download_reference,  # "Download Python source code", ...
    addnodes.highlightlang,
    addnodes.toctree,
    addnodes.versionmodified,
)

# ``desc`` is an ``Admonition`` subclass for historical reasons, so admonitions can
# only be skipped once autodoc content has been excluded from the check.
_ADMONITION_EXCEPTIONS: tuple[type[nodes.Node], ...] = (addnodes.desc,)

# Subtrees identified by CSS class rather than node type.
_SKIP_CLASSES = frozenset(
    {
        'pyvista-plot-source',  # code narration from the plot directive
        'sphx-glr-download',
        'sphx-glr-download-link-note',
        'sphx-glr-footer',
        'sphx-glr-script-out',
        'sphx-glr-signature',
        'sphx-glr-timing',
        'toctree-wrapper',
    }
)


def setup(app: Sphinx) -> None:
    """Wire up Open Graph descriptions.

    Called by :mod:`pyvista.ext.plot_directive`; this module is not a Sphinx
    extension of its own.
    """
    _opengraph.add_auto_config_value(app, CONFIG_VALUE)
    # Must run before ``sphinxext.opengraph`` renders its tags at the default priority
    app.connect('html-page-context', _set_description, priority=400)


def _set_description(  # noqa: PLR0917
    app: Sphinx,
    pagename: str,  # noqa: ARG001
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    """Override ``og:description`` with the page's leading prose."""
    if doctree is None or not getattr(app.config, CONFIG_VALUE):
        return
    fields = _opengraph.page_fields(app, context)
    if fields is None or 'og:description' in fields:
        return

    length = _description_length(app, fields)
    description = _page_description(doctree, length)
    if not description:
        return

    fields['og:description'] = description
    if app.config.ogp_enable_meta_description and 'name="description"' not in context.get(
        'metatags', ''
    ):
        # ``sphinxext-opengraph`` only fills this in from its own parser, so write it
        # here to keep the plain description and ``og:description`` in agreement
        context['metatags'] += f'\n<meta name="description" content="{description}" />'


def _description_length(app: Sphinx, fields: dict[str, str]) -> int:
    """Return the description budget, honouring the per-page override."""
    try:
        return int(fields.get('ogp_description_length', app.config.ogp_description_length))
    except (TypeError, ValueError):
        return app.config.ogp_description_length


def _page_description(doctree: nodes.document, length: int) -> str:
    """Return the first ``length`` characters of a page's prose."""
    if length <= 0:
        return ''
    paragraphs: list[str] = []
    total = 0
    for paragraph in _prose_paragraphs(doctree):
        text = ' '.join(''.join(_prose_text(paragraph)).split())
        # A paragraph of nothing but download links leaves only its separators behind
        if not any(character.isalnum() for character in text):
            continue
        paragraphs.append(text)
        total += len(text) + 1
        if total > length:
            break
    return html.escape(_truncate(' '.join(paragraphs), length), quote=True)


def _prose_paragraphs(node: nodes.Node) -> Iterator[nodes.paragraph]:
    """Yield the page's prose paragraphs in document order."""
    if _is_skipped(node):
        return
    if isinstance(node, nodes.paragraph):
        yield node
        return
    for child in node.children:
        yield from _prose_paragraphs(child)


def _prose_text(node: nodes.Node) -> Iterator[str]:
    """Yield a paragraph's text, leaving out anything that is not prose."""
    if isinstance(node, nodes.Text):
        yield node.astext()
    elif not _is_skipped(node):
        for child in node.children:
            yield from _prose_text(child)


def _is_skipped(node: nodes.Node) -> bool:
    """Return whether a subtree is page furniture rather than prose."""
    if isinstance(node, nodes.Text):
        return True
    if isinstance(node, _SKIP_NODES):
        return True
    if isinstance(node, nodes.Admonition) and not isinstance(node, _ADMONITION_EXCEPTIONS):
        return True
    return bool(_SKIP_CLASSES.intersection(node.get('classes', ())))


def _truncate(text: str, length: int) -> str:
    """Shorten *text* to at most *length* characters, preferring a word boundary."""
    if len(text) <= length:
        return text
    truncated = text[: max(length - 3, 0)]
    head, separator, _ = truncated.rpartition(' ')
    # Only back up to the previous word when that still leaves a usable description
    if separator and len(head) * 2 >= length:
        truncated = head
    return truncated.rstrip().rstrip(',;:') + '...'
