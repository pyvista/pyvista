"""Dump each page's lead paragraph to ``searchsummaries.json`` for search results."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx import addnodes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx

MAX_LENGTH = 300

# Containers whose paragraphs never summarize a page.
_SKIP_ANCESTORS = (
    nodes.Admonition,
    nodes.table,
    nodes.topic,
    nodes.sidebar,
    nodes.field_list,
    nodes.footnote,
    nodes.citation,
    nodes.figure,
    addnodes.versionmodified,
)


def _ancestors(node: nodes.Node) -> Iterator[nodes.Element]:
    """Yield the node's ancestors, innermost first."""
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def _prose(paragraph: nodes.paragraph) -> str:
    """Return the paragraph's text without download-link labels."""
    parts = [
        text.astext()
        for text in paragraph.findall(nodes.Text)
        if not any(
            isinstance(ancestor, addnodes.download_reference) for ancestor in _ancestors(text)
        )
    ]
    return re.sub(r'\s+', ' ', ''.join(parts)).strip()


def _lead_paragraph(doctree: nodes.document) -> str:
    """Return the document's first prose paragraph, or ``''`` if it has none."""
    for paragraph in doctree.findall(nodes.paragraph):
        # ``desc`` subclasses ``Admonition`` but holds the documented object's own summary.
        if any(
            isinstance(ancestor, _SKIP_ANCESTORS) and not isinstance(ancestor, addnodes.desc)
            for ancestor in _ancestors(paragraph)
        ):
            continue
        text = _prose(paragraph)
        if re.search('[A-Za-z]', text):
            return text[:MAX_LENGTH]
    return ''


def dump_search_summaries(app: Sphinx, exception: Exception | None) -> None:
    """Write ``searchsummaries.json`` mapping each docname to its lead paragraph.

    ``search_summaries.js`` renders search result snippets from this index so
    the search page does not have to fetch every matching page.
    """
    if exception is not None or app.builder.name not in ('html', 'dirhtml'):
        return
    summaries = {}
    for docname in sorted(app.env.all_docs):
        try:
            doctree = app.env.get_doctree(docname)
        except OSError:
            continue
        text = _lead_paragraph(doctree)
        if text:
            summaries[docname] = text
    path = Path(app.outdir) / 'searchsummaries.json'
    path.write_text(json.dumps(summaries, ensure_ascii=False), encoding='utf-8')
