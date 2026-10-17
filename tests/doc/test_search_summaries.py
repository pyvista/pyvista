"""Test the lead-paragraph extraction behind the search result snippets."""

from __future__ import annotations

from docutils import nodes
from docutils.utils import new_document
from sphinx import addnodes

from doc.source import make_search_summaries


def _paragraph(text: str) -> nodes.paragraph:
    return nodes.paragraph(text=text)


def _section(title: str, *children: nodes.Node) -> nodes.section:
    section = nodes.section()
    section += nodes.title(text=title)
    section.extend(children)
    return section


def _api_page(*body: nodes.Node) -> nodes.document:
    """Mimic an autosummary page: one ``desc`` whose sections were hoisted after it."""
    document = new_document('page')
    desc = addnodes.desc()
    content = addnodes.desc_content()
    content.extend(body)
    desc += content
    document += desc
    return document


def test_documented_object_summary_wins_over_hoisted_sections():
    document = _api_page(_paragraph('Transform this mesh with a 4x4 transform.'))
    document += _section('Examples', _paragraph('Translate a mesh by (50, 100, 200).'))

    assert make_search_summaries._lead_paragraph(document) == (
        'Transform this mesh with a 4x4 transform.'
    )


def test_admonitions_and_version_notes_inside_the_object_are_skipped():
    warning = nodes.warning()
    warning += _paragraph('Shear transformations are not supported.')
    changed = addnodes.versionmodified()
    changed += _paragraph('Changed in version 0.48.0: inplace must be specified.')
    document = _api_page(warning, changed, _paragraph('The real summary.'))

    assert make_search_summaries._lead_paragraph(document) == 'The real summary.'


def test_see_also_is_still_skipped():
    seealso = addnodes.seealso()
    seealso += _paragraph('Transform Describe linear transformations.')
    document = _api_page(seealso)
    document += _section('Notes', _paragraph('A note about the method.'))

    assert make_search_summaries._lead_paragraph(document) == 'A note about the method.'


def test_plain_page_uses_its_first_prose_paragraph():
    document = new_document('page')
    document += _section('Title', _paragraph('...'), _paragraph('First real sentence.'))

    assert make_search_summaries._lead_paragraph(document) == 'First real sentence.'
