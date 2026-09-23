"""Test that docstring/page sections render as real, hoistable headings.

Three independent mechanisms feed into this, none of which fail loudly if they
regress -- the section just silently reverts to an unlinkable rubric or
admonition, or disappears from the "on this page" navbar:

- ``_str_header`` patches numpydoc to emit real headings (not ``.. rubric::``) for
  sections like Examples, and ``hoist_docstring_sections`` lifts those out of the
  ``desc`` node they're generated inside, to page level.
- ``promote_seealso_admonitions`` additionally turns a literal ``.. seealso::``
  admonition -- written directly in a docstring instead of using numpydoc's own
  "See Also" syntax, as ``pyvista.examples.downloads`` does -- into a real section
  before the hoist above runs.
- ``doc/source/_templates/autosummary/class.rst`` renders "Methods" and
  "Attributes" as real page-level headings directly in the class page template,
  outside the docstring entirely.
"""

from __future__ import annotations

from pathlib import Path

from conftest import BUILD_HTML_DIR
import pytest

# A page whose docstring uses numpydoc's own "See Also" section syntax, and has
# "Methods"/"Attributes" sections rendered by the class page template.
CLASS_PAGE = 'pyvista.Plotter.html'

# A page whose docstring instead writes `.. seealso::` directly, in Examples.
RAW_SEEALSO_PAGE = 'pyvista.examples.downloads.download_bunny.html'

# A page whose raw `.. seealso::` is written well *before* References/Examples --
# unlike RAW_SEEALSO_PAGE's, which is already last. Catches promote_seealso_admonitions
# not repositioning the promoted section to match the others' order.
CELLTYPE_PAGE = 'pyvista.CellType.html'


def find_api_page(filename: str) -> Path:
    """Return a generated single-object API page.

    Fails rather than skips when the page is missing: skipping would silently
    stop testing the feature, and nobody would know to update the test.
    """
    page = next(Path(BUILD_HTML_DIR).rglob(filename), None)
    assert page is not None, (
        f'{filename} not found under {BUILD_HTML_DIR}. If the API doc layout changed, point '
        f'this test at another page with the section under test.'
    )
    return page


def assert_is_real_heading(html: str, name: str) -> None:
    """Assert ``name`` renders as a real ``<h2>`` heading, not a rubric."""
    assert f'<p class="rubric">{name}</p>' not in html
    assert f'<h2>{name}' in html


def heading_index(html: str, name: str) -> int:
    """Return the position of ``name``'s real heading in the page."""
    return html.index(f'<h2>{name}')


@pytest.mark.parametrize('page', [CLASS_PAGE, RAW_SEEALSO_PAGE])
def test_examples_is_hoisted_section(page: str):
    """Confirm "Examples" renders as a real heading on both kinds of page."""
    html = find_api_page(page).read_text(encoding='utf-8')
    assert_is_real_heading(html, 'Examples')


def _assert_see_also_is_hoisted_section(html: str) -> None:
    assert_is_real_heading(html, 'See Also')
    assert 'admonition seealso' not in html
    assert heading_index(html, 'Examples') < heading_index(html, 'See Also')
    # Both pages are heavily cross-referenced, so "Used In" should always be
    # present -- but don't let that assumption itself sink the ordering check.
    if '<h2>Used In' in html:
        assert heading_index(html, 'See Also') < heading_index(html, 'Used In')


def test_numpydoc_see_also_is_hoisted_section():
    """Confirm a numpydoc "See Also" section renders as a real heading."""
    html = find_api_page(CLASS_PAGE).read_text(encoding='utf-8')
    _assert_see_also_is_hoisted_section(html)


def test_raw_seealso_admonition_is_hoisted_section():
    """Confirm a literal ``.. seealso::`` written in a docstring is promoted too."""
    html = find_api_page(RAW_SEEALSO_PAGE).read_text(encoding='utf-8')
    _assert_see_also_is_hoisted_section(html)


def test_raw_seealso_written_before_examples_is_still_reordered_after():
    """Confirm a ``.. seealso::`` written before References/Examples still ends up after."""
    html = find_api_page(CELLTYPE_PAGE).read_text(encoding='utf-8')
    _assert_see_also_is_hoisted_section(html)
    assert heading_index(html, 'Examples') < heading_index(html, 'See Also')
    assert heading_index(html, 'See Also') < heading_index(html, 'Used In')


def test_methods_is_real_section():
    """Confirm "Methods" -- rendered by the class page template -- is a real heading."""
    html = find_api_page(CLASS_PAGE).read_text(encoding='utf-8')
    assert_is_real_heading(html, 'Methods')


def test_attributes_is_real_section():
    """Confirm "Attributes" -- rendered by the class page template -- is a real heading."""
    html = find_api_page(CLASS_PAGE).read_text(encoding='utf-8')
    assert_is_real_heading(html, 'Attributes')
