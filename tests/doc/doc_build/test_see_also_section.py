"""Test that "See Also" renders as a real, hoisted section, not an admonition.

`conf.py` patches numpydoc so its own "See Also" section renders as a heading
instead of a `.. seealso::` admonition, and separately promotes any literal
`.. seealso::` written directly in a docstring (as `pyvista.examples.downloads`
does) into the same kind of section. Neither fails loudly if it breaks -- the
section just silently reverts to an unlinkable admonition.
"""

from __future__ import annotations

from pathlib import Path

from conftest import BUILD_HTML_DIR

# A page whose docstring uses numpydoc's own "See Also" section syntax.
NUMPYDOC_SEE_ALSO_PAGE = 'pyvista.Plotter.html'

# A page whose docstring instead writes `.. seealso::` directly, in Examples.
RAW_SEEALSO_PAGE = 'pyvista.examples.downloads.download_bunny.html'


def find_api_page(filename: str) -> Path:
    """Return a generated single-object API page.

    Fails rather than skips when the page is missing: skipping would silently
    stop testing the feature, and nobody would know to update the test.
    """
    page = next(Path(BUILD_HTML_DIR).rglob(filename), None)
    assert page is not None, (
        f'{filename} not found under {BUILD_HTML_DIR}. If the API doc layout changed, point '
        f'this test at another single-object page with a "See Also" section.'
    )
    return page


def assert_see_also_is_hoisted_section(html: str) -> None:
    """Assert "See Also" is a real heading, ordered after Examples and before Used In."""
    assert 'admonition seealso' not in html
    assert '<h2>See Also' in html

    examples_idx = html.index('<h2>Examples')
    see_also_idx = html.index('<h2>See Also')
    assert examples_idx < see_also_idx

    used_in_idx = html.find('<h2>Used In')
    if used_in_idx != -1:
        assert see_also_idx < used_in_idx


def test_numpydoc_see_also_is_hoisted_section():
    """Confirm a numpydoc "See Also" section renders as a real heading."""
    html = find_api_page(NUMPYDOC_SEE_ALSO_PAGE).read_text(encoding='utf-8')
    assert_see_also_is_hoisted_section(html)


def test_raw_seealso_admonition_is_hoisted_section():
    """Confirm a literal ``.. seealso::`` written in a docstring is promoted too."""
    html = find_api_page(RAW_SEEALSO_PAGE).read_text(encoding='utf-8')
    assert_see_also_is_hoisted_section(html)
