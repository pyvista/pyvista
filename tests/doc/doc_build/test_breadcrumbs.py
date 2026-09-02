"""Test the breadcrumb trail in the article header of built pages."""

from __future__ import annotations

from pathlib import Path
import re

from conftest import BUILD_HTML_DIR

_BREADCRUMBS_RE = re.compile(r'<ul class="bd-breadcrumbs">(.*?)</ul>', re.DOTALL)
_ITEM_RE = re.compile(r'<li class="breadcrumb-item[^"]*">(.*?)</li>', re.DOTALL)
_TAG_RE = re.compile(r'<[^>]+>')


def breadcrumbs(path: Path) -> list[str]:
    """Return a page's breadcrumb labels, the home icon included as ``''``."""
    match = _BREADCRUMBS_RE.search(path.read_text())
    assert match is not None, f'{path} has no breadcrumb trail'
    return [_TAG_RE.sub('', item).strip() for item in _ITEM_RE.findall(match.group(1))]


def test_member_page_links_back_to_its_class():
    page = Path(BUILD_HTML_DIR) / 'api/utilities/_autosummary/pyvista.Transform.decompose.html'

    trail = breadcrumbs(page)

    assert trail[-2:] == ['Transform', 'Transform.decompose']


def test_gallery_example_belongs_to_its_gallery_not_a_tag_page():
    """Tag pages list every tagged example, which must not make them its parent."""
    page = Path(BUILD_HTML_DIR) / 'examples/01-filter/image_fft.html'

    trail = breadcrumbs(page)

    assert 'Examples' in trail
    assert not any(label.startswith('Tag') for label in trail)
