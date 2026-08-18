"""Sanity checks for Open Graph link previews against the real documentation build."""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re

from conftest import HTML_DIR
import pytest

pytestmark = pytest.mark.needs_docs_build

# Same value as `ogp_site_url` in `conf.py`
OGP_SITE_URL = 'https://docs.pyvista.org/'

_META_TAG = re.compile(r'<meta\b[^>]*>')
_META_KEY = re.compile(r'\b(?:property|name)="([^"]+)"')
_META_CONTENT = re.compile(r'\bcontent="([^"]*)"')
_PAGE_IMAGE = re.compile(r'<img\b[^>]*\bsrc="[^"]*/_images/([^"]+)"')


def meta_tags(page: Path) -> dict[str, str]:
    """Return a built page's ``<meta>`` tags, keyed by ``property`` or ``name``."""
    tags: dict[str, str] = {}
    for tag in _META_TAG.findall(page.read_text(encoding='utf-8')):
        key = _META_KEY.search(tag)
        content = _META_CONTENT.search(tag)
        if key is not None and content is not None:
            tags.setdefault(key.group(1), html.unescape(content.group(1)))
    return tags


def page_images(page: Path) -> list[str]:
    """Return the filenames of the images a page shows, in the order it shows them."""
    images = _PAGE_IMAGE.findall(page.read_text(encoding='utf-8'))
    assert images, f'{page} shows no images'
    return images


@dataclass(frozen=True)
class OpenGraphPage:
    """A page checked for its Open Graph preview."""

    id: str
    path: str
    description: str
    #: One-based position of the expected preview among the images the page shows.
    #: Ignored when `image` is set.
    image_number: int = 1
    #: Exact expected `og:image` URL, for a page whose preview isn't one of its
    #: own images (e.g. the root page's `ogp_image` fallback).
    image: str | None = None


OPENGRAPH_PAGES = (
    OpenGraphPage(
        id='prose',
        # A hand-written page selecting its image with ``.. autoopengraph_thumbnail::``
        path='user-guide/what-is-a-mesh.html',
        description='In PyVista, a mesh is any spatially referenced information',
        image_number=7,
    ),
    OpenGraphPage(
        id='gallery',
        # A gallery example selecting its image with ``sphinx_gallery_thumbnail_number``
        path='examples/00-load/create_circular_arc.html',
        description='Generate arc geometry with pyvista.CircularArc()',
        image_number=2,
    ),
    OpenGraphPage(
        id='api',
        # ``.. autoopengraph_thumbnail::`` inside an Examples section numpydoc wraps
        # in ``.. pyvista-plot::`` on its own
        path='api/core/_autosummary/pyvista.ImageDataFilters.crop.html',
        description='Crop this image to remove points at its boundaries.',
        image_number=4,
    ),
    OpenGraphPage(
        id='autoenum-gallery',
        # Images are plain ``.. image::`` directives rather than the plot directive,
        # to confirm selection counts image nodes generically
        path='api/utilities/_autosummary/pyvista.CellType.html',
        description='Define types of cells.',
        image_number=13,
    ),
    OpenGraphPage(
        id='root',
        # Opts out of selecting one of its own images with
        # ``.. autoopengraph_thumbnail:: none``, so its preview is the site-wide
        # default rather than one of its own real content images below the fold
        path='index.html',
        description=(
            'PyVista is the foundational Python library for 3D visualization and mesh '
            'analysis in scientific computing and engineering.'
        ),
        image=f'{OGP_SITE_URL}_static/pyvista_banner_small.png',
    ),
)


@pytest.mark.parametrize('page', OPENGRAPH_PAGES, ids=lambda page: page.id)
def test_opengraph_description(page: OpenGraphPage):
    """Confirm the page's og:description matches what's expected."""
    path = Path(HTML_DIR) / page.path
    assert path.is_file(), f'{path} not found. Build the documentation first.'

    description = meta_tags(path).get('og:description')

    assert description is not None, f'{page.path} has no og:description'
    assert description.startswith(page.description)


@pytest.mark.parametrize('page', OPENGRAPH_PAGES, ids=lambda page: page.id)
def test_opengraph_image(page: OpenGraphPage):
    """Confirm the page's og:image matches what's expected."""
    path = Path(HTML_DIR) / page.path
    assert path.is_file(), f'{path} not found. Build the documentation first.'

    if page.image is not None:
        expected = page.image
    else:
        expected = f'{OGP_SITE_URL}_images/{page_images(path)[page.image_number - 1]}'

    assert meta_tags(path).get('og:image') == expected
