"""Test the 404 page that is served for any missing path."""

from __future__ import annotations

from pathlib import Path
import re

from conftest import BUILD_HTML_DIR

_ASSET_URL_RE = re.compile(r'<(?:script|link|img)\b[^>]*\b(?:src|href)="([^"]+)"')


def notfound_page() -> str:
    """Return the built 404 page."""
    return (Path(BUILD_HTML_DIR) / '404.html').read_text(encoding='utf-8')


def test_notfound_page_has_the_suggestion_container_and_script():
    page = notfound_page()

    assert '<div id="notfound"></div>' in page
    assert re.search(r'<script src="/_static/notfound\.js(?:\?v=\w+)?"></script>', page)
    for fallback in ('/search.html', '/api/index.html', '/examples/index.html'):
        assert f'<a href="{fallback}">' in page


def test_notfound_page_assets_resolve_from_any_depth():
    """The page is served for ``/version/x.y/...`` too, so no asset URL may be relative."""
    urls = _ASSET_URL_RE.findall(notfound_page())

    assert urls
    assert [url for url in urls if not (url.startswith('/') or '://' in url)] == []
