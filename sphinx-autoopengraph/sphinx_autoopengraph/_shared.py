"""Shared plumbing for this extension's ``sphinxext-opengraph`` integration.

Both halves of the integration -- the ``og:image`` support in
``sphinx_autoopengraph._image`` and the ``og:description`` support in
``sphinx_autoopengraph._description`` -- are meant to be transparent: a
project only has to enable both ``sphinx_autoopengraph`` and
``sphinxext.opengraph``, and set ``ogp_site_url``. This module holds the small
amount of machinery they share to make that work.

Enabling ``sphinx_autoopengraph`` is itself the opt-in, independent of any other
extension. Each of the two features can be turned off individually with
``autoopengraph_image`` / ``autoopengraph_description`` (both default ``True``);
a project that wants neither should not enable this extension at all.

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx

#: The extension this integrates with. Both features are no-ops without it.
EXTENSION = 'sphinxext.opengraph'


def is_enabled(app: Sphinx) -> bool:  # numpydoc ignore=RT01
    """Return whether ``sphinxext.opengraph`` is enabled for this build."""
    return EXTENSION in app.extensions


def page_fields(app: Sphinx, context: dict[str, Any]) -> dict[str, str] | None:
    """Return the per-page Open Graph field list, ready to be added to.

    ``sphinxext-opengraph`` reads its per-page overrides from the page's metadata
    (``context['meta']``), which is the same dict Sphinx stores in the build
    environment. A copy is installed in its place so that overrides added for one
    build of a page never leak into the next one, and so that
    ``sphinxext-opengraph`` popping entries out of it cannot corrupt the
    environment.

    Returns
    -------
    dict[str, str] | None
        The page's Open Graph fields, or ``None`` if the page opts out of Open
        Graph entirely.

    """
    if app.builder.name == 'epub':
        return None
    fields = dict(context.get('meta') or {})
    if 'ogp_disable' in fields:
        return None
    context['meta'] = fields
    return fields
