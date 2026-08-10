"""Shared plumbing for PyVista's ``sphinxext-opengraph`` integration.

Both halves of the integration -- the ``og:image`` support in
:mod:`pyvista.ext.plot_directive` and the ``og:description`` support in
:mod:`pyvista.ext._opengraph_description` -- are meant to be transparent: a
project only has to enable ``sphinxext.opengraph`` and set ``ogp_site_url``.
This module holds the small amount of machinery they share to make that work.

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from sphinx.errors import ExtensionError

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.config import Config

#: The extension PyVista integrates with. Both features are no-ops without it.
EXTENSION = 'sphinxext.opengraph'


def add_auto_config_value(app: Sphinx, name: str) -> None:
    """Register a tri-state ``True``/``False``/``None`` opt-out config value.

    ``None`` means "follow ``sphinxext.opengraph``", which is resolved to a plain
    boolean once the configuration is known.
    """
    app.add_config_value(name, default=None, rebuild='html', types=frozenset({bool, type(None)}))
    app.connect('config-inited', lambda app, config: _resolve(app, config, name))


def _resolve(app: Sphinx, config: Config, name: str) -> None:
    """Replace a ``None`` config value with whether ``sphinxext.opengraph`` is enabled."""
    enabled = getattr(config, name)
    available = EXTENSION in app.extensions
    if enabled is None:
        setattr(config, name, available)
    elif enabled and not available:
        msg = (
            f"'{name} = True' requires the '{EXTENSION}' extension. Add '{EXTENSION}' to "
            f"'extensions' in your conf.py, or set '{name} = False'."
        )
        raise ExtensionError(msg)


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
