"""Automatic Open Graph link previews for Sphinx documentation.

Add this extension to give every page of a Sphinx project a real
`Open Graph <https://ogp.me>`_ link preview -- an image the page actually shows and
a description written from its own opening prose -- instead of one site-wide
default repeated on every page:

.. code-block:: python

    extensions = [
        'sphinx_autoopengraph',
        'sphinxext.opengraph',
    ]

    ogp_site_url = 'https://docs.example.org/'

Nothing here is specific to any one plot-generating extension: image selection
numbers whatever images a page has, regardless of source, and description
extraction works on any page of prose. It pairs well with things like PyVista's
``pyvista.ext.plot_directive``, but does not require it, or anything else beyond
plain Sphinx and ``sphinxext-opengraph``.

Both halves can be turned off individually with ``autoopengraph_image`` and
``autoopengraph_description``, which each default to ``True`` -- see
``sphinx_autoopengraph._image`` and ``sphinx_autoopengraph._description`` for
the two halves.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _description
from . import _image

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__version__ = '0.1.0.dev0'


def setup(app: Sphinx) -> dict[str, bool | str]:
    """Set up automatic Open Graph link previews.

    Returns
    -------
    dict[str, bool | str]
        Sphinx extension metadata.

    """
    _image.setup(app)
    _description.setup(app)
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
        'version': __version__,
    }
