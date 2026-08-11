"""Automatic Open Graph link previews for Sphinx documentation.

Add this extension on its own to give every page of a Sphinx project a real
`Open Graph <https://ogp.me>`_ link preview -- an image the page actually shows and
a description written from its own opening prose -- instead of one site-wide
default repeated on every page:

.. code-block:: python

    extensions = [
        'pyvista.ext.opengraph',
        'sphinxext.opengraph',
    ]

    ogp_site_url = 'https://docs.example.org/'

Nothing here is specific to PyVista or to :mod:`pyvista.ext.plot_directive`. Image
selection numbers whatever images a page has, regardless of source. Description
extraction works on any page of prose. The two extensions are independent --
enabling one does not enable the other -- but they are meant to be used together:
a project using the plot directive that wants this too has to add both. See
``pyvista.ext._opengraph_image`` and ``pyvista.ext._opengraph_description`` for
the two halves, and :ref:`opengraph_docs` for the user-facing documentation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyvista.ext import _opengraph_description
from pyvista.ext import _opengraph_image

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def setup(app: Sphinx) -> dict[str, bool]:
    """Set up PyVista's Open Graph integration.

    Returns
    -------
    dict[str, bool]
        Sphinx extension metadata.

    """
    _opengraph_image.setup(app)
    _opengraph_description.setup(app)
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
