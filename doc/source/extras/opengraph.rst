.. _opengraph_docs:

Open Graph Link Previews
=========================

When someone shares a link to your documentation -- on social media, in a chat
app, anywhere that unfurls links -- the preview card that appears is built from
the page's `Open Graph <https://ogp.me>`_ metadata. PyVista fills that metadata
in for you, so every page previews with an image it actually shows and a
description written from its own opening prose, instead of the site-wide
defaults you would otherwise get on every page alike.

Enabling
---------

Nothing here is specific to plotting, or to any particular directive. Enable it on
its own and tell ``sphinxext-opengraph`` where your documentation is published:

.. code-block:: python

    extensions = [
        "pyvista.ext.opengraph",
        "sphinxext.opengraph",
    ]

    ogp_site_url = "https://docs.example.org/"

Both extensions are required: ``pyvista.ext.opengraph`` is what chooses each
page's image and description, and
`sphinxext-opengraph <https://github.com/wpilibsuite/sphinxext-opengraph>`_ is
what writes the tags -- without it, PyVista does nothing here. Nothing else needs
configuring; a page's preview image is chosen from whatever images it has, from
any source, and its description from its own prose, whether or not the page has
any images at all.

If you also use :mod:`pyvista's plot directive <pyvista.ext.plot_directive>`, it
enables this extension for you -- you do not need to list both.

If you would rather keep one half of the integration and not the other, turn
either off individually:

.. code-block:: python

    pyvista_opengraph_image = False
    pyvista_opengraph_description = False

Left unset, both follow ``sphinxext.opengraph``: enabled whenever it is. Setting
either to ``True`` without ``sphinxext.opengraph`` in ``extensions`` is an error
rather than a silent no-op.

Choosing the Preview Image
---------------------------

By default a page previews the first image it shows. When the first image is
only scene-setting, pick a different one with the
``pyvista-opengraph-thumbnail`` directive:

.. code-block:: rst

   .. pyvista-opengraph-thumbnail:: 2

The argument is the one-based position of the image among *all* images on the
page, in the order they appear. It counts images, not files, so it is unaffected
by how generated filenames happen to be numbered. Negative values count
backwards from the last image.

The directive renders nothing and can go anywhere on the page, so you can put it
next to the code it refers to rather than at the top. In a docstring, the
natural place is the start of the ``Examples`` section:

.. code-block:: rst

   Examples
   --------
   .. pyvista-opengraph-thumbnail:: 2

   Create a sphere.

   >>> import pyvista as pv
   >>> pv.Sphere().plot()

   Clip it, which is what this page is really about.

   >>> pv.Sphere().clip().plot()

Two things to be aware of when using it:

- A page has a single ``<head>``, so it gets a single link preview -- section
  anchors cannot have their own. Using the directive twice on one page warns and
  keeps the first selection. This can happen without either docstring being
  wrong, on pages that document several objects at once with ``:members:``.
- Selecting an image the page does not have also warns, and falls back to the
  first image. That warning is suppressed while ``pyvista_plot_skip`` or
  ``pyvista_plot_skip_optional`` is enabled, since those builds deliberately
  render fewer images.

Pages with no images at all keep whatever site-wide ``ogp_image`` you have
configured.

Sphinx-Gallery Examples
------------------------

Gallery examples already have a thumbnail, and their preview always matches it,
so a shared link shows the same picture as the gallery. PyVista uses the full
resolution version of that image rather than the gallery's own thumbnail file,
which is too small to preview well.

Using ``pyvista-opengraph-thumbnail`` in a gallery example is an error. Select
the image with Sphinx-Gallery's own comment instead:

.. code-block:: python

   # sphinx_gallery_thumbnail_number = 2

Preview Descriptions
----------------------

PyVista describes each page with its leading paragraphs of real prose, up to
``ogp_description_length`` characters, skipping signatures, parameter tables,
code blocks, admonitions, download links, captions and navigation. For a
docstring that is its summary and the paragraphs following it; for a gallery
example it is the example's introduction. The plain ``description`` meta tag is
set to match, unless ``ogp_enable_meta_description`` is disabled.

This replaces ``sphinxext-opengraph``'s own description, which is built from
every piece of text on a page until it has enough characters. That works poorly
for generated pages: autodoc output is invisible to it, because Sphinx wraps
docstrings in a node it treats as an admonition and skips, so API pages end up
described by whatever sits outside that node -- or not described at all.
Sphinx-Gallery pages instead pick up download links, the timing footer and the
"Gallery generated by Sphinx-Gallery" signature.


API Reference
==============

.. automodule::
   pyvista.ext.opengraph
