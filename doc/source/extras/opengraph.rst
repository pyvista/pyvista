.. _opengraph_docs:

Open Graph Link Previews
========================

When someone shares a link to your documentation -- on social media, in a chat
app, anywhere that unfurls links -- the preview card that appears is built from
the page's `Open Graph <https://ogp.me>`_ metadata. PyVista's documentation uses
`sphinx-autoopengraph <https://github.com/user27182/sphinx-autoopengraph>`_ for
this, so every page previews with an image it actually shows and a description
written from its own opening prose, instead of the same site-wide defaults on
every page alike.

It is a separate, independent extension -- nothing about it is specific to
PyVista or to :mod:`pyvista.ext.plot_directive`, and enabling one does not
enable the other, so a project using the plot directive that wants this too
has to add both:

.. code-block:: python

    extensions = [
        "pyvista.ext.plot_directive",
        "sphinx_autoopengraph",
        "sphinxext.opengraph",
    ]

    ogp_site_url = "https://docs.example.org/"

By default a page previews the first image it shows. Pick a different one with
the ``autoopengraph_thumbnail`` directive, which can go anywhere on the page
and renders nothing -- in a docstring, the natural place is the start of the
``Examples`` section:

.. code-block:: rst

   Examples
   --------
   .. autoopengraph_thumbnail:: 2

   Create a sphere.

   >>> import pyvista as pv
   >>> pv.Sphere().plot()

   Clip it, which is what this page is really about.

   >>> pv.Sphere().clip().plot()

See the `sphinx-autoopengraph README
<https://github.com/user27182/sphinx-autoopengraph#readme>`_ for full
documentation: configuration, the two independent ``autoopengraph_image`` /
``autoopengraph_description`` switches, and how Sphinx-Gallery examples are
handled.
